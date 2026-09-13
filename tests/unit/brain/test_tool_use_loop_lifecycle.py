"""A suspended provider belongs to one run, through results and every exit."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from jarvis.brain.loop_control import LoopControl, VerifyOutcome
from jarvis.brain.tool_use_loop import ToolUseLoop
from jarvis.core.protocols import BrainDelta, BrainMessage, BrainRequest, ToolResult
from jarvis.safety.tool_executor import VOICE_CONFIRM_SENTINEL


class ReadTool:
    name = "wiki-list"
    description = "Read the test value"
    schema: dict[str, Any] = {"type": "object", "properties": {}}


class TestExecutor:
    __test__ = False

    def __init__(self, *, defer: bool = False, pause: bool = False) -> None:
        self.defer = defer
        self.pause = pause
        self.started = asyncio.Event()
        self.calls = 0

    async def execute(self, tool: Any, args: Any, **kwargs: Any) -> ToolResult:
        self.calls += 1
        self.started.set()
        if self.pause:
            await asyncio.Event().wait()
        if self.defer:
            return ToolResult(
                success=False,
                output={"tool_name": tool.name, "trace_id": str(kwargs["trace_id"])},
                error=VOICE_CONFIRM_SENTINEL,
            )
        return ToolResult(success=True, output="amber teapot")


class SuspendedBrain:
    """Like app-server, retain a request until its matching result or release."""

    def __init__(
        self, *, concurrent: bool = False, resume_error: Exception | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        self.requests: list[BrainRequest] = []
        self.pending: dict[str, str] = {}
        self.released: list[str] = []
        self.concurrent = concurrent
        self.both_started = asyncio.Event()
        self.start_count = 0
        self.resume_error = resume_error
        self.cleanup_error = cleanup_error

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        loop_id = getattr(req, "tool_loop_id", None)
        assert isinstance(loop_id, str) and loop_id, "missing provider continuation scope"
        self.requests.append(req)
        results = [message for message in req.messages if message.role == "tool"]
        if not results:
            assert loop_id not in self.pending, "concurrent runs reused a continuation"
            call_id = f"call-{loop_id}"
            self.pending[loop_id] = call_id
            if self.concurrent:
                self.start_count += 1
                if self.start_count == 2:
                    self.both_started.set()
                await self.both_started.wait()
            yield BrainDelta(tool_call={"id": call_id, "name": "wiki-list", "input": {}})
            yield BrainDelta(finish_reason="tool_use")
            return
        assert results[-1].tool_call_id == self.pending[loop_id]
        if self.resume_error:
            raise self.resume_error
        payload = json.loads(results[-1].content[0]["content"])
        yield BrainDelta(content=payload["output"])
        yield BrainDelta(finish_reason="stop")

    async def release_tool_loop(self, loop_id: str) -> None:
        await asyncio.sleep(0)
        self.released.append(loop_id)
        self.pending.pop(loop_id, None)
        if self.cleanup_error:
            raise self.cleanup_error


def make_loop(brain: Any, executor: TestExecutor | None = None, **kwargs: Any) -> ToolUseLoop:
    return ToolUseLoop(brain, {"wiki-list": ReadTool()}, executor or TestExecutor(), **kwargs)


async def test_tool_result_resumes_same_scope_and_success_releases_it() -> None:
    brain = SuspendedBrain()
    result = await make_loop(brain).run([], user_utterance="list my wiki")

    assert result.text == "amber teapot"
    assert len(brain.requests) == 2
    assert brain.requests[0].tool_loop_id == brain.requests[1].tool_loop_id
    assert brain.released == [brain.requests[0].tool_loop_id]
    assert not brain.pending


async def test_concurrent_runs_on_one_loop_cannot_share_provider_continuations() -> None:
    brain = SuspendedBrain(concurrent=True)
    loop = make_loop(brain)
    first, second = await asyncio.wait_for(asyncio.gather(
        loop.run([], user_utterance="list my wiki"),
        loop.run([], user_utterance="list my wiki"),
    ), timeout=2)

    assert first.text == second.text == "amber teapot"
    assert len(set(brain.released)) == 2
    assert len(brain.released) == 2
    assert not brain.pending


async def test_deferred_approval_releases_suspended_call_without_resuming_provider() -> None:
    brain = SuspendedBrain()
    result = await make_loop(brain, TestExecutor(defer=True)).run(
        [], user_utterance="list my wiki", voice_confirm=True,
    )

    assert result.finish_reason == "voice_confirm_pending"
    assert len(brain.requests) == 1
    assert brain.released == [brain.requests[0].tool_loop_id]
    assert not brain.pending


@pytest.mark.parametrize(
    "cleanup_error", [None, RuntimeError("cleanup failed"), asyncio.CancelledError()],
)
async def test_cancel_during_tool_execution_releases_suspended_call(
    cleanup_error: BaseException | None,
) -> None:
    brain = SuspendedBrain(cleanup_error=cleanup_error)
    executor = TestExecutor(pause=True)
    task = asyncio.create_task(make_loop(brain, executor).run([], user_utterance="list my wiki"))
    await asyncio.wait_for(executor.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert brain.released == [brain.requests[0].tool_loop_id]
    assert not brain.pending


async def test_tool_owned_response_releases_without_another_provider_round() -> None:
    class SuppressingRead(ReadTool):
        suppress_response = True

    brain = SuspendedBrain()
    loop = ToolUseLoop(brain, {"wiki-list": SuppressingRead()}, TestExecutor())
    result = await loop.run([], user_utterance="list my wiki")

    assert result.text == "amber teapot"
    assert result.finish_reason == "suppress_response"
    assert len(brain.requests) == 1
    assert brain.released == [brain.requests[0].tool_loop_id]
    assert not brain.pending


@pytest.mark.parametrize(
    "cleanup_error", [RuntimeError("cleanup failed"), asyncio.CancelledError()],
)
async def test_provider_error_survives_cleanup_failure(cleanup_error: BaseException) -> None:
    original = RuntimeError("provider failed while suspended")
    brain = SuspendedBrain(resume_error=original, cleanup_error=cleanup_error)

    with pytest.raises(RuntimeError) as raised:
        await make_loop(brain).run([], user_utterance="list my wiki")
    assert raised.value is original
    assert len(brain.released) == 1
    assert not brain.pending


async def test_cleanup_error_does_not_replace_successful_answer() -> None:
    brain = SuspendedBrain(cleanup_error=RuntimeError("cleanup failed"))
    result = await make_loop(brain).run([], user_utterance="list my wiki")
    assert result.text == "amber teapot"
    assert len(brain.released) == 1


async def test_cancel_from_cleanup_remains_cancellation_without_prior_error() -> None:
    brain = SuspendedBrain(cleanup_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await make_loop(brain).run([], user_utterance="list my wiki")
    assert len(brain.released) == 1


async def test_legacy_provider_without_release_hook_keeps_working() -> None:
    class LegacyBrain:
        async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
            yield BrainDelta(content="legacy answer", finish_reason="stop")

    result = await make_loop(LegacyBrain()).run([])
    assert result.text == "legacy answer"


async def test_verifier_request_keeps_the_run_scope() -> None:
    class AnswerBrain:
        def __init__(self) -> None:
            self.requests: list[BrainRequest] = []
            self.released: list[str] = []

        async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
            self.requests.append(req)
            yield BrainDelta(content="answer", finish_reason="stop")

        async def release_tool_loop(self, loop_id: str) -> None:
            self.released.append(loop_id)

    async def verify(request: Any, ask: Any) -> VerifyOutcome:
        await ask("Check evidence", "Does this answer follow?")
        return VerifyOutcome(accepted=True)

    brain = AnswerBrain()
    await make_loop(brain, loop_control=LoopControl(verify=verify)).run(
        [BrainMessage(role="user", content="hello")],
    )
    assert len(brain.requests) == 2
    loop_id = getattr(brain.requests[0], "tool_loop_id", None)
    assert loop_id
    assert brain.requests[1].tool_loop_id == loop_id
    assert brain.released == [loop_id]
