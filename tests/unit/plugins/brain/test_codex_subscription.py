"""The subscription adapter suspends native calls at the Jarvis tool boundary."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from jarvis.brain.tool_use_loop import ToolUseLoop
from jarvis.brain.usage_meter import meter_brain
from jarvis.core.protocols import BrainMessage, BrainRequest, ImageBlock, ToolResult
from jarvis.plugins.brain.codex_subscription import CodexSubscriptionBrain
from jarvis.safety.tool_executor import VOICE_CONFIRM_SENTINEL


TOOL = {"name": "wiki-recall", "description": "Recall shared memory",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}


def call(*, rpc: int | str = 47, call_id: str = "memory-call", name: str = "wiki-recall",
         args: Any = None, thread: str = "thread-1") -> dict:
    return {"id": rpc, "method": "item/tool/call", "params": {
        "threadId": thread, "turnId": "turn-1", "callId": call_id, "tool": name,
        "arguments": {"query": "teapot"} if args is None else args,
    }}


def final(text: str = "Remembered.", *, thread: str = "thread-1") -> list[dict]:
    return [
        {"method": "item/agentMessage/delta", "params": {
            "threadId": thread, "turnId": "turn-1", "itemId": "answer", "delta": text}},
        {"method": "item/completed", "params": {"threadId": thread, "turnId": "turn-1",
            "item": {"id": "answer", "type": "agentMessage", "text": text}}},
        {"method": "turn/completed", "params": {"threadId": thread,
            "turn": {"id": "turn-1", "status": "completed"}}},
    ]


class FakeTransport:
    def __init__(self, events: list[dict | BaseException] | None = None) -> None:
        self.events = list(events or [])
        self.requests: list[tuple[str, dict]] = []
        self.responses: list[tuple[int | str, dict]] = []
        self.releases: list[tuple[str, str]] = []
        self.started = 0
        self.waiting = asyncio.Event()
        self.thread_count = 0
        self.release_error: BaseException | None = None
        self.start_error: BaseException | None = None

    async def ensure_started(self) -> None:
        self.started += 1

    async def request(self, method: str, params: dict) -> dict:
        self.requests.append((method, params))
        if method == "thread/start":
            self.thread_count += 1
            return {"thread": {"id": f"thread-{self.thread_count}"}}
        if method == "turn/start":
            if self.start_error:
                raise self.start_error
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(method)

    async def next_event(self, thread_id: str, timeout_s: float) -> dict:
        assert timeout_s > 0
        for index, event in enumerate(self.events):
            if isinstance(event, BaseException):
                raise self.events.pop(index)
            if event.get("params", {}).get("threadId", thread_id) == thread_id:
                return self.events.pop(index)
        self.waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def respond(self, request_id: int | str, result: dict) -> None:
        self.responses.append((request_id, result))

    async def release_thread(self, thread_id: str, turn_id: str = "") -> None:
        self.releases.append((thread_id, turn_id))
        if self.release_error:
            raise self.release_error

    async def close(self) -> None:
        pass

    async def probe(self, *, model: str) -> dict:
        return {"ready": True, "auth_mode": "chatgpt", "model": model,
                "transport": "codex-app-server", "version": "0.test"}


def request(**kwargs: Any) -> BrainRequest:
    return BrainRequest(messages=(BrainMessage("user", "Recall my teapot."),),
                        tools=(TOOL,), system="Jarvis shared memory: amber teapot.\n",
                        tool_loop_id="loop-1", **kwargs)


def with_result(req: BrainRequest, *, call_id: str = "memory-call", success: bool = True,
                images: tuple[ImageBlock, ...] = ()) -> BrainRequest:
    messages = req.messages + (
        BrainMessage("assistant", [{"type": "tool_use", "id": call_id,
                                     "name": "wiki-recall", "input": {"query": "teapot"}}]),
        BrainMessage("tool", [{"type": "tool_result", "tool_use_id": call_id,
                               "content": json.dumps({"success": success, "output": "amber",
                                                       "error": None if success else "denied"})}],
                     tool_call_id=call_id, name="wiki-recall"),
    )
    if images:
        messages += (BrainMessage("user", "Tool screenshot", images=images),)
    return replace(req, messages=messages)


async def collect(brain: Any, req: BrainRequest) -> list:
    return [delta async for delta in brain.complete(req)]


async def test_system_memory_history_and_images_reach_native_request_without_loss() -> None:
    transport = FakeTransport(final())
    brain = CodexSubscriptionBrain(transport=transport)
    messages = (
        BrainMessage("system", "Historical system note"),
        BrainMessage("user", "Original question", images=(ImageBlock("image/png", "cmVk"),)),
        BrainMessage("assistant", [{"type": "tool_use", "id": "old", "name": "wiki-recall", "input": {}}]),
        BrainMessage("tool", "Old memory", tool_call_id="old", name="wiki-recall"),
        BrainMessage("user", "Current question"),
    )
    req = replace(request(), messages=messages)
    chunks = await collect(brain, req)
    start = transport.requests[0][1]
    assert start["baseInstructions"] == req.system
    assert start["dynamicTools"] == [{"type": "function", "name": TOOL["name"],
        "description": TOOL["description"], "inputSchema": TOOL["input_schema"]}]
    assert start["model"] == "gpt-5.5"
    assert start["ephemeral"] is True
    assert start["approvalPolicy"] == "never" and start["sandbox"] == "read-only"
    assert start["environments"] == [] and start["selectedCapabilityRoots"] == []
    assert start["runtimeWorkspaceRoots"] == []
    assert start["allowProviderModelFallback"] is False
    inputs = transport.requests[1][1]["input"]
    history = json.loads(inputs[0]["text"])
    assert [entry["role"] for entry in history["messages"]] == [m.role for m in messages]
    assert history["messages"][2]["content"] == messages[2].content
    assert history["messages"][3]["tool_call_id"] == "old"
    assert inputs[1] == {"type": "image", "url": "data:image/png;base64,cmVk"}
    assert "".join(d.content or "" for d in chunks) == "Remembered."
    assert chunks[-1].finish_reason == "stop"
    assert transport.releases == [("thread-1", "turn-1")]


async def test_dynamic_rpc_waits_for_jarvis_result_then_resumes_same_turn() -> None:
    transport = FakeTransport([call(), *final()])
    brain = CodexSubscriptionBrain(transport=transport)
    req = request()
    chunks = await collect(brain, req)
    assert [d.tool_call for d in chunks if d.tool_call] == [
        {"id": "memory-call", "name": "wiki-recall", "input": {"query": "teapot"}}]
    assert chunks[-1].finish_reason == "tool_use"
    assert transport.responses == [] and transport.releases == []
    chunks = await collect(brain, with_result(req))
    assert transport.responses[0][0] == 47  # JSON-RPC id, not model callId.
    payload = transport.responses[0][1]
    assert payload["success"] is True
    assert json.loads(payload["contentItems"][0]["text"])["output"] == "amber"
    assert [method for method, _ in transport.requests] == ["thread/start", "turn/start"]
    assert chunks[-1].finish_reason == "stop"
    await brain.release_tool_loop("loop-1")
    assert transport.releases == [("thread-1", "turn-1")]


async def test_tool_images_and_late_user_steering_are_forwarded_without_old_images() -> None:
    transport = FakeTransport([call(), *final()])
    brain = CodexSubscriptionBrain(transport=transport)
    req = replace(request(), messages=(BrainMessage("user", "Read", images=(ImageBlock("image/png", "cmVk"),)),))
    await collect(brain, req)
    follow = with_result(req, success=False, images=(ImageBlock("image/png", "Ymx1ZQ=="),))
    follow = replace(follow, messages=follow.messages + (BrainMessage("user", "Reply in Russian now."),))
    await collect(brain, follow)
    result = transport.responses[0][1]
    assert result["success"] is False
    assert [c for c in result["contentItems"] if c["type"] == "inputImage"] == [
        {"type": "inputImage", "imageUrl": "data:image/png;base64,Ymx1ZQ=="}]
    assert "Reply in Russian now." in json.dumps(result)


async def test_aliases_preserve_exact_jarvis_names_and_avoid_collisions() -> None:
    transport = FakeTransport()
    brain = CodexSubscriptionBrain(transport=transport)
    names = ["mcp.gmail/read", "jarvis_tool_0", "mcp.gmail read", "wiki-recall"]
    req = replace(request(), tools=tuple({**TOOL, "name": name} for name in names))
    task = asyncio.create_task(collect(brain, req))
    await transport.waiting.wait()
    specs = transport.requests[0][1]["dynamicTools"]
    wire_names = [spec["name"] for spec in specs]
    assert len(set(wire_names)) == len(names)
    assert wire_names[1] == "jarvis_tool_0"
    assert all("." not in name and "/" not in name and " " not in name for name in wire_names)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    transport2 = FakeTransport([call(name=wire_names[0])])
    brain2 = CodexSubscriptionBrain(transport=transport2)
    chunks = await collect(brain2, req)
    assert next(d.tool_call for d in chunks if d.tool_call)["name"] == names[0]
    await brain2.release_tool_loop("loop-1")


async def test_sequential_native_calls_each_wait_for_their_own_result() -> None:
    transport = FakeTransport([call(), call(rpc="rpc-next", call_id="next"), *final()])
    brain = CodexSubscriptionBrain(transport=transport)
    req = request()
    await collect(brain, req)
    follow = with_result(req)
    chunks = await collect(brain, follow)
    assert next(d.tool_call for d in chunks if d.tool_call)["id"] == "next"
    assert len(transport.responses) == 1
    await collect(brain, with_result(follow, call_id="next"))
    assert [rpc for rpc, _ in transport.responses] == [47, "rpc-next"]


@pytest.mark.parametrize("event", [
    call(name="unadvertised-tool"), call(args="not an object"),
    {"method": "transport/error", "params": {"message": "native tool denied"}},
    {"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "failed"}}},
    RuntimeError("transport died"),
])
async def test_protocol_and_transport_failures_release_without_claiming_success(event: Any) -> None:
    transport = FakeTransport([event])
    brain = CodexSubscriptionBrain(transport=transport)
    with pytest.raises(RuntimeError):
        await collect(brain, request())
    assert transport.releases == [("thread-1", "turn-1")]


async def test_missing_result_cannot_resume_or_leak_a_suspended_rpc() -> None:
    transport = FakeTransport([call()])
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, request())
    with pytest.raises(RuntimeError, match="result"):
        await collect(brain, with_result(request(), call_id="wrong"))
    assert transport.responses == []
    assert transport.releases == [("thread-1", "turn-1")]


async def test_standalone_text_works_but_cannot_suspend_an_unscoped_tool_call() -> None:
    transport = FakeTransport(final())
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, replace(request(), tool_loop_id=None, tools=()))
    assert len(transport.releases) == 1
    transport = FakeTransport([call()])
    brain = CodexSubscriptionBrain(transport=transport)
    with pytest.raises(RuntimeError, match="tool_loop_id"):
        await collect(brain, replace(request(), tool_loop_id=None))
    assert len(transport.releases) == 1


async def test_concurrent_loops_have_independent_native_threads_and_cleanup() -> None:
    transport = FakeTransport([call(), call(thread="thread-2")])
    brain = CodexSubscriptionBrain(transport=transport)
    results = await asyncio.gather(collect(brain, request()),
        collect(brain, replace(request(), tool_loop_id="loop-2")))
    assert all(chunks[-1].finish_reason == "tool_use" for chunks in results)
    await brain.release_tool_loop("loop-1")
    assert transport.releases == [("thread-1", "turn-1")]
    await brain.release_tool_loop("loop-2")
    assert transport.releases == [("thread-1", "turn-1"), ("thread-2", "turn-1")]


@pytest.mark.parametrize("cleanup_error", [None, RuntimeError("cleanup failed"), asyncio.CancelledError()])
async def test_cancellation_keeps_its_identity_and_releases_native_thread(cleanup_error: Any) -> None:
    transport = FakeTransport()
    transport.release_error = cleanup_error
    brain = CodexSubscriptionBrain(transport=transport)
    task = asyncio.create_task(collect(brain, request()))
    await transport.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.releases == [("thread-1", "turn-1")]


async def test_turn_start_failure_still_releases_created_thread() -> None:
    transport = FakeTransport()
    transport.start_error = ValueError("start failed")
    transport.release_error = RuntimeError("cleanup failed")
    brain = CodexSubscriptionBrain(transport=transport)
    with pytest.raises(ValueError, match="start failed"):
        await collect(brain, request())
    assert transport.releases == [("thread-1", "")]


async def test_metered_provider_releases_pending_approval_through_tool_loop() -> None:
    class MemoryTool:
        name = TOOL["name"]
        description = TOOL["description"]
        schema = TOOL["input_schema"]

    class ConfirmExecutor:
        async def execute(self, tool: Any, args: Any, **kwargs: Any) -> ToolResult:
            return ToolResult(False, {"tool_name": tool.name, "trace_id": str(kwargs["trace_id"])},
                              error=VOICE_CONFIRM_SENTINEL)

    transport = FakeTransport([call()])
    brain = meter_brain(CodexSubscriptionBrain(transport=transport), "codex-subscription")
    loop = ToolUseLoop(brain, {TOOL["name"]: MemoryTool()}, ConfirmExecutor())
    result = await loop.run([BrainMessage("user", "Recall my teapot.")])
    assert result.finish_reason == "voice_confirm_pending"
    assert transport.responses == []
    assert transport.releases == [("thread-1", "turn-1")]


def test_capabilities_only_claim_vision_for_the_verified_model() -> None:
    brain = CodexSubscriptionBrain(transport=FakeTransport())
    assert brain.name == "codex-subscription"
    assert brain.model == "gpt-5.5"
    assert brain.context_window == 200_000
    assert brain.supports_vision and brain.native_system_prompt and brain.can_call_tools()
    assert not CodexSubscriptionBrain(model="gpt-6-astra", transport=FakeTransport()).supports_vision


async def test_probe_reports_actual_model_without_starting_a_turn() -> None:
    transport = FakeTransport()
    brain = CodexSubscriptionBrain(model="gpt-6-astra", transport=transport)
    status = await brain.probe()
    assert status["model"] == "gpt-6-astra"
    assert status["transport"] == "codex-app-server"
    assert transport.requests == []


async def test_changed_system_or_parallel_completion_cannot_corrupt_pending_context() -> None:
    transport = FakeTransport()
    brain = CodexSubscriptionBrain(transport=transport)
    task = asyncio.create_task(collect(brain, request()))
    await transport.waiting.wait()
    with pytest.raises(RuntimeError, match="Concurrent"):
        await collect(brain, request())
    assert transport.releases == []  # The first call still owns its context.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    transport = FakeTransport([call()])
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, request())
    with pytest.raises(RuntimeError, match="system"):
        await collect(brain, replace(with_result(request()), system="Different identity"))
    assert transport.responses == []
    assert transport.releases == [("thread-1", "turn-1")]


async def test_final_tool_budget_is_conveyed_to_codex_before_resuming() -> None:
    transport = FakeTransport([call(), *final()])
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, request())
    await collect(brain, replace(with_result(request()), tools=()))
    feedback = transport.responses[0][1]["contentItems"]
    assert any("No further tools" in item.get("text", "") for item in feedback)


async def test_reasoning_effort_unsupported_levels_map_to_verified_model_levels() -> None:
    transport = FakeTransport(final())
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, request(reasoning_effort="minimal"))
    assert transport.requests[1][1]["effort"] == "low"


async def test_unknown_model_images_fail_before_any_provider_request() -> None:
    transport = FakeTransport()
    brain = CodexSubscriptionBrain(model="gpt-6-astra", transport=transport)
    req = replace(request(), messages=(BrainMessage("user", "See", images=(ImageBlock("image/png", "cmVk"),)),))
    with pytest.raises(RuntimeError, match="Image support"):
        await collect(brain, req)
    assert transport.requests == []


async def test_closing_partially_consumed_stream_releases_created_thread() -> None:
    transport = FakeTransport(final())
    brain = CodexSubscriptionBrain(transport=transport)
    stream = brain.complete(request())
    assert (await anext(stream)).content == "Remembered."
    await stream.aclose()
    assert transport.releases == [("thread-1", "turn-1")]


async def test_usage_is_reported_once_with_uncached_input_count() -> None:
    transport = FakeTransport([{"method": "thread/tokenUsage/updated", "params": {
        "threadId": "thread-1", "tokenUsage": {"last": {
            "inputTokens": 200, "cachedInputTokens": 150, "outputTokens": 12}}}}, *final()])
    chunks = await collect(CodexSubscriptionBrain(transport=transport), request())
    assert [delta.usage for delta in chunks if delta.usage] == [
        {"input_tokens": 50, "cache_hit_tokens": 150, "output_tokens": 12}]


async def test_schema_errors_do_not_start_or_leak_a_native_thread() -> None:
    transport = FakeTransport()
    brain = CodexSubscriptionBrain(transport=transport)
    with pytest.raises(RuntimeError, match="Duplicate"):
        await collect(brain, replace(request(), tools=(TOOL, TOOL)))
    assert transport.requests == []


async def test_protocol_failure_has_a_safe_public_message() -> None:
    transport = FakeTransport([call(name="unadvertised-sensitive-value")])
    brain = CodexSubscriptionBrain(transport=transport)
    with pytest.raises(RuntimeError) as caught:
        await collect(brain, request())
    assert caught.value.public_message
    assert "unadvertised-sensitive-value" not in caught.value.public_message


async def test_retryable_native_error_waits_for_same_turn_to_recover() -> None:
    transport = FakeTransport([{"method": "error", "params": {
        "threadId": "thread-1", "turnId": "turn-1", "willRetry": True,
        "error": {"message": "Reconnecting"},
    }}, *final()])
    chunks = await collect(CodexSubscriptionBrain(transport=transport), request())
    assert "".join(delta.content or "" for delta in chunks) == "Remembered."
    assert transport.releases == [("thread-1", "turn-1")]


@pytest.mark.parametrize("will_retry", [False, None, "true"])
async def test_terminal_or_malformed_native_retry_flag_is_not_ignored(will_retry: Any) -> None:
    transport = FakeTransport([{"method": "error", "params": {
        "threadId": "thread-1", "turnId": "turn-1", "willRetry": will_retry,
        "error": {"message": "Failed"},
    }}])
    with pytest.raises(RuntimeError):
        await collect(CodexSubscriptionBrain(transport=transport), request())
    assert transport.releases == [("thread-1", "turn-1")]


async def test_verified_56_delivers_tool_image_before_resuming_pending_call():
    """Code-mode models need image feedback enqueued before the tool returns."""
    class ImageTransport(FakeTransport):
        async def request(self, method, params):
            if method == "turn/steer":
                assert not self.responses, "image must arrive before model continuation"
                self.requests.append((method, params))
                return {"turnId": "turn-1"}
            return await super().request(method, params)
    transport = ImageTransport([call(), *final("The tool image is blue.")])
    brain = CodexSubscriptionBrain(model="gpt-5.6-sol", transport=transport)
    req = request()
    await collect(brain, req)
    chunks = await collect(brain, with_result(req, images=(ImageBlock("image/png", "Ymx1ZQ=="),)))
    steer = [params for method, params in transport.requests if method == "turn/steer"]
    assert len(steer) == 1
    assert steer[0]["threadId"] == "thread-1"
    assert steer[0]["expectedTurnId"] == "turn-1"
    assert steer[0]["input"][1] == {"type": "image", "url": "data:image/png;base64,Ymx1ZQ=="}
    assert "untrusted tool output" in steer[0]["input"][0]["text"]
    assert "blue" in "".join(chunk.content or "" for chunk in chunks)
    assert len(transport.responses) == 1


async def test_retiring_model_finishes_pending_tool_then_closes_transport():
    class ClosingTransport(FakeTransport):
        closed = False
        async def close(self):
            self.closed = True
    transport = ClosingTransport([call(), *final()])
    brain = CodexSubscriptionBrain(transport=transport)
    req = request()
    await collect(brain, req)
    brain.retire()
    assert not transport.closed
    await collect(brain, with_result(req))
    assert transport.closed


async def test_retiring_idle_model_releases_its_process():
    class ClosingTransport(FakeTransport):
        closed = False
        async def close(self):
            self.closed = True
    transport = ClosingTransport(final())
    brain = CodexSubscriptionBrain(transport=transport)
    await collect(brain, request())
    brain.retire()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.closed
