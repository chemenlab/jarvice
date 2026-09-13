"""Temporary memory providers release streams and owned CLI processes."""

import asyncio
from types import SimpleNamespace

import pytest

from jarvis.core.protocols import BrainDelta, BrainRequest
from jarvis.memory.wiki.provider_chain import complete_with_fallback


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "aggregate_error"])
async def test_memory_provider_is_closed_on_every_exit(outcome, monkeypatch):
    closed = []

    class Brain:
        async def complete(self, request):
            if outcome == "timeout":
                await asyncio.sleep(10)
            if outcome == "failure":
                raise ValueError("fixture failure")
            yield BrainDelta(content="fixture")

        async def close(self):
            closed.append(True)

    monkeypatch.setattr("jarvis.memory.wiki.curator_llm.instantiate_curator_brain", lambda *a, **k: Brain())

    async def aggregate(stream):
        return SimpleNamespace(text="".join([d.content async for d in stream]))

    def broken_aggregate(stream):
        raise ValueError("fixture aggregate failure")

    await complete_with_fallback(registry=object(), chain=[("fixture", "model")],
        request=BrainRequest(messages=()), timeout_s=0.01, label="fixture", record_health=False,
        aggregate=broken_aggregate if outcome == "aggregate_error" else aggregate)
    assert closed == [True]
