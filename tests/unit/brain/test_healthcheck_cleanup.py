"""A model availability check must release its temporary provider resources."""

import asyncio

import pytest

from jarvis.brain.healthcheck import BrainHealthChecker
from jarvis.core.protocols import BrainDelta


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
async def test_probe_closes_stream_before_provider(outcome):
    events = []

    class Brain:
        async def complete(self, req):
            try:
                if outcome == "timeout":
                    await asyncio.sleep(10)
                if outcome == "failure":
                    raise ValueError("unavailable")
                yield BrainDelta(content="hi")
            finally:
                events.append("stream closed")

        async def close(self):
            events.append("provider closed")

    class Registry:
        def instantiate(self, provider, **kwargs):
            return Brain()

    result = await BrainHealthChecker(Registry()).probe("fixture", "model", timeout_s=0.01)
    assert result.ok is (outcome == "success")
    assert events == ["stream closed", "provider closed"]
