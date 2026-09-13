"""A subscription main brain never constructs a second API LLM for voice prose."""

from __future__ import annotations

import pytest

from jarvis.brain import factory
from jarvis.brain.ack_brain.providers import REGISTRY
from jarvis.core.config import JarvisConfig


@pytest.mark.parametrize("builder", [
    factory.build_ack_brain,
    factory.build_spawn_announcer,
    factory.build_readback_composer,
])
def test_voice_prose_builders_do_not_create_api_provider_for_codex_main(
    monkeypatch, builder
) -> None:
    cfg = JarvisConfig()
    cfg.brain.primary = "codex-subscription"
    cfg.ack_brain.provider = "gemini"
    cfg.ack_brain.enabled = True
    cfg.ack_brain.preamble_enabled = True
    cfg.ack_brain.spawn_announcements = True
    api_providers: list[object] = []

    class APIProvider:
        def __init__(self, config):
            api_providers.append(config)

    monkeypatch.setitem(REGISTRY, "gemini", APIProvider)

    built = builder(cfg)

    assert api_providers == []
    if builder is factory.build_ack_brain:
        assert built is None
    elif builder is factory.build_readback_composer:
        assert built.has_llm is False
