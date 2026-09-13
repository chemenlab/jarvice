"""A subscription-only main brain must never borrow an API model."""
from __future__ import annotations

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainTierConfig, JarvisConfig


def manager() -> BrainManager:
    cfg = JarvisConfig()
    cfg.brain.primary = "codex-subscription"
    cfg.brain.deep_brain = "gemini"
    cfg.brain.tool_model = BrainTierConfig(provider="gemini")
    return BrainManager(cfg, EventBus())


@pytest.mark.parametrize("level", ["fast", "deep", "code"])
def test_subscription_keeps_cli_default_and_never_falls_back(level):
    brain = manager()
    brain._dead_providers.add("codex-subscription")
    assert brain._build_fallback_chain(level) == [("codex-subscription", None)]


def test_delegated_tools_and_vision_keep_main_brain():
    brain = manager()
    foreign = [("gemini", "foreign-model")]
    assert brain._hoist_tool_model(foreign) == [("codex-subscription", None)]
    assert brain._lead_vision_chain(foreign) == [("codex-subscription", None)]


def test_tool_model_unavailability_is_blocked_not_api_fallback():
    brain = manager()
    seen = []

    def status(provider, model=None):
        seen.append(provider)
        return dict(provider=provider, model=model, ready=False,
                    reason="missing_credential", tools=False, vision=False)

    brain.tool_model_candidate_status = status
    verdict = brain.resolve_tool_model([("gemini", "foreign-model")])
    assert verdict["state"] == "blocked"
    assert verdict["configured_provider"] == "codex-subscription"
    assert seen == ["codex-subscription"]


def test_subscription_available_in_shared_memory_chat_catalog():
    from jarvis.agent_chat.catalog import rows_for
    from jarvis.agent_chat.service import resolve_runner
    from jarvis.ui.web.provider_spec import get_spec

    rows = {row.id: row for row in rows_for("jarvis")}
    assert rows["codex-subscription"].runner == "brain"
    assert resolve_runner("codex-subscription", surface="jarvis") == "brain"
    spec = get_spec("codex-subscription")
    assert spec.brain_switchable
    assert spec.secret_keys == ()
    assert spec.exclusive_main_brain
