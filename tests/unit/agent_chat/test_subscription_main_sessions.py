"""Main-brain subscription selection carries old text chats with their history."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jarvis.agent_chat import runner_brain, service
from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from tests.fakes.fake_brain_manager import FakeBrainManager


async def test_composer_catalog_exposes_live_main_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from jarvis.ui.web.agent_chat_routes import get_catalog

    fake = FakeBrainManager(fast_model="gpt-5.6-sol")
    fake.active_provider = "codex-subscription"
    monkeypatch.setattr(runner_brain, "brain_manager", lambda: fake)
    svc = AgentChatService(AgentChatStore(":memory:"))
    monkeypatch.setattr("jarvis.ui.web.agent_chat_routes._service", lambda request: svc)
    catalog = await get_catalog(SimpleNamespace(), surface="jarvis")
    assert catalog["main_selection"] == {"provider": "codex-subscription", "model": "gpt-5.6-sol"}
    row = next(p for p in catalog["providers"] if p["id"] == "codex-subscription")
    assert row["models_source"] == "live"
    assert row["default_model"] == "gpt-5.6-sol"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.core.config.get_provider_secret", lambda *_: None)


async def finish(queue: asyncio.Queue) -> list[dict]:
    events = []
    async with asyncio.timeout(5):
        while True:
            event = await queue.get()
            events.append(event)
            if event["kind"] == "turn_finished":
                return events


@pytest.mark.parametrize("surface", ["jarvis", "local-models", "society"])
async def test_old_brain_surface_moves_to_main_before_turn_and_preserves_history(
    surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBrainManager(fast_model="gpt-5.5")
    fake.active_provider = "codex-subscription"
    monkeypatch.setattr(runner_brain, "brain_manager", lambda: fake)
    monkeypatch.setattr(runner_brain, "_agent_secret", lambda *_: None)
    store = AgentChatStore(":memory:")
    svc = AgentChatService(store)
    session = store.create_session(provider="gemini", model="gemini-old", effort="high",
        cwd=str(tmp_path), title="Old test chat", permission_mode="ask", surface=surface)
    old_event = store.append_event(session.session_id, make_event("user_message", {"text": "My code is amber."}))
    store.append_event(session.session_id, make_event("assistant_text", {"message_id": "old", "text": "Stored."}))
    # Use the real brain runner for the primary Jarvis surface. Other kits can
    # bring unrelated setup/agent context; inspect their actual persisted handle.
    seen = []
    if surface != "jarvis":
        async def run(handle, text, **kwargs):
            seen.append(handle)
            await handle.emit(make_event("turn_finished", {"status": "done"}))
        monkeypatch.setattr(service, "run_brain_turn", run)
    queue = svc.subscribe(session.session_id)
    await svc.send(session.session_id, "What is my code?")
    events = await finish(queue)
    patched = store.get_session(session.session_id)
    assert patched is not None
    assert (patched.provider, patched.model) == ("codex-subscription", "gpt-5.5")
    assert patched.title == "Old test chat" and patched.cwd == str(tmp_path)
    assert patched.permission_mode == "ask"
    assert store.list_events(session.session_id)[0] == old_event
    update = next(e for e in events if e["kind"] == "session_updated")
    start = next(e for e in events if e["kind"] == "turn_started")
    assert update["seq"] < start["seq"]
    assert update["payload"]["provider"] == start["payload"]["provider"] == patched.provider
    assert update["payload"]["model"] == start["payload"]["model"] == patched.model
    if surface == "jarvis":
        override = fake.calls[0][1]["turn_override"]
        assert (override.provider, override.model) == (patched.provider, patched.model)
        assert override.credential_scope is None  # Same provider cache as voice.
        assert any(m.content == "My code is amber." for m in fake.calls[0][1]["history_override"])
    else:
        assert seen[0].session.provider == patched.provider


async def test_detached_agent_cli_session_keeps_its_own_provider_and_vendor_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBrainManager(fast_model="gpt-5.5")
    fake.active_provider = "codex-subscription"
    monkeypatch.setattr(runner_brain, "brain_manager", lambda: fake)
    store = AgentChatStore(":memory:")
    svc = AgentChatService(store)
    session = svc.create_session(provider="openai-codex", model="gpt-6-astra", cwd=str(tmp_path), surface="agent")
    store.update_session(session.session_id, vendor_session="vendor-old")
    seen = []
    async def run(handle, text, runner, **kwargs):
        seen.append(handle.session)
        await handle.emit(make_event("turn_finished", {"status": "done"}))
        return None
    monkeypatch.setattr(service, "run_cli_turn", run)
    queue = svc.subscribe(session.session_id)
    await svc.send(session.session_id, "Test")
    events = await finish(queue)
    assert seen[0].provider == "openai-codex" and seen[0].model == "gpt-6-astra"
    assert seen[0].vendor_session == "vendor-old"
    assert not any(e["kind"] == "session_updated" for e in events)


async def test_nonexclusive_main_keeps_the_chat_pick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBrainManager()
    fake.active_provider = "gemini"
    monkeypatch.setattr(runner_brain, "brain_manager", lambda: fake)
    monkeypatch.setattr(runner_brain, "_agent_secret", lambda *_: None)
    svc = AgentChatService(AgentChatStore(":memory:"))
    session = svc.create_session(provider="openai", model="chat-pick", cwd=str(tmp_path), surface="jarvis")
    queue = svc.subscribe(session.session_id)
    await svc.send(session.session_id, "Test")
    events = await finish(queue)
    assert fake.calls[0][1]["turn_override"].provider == "openai"
    assert not any(e["kind"] == "session_updated" for e in events)


def test_runner_refuses_stale_selection_if_main_changed_after_session_normalization(tmp_path: Path) -> None:
    fake = FakeBrainManager(fast_model="gpt-5.5")
    fake.active_provider = "codex-subscription"
    session = AgentChatStore(":memory:").create_session(provider="gemini", model="old", effort="high",
        cwd=str(tmp_path), surface="jarvis")
    with pytest.raises(ValueError, match="main brain"):
        runner_brain.build_override(session, fake, stance="ask", cwd=tmp_path, ref="test")
