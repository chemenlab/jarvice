"""Selection validates the real subscription transport before persisting."""
import asyncio
import time
from types import SimpleNamespace

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jarvis.core.config import JarvisConfig
from jarvis.ui.web import main_brain_routes as routes


@pytest.fixture(autouse=True)
def reset_probe_cache(monkeypatch):
    monkeypatch.setattr(routes, "_PROBE_CACHE", None, raising=False)
    monkeypatch.setattr(routes, "_PROBE_TASK", None, raising=False)


def client(monkeypatch, ready=True):
    app = FastAPI()
    app.state.config = JarvisConfig()
    app.state.config.voice.mode = "pipeline"
    app.state.brain = SimpleNamespace(active_provider="gemini")
    app.include_router(routes.router)

    async def probe(*, force=False):
        return dict(ready=ready, auth_mode="chatgpt" if ready else None,
                    transport="codex-app-server", model="gpt-5.5", version="0.153.4",
                    speech_key_available=True)

    monkeypatch.setattr(routes, "probe_subscription", probe)
    monkeypatch.setattr(routes, "speech_key_available", lambda: True)
    return TestClient(app)


def test_status_distinguishes_active_brain_from_voice(monkeypatch):
    response = client(monkeypatch).get("/api/settings/main-brain")
    assert response.status_code == 200
    body = response.json()
    assert body["active_provider"] == "gemini"
    assert body["codex"] is None
    assert body["voice"]["speech_key_available"] is None
    assert body["voice"]["mode"] == "pipeline"


def test_activation_reports_restart_and_writes_only_after_validation(monkeypatch):
    writes = []
    monkeypatch.setattr(routes, "persist_subscription", lambda model: writes.append(True))
    response = client(monkeypatch).put("/api/settings/main-brain", json={"provider": "codex-subscription"})
    assert response.status_code == 200
    assert response.json()["requires_restart"] is True
    assert writes == [True]


def test_missing_subscription_never_changes_selection(monkeypatch):
    monkeypatch.setattr(routes, "persist_subscription", lambda model: pytest.fail("unexpected write"))
    response = client(monkeypatch, ready=False).put("/api/settings/main-brain", json={"provider": "codex-subscription"})
    assert response.status_code == 409


def test_existing_speech_only_pipeline_can_switch_the_shared_brain_live(monkeypatch):
    c = client(monkeypatch)
    cfg = c.app.state.config
    cfg.stt.provider = "gemini-api"
    cfg.tts.provider = "gemini-flash-tts"
    cfg.voice.profile = ""
    brain = c.app.state.brain

    async def switch(provider, persist=False):
        brain.active_provider = provider

    brain.switch = switch
    monkeypatch.setattr(routes, "persist_subscription", lambda model: None)
    response = c.put("/api/settings/main-brain", json={"provider": "codex-subscription"})
    assert response.status_code == 200
    assert response.json()["applied_live"] is True
    assert response.json()["requires_restart"] is False
    assert cfg.brain.primary == "codex-subscription"


def test_get_does_not_probe_transport_or_keychain(monkeypatch):
    c = client(monkeypatch)
    async def forbidden():
        pytest.fail("GET must not start a Codex process")
    monkeypatch.setattr(routes, "probe_subscription", forbidden)
    monkeypatch.setattr(routes, "speech_key_available", lambda: pytest.fail("GET must not query Keychain"))
    response = c.get("/api/settings/main-brain")
    assert response.status_code == 200
    assert response.json()["active_provider"] == "gemini"


def test_post_probe_has_explicit_independent_result(monkeypatch):
    c = client(monkeypatch)
    response = c.post("/api/settings/main-brain/probe")
    assert response.status_code == 200
    assert response.json()["auth_mode"] == "chatgpt"
    assert response.json()["speech_key_available"] is True


async def test_probe_singleflight_survives_one_cancelled_waiter_and_caches(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def perform():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"ready": True, "auth_mode": "chatgpt", "speech_key_available": True}
    monkeypatch.setattr(routes, "_perform_subscription_probe", perform, raising=False)
    first = asyncio.create_task(routes.probe_subscription())
    await started.wait()
    second = asyncio.create_task(routes.probe_subscription())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert (await second)["ready"] is True
    assert (await routes.probe_subscription())["auth_mode"] == "chatgpt"
    assert calls == 1
    # Explicit refresh skips a completed cache, but still shares an active task.
    await routes.probe_subscription(force=True)
    assert calls == 2


def test_get_reports_only_fresh_cached_auth_and_never_infers_it_from_active(monkeypatch):
    c = client(monkeypatch)
    c.app.state.brain.active_provider = "codex-subscription"
    cached = {"ready": False, "auth_mode": None, "speech_key_available": False}
    monkeypatch.setattr(routes, "_PROBE_CACHE", (time.monotonic(), cached), raising=False)
    body = c.get("/api/settings/main-brain").json()
    assert body["codex"]["ready"] is False
    assert body["voice"]["speech_key_available"] is False
    monkeypatch.setattr(routes, "_PROBE_CACHE", (time.monotonic() - 61, cached))
    assert c.get("/api/settings/main-brain").json()["codex"] is None


def test_activation_keeps_selected_56_and_reports_model_in_fast_status(monkeypatch):
    from jarvis.core.config import BrainProviderConfig
    c = client(monkeypatch)
    c.app.state.config.brain.providers["codex-subscription"] = BrainProviderConfig(model="gpt-5.6-sol")
    writes = []
    monkeypatch.setattr(routes, "persist_subscription", lambda model: writes.append(model))
    response = c.put("/api/settings/main-brain", json={"provider":"codex-subscription"})
    assert response.status_code == 200
    assert writes == ["gpt-5.6-sol"]
    c.app.state.brain.active_provider = "codex-subscription"
    assert c.get("/api/settings/main-brain").json()["active_model"] == "gpt-5.6-sol"
