"""GET/PUT /api/settings/ui-language — the interface (display) language backend.

This gives the formerly frontend-only localStorage UI language a backend home so
a voice command / the Control API can change it and the open UI switches live
(a UiLanguageChanged event is broadcast over /ws). JARVIS_CONFIG points the
writer at a temp file so the real jarvis.toml is never touched.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jarvis.core import config as cfg_mod
from jarvis.core.bus import EventBus
from jarvis.core.events import UiLanguageChanged
from jarvis.ui.web.settings_routes import router as settings_router


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    config_file = tmp_path / "jarvis.toml"
    config_file.write_text('[ui]\nlanguage = "en"\n', encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG", str(config_file))

    bus = EventBus()
    captured: list[UiLanguageChanged] = []

    async def _capture(ev: UiLanguageChanged) -> None:
        captured.append(ev)

    bus.subscribe(UiLanguageChanged, _capture)

    app = FastAPI()
    app.state.bus = bus
    app.state.config = cfg_mod.load_config()
    app.include_router(settings_router)
    return TestClient(app), config_file, captured


def test_get_returns_current_ui_language(ctx) -> None:
    tc, _, _ = ctx
    body = tc.get("/api/settings/ui-language").json()
    assert body["language"] == "en"
    assert set(body["options"]) == {"en", "de", "es", "ru"}


@pytest.mark.parametrize("language", ["de", "ru"])
def test_put_persists_and_broadcasts(ctx, language) -> None:
    tc, config_file, captured = ctx
    before = cfg_mod.load_config()
    res = tc.put("/api/settings/ui-language", json={"language": language})
    assert res.status_code == 200, res.text
    assert res.json() == {"ok": True, "language": language, "persisted": True}
    # Landed in the env-pointed config file (NOT the real jarvis.toml).
    assert f'language = "{language}"' in config_file.read_text(encoding="utf-8")
    # GET now reflects it.
    assert tc.get("/api/settings/ui-language").json()["language"] == language
    # And it was broadcast so every open frontend switches live.
    assert any(ev.language == language for ev in captured)
    cfg_mod.clear_config_cache()
    after = cfg_mod.load_config()
    assert after.ui.language == language
    assert after.brain == before.brain
    assert after.voice == before.voice
    assert after.stt == before.stt
    assert after.tts == before.tts


def test_put_rejects_unknown_language(ctx) -> None:
    tc, _, _ = ctx
    assert tc.put("/api/settings/ui-language", json={"language": "zh"}).status_code == 400
