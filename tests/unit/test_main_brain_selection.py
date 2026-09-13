"""Main subscription selection is one durable, reversible speech cutover."""
import tomllib

from jarvis.core import config_writer


def test_subscription_selection_is_atomic_and_retains_previous_voice(tmp_path, monkeypatch):
    path = tmp_path / "jarvis.toml"
    path.write_text('[brain]\nprimary="gemini"\n[voice]\nmode="realtime"\n'
                    'profile=""\n[stt]\nprovider="old-stt"\n'
                    '[tts]\nprovider="old-tts"\n[ui]\nlanguage="ru"\n')
    monkeypatch.setattr(config_writer, "_sync_brain_primary_drift_soll", lambda *a: None)  # i18n-allow: existing persistence identifier
    monkeypatch.setattr(config_writer, "_sync_stt_provider_drift_soll", lambda *a: None)  # i18n-allow: existing persistence identifier
    monkeypatch.setattr(config_writer, "_sync_tts_provider_drift_soll", lambda *a: None)  # i18n-allow: existing persistence identifier
    monkeypatch.setattr(config_writer, "_sync_brain_provider_model_drift_soll", lambda *a, **k: None)  # i18n-allow: existing persistence identifier
    config_writer.set_subscription_main_brain(path=path)
    doc = tomllib.loads(path.read_text())
    assert doc["brain"]["primary"] == "codex-subscription"
    assert doc["brain"]["tool_model"]["provider"] == "codex-subscription"
    assert doc["brain"]["providers"]["codex-subscription"]["model"] == "gpt-5.5"
    assert doc["voice"]["mode"] == "pipeline"
    assert doc["voice"]["profile"] == ""
    assert doc["stt"]["provider"] == "gemini-api"
    assert doc["tts"]["provider"] == "gemini-flash-tts"
    assert doc["ui"]["language"] == "ru"
    assert doc["subscription_main_restore"]["voice_mode"] == "realtime"
    config_writer.set_subscription_main_brain(path=path)
    assert tomllib.loads(path.read_text())["subscription_main_restore"] == doc["subscription_main_restore"]


def test_subscription_selection_persists_requested_model_in_all_brain_slots(tmp_path, monkeypatch):
    path = tmp_path / "jarvis.toml"
    path.write_text('[brain]\nprimary="gemini"\n[voice]\nmode="realtime"\n', encoding="utf-8")
    for name in ("_sync_brain_primary_drift_soll", "_sync_stt_provider_drift_soll", "_sync_tts_provider_drift_soll"):  # i18n-allow: existing persistence identifier
        monkeypatch.setattr(config_writer, name, lambda *a: None)
    syncs = []
    monkeypatch.setattr(config_writer, "_sync_brain_provider_model_drift_soll", lambda *a, **k: syncs.append((a, k)))  # i18n-allow: existing persistence identifier
    config_writer.set_subscription_main_brain(model="gpt-5.6-sol", path=path)
    assert syncs == [(("codex-subscription",), {"model": "gpt-5.6-sol", "deep_model": None})]
    doc = tomllib.loads(path.read_text())
    assert doc["brain"]["providers"]["codex-subscription"]["model"] == "gpt-5.6-sol"
    assert doc["brain"]["router"]["model"] == ""
    assert doc["voice"]["mode"] == "pipeline"

    from jarvis.brain.manager import BrainManager
    from jarvis.core.bus import EventBus
    from jarvis.core.config import load_config

    config_writer.set_brain_provider_model("codex-subscription", model="gpt-5.5", path=path)
    cfg = load_config(path)
    restarted = BrainManager.from_tier_config("router", config=cfg, bus=EventBus())
    assert restarted._fast_model("codex-subscription") == "gpt-5.5"
