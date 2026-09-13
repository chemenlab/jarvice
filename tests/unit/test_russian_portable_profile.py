"""Portable profile writes nested wake settings and keeps paths machine-local."""
import tomllib
from pathlib import Path

from jarvis.core import config_writer as writer
from scripts.configure_russian_assistant import configure


def test_portable_russian_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(writer, "_config_soll_path", lambda: tmp_path / "absent-baseline.json")  # i18n-allow: existing persistence identifier
    monkeypatch.setattr(writer, "_set_user_env_var", lambda *_args: None)
    path = tmp_path / "jarvis.toml"
    configure(path, tmp_path / "wake")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config["brain"]["providers"]["codex-subscription"]["model"] == "gpt-5.6-sol"
    assert config["brain"]["router"]["model"] == ""
    assert config["trigger"]["wake_word"]["phrase"] == "Привет, Джарвис"
    assert config["trigger"]["wake_word"]["language"] == "ru"
    assert config["trigger"]["wake_word"]["engine"] == "stt_match"
    assert config["sessions"]["retention_days"] == 0
    assert config["voice"]["mode"] == "pipeline"
    assert config["stt"]["provider"] == "gemini-api"
    assert config["tts"]["provider"] == "gemini-flash-tts"
    assert "token" not in config and "api_key" not in config
