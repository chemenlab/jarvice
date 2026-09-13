"""Local Finder launches keep their installation's isolated paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.setup import macos_launcher_entry as entry


@pytest.fixture
def local_install(tmp_path: Path, monkeypatch):
    """Exercise the entry point without importing or starting the desktop app."""
    monkeypatch.setattr(entry, "__file__", str(tmp_path / "jarvis/setup/entry.py"))
    environment: dict[str, str] = {}
    monkeypatch.setattr(entry, "os", SimpleNamespace(environ=environment, chdir=entry.os.chdir))
    monkeypatch.chdir(tmp_path)
    return tmp_path / ".jarvis-local-launch.json", environment


def test_profile_is_applied_before_launcher_import(local_install, monkeypatch) -> None:
    profile_path, environment = local_install
    profile = {
        "JARVIS_CONFIG": "/private/local profile/jarvis.toml",
        "JARVIS_DATA_DIR": "/private/local profile/data",
        "LOCALAPPDATA": "/private/local profile/user",
        "JARVIS_INSTANCE": "local-preview",
        "JARVIS_BIND_HOST": "127.0.0.1",
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    at_import = {}

    def import_launcher(name):
        assert name == "jarvis.ui.web.launcher"
        at_import.update(environment)
        return SimpleNamespace(main=lambda: 17)

    monkeypatch.setattr(entry, "import_module", import_launcher)

    assert entry.main([]) == 17
    assert at_import == profile


def test_profile_is_applied_before_identity_probe(local_install, monkeypatch) -> None:
    profile_path, environment = local_install
    profile_path.write_text('{"JARVIS_INSTANCE": "local-preview"}', encoding="utf-8")
    at_probe = {}

    def probe(destination):
        at_probe.update(environment)
        destination.write_text("identity checked", encoding="utf-8")
        return 7

    monkeypatch.setattr(entry, "_write_identity_probe", probe)
    destination = profile_path.with_name("identity.json")

    assert entry.main(["--jarvis-identity-probe", str(destination)]) == 7
    assert at_probe == {"JARVIS_INSTANCE": "local-preview"}
    assert destination.read_text(encoding="utf-8") == "identity checked"


def test_explicit_environment_wins_over_profile(local_install, monkeypatch) -> None:
    profile_path, environment = local_install
    profile_path.write_text(
        '{"JARVIS_CONFIG": "/profile/config.toml", "JARVIS_INSTANCE": "profile"}',
        encoding="utf-8",
    )
    environment.update(JARVIS_CONFIG="/explicit/config.toml", JARVIS_INSTANCE="")
    monkeypatch.setattr(entry, "import_module", lambda _: SimpleNamespace(main=lambda: 0))

    assert entry.main([]) == 0
    assert environment == {"JARVIS_CONFIG": "/explicit/config.toml", "JARVIS_INSTANCE": ""}


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "null",
        '"not an object"',
        '{"JARVIS_DATA_DIR": 123}',
        '{"JARVIS_DATA_DIR": null}',
        '{"JARVIS_DATA_DIR": true}',
        '{"JARVIS_DATA_DIR": []}',
        '{"JARVIS_DATA_DIR": ""}',
        '{"JARVIS_DATA_DIR": "   "}',
        '{"JARVIS_DATA_DIR": "bad\\u0000path"}',
        '{"JARVIS_CONFIG": "/valid.toml", "UNSUPPORTED_SETTING": "private-value"}',
        '{"JARVIS_CONFIG": "/valid.toml", "JARVIS_INSTANCE": 123}',
        "{invalid json",
    ],
)
def test_invalid_profile_fails_before_any_application_side_effect(
    local_install, monkeypatch, capsys, contents
) -> None:
    profile_path, environment = local_install
    profile_path.write_text(contents, encoding="utf-8")

    def unexpected_import(_):
        pytest.fail("An invalid launch profile must not import the application")

    monkeypatch.setattr(entry, "import_module", unexpected_import)

    assert entry.main([]) == 2
    assert environment == {}
    error = capsys.readouterr().err
    assert ".jarvis-local-launch.json" in error
    assert "Invalid" in error
    assert "private-value" not in error


def test_absent_profile_keeps_normal_launch(local_install, monkeypatch) -> None:
    _, environment = local_install
    environment["JARVIS_INSTANCE"] = "explicit"
    monkeypatch.setattr(entry, "import_module", lambda _: SimpleNamespace(main=lambda: 23))

    assert entry.main([]) == 23
    assert environment == {"JARVIS_INSTANCE": "explicit"}
