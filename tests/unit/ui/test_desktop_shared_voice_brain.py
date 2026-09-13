"""Desktop voice must not build an independent fallback for a selected Codex main."""

from types import SimpleNamespace

import pytest

from jarvis.ui import desktop_app


def test_codex_voice_keeps_shared_callable_brain() -> None:
    cfg = SimpleNamespace(brain=SimpleNamespace(primary="codex-subscription"))

    async def shared_brain(text):
        return text

    assert desktop_app._voice_brain_for_config(cfg, shared_brain) is shared_brain


@pytest.mark.parametrize("brain", [None, SimpleNamespace(respond=lambda: "fallback")])
def test_codex_voice_rejects_missing_shared_brain(brain, monkeypatch) -> None:
    from jarvis.brain import factory

    cfg = SimpleNamespace(brain=SimpleNamespace(primary="codex-subscription"))
    built: list[str] = []
    monkeypatch.setattr(factory, "build_default_brain", lambda **kw: built.append(kw["tier"]))

    with pytest.raises(ValueError, match="shared brain"):
        desktop_app._voice_brain_for_config(cfg, brain)

    assert built == []


def test_other_main_preserves_legacy_voice_fallback(monkeypatch) -> None:
    from jarvis.brain import factory

    cfg = SimpleNamespace(brain=SimpleNamespace(primary="gemini"))
    fallback = object()
    monkeypatch.setattr(factory, "build_default_brain", lambda **kw: fallback)

    assert desktop_app._voice_brain_for_config(cfg, None) is fallback
