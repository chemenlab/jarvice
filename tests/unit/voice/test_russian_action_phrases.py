"""Russian action readbacks must preserve the resolved turn language."""

from string import Formatter

from jarvis.voice.action_phrases import _PHRASES, action_phrase, resolve_phrase_language


def test_explicit_russian_pin_applies_to_ambiguous_action_input():
    assert resolve_phrase_language("ru", "OK") == "ru"
    assert action_phrase("wiki_saved", "ru") == "Сохранено в памяти."
    assert action_phrase("cu_dispatch_ack", "ru").startswith("Сейчас выполню")


def test_every_action_readback_has_russian_and_preserves_fields():
    def fields(text):
        return {field for _, field, _, _ in Formatter().parse(text) if field}

    for key, variants in _PHRASES.items():
        assert variants.get("ru"), key
        assert fields(variants["ru"]) == fields(variants["en"]), key
