"""Non-Latin wake phrases must survive the generic matcher and grammar path."""
from jarvis.speech.wake_constants import normalize_phrase, normalize_phrase_for_match, phrase_core_for_match


def test_cyrillic_phrase_is_not_discarded_by_tokenization():
    assert normalize_phrase("Привет, Джарвис!") == ["привет", "джарвис"]
    assert normalize_phrase_for_match("Привет, Джарвис!") == ["привет", "джарвис"]
    assert phrase_core_for_match("Привет, Джарвис!") == ["джарвис"]
