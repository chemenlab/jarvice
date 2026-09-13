"""Restarting the recorder preserves old text unless retention was opted into."""
from jarvis.core.bus import EventBus
from jarvis.sessions.init import bootstrap_sessions, shutdown_sessions
from jarvis.sessions.store import SessionStore


def test_default_boot_keeps_old_conversation_and_explicit_retention_prunes(tmp_path):
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    store.open()
    store.upsert_session(session_id="old", started_ms=1000, language="ru")
    store.upsert_turn(turn_id="turn", session_id="old", idx=0, started_ms=1000)
    store.finalize_turn(turn_id="turn", ended_ms=2000, user_text="Synthetic historical fact", user_lang="ru",
        jarvis_text="Remembered", jarvis_lang="ru", tier="deep", provider="codex-subscription", model="gpt-5.6-sol",
        tokens_in=1, tokens_out=1, cost_usd=0, latency_total_ms=10, tool_calls=[])
    store.close()
    stack = bootstrap_sessions(bus=EventBus(), db_path=path)
    try:
        assert stack["store"].get_session("old") is not None
        assert stack["store"].get_turns("old")[0].user_text == "Synthetic historical fact"
    finally:
        shutdown_sessions(stack)
    opted_in = bootstrap_sessions(bus=EventBus(), db_path=path, retention_days=30)
    try:
        assert opted_in["store"].get_session("old") is None
    finally:
        shutdown_sessions(opted_in)
