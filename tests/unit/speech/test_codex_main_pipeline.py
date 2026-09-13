"""The subscription main brain owns conversational voice turns and their wording."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest

from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import AudioChunk
from jarvis.speech.pipeline import SpeechPipeline, TurnTakingState
from jarvis.voice.subscription_profile import CODEX_SUBSCRIPTION_VOICE_PROFILE


class _Brain:
    reply_language = "auto"
    conversation_language = ""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def __call__(self, text: str) -> str:
        return "Завтра встреча в десять утра."

    async def generate_stream(self, text: str, **kwargs):
        self.requests.append(text)
        yield await self(text)


class _TTS:
    supports_streaming = True
    name = "test-tts"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(self, text: str, language_code: str | None = None):
        self.calls.append((text, language_code))
        yield AudioChunk(pcm=b"\0\0", sample_rate=24000, timestamp_ns=0)


class _Player:
    async def play_chunks(self, chunks, **kwargs):
        async for _ in chunks:
            pass

    def stop(self) -> None:
        pass


def _config(primary: str = "codex-subscription") -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.brain.primary = primary
    cfg.voice.mode = "pipeline"
    return cfg


def _pipeline(config, brain=None, **kwargs) -> SpeechPipeline:
    return SpeechPipeline(
        config=config,
        brain_callback=brain,
        tts=_TTS(),
        stt=SimpleNamespace(name="test-stt"),
        enable_local_whisper=False,
        enable_whisper_wake=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_main_brain_bypasses_legacy_voice_profile_on_construction() -> None:
    cfg = _config()
    cfg.voice.profile = CODEX_SUBSCRIPTION_VOICE_PROFILE
    brain = _Brain()
    pipeline = _pipeline(cfg, brain)

    assert pipeline._brain is brain
    result = [chunk async for chunk in pipeline._brain.generate_stream("Привет")]

    assert result == ["Завтра встреча в десять утра."]
    assert brain.requests == ["Привет"]
    assert cfg.voice.profile == CODEX_SUBSCRIPTION_VOICE_PROFILE


def test_applying_legacy_profile_keeps_shared_main_brain() -> None:
    cfg = _config("gemini")
    brain = _Brain()
    pipeline = _pipeline(cfg, brain)
    pipeline.apply_voice_profile(CODEX_SUBSCRIPTION_VOICE_PROFILE)
    cfg.brain.primary = "codex-subscription"

    pipeline.apply_voice_profile(CODEX_SUBSCRIPTION_VOICE_PROFILE)

    assert pipeline._brain is brain


def test_subscription_main_never_falls_back_to_echo_without_shared_brain() -> None:
    with pytest.raises(ValueError, match="shared brain"):
        _pipeline(_config())


@pytest.mark.asyncio
async def test_live_main_switch_stops_existing_flash_ack() -> None:
    api_requests: list[str] = []

    class FlashAck:
        async def run(self, utterance: str, **kwargs):
            api_requests.append(utterance)
            return None

    cfg = _config("gemini")
    pipeline = _pipeline(cfg, _Brain(), ack_brain=FlashAck())
    cfg.brain.primary = "codex-subscription"

    await pipeline._spawn_flash_brain_ack("Проверь календарь", "ru")

    assert api_requests == []


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_pending_api_ack_cannot_speak_after_main_switch(streaming) -> None:
    from jarvis.core.bus import EventBus
    from jarvis.core.events import AnnouncementRequested

    started = asyncio.Event()
    release = asyncio.Event()

    class FlashAck:
        async def run(self, *args, **kwargs):
            started.set()
            await release.wait()
            return "I'm checking the calendar."

        async def run_stream(self, *args, **kwargs):
            yield await self.run(*args, **kwargs)

    cfg = _config("gemini")
    cfg.ack_brain.streaming = streaming
    cfg.ack_brain.suppress_if_brain_faster_than_ms = 0
    cfg.ack_brain.ack_continuation_grace_ms = 0
    bus = EventBus()
    published: list[str] = []

    async def record(event):
        published.append(event.text)

    bus.subscribe(AnnouncementRequested, record)
    pipeline = _pipeline(cfg, _Brain(), ack_brain=FlashAck(), bus=bus)
    pipeline._player = _Player()
    pipeline._turn_state = TurnTakingState.PROCESSING
    task = asyncio.create_task(pipeline._spawn_flash_brain_ack("Check the calendar", "en"))
    await started.wait()

    cfg.brain.primary = "codex-subscription"
    release.set()
    await task

    assert published == []


@pytest.mark.asyncio
async def test_live_main_switch_preserves_deterministic_tool_ack() -> None:
    from jarvis.core.events import AnnouncementRequested

    cfg = _config("gemini")
    cfg.ack_brain.grounded_ack_commit_grace_ms = 0
    pipeline = _pipeline(cfg, _Brain(), ack_brain=object())
    pipeline._player = _Player()
    pipeline._turn_state = TurnTakingState.PROCESSING
    cfg.brain.primary = "codex-subscription"

    await pipeline._on_announcement(AnnouncementRequested(
        source_layer="brain.router.ack", text="I'm checking the calendar.",
        language="en", kind="preamble",
    ))

    assert pipeline._tts.calls == [("I'm checking the calendar.", "en-US")]


@pytest.mark.asyncio
async def test_russian_transcript_reaches_shared_brain_and_russian_tts() -> None:
    brain = _Brain()
    pipeline = _pipeline(_config(), brain)
    pipeline._player = _Player()
    pipeline._latency_tracker = None

    async def never_barge(**kwargs):
        await asyncio.Event().wait()
        return False

    pipeline._barge_monitor = never_barge
    utterance = "Напомни мне пожалуйста когда завтра встреча"
    language = pipeline._output_language("ru-RU", utterance)

    await pipeline._brain_streaming(utterance, language)

    assert brain.requests == [utterance]
    assert pipeline._tts.calls == [("Завтра встреча в десять утра.", "ru-RU")]


@pytest.mark.asyncio
async def test_live_main_switch_uses_deterministic_instant_ack(monkeypatch) -> None:
    from jarvis.core.bus import EventBus
    from jarvis.core.events import AnnouncementRequested
    from jarvis.voice import instant_ack

    api_requests: list[dict] = []

    class Composer:
        has_llm = True

        async def compose(self, **kwargs):
            api_requests.append(kwargs)
            return ""

    cfg = _config("gemini")
    brain = _Brain()
    brain._readback_composer = Composer()
    bus = EventBus()
    pipeline = _pipeline(cfg, brain, bus=bus)
    pipeline._player = _Player()
    pipeline._turn_state = TurnTakingState.PROCESSING
    pipeline._brain_first_frame_played = False
    cfg.brain.primary = "codex-subscription"
    spoken: list[str] = []
    async def record(event):
        spoken.append(event.text)

    bus.subscribe(AnnouncementRequested, record)
    monkeypatch.setitem(instant_ack._POOLS, instant_ack.WorkClass.RESEARCH, {
        "en": ("I'm looking that up online.",),
    })

    pipeline._arm_instant_ack("What's the weather in Berlin right now?", "en")
    await pipeline._instant_ack_task
    pipeline._cancel_spawn_heartbeats()
    if pipeline._instant_progress_task is not None:
        pipeline._instant_progress_task.cancel()

    assert api_requests == []
    assert spoken == ["I'm looking that up online."]
