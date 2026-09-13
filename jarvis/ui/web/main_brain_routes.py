"""Explicit subscription brain setup and independent speech status."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/settings", tags=["settings"])
log = logging.getLogger(__name__)
_PROBE_TTL_S = 60.0
_PROBE_CACHE: tuple[float, dict[str, Any]] | None = None
_PROBE_TASK: asyncio.Task[dict[str, Any]] | None = None


async def _perform_subscription_probe() -> dict[str, Any]:
    from jarvis.plugins.brain.codex_subscription import CodexSubscriptionBrain

    brain = None
    try:
        brain = CodexSubscriptionBrain(model="gpt-5.5")
        result = await brain.probe()
    except Exception as exc:
        log.error("Codex readiness failed (%s)", type(exc).__name__)
        result = {
            "ready": False,
            "auth_mode": None,
            "transport": "codex-app-server",
            "model": "gpt-5.5",
            "version": None,
            "error": "Codex readiness check failed. Restart Jarvis and retry.",
        }
    finally:
        if brain is not None:
            try:
                await brain.close()
            except Exception as exc:
                log.warning("Codex readiness cleanup failed (%s)", type(exc).__name__)
    # Keychain access is part of the explicit probe, never the fast status GET.
    try:
        result["speech_key_available"] = await asyncio.to_thread(speech_key_available)
    except Exception:
        result["speech_key_available"] = False
    return result


def cached_subscription_probe() -> dict[str, Any] | None:
    if _PROBE_CACHE is None or time.monotonic() - _PROBE_CACHE[0] >= _PROBE_TTL_S:
        return None
    return dict(_PROBE_CACHE[1])


async def probe_subscription(*, force: bool = False) -> dict[str, Any]:
    """Share one safe auth/transport probe across cards and section health.

    No model turn is started, so readiness never claims available turn quota.
    A disconnected browser waiter does not cancel a shared child process.
    """
    global _PROBE_TASK
    cached = cached_subscription_probe()
    if not force and cached is not None:
        return cached
    current_loop = asyncio.get_running_loop()
    if _PROBE_TASK is None or _PROBE_TASK.done() or _PROBE_TASK.get_loop() is not current_loop:

        async def run() -> dict[str, Any]:
            global _PROBE_CACHE, _PROBE_TASK
            try:
                result = await _perform_subscription_probe()
                _PROBE_CACHE = (time.monotonic(), dict(result))
                return result
            finally:
                if _PROBE_TASK is asyncio.current_task():
                    _PROBE_TASK = None

        _PROBE_TASK = asyncio.create_task(run(), name="codex-subscription-readiness")
    return dict(await asyncio.shield(_PROBE_TASK))


def speech_key_available() -> bool:
    from jarvis.core.config import get_provider_secret

    return bool(get_provider_secret("gemini"))


def persist_subscription(model: str) -> None:
    from jarvis.core.config import resolve_config_path
    from jarvis.core.config_writer import set_subscription_main_brain

    set_subscription_main_brain(model=model, path=resolve_config_path())


@router.get("/main-brain")
async def main_brain_status(request: Request) -> dict[str, Any]:
    cfg = request.app.state.config
    brain = getattr(request.app.state, "brain", None)
    active = getattr(brain, "active_provider", None) or cfg.brain.primary
    pending = getattr(request.app.state, "pending_main_brain", None)
    cached = cached_subscription_probe()
    from jarvis.ui.web.provider_routes import _current_brain_model

    return {
        "active_provider": active,
        "active_model": _current_brain_model(cfg, active),
        "configured_provider": pending or cfg.brain.primary,
        "requires_restart": bool(pending),
        "codex": cached,
        "voice": {
            "mode": cfg.voice.mode,
            "stt_provider": cfg.stt.provider,
            "tts_provider": cfg.tts.provider,
            "realtime_provider": getattr(cfg.voice, "realtime_provider", None),
            "speech_key_available": cached.get("speech_key_available") if cached else None,
        },
    }


@router.post("/main-brain/probe")
async def main_brain_probe(force: bool = False) -> dict[str, Any]:
    return await probe_subscription(force=force)


class MainBrainSelection(BaseModel):
    provider: Literal["codex-subscription"]


@router.put("/main-brain")
async def select_main_brain(body: MainBrainSelection, request: Request) -> dict[str, Any]:
    status = await probe_subscription(force=True)
    if not status.get("ready") or status.get("auth_mode") != "chatgpt":
        raise HTTPException(409, detail=status.get("error") or "Sign in with codex login first.")
    if not speech_key_available():
        raise HTTPException(409, detail="Save a Gemini key for speech recognition and synthesis.")
    cfg = request.app.state.config
    from jarvis.ui.web.provider_routes import _current_brain_model

    model = _current_brain_model(cfg, body.provider)
    persist_subscription(model)
    brain = getattr(request.app.state, "brain", None)
    compatible_voice = (
        cfg.voice.mode == "pipeline"
        and not cfg.voice.profile
        and cfg.stt.provider == "gemini-api"
        and cfg.tts.provider == "gemini-flash-tts"
    )
    applied = False
    if compatible_voice and callable(getattr(brain, "switch", None)):
        assert brain is not None
        from jarvis.core.config import BrainProviderConfig, BrainTierConfig

        if callable(getattr(brain, "apply_provider_model", None)):
            brain.apply_provider_model(body.provider, model)
        await brain.switch(body.provider, persist=False)
        applied = brain.active_provider == body.provider
        if applied:
            cfg.brain.primary = body.provider
            cfg.brain.deep_brain = body.provider
            cfg.brain.tool_model = BrainTierConfig(provider=body.provider)
            cfg.brain.providers[body.provider] = BrainProviderConfig(model=model)
    request.app.state.pending_main_brain = None if applied else body.provider
    return {
        "ok": True,
        "new_provider": body.provider,
        "persisted": True,
        "applied_live": applied,
        "requires_restart": not applied,
        "codex": status,
    }
