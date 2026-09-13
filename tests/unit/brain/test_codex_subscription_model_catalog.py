"""The subscription picker comes from the authenticated installed CLI only."""
from __future__ import annotations

import pytest

from jarvis.brain.model_catalog import ModelCatalog, catalog_spec


@pytest.fixture
def cli(monkeypatch):
    state = {"calls": [], "closed": 0, "fail": False}

    class Transport:
        async def request(self, method, params):
            state["calls"].append((method, params))
            if state["fail"]:
                raise RuntimeError("No authenticated CLI catalog")
            assert method == "model/list"
            if params.get("cursor") == "next-page":
                return {"data": [{"id": "new-native-model", "displayName": "New native model", "inputModalities": ["text", "image"]}], "nextCursor": None}
            return {"data": [
                {"id": "gpt-5.5", "displayName": "GPT-5.5", "inputModalities": ["text", "image"]},
                {"id": "hidden-model", "displayName": "Hidden", "hidden": True},
            ], "nextCursor": "next-page"}

        async def close(self):
            state["closed"] += 1

    monkeypatch.setattr("jarvis.codex_brain_transport.CodexBrainTransport", Transport)
    monkeypatch.setattr("jarvis.core.config.get_provider_secret", lambda *_: pytest.fail("Subscription catalog must never request an API key"))
    return state


@pytest.mark.asyncio
async def test_live_cli_catalog_paginates_and_keeps_verified_vision_boundary(cli, tmp_path):
    spec = catalog_spec("codex-subscription")
    assert spec is not None and spec.live and spec.curated == ()
    result = await ModelCatalog(cache_path=tmp_path / "catalog.json").list_models("codex-subscription")
    assert result.source == "live"
    models = {model.id: model for model in result.models}
    assert set(models) == {"gpt-5.5", "new-native-model"}
    assert models["new-native-model"].label == "New native model"
    assert "image" in models["gpt-5.5"].input_modalities
    assert "image" not in models["new-native-model"].input_modalities
    assert cli["calls"][0][1]["includeHidden"] is False
    assert cli["calls"][1][1]["cursor"] == "next-page"
    assert cli["closed"] == 1


@pytest.mark.asyncio
async def test_catalog_cache_and_refresh_never_invent_models(cli, tmp_path):
    catalog = ModelCatalog(cache_path=tmp_path / "catalog.json")
    first = await catalog.list_models("codex-subscription")
    cached = await catalog.list_models("codex-subscription")
    assert cached.source == "cache" and cached.models == first.models
    assert cli["closed"] == 1
    await catalog.list_models("codex-subscription", force_refresh=True)
    assert cli["closed"] == 2
    cli["fail"] = True
    stale = await catalog.list_models("codex-subscription", force_refresh=True)
    assert stale.source == "cache" and stale.models == first.models
    assert cli["closed"] == 3


@pytest.mark.asyncio
async def test_unavailable_cli_has_no_fabricated_fallback(cli, tmp_path):
    cli["fail"] = True
    result = await ModelCatalog(cache_path=tmp_path / "catalog.json").list_models("codex-subscription")
    assert result.models == () and result.source == "static"
    assert cli["closed"] == 1
