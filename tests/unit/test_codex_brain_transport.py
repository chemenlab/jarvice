"""Subscription transport boundaries with a deterministic JSONL peer."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jarvis import codex_brain_transport as transport_module
from jarvis.codex_brain_transport import CodexBrainTransport, CodexBrainTransportError


from tests.fakes.codex_jsonl_peer import FakeProcess, FakeTree


@pytest.fixture
def peer(monkeypatch, tmp_path):
    processes, trees, launches = [], [], []
    auth = {"type": "chatgpt"}

    async def spawn(*command, **kwargs):
        p = FakeProcess(command, auth["type"])
        processes.append(p)
        launches.append(kwargs)
        return p

    def tree_factory(_name):
        tree = FakeTree()
        trees.append(tree)
        return tree

    monkeypatch.setattr(transport_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(transport_module, "make_process_tree", tree_factory)
    return processes, trees, launches, auth


@pytest.mark.asyncio
async def test_auth_stays_subscription_and_mcp_is_disabled_before_work(peer, monkeypatch, tmp_path):
    processes, trees, launches, _ = peer
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_HOME", "/already-configured-codex-home")
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    try:
        await client.ensure_started()
        assert len(processes) == 2
        assert trees[0].closed
        assert all("OPENAI_API_KEY" not in launch["env"] and "CODEX_API_KEY" not in launch["env"] for launch in launches)
        assert all(launch["env"]["CODEX_HOME"] == "/already-configured-codex-home" for launch in launches)
        assert not any(arg.startswith(("openai_base_url=", "chatgpt_base_url=")) for process in processes for arg in process.command)
        assert not any(frame.get("method") == "thread/start" for frame in processes[0].sent)
        await client.request("thread/start", {"dynamicTools": []})
        assert any(frame.get("method") == "thread/start" for frame in processes[1].sent)
    finally:
        await client.close()
    assert all(tree.closed for tree in trees)


@pytest.mark.asyncio
async def test_api_account_fails_closed_without_starting_a_thread(peer, tmp_path):
    processes, _, _, auth = peer
    auth["type"] = "apiKey"
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    with pytest.raises(CodexBrainTransportError, match="ChatGPT"):
        await client.ensure_started()
    assert not any(f.get("method") == "thread/start" for p in processes for f in p.sent)
    await client.close()


@pytest.mark.asyncio
async def test_dynamic_request_waits_for_external_tool_result(peer, tmp_path):
    processes, _, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    await client.ensure_started()
    await client.request("thread/start", {})
    p = processes[-1]
    frame = {"id": "tool-rpc", "method": "item/tool/call", "params": {"threadId": "thread-1", "turnId": "turn-1", "callId": "call-1", "tool": "echo", "arguments": {"text": "hello"}}}
    p.emit(frame)
    assert await client.next_event("thread-1", 1) == frame
    assert not any(f.get("id") == "tool-rpc" for f in p.sent)
    result = {"contentItems": [{"type": "inputText", "text": "hello"}], "success": True}
    await client.respond("tool-rpc", result)
    assert p.sent[-1] == {"id": "tool-rpc", "result": result}
    await client.close()


@pytest.mark.asyncio
async def test_unexpected_approval_is_denied_and_not_exposed_as_jarvis_tool(peer, tmp_path):
    processes, _, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    await client.ensure_started()
    await client.request("thread/start", {})
    p = processes[-1]
    p.emit({"id": 999, "method": "item/commandExecution/requestApproval", "params": {"threadId": "thread-1", "command": "never execute"}})
    event = await client.next_event("thread-1", 1)
    assert event["method"] == "transport/error"
    assert {"id": 999, "result": {"decision": "decline"}} in p.sent
    await client.close()


@pytest.mark.asyncio
async def test_release_resolves_pending_call_and_interrupts_exact_turn(peer, tmp_path):
    processes, _, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    await client.ensure_started()
    await client.request("thread/start", {})
    p = processes[-1]
    p.emit({"id": "pending", "method": "item/tool/call", "params": {"threadId": "thread-1", "turnId": "turn-1", "callId": "call-1", "tool": "echo", "arguments": {}}})
    await client.next_event("thread-1", 1)
    await client.release_thread("thread-1", "turn-1")
    assert any(f.get("id") == "pending" and f.get("result", {}).get("success") is False for f in p.sent)
    assert any(f.get("method") == "turn/interrupt" and f["params"] == {"threadId": "thread-1", "turnId": "turn-1"} for f in p.sent)
    await client.close()


@pytest.mark.asyncio
async def test_disconnected_peer_unblocks_pending_request(peer, tmp_path):
    processes, _, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    await client.ensure_started()
    pending = asyncio.create_task(client.request("wait/forever", {}))
    await asyncio.sleep(0)
    processes[-1].finish()
    with pytest.raises(CodexBrainTransportError):
        await asyncio.wait_for(pending, 1)
    await client.close()


@pytest.mark.asyncio
async def test_probe_reports_only_safe_metadata_and_starts_no_model_turn(peer, tmp_path):
    processes, _, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    status = await client.probe(model="gpt-5.5")
    assert status == {"ready": True, "auth_mode": "chatgpt", "transport": "codex-app-server", "model": "gpt-5.5", "version": "0.153.4"}
    assert not any(f.get("method") in {"thread/start", "turn/start"} for p in processes for f in p.sent)
    assert "private@example.invalid" not in json.dumps(status)
    assert "do-not-log" not in json.dumps(status)
    await client.close()


@pytest.mark.asyncio
async def test_rpc_timeout_reaps_the_process_instead_of_leaving_a_live_turn(peer, tmp_path):
    processes, trees, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path, request_timeout_s=0.01)
    await client.ensure_started()
    with pytest.raises(CodexBrainTransportError, match="timed out"):
        await client.request("wait/forever", {})
    assert processes[-1].returncode is not None
    assert trees[-1].closed
    await client.close()


@pytest.mark.asyncio
async def test_default_runtime_uses_configured_jarvis_data_directory(peer, monkeypatch, tmp_path):
    from jarvis.core import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    client = CodexBrainTransport(binary_path="codex")
    try:
        assert (await client.probe())["ready"] is True
        assert client._root.parent == tmp_path / "codex-brain"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancelled_rpc_reaps_child_before_returning(peer, tmp_path):
    processes, trees, _, _ = peer
    client = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path)
    await client.ensure_started()
    pending = asyncio.create_task(client.request("wait/forever", {}))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert processes[-1].returncode is not None
    assert trees[-1].closed
    await client.close()
