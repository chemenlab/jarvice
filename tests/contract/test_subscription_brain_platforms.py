"""The subscription boundary and honest capability failure on every OS family.

Windows/Linux process launches are emulated here; macOS also has live evidence.
"""

import os
from types import SimpleNamespace

import pytest

from jarvis import codex_brain_transport as module
from jarvis.codex_brain_transport import CodexBrainTransport
from tests.fakes.codex_jsonl_peer import FakeProcess, FakeTree


@pytest.mark.parametrize("family,os_name,flags", [
    ("windows", "nt", 0x08000000), ("macos", "posix", 0), ("linux", "posix", 0),
])
async def test_platform_launch_is_contained_keyless_and_closes(family, os_name, flags, tmp_path, monkeypatch):
    launches, trees, processes = [], [], []
    monkeypatch.setattr(module, "os", SimpleNamespace(name=os_name, environ={
        **os.environ, "OPENAI_API_KEY": "never-inherit", "CODEX_HOME": "/existing-login",
    }))
    monkeypatch.setattr(module, "NO_WINDOW_CREATIONFLAGS", flags)

    async def spawn(*command, **kwargs):
        launches.append(kwargs)
        proc = FakeProcess(command)
        processes.append(proc)
        return proc

    def tree_factory(name):
        tree = FakeTree()
        trees.append(tree)
        return tree

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(module, "make_process_tree", tree_factory)
    transport = CodexBrainTransport(binary_path="codex", runtime_dir=tmp_path / family)
    status = await transport.probe(model="gpt-5.6-sol")
    assert status["ready"] and status["auth_mode"] == "chatgpt"
    assert launches
    for launch in launches:
        assert launch["creationflags"] == flags
        assert launch["start_new_session"] is (os_name != "nt")
        assert "OPENAI_API_KEY" not in launch["env"]
        assert launch["env"]["CODEX_HOME"] == "/existing-login"
    await transport.close()
    assert all(tree.closed for tree in trees)
    assert all(proc.returncode is not None for proc in processes)


@pytest.mark.parametrize("family", ["windows", "macos", "linux"])
async def test_missing_cli_degrades_without_an_api_fallback(family, tmp_path, monkeypatch):
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr("jarvis.core.path_augment.ensure_cli_paths", lambda: None)
    client = CodexBrainTransport(runtime_dir=tmp_path / family)
    try:
        status = await client.probe()
        assert status["ready"] is False
        assert "not installed" in status["error"]
    finally:
        await client.close()
