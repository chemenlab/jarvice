"""JSONL app-server transport for the main ChatGPT-subscription brain.

Codex owns credentials in its existing home. This module never reads or copies
authentication files and never falls back to an API key. Dynamic tool requests
remain pending until Jarvis's normal ToolExecutor supplies their results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from jarvis.core.process_tree import make_process_tree
from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

log = logging.getLogger(__name__)
_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "apps",
    "browser_use",
    "computer_use",
    "hooks",
    "image_generation",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "remote_plugin",
    "network_proxy",
    "remote_control",
    "skill_search",
    "workspace_dependencies",
    "web_search_request",
    "external_agent_memory_import",
    "chronicle",
    "view_image",
    "sleep_tool",
    "code_mode",
    "code_mode_only",
    "skill_mcp_dependency_install",
)
_DENY_APPROVAL_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "execCommandApproval",
    "applyPatchApproval",
}
_NO_ACTION_INSTRUCTIONS = (
    "You are the reasoning component of Personal Jarvis. Use only the dynamic "
    "tools supplied by Jarvis. Jarvis executes tools and owns permissions, "
    "memory, files, and external actions. Never use native shell, file, browser, "
    "MCP, skill, or computer tools."
)


class CodexBrainTransportError(RuntimeError):
    """The subscription transport could not safely complete an operation."""

    public_message = (
        "Не удалось подключиться к Codex по подписке ChatGPT. "
        "Проверьте вход в Codex CLI и повторите запрос."
    )

    def __init__(self, message: str, *, public_message: str | None = None) -> None:
        super().__init__(message)
        if public_message is not None:
            self.public_message = public_message


def _child_environment() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().endswith("API_KEY")
        and name.upper()
        not in {
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "AZURE_OPENAI_ENDPOINT",
            "CODEX_BASE_URL",
            "OPENAI_ACCESS_TOKEN",
        }
    }
    # HOME and CODEX_HOME are deliberately inherited without modification.
    env["CODEX_INTERNAL_APP_SERVER_REMOTE_CONTROL_DISABLED"] = "1"
    env["RUST_LOG"] = "error"
    return env


class CodexBrainTransport:
    """One lazy, contained app-server process with independent thread queues."""

    def __init__(
        self,
        binary_path: str | None = None,
        runtime_dir: Path | None = None,
        request_timeout_s: float = 30.0,
    ) -> None:
        self._binary_path = binary_path
        self._runtime_dir = runtime_dir
        self._request_timeout_s = request_timeout_s
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._process: Any = None
        self._tree: Any = None
        self._reader: asyncio.Task | None = None
        self._stderr_reader: asyncio.Task | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._root: Path | None = None
        self._ready = False
        self._closed = False
        self._stopping = False
        self._fatal: CodexBrainTransportError | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._server_requests: dict[int | str, str] = {}
        self.cli_version: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready and self._process is not None and self._process.returncode is None

    async def probe(self, *, model: str = "gpt-5.5") -> dict[str, Any]:
        """Check protocol and login readiness without creating a model turn."""
        status: dict[str, Any] = {
            "ready": False,
            "auth_mode": None,
            "transport": "codex-app-server",
            "model": model,
            "version": self.cli_version,
        }
        try:
            await self.ensure_started()
        except (CodexBrainTransportError, OSError) as exc:
            status["error"] = (
                str(exc)
                if isinstance(exc, CodexBrainTransportError)
                else "Codex app-server could not be started."
            )
        else:
            status.update(ready=True, auth_mode="chatgpt", version=self.cli_version)
        return status

    def _prepare_workspace(self) -> None:
        if self._temporary is not None:
            return
        runtime_dir = self._runtime_dir
        if runtime_dir is None:
            from jarvis.core.config import DATA_DIR

            runtime_dir = Path(DATA_DIR) / "codex-brain"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="transport-", dir=runtime_dir)
        self._root = Path(self._temporary.name)
        for name in ("workspace", "logs", "sqlite"):
            (self._root / name).mkdir()
        (self._root / "instructions.md").write_text(_NO_ACTION_INSTRUCTIONS, encoding="utf-8")

    def _command(self, disabled_mcp: tuple[str, ...]) -> list[str]:
        binary: str | None
        if self._binary_path:
            binary = self._binary_path
        else:
            from jarvis.core.path_augment import ensure_cli_paths

            ensure_cli_paths()
            binary = shutil.which("codex")
        if not binary:
            raise CodexBrainTransportError("Codex CLI is not installed or is not on PATH.")
        assert self._root is not None
        command = [binary, "app-server", "--stdio", "--strict-config"]
        for feature in _DISABLED_FEATURES:
            command.extend(("--disable", feature))
        settings: dict[str, Any] = {
            "approval_policy": "never",
            "sandbox_mode": "read-only",
            "forced_login_method": "chatgpt",
            "model_provider": "openai",
            "web_search": "disabled",
            "notify": [],
            "features.code_mode_host": True,
            "history.persistence": "none",
            "analytics.enabled": False,
            "project_doc_max_bytes": 0,
            "include_apps_instructions": False,
            "include_environment_context": False,
            "include_collaboration_mode_instructions": False,
            "include_permissions_instructions": False,
            "memories.dedicated_tools": False,
            "memories.generate_memories": False,
            "memories.use_memories": False,
            "orchestrator.mcp.enabled": False,
            "orchestrator.skills.enabled": False,
            "skills.bundled.enabled": False,
            "skills.include_instructions": False,
            "model_instructions_file": str(self._root / "instructions.md"),
            "log_dir": str(self._root / "logs"),
            "sqlite_home": str(self._root / "sqlite"),
            "otel.exporter": "none",
            "otel.trace_exporter": "none",
            "otel.metrics_exporter": "none",
            "otel.log_user_prompt": False,
        }
        for name in disabled_mcp:
            # Codex's CLI override parser splits dotted paths literally; quoted
            # segments become part of the key instead of TOML quoted keys.
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise CodexBrainTransportError("An inherited MCP name cannot be disabled safely.")
            settings[f"mcp_servers.{name}.enabled"] = False
        for key, value in settings.items():
            command.extend(("-c", f"{key}={json.dumps(value, separators=(',', ':'))}"))
        return command

    async def ensure_started(self) -> None:
        if self.ready:
            return
        async with self._start_lock:
            if self.ready:
                return
            if self._closed:
                raise CodexBrainTransportError("Codex subscription transport is closed.")
            self._prepare_workspace()
            try:
                await self._stop_process()
                await self._spawn(())
                config = await self._read_config()
                servers = config.get("mcp_servers") or {}
                if not isinstance(servers, dict):
                    raise CodexBrainTransportError("Could not verify Codex MCP configuration.")
                names = tuple(servers)
                # An empty table merges with the user's servers instead of
                # deleting them. Discover names without initializing servers,
                # then disable each in launch flags BEFORE any thread exists.
                if any(
                    not isinstance(row, dict) or row.get("enabled", True)
                    for row in servers.values()
                ):
                    del config, servers
                    await self._stop_process()
                    await self._spawn(names)
                    config = await self._read_config()
                    servers = config.get("mcp_servers") or {}
                if any(
                    not isinstance(row, dict) or row.get("enabled", True)
                    for row in servers.values()
                ):
                    raise CodexBrainTransportError("An inherited MCP server remains enabled.")
                del config, servers
                account = await self._rpc("account/read", {"refreshToken": False})
                auth_type = (account.get("account") or {}).get("type")
                del account
                if auth_type != "chatgpt":
                    raise CodexBrainTransportError(
                        "Sign in to Codex with ChatGPT; API-key authentication is not accepted.",
                        public_message=(
                            "Войдите в Codex CLI через аккаунт ChatGPT. "
                            "Для этого режима нужен вход по подписке."
                        ),
                    )
                self._ready = True
            except BaseException:
                await self._stop_process()
                raise

    async def _read_config(self) -> dict:
        result = await self._rpc("config/read", {"includeLayers": False})
        config = result.get("config")
        if not isinstance(config, dict):
            raise CodexBrainTransportError("Could not verify Codex effective configuration.")
        required = {
            "approval_policy": "never",
            "sandbox_mode": "read-only",
            "model_provider": "openai",
            "forced_login_method": "chatgpt",
            "web_search": "disabled",
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise CodexBrainTransportError(
                "Codex effective settings do not preserve the subscription tool boundary."
            )
        # Never log or persist this response: arbitrary MCP env fields can be
        # credentials. Only server names are needed for disabled launch flags.
        return config

    async def _spawn(self, disabled_mcp: tuple[str, ...]) -> None:
        tree = make_process_tree("codex-subscription-brain")
        if not tree.supports_containment:
            tree.close()
            raise CodexBrainTransportError("Process-tree containment is unavailable on this host.")
        self._tree = tree
        self._fatal = None
        self._stopping = False
        assert self._root is not None
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command(disabled_mcp),
                cwd=self._root / "workspace",
                env=_child_environment(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                creationflags=NO_WINDOW_CREATIONFLAGS,
                start_new_session=os.name != "nt",
            )
            self._process = process
            tree.assign(process.pid)
        except BaseException:
            tree.close()
            if self._process is not None:
                self._process.kill()
                await self._process.wait()
            raise
        self._reader = asyncio.create_task(self._read_loop(process))
        self._stderr_reader = asyncio.create_task(self._drain_stderr(process))
        initialized = await self._rpc(
            "initialize",
            {
                "clientInfo": {"name": "personal_jarvis_brain", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        version = re.search(r"/(\d+\.\d+\.\d+)", str(initialized.get("userAgent", "")))
        self.cli_version = version.group(1) if version else None
        await self._write({"method": "initialized", "params": {}})

    async def request(self, method: str, params: dict) -> dict:
        await self.ensure_started()
        result = await self._rpc(method, params)
        if method == "thread/start":
            thread_id = (result.get("thread") or {}).get("id")
            if isinstance(thread_id, str):
                self._queues.setdefault(thread_id, asyncio.Queue(maxsize=4096))
        return result

    async def _rpc(self, method: str, params: dict) -> dict:
        if self._fatal is not None:
            raise self._fatal
        self._next_id += 1
        request_id = self._next_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with asyncio.timeout(self._request_timeout_s):
                await self._write({"id": request_id, "method": method, "params": params})
                result = await future
            if not isinstance(result, dict):
                raise CodexBrainTransportError(f"Invalid Codex {method} response.")
            return result
        except TimeoutError as exc:
            future.cancel()
            await self._stop_process()
            raise CodexBrainTransportError(f"Codex {method} timed out.") from exc
        except asyncio.CancelledError:
            # A cancelled thread/start can otherwise create a native thread
            # after its caller has lost the ID required to interrupt it.
            future.cancel()
            await self._stop_process()
            raise
        finally:
            self._pending.pop(request_id, None)

    async def _write(self, frame: dict) -> None:
        async with self._write_lock:
            process = self._process
            if process is None or process.returncode is not None or process.stdin is None:
                raise CodexBrainTransportError("Codex app-server is disconnected.")
            try:
                process.stdin.write((json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8"))
                await process.stdin.drain()
            except (OSError, RuntimeError) as exc:
                raise CodexBrainTransportError("Codex app-server input closed.") from exc

    def _fail(self, message: str) -> None:
        self._fatal = CodexBrainTransportError(message)
        self._ready = False
        for future in self._pending.values():
            if not future.done():
                future.set_exception(self._fatal)
        for queue in self._queues.values():
            with suppress(asyncio.QueueFull):
                queue.put_nowait(self._fatal)

    def _publish(self, thread_id: str, frame: dict) -> None:
        queue = self._queues.get(thread_id)
        if queue is None:
            return
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            self._fail("Codex notification queue exceeded its bounded capacity.")

    async def _read_loop(self, process: Any) -> None:
        try:
            while line := await process.stdout.readline():
                frame = json.loads(line.decode("utf-8"))
                if not isinstance(frame, dict):
                    raise ValueError("Non-object frame")
                method = frame.get("method")
                request_id = frame.get("id")
                params = frame.get("params") or {}
                thread_id = params.get("threadId") if isinstance(params, dict) else None
                if request_id is not None and method:
                    if (
                        method == "item/tool/call"
                        and isinstance(thread_id, str)
                        and thread_id in self._queues
                    ):
                        self._server_requests[request_id] = thread_id
                        self._publish(thread_id, frame)
                    else:
                        if method in _DENY_APPROVAL_METHODS:
                            await self._write({"id": request_id, "result": {"decision": "decline"}})
                        elif method == "item/tool/call":
                            await self._write(
                                {"id": request_id, "result": {"contentItems": [], "success": False}}
                            )
                        else:
                            await self._write(
                                {
                                    "id": request_id,
                                    "error": {
                                        "code": -32000,
                                        "message": "Only Jarvis dynamic tool calls are allowed.",
                                    },
                                }
                            )
                        if isinstance(thread_id, str):
                            self._publish(
                                thread_id,
                                {
                                    "method": "transport/error",
                                    "params": {
                                        "message": "Codex requested an action outside Jarvis tools."
                                    },
                                },
                            )
                elif request_id is not None:
                    future = self._pending.get(request_id)
                    if future is not None and not future.done():
                        if "error" in frame:
                            code = (frame.get("error") or {}).get("code")
                            future.set_exception(
                                CodexBrainTransportError(
                                    f"Codex app-server rejected a request (code {code})."
                                )
                            )
                        else:
                            future.set_result(frame.get("result"))
                elif method and isinstance(thread_id, str):
                    self._publish(thread_id, frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never log frame contents
            log.warning("Codex brain JSONL reader stopped (%s)", type(exc).__name__)
        finally:
            if not self._stopping and process is self._process:
                self._fail("Codex app-server disconnected.")

    async def _drain_stderr(self, process: Any) -> None:
        try:
            while data := await process.stderr.read(8192):
                log.debug("Codex brain diagnostic received (%d bytes; content redacted)", len(data))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - diagnostics are nonessential
            log.debug("Codex diagnostic stream stopped (%s)", type(exc).__name__)

    async def next_event(self, thread_id: str, timeout_s: float) -> dict:
        if self._fatal is not None:
            raise self._fatal
        queue = self._queues.get(thread_id)
        if queue is None:
            raise CodexBrainTransportError("Unknown Codex subscription thread.")
        try:
            event = await asyncio.wait_for(queue.get(), timeout_s)
        except TimeoutError as exc:
            raise CodexBrainTransportError("Codex subscription turn timed out.") from exc
        if isinstance(event, Exception):
            raise event
        return event

    async def respond(self, request_id: int | str, result: dict) -> None:
        if request_id not in self._server_requests:
            raise CodexBrainTransportError("Codex tool request is no longer pending.")
        await self._write({"id": request_id, "result": result})
        self._server_requests.pop(request_id, None)

    async def release_thread(self, thread_id: str, turn_id: str = "") -> None:
        try:
            if self._process is not None and self._process.returncode is None:
                # Interrupt before resolving outstanding calls, so cancellation
                # cannot release another model step into an abandoned turn.
                if turn_id:
                    with suppress(CodexBrainTransportError):
                        await self._rpc(
                            "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
                        )
                for request_id, owner in tuple(self._server_requests.items()):
                    if owner == thread_id:
                        with suppress(CodexBrainTransportError):
                            await self.respond(request_id, {"contentItems": [], "success": False})
                with suppress(CodexBrainTransportError):
                    await self._rpc("thread/unsubscribe", {"threadId": thread_id})
        finally:
            self._queues.pop(thread_id, None)
            for request_id, owner in tuple(self._server_requests.items()):
                if owner == thread_id:
                    self._server_requests.pop(request_id, None)

    async def _stop_process(self) -> None:
        self._ready = False
        self._stopping = True
        process = self._process
        if process is not None:
            if process.stdin is not None:
                with suppress(OSError, RuntimeError):
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                if self._tree is not None:
                    self._tree.close()
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if self._tree is not None:
            self._tree.close()
        for task in (self._reader, self._stderr_reader):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self._process = self._tree = self._reader = self._stderr_reader = None
        self._fail("Codex app-server stopped.")

    async def close(self) -> None:
        self._closed = True
        await self._stop_process()
        self._queues.clear()
        self._server_requests.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
