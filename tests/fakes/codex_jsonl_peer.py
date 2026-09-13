"""Deterministic app-server process and containment fakes for transport contracts."""

import asyncio
import json

class FakeTree:
    supports_containment = True

    def __init__(self):
        self.pids = []
        self.closed = False

    def assign(self, pid):
        self.pids.append(pid)

    def close(self):
        self.closed = True


class FakeInput:
    def __init__(self, process):
        self.process = process

    def write(self, data):
        frame = json.loads(data)
        self.process.sent.append(frame)
        if "method" not in frame:
            return
        method = frame["method"]
        result = {}
        if method == "initialize":
            result = {"userAgent": "Codex Desktop/0.153.4 (test peer)"}
        if method == "config/read":
            disabled = any('mcp_servers.outside.enabled=false' == arg for arg in self.process.command)
            result = {"config": {"approval_policy": "never", "sandbox_mode": "read-only", "model_provider": "openai", "forced_login_method": "chatgpt", "web_search": "disabled", "mcp_servers": {"outside": {"enabled": not disabled, "env": {"TOKEN": "do-not-log"}}}}}
        elif method == "account/read":
            result = {"account": {"type": self.process.auth, "email": "private@example.invalid"}}
        elif method == "thread/start":
            result = {"thread": {"id": "thread-1"}}
        elif method == "turn/start":
            result = {"turn": {"id": "turn-1"}}
        elif method == "wait/forever":
            return
        if "id" in frame:
            self.process.emit({"id": frame["id"], "result": result})

    async def drain(self):
        return None

    def close(self):
        self.process.finish()


class FakeProcess:
    def __init__(self, command, auth="chatgpt"):
        self.command = command
        self.auth = auth
        self.pid = 43210
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()
        self.stdin = FakeInput(self)
        self.sent = []
        self.done = asyncio.Event()

    def emit(self, frame):
        self.stdout.feed_data((json.dumps(frame) + "\n").encode())

    def finish(self):
        self.returncode = 0
        if not self.stdout.at_eof():
            self.stdout.feed_eof()
        self.done.set()

    async def wait(self):
        await self.done.wait()
        return self.returncode

    def terminate(self):
        self.finish()

    def kill(self):
        self.finish()


