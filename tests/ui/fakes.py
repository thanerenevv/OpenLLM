"""Fakes injected into RunScreen: a scriptable server and a scriptable chat stream."""
import asyncio

from openllm_app.client import ChatEvent


class FakeServer:
    def __init__(self, argv, host, port, on_line, on_state):
        self.argv, self.host, self.port = argv, host, port
        self._on_line, self._on_state = on_line, on_state
        self.state, self.failure_reason, self.pid = "idle", None, None
        self.stopped, self._lines = False, []

    def start(self):
        self.set_state("starting")

    def stop(self, timeout=10.0):
        self.stopped = True
        self.set_state("stopped")

    def last_lines(self, n=30):
        return self._lines[-n:]

    def emit_line(self, line):
        self._lines.append(line)
        self._on_line(line)

    def set_state(self, state, reason=None):
        self.state = state
        if reason:
            self.failure_reason = reason
        self._on_state(state)


class Factory:
    """server_factory that remembers the server it built."""

    def __init__(self):
        self.server = None

    def __call__(self, argv, host, port, on_line, on_state):
        self.server = FakeServer(argv, host, port, on_line, on_state)
        return self.server


def scripted_chat(events, gate: asyncio.Event | None = None):
    """chat_fn replaying `events`; the string 'WAIT' blocks until gate is set."""
    calls = []

    async def chat(base_url, api_key, model_id, messages, max_tokens):
        calls.append({"base_url": base_url, "api_key": api_key, "model_id": model_id,
                      "messages": [dict(m) for m in messages], "max_tokens": max_tokens})
        for e in events:
            if e == "WAIT":
                await gate.wait()
            else:
                yield e

    chat.calls = calls
    return chat


def ev(kind, text="", **data):
    return ChatEvent(kind, text, data or None)
