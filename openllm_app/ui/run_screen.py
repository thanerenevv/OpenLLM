"""Run screen: header status, Chat / Log / Connect tabs, and stop-the-server control."""
import asyncio
import threading
from typing import Callable

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Input, Static, TabbedContent, TabPane

from .. import client
from ..core import cmdline, snippets
from ..server import ServerProcess, install_cleanup
from .model_screen import memory_human
from .widgets import ChatPane, LogPane, SnippetBox

STATUS_WORDS = {"idle": "starting", "starting": "loading model…", "ready": "ready",
                "stopping": "stopping", "stopped": "stopped", "failed": "failed"}
SNIPPETS = (("cline", "Cline"), ("opencode", "opencode"), ("curl", "curl"))
EMPTY_ANSWER = ("No answer: reasoning used all tokens. Raise the max output tokens "
                "or lower the reasoning effort and try again.")


def tokens_k(n: int) -> str:
    return f"{n / 1024:.1f}K" if n >= 1024 else str(n)


class ConfirmStop(ModalScreen[bool]):
    BINDINGS = [Binding("y", "yes", "Stop"), Binding("n,escape", "no", "Keep going")]
    DEFAULT_CSS = """
    ConfirmStop { align: center middle; }
    ConfirmStop > Vertical { width: 56; height: auto; border: thick $error; padding: 1 2; background: $surface; }
    ConfirmStop Horizontal { height: 3; margin-top: 1; }
    """

    def compose(self):
        with Vertical():
            yield Static("A reply is still generating. Stop the server anyway?")
            with Horizontal():
                yield Button("Stop server (y)", id="yes", variant="error")
                yield Button("Keep going (n)", id="no")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "stop", "Back (stops server)"),
        Binding("s", "stop", "Stop server"),
        Binding("ctrl+c", "cancel", "Stop generating", priority=True),
        Binding("y", "copy_snippet", "Copy snippet", show=False),
    ]

    DEFAULT_CSS = """
    RunScreen #run-header { padding: 0 1; text-style: bold; background: $boost; }
    RunScreen #run-header.ready { color: $success; }
    RunScreen #run-header.failed { color: $error; }
    RunScreen #fail { display: none; height: auto; border: round $error; padding: 0 1; }
    RunScreen #fail.shown { display: block; }
    RunScreen #chat-input { dock: bottom; }
    """

    def __init__(self, model_id: str, settings: dict, api_key: str | None,
                 server_factory: Callable | None = None, chat_fn: Callable | None = None) -> None:
        super().__init__()
        self.model_id, self.settings, self.api_key = model_id, dict(settings), api_key
        self._server_factory, self._chat_fn = server_factory, chat_fn or client.stream_chat
        self.server = None
        self.server_state = "idle"
        self.messages: list[dict] = []
        self.generating = False
        self.ctx_used = 0
        self.tps: float | None = None
        self._chat_worker = None
        self._stopping = self._closing = False
        self._fail_text = ""
        self._ui_thread = threading.get_ident()

    # ----------------------------------------------------------- derived values
    @property
    def port(self) -> int:
        return int(self.settings.get("port", 8080))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def model(self) -> dict:
        return self.app.registry["models"][self.model_id]

    @property
    def header_text(self) -> str:
        parts = [f"● {self.model['name']}", STATUS_WORDS[self.server_state], f":{self.port}",
                 f"ctx {tokens_k(self.ctx_used)}/{memory_human(self.settings['ctx'])}"]
        if self.tps:
            parts.append(f"{self.tps:.1f} tok/s")
        return "  ".join(parts)

    @property
    def fail_text(self) -> str:
        return self._fail_text

    @property
    def connect_text(self) -> str:
        return "\n".join(box.snippet for box in self.query(SnippetBox))

    # ------------------------------------------------------------------ layout
    def _snippet_settings(self) -> dict:
        s = {**self.settings, "model_id": self.model_id, "model_name": self.model["name"], "port": self.port}
        if s.get("mode") == "chat":
            s["expose"] = "local"
        return s

    def compose(self):
        snip = snippets.client_snippets(self._snippet_settings(), self.api_key)
        yield Static(self.header_text, id="run-header")
        with Vertical(id="fail"):
            yield Static("", id="fail-text", markup=False)
            yield Button("Back to settings", id="back", variant="primary")
        with TabbedContent(initial="chat-tab"):
            with TabPane("Chat", id="chat-tab"):
                yield ChatPane(id="transcript")
                yield Input(placeholder="Loading model…", id="chat-input", disabled=True)
            with TabPane("Log", id="log-tab"):
                yield LogPane(id="log")
            with TabPane("Connect", id="connect-tab"):
                with VerticalScroll():
                    yield Static(f"Base URL: {snip['base_url']}\nAPI key:  {snip['api_key']}\n"
                                 f"Model ID: {self.model_id}", markup=False)
                    for key, title in SNIPPETS:
                        yield Static(f"{title}  (focus + y to copy)", classes="h")
                        yield SnippetBox(snip[key], id=f"snip-{key}")
        yield Footer()

    # --------------------------------------------------------------- lifecycle
    def on_mount(self) -> None:
        s = {**self.settings, "mode": "server", "model_id": self.model_id, "port": self.port}
        if self.settings.get("mode") == "chat":
            s["expose"] = "local"
        rt = self.app.registry["runtimes"][self.settings.get("runtime", self.model["runtime"])]
        key = self.api_key if s.get("expose") == "lan" else None
        argv = cmdline.build_command(self.app.root, rt, self.model, s, key)
        real = self._server_factory is None
        self.server = (self._server_factory or ServerProcess)(argv, "127.0.0.1", self.port,
                                                                self._line_from_thread, self._state_from_thread)
        if real:
            install_cleanup(self.server)
        self.server.start()

    def on_unmount(self) -> None:
        self._closing = True
        if self.server is not None:
            self.server.stop(3.0)

    def _marshal(self, fn, *args) -> None:
        if self._closing:
            return
        if threading.get_ident() == self._ui_thread:
            fn(*args)
            return
        try:
            self.app.call_from_thread(fn, *args)
        except RuntimeError:  # app already shutting down
            pass

    def _line_from_thread(self, line: str) -> None:
        self._marshal(self._on_line, line)

    def _state_from_thread(self, state: str) -> None:
        self._marshal(self._on_state, state)

    # ----------------------------------------------------------------- reactions
    def _refresh_header(self) -> None:
        header = self.query_one("#run-header", Static)
        header.update(self.header_text)
        header.set_class(self.server_state == "ready", "ready")
        header.set_class(self.server_state == "failed", "failed")

    def _on_line(self, line: str) -> None:
        self.query_one(LogPane).add(line)

    def _on_state(self, state: str) -> None:
        self.server_state = state
        self._refresh_header()
        box = self.query_one("#chat-input", Input)
        if state == "ready":
            box.disabled = False
            box.placeholder = "Message (↵ to send, ctrl+c stops a reply)"
            box.focus()
        elif state in ("failed", "stopping", "stopped"):
            box.disabled = True
            box.placeholder = "Server is not running"
        if state == "failed":
            self._cancel_chat("Server stopped unexpectedly.")
            reason = getattr(self.server, "failure_reason", None) or "The server stopped."
            tail = "\n".join(self.server.last_lines(30))
            self._fail_text = f"{reason}\n\n{tail}".rstrip()
            self.query_one("#fail-text", Static).update(self._fail_text)
            self.query_one("#fail").add_class("shown")

    # -------------------------------------------------------------------- chat
    @on(Input.Submitted, "#chat-input")
    def _submit(self, ev: Input.Submitted) -> None:
        ev.stop()
        text = ev.value.strip()
        if not text or self.generating or self.server_state != "ready":
            return
        ev.input.value = ""
        self.messages.append({"role": "user", "content": text})
        self.query_one(ChatPane).add_user(text)
        self.generating = True
        self._chat_worker = self.run_worker(self._chat_task(), exclusive=True, group="chat", name="chat")

    async def _chat_task(self) -> None:
        pane = self.query_one(ChatPane)
        reply: list[str] = []
        try:
            async for e in self._chat_fn(self.base_url, self.api_key, self.model_id,
                                         list(self.messages), int(self.settings.get("max_tokens", 16384))):
                if e.kind == "reasoning":
                    pane.append_reasoning(e.text)
                elif e.kind == "content":
                    reply.append(e.text)
                    pane.append_content(e.text)
                elif e.kind == "usage" and e.data:
                    self.ctx_used = e.data["prompt_tokens"] + e.data["completion_tokens"]
                    self._refresh_header()
                elif e.kind == "timings" and e.data:
                    self.tps = e.data.get("predicted_per_second") or self.tps
                    self._refresh_header()
                elif e.kind == "done" and e.data and e.data.get("empty_answer"):
                    pane.add_notice(EMPTY_ANSWER, error=True)
                elif e.kind == "error":
                    pane.add_notice(e.text, error=True)
        finally:
            pane.end_reply()
            self.generating = False
            if reply:
                self.messages.append({"role": "assistant", "content": "".join(reply)})
            elif self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()  # keep user/assistant turns alternating for the chat template

    def _cancel_chat(self, notice: str) -> None:
        if self._chat_worker is not None and self.generating:
            self._chat_worker.cancel()
            self.generating = False
            self.query_one(ChatPane).add_notice(notice, error=notice != "(stopped)")

    # ------------------------------------------------------------------ actions
    def action_cancel(self) -> None:
        if self.generating:
            self._cancel_chat("(stopped)")
        else:
            self.notify("Nothing is generating. Esc goes back and stops the server; ctrl+q quits.")

    def action_stop(self) -> None:
        if self._stopping:
            return
        if self.generating:
            self.app.push_screen(ConfirmStop(), self._after_confirm)
        else:
            self._stop_and_back()

    def _after_confirm(self, yes: bool | None) -> None:
        if yes:
            self._stop_and_back()

    @on(Button.Pressed, "#back")
    def _back_pressed(self) -> None:
        self._stop_and_back()

    def _stop_and_back(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._chat_worker is not None:
            self._chat_worker.cancel()
        self.run_worker(self._stop_thread, thread=True, name="stop")

    def _stop_thread(self) -> None:
        self.server.stop()
        self.app.call_from_thread(self._leave)

    def _leave(self) -> None:
        if self.app.screen is self:
            self.app.pop_screen()

    def action_copy_snippet(self) -> None:
        box = self.focused
        if isinstance(box, SnippetBox):
            self.app.copy_to_clipboard(box.snippet)
            self.notify("Copied.")
