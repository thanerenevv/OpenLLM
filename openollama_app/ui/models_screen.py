"""Models screen: pick a model, tune settings with a live memory gauge, start or download."""
import secrets

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, ProgressBar, RadioButton, RadioSet, Static

from ..core import downloads, memory
from ..core.registry import model_path, runtime_bin
from ..core.state import merge_defaults, save_state
from .widgets import MemoryGauge

CTX_PRESETS = (8192, 16384, 32768, 65536, 131072)
KV_OPTIONS = [("f16", "f16  (best quality)"), ("q8_0", "q8_0 (half the memory, near-lossless)"),
              ("q4_0", "q4_0 (quarter the memory)")]
MODE_OPTIONS = [("server", "OpenAI-compatible API server"), ("chat", "Terminal chat")]
EXPOSE_OPTIONS = [("local", "This Mac only (127.0.0.1)"), ("lan", "My network (0.0.0.0, API key)")]


class ModelsScreen(Screen):
    BINDINGS = [
        Binding("enter", "start", "Start"),
        Binding("d", "download", "Download"),
        Binding("l", "last", "Last settings"),
        Binding("q", "quit_app", "Quit"),
    ]

    DEFAULT_CSS = """
    ModelsScreen #title { text-style: bold; padding: 0 1; background: $boost; }
    ModelsScreen #left { width: 34; border-right: solid $primary; padding: 0 1; }
    ModelsScreen #right { padding: 0 2; }
    ModelsScreen .h { text-style: bold; margin-top: 1; }
    ModelsScreen RadioSet { layout: horizontal; height: auto; border: none; padding: 0; }
    ModelsScreen Horizontal.row { height: 3; }
    ModelsScreen Button.step { min-width: 5; width: 5; }
    ModelsScreen #ctx-label { width: 10; content-align: center middle; height: 3; }
    ModelsScreen #ctx-custom { width: 30; margin-left: 2; }
    ModelsScreen #status.error { color: $error; }
    ModelsScreen #dl { display: none; }
    ModelsScreen #start { margin-top: 1; }
    """

    class StartRequested(Message):
        def __init__(self, model_id: str, settings: dict, api_key: str | None) -> None:
            super().__init__()
            self.model_id, self.settings, self.api_key = model_id, settings, api_key

    # ------------------------------------------------------------------ setup
    def __init__(self) -> None:
        super().__init__()
        self.model_id = ""
        self.settings: dict = {}
        self._status_text = ""
        self._status_error = False
        self._port_ok = True
        self._downloading = False

    def _ensure_init(self) -> None:
        if not self.model_id:
            self.model_id = next(iter(self.app.registry["models"]))
            self.settings = merge_defaults(self.model, self.app.state.get(self.model_id, {}))

    @property
    def model(self) -> dict:
        return self.app.registry["models"][self.model_id]

    @property
    def runtime(self) -> dict:
        return self.app.registry["runtimes"][self.model["runtime"]]

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def status_is_error(self) -> bool:
        return self._status_error

    def _missing(self, model_id: str | None = None) -> list[str]:
        m = self.app.registry["models"][model_id or self.model_id]
        rt = self.app.registry["runtimes"][m["runtime"]]
        out = []
        if not runtime_bin(self.app.root, rt, "llama-server").exists():
            out.append("runtime")
        if not model_path(self.app.root, m).exists():
            out.append("model")
        return out

    def _presets(self) -> list[int]:
        return [c for c in CTX_PRESETS if c <= self.model["max_ctx"]]

    # --------------------------------------------------------------- layout
    def _radios(self, group: str, options: list[tuple[str, str]]) -> RadioSet:
        return RadioSet(*[RadioButton(label, value=(val == self.settings[group]), name=val, id=f"{group}-{val}")
                          for val, label in options], id=group)

    def compose(self):
        self._ensure_init()
        s, m = self.settings, self.model
        models = self.app.registry["models"]
        yield Static("OpenOllama - local model launcher", id="title")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("Models", classes="h")
                yield ListView(*[ListItem(Label(self._model_label(i)), name=i) for i in models],
                               initial_index=list(models).index(self.model_id), id="models")
            with VerticalScroll(id="right"):
                yield Static("Mode", classes="h")
                yield self._radios("mode", MODE_OPTIONS)
                yield Static("KV cache precision (lower = more context in RAM)", classes="h")
                yield self._radios("kv", KV_OPTIONS)
                yield Static("Context length", classes="h")
                with Horizontal(classes="row"):
                    yield Button("◂", id="ctx-down", classes="step")
                    yield Label(memory_human(s["ctx"]), id="ctx-label")
                    yield Button("▸", id="ctx-up", classes="step")
                    yield Input(placeholder="custom, e.g. 48k or 49152", id="ctx-custom")
                yield Static("Reasoning effort", classes="h")
                yield self._radios("effort", [(e, e) for e in m["reasoning_efforts"]])
                with Vertical(id="server-opts"):
                    yield Static("Who can reach the API?", classes="h")
                    yield self._radios("expose", EXPOSE_OPTIONS)
                    yield Static("Port", classes="h")
                    yield Input(value=str(s["port"]), type="integer", id="port")
                with Vertical(id="chat-opts"):
                    yield Static("Max output tokens per reply (thinking counts; keep >= 16384)", classes="h")
                    yield Input(value=str(s["max_tokens"]), type="integer", id="max-tokens")
                gauge = MemoryGauge(id="gauge")
                gauge.update_estimate(*self._estimate())
                yield gauge
                yield ProgressBar(total=100, show_eta=False, id="dl")
                yield Static(self._status_text, id="status", classes="error" if self._status_error else "")
                yield Button("Start", id="start", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._apply_mode_visibility()

    def _model_label(self, model_id: str) -> str:
        m = self.app.registry["models"][model_id]
        return f"{m['name']}  " + ("(not downloaded)" if self._missing(model_id) else "✓")

    def _estimate(self) -> tuple[float, float]:
        used = memory.estimate_gb(self.model, self.settings["ctx"], self.settings["kv"])
        return used, memory.gpu_budget_gb(memory.total_ram_gb())

    def _apply_mode_visibility(self) -> None:
        server = self.settings["mode"] == "server"
        self.query_one("#server-opts").display = server
        self.query_one("#chat-opts").display = not server

    def _changed(self) -> None:
        self.query_one(MemoryGauge).update_estimate(*self._estimate())
        self.query_one("#ctx-label", Label).update(memory_human(self.settings["ctx"]))
        self._apply_mode_visibility()

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_text, self._status_error = text, error
        status = self.query_one("#status", Static)
        status.update(text)
        status.set_class(error, "error")

    # ------------------------------------------------------------ settings events
    @on(RadioSet.Changed)
    def _radio_changed(self, ev: RadioSet.Changed) -> None:
        ev.stop()
        group, val = ev.radio_set.id, ev.pressed.name
        if group in ("mode", "kv", "effort", "expose") and self.settings.get(group) != val:
            self.settings[group] = val
            self._changed()

    @on(Button.Pressed, "#ctx-up")
    def _ctx_up(self) -> None:
        bigger = [c for c in self._presets() if c > self.settings["ctx"]]
        if bigger:
            self.settings["ctx"] = bigger[0]
            self._changed()

    @on(Button.Pressed, "#ctx-down")
    def _ctx_down(self) -> None:
        smaller = [c for c in self._presets() if c < self.settings["ctx"]]
        if smaller:
            self.settings["ctx"] = smaller[-1]
            self._changed()

    @on(Input.Submitted, "#ctx-custom")
    def _ctx_custom(self, ev: Input.Submitted) -> None:
        ev.stop()
        try:
            n = memory.parse_tokens(ev.value)
        except ValueError:
            self._set_status("Context must be a positive number like 48k or 49152", error=True)
            return
        ctx = memory.clamp_ctx(n, self.model["max_ctx"])
        self.settings["ctx"] = ctx
        self._changed()
        self._set_status("" if ctx == n else f"Context limited to {memory_human(ctx)} for this model")

    @on(Input.Changed, "#port")
    def _port_changed(self, ev: Input.Changed) -> None:
        ev.stop()
        if ev.value.isdigit() and 1 <= int(ev.value) <= 65535:
            self.settings["port"] = int(ev.value)
            self._port_ok = True
            if self._status_error and "Port" in self._status_text:
                self._set_status("")
        else:
            self._port_ok = False
            self._set_status("Port must be a number from 1 to 65535", error=True)

    @on(Input.Changed, "#max-tokens")
    def _max_tokens_changed(self, ev: Input.Changed) -> None:
        ev.stop()
        if ev.value.isdigit() and int(ev.value) >= 1:
            self.settings["max_tokens"] = int(ev.value)

    @on(ListView.Highlighted, "#models")
    def _model_highlighted(self, ev: ListView.Highlighted) -> None:
        ev.stop()
        new = ev.item.name if ev.item else None
        if new and new != self.model_id:
            self.model_id = new
            self.settings = merge_defaults(self.model, self.app.state.get(new, {}))
            self.refresh(recompose=True)
            self.call_after_refresh(self._focus_models)  # recompose rebuilds the list and drops focus

    def _focus_models(self) -> None:
        self.query_one("#models", ListView).focus()

    @on(ListView.Selected, "#models")
    def _model_selected(self, ev: ListView.Selected) -> None:
        ev.stop()
        self.action_start()

    @on(Button.Pressed, "#start")
    def _start_pressed(self) -> None:
        self.action_start()

    # ------------------------------------------------------------------ actions
    def action_quit_app(self) -> None:
        self.app.exit()

    def action_last(self) -> None:
        self.settings = merge_defaults(self.model, self.app.state.get(self.model_id, {}))
        self._port_ok = True
        self._set_status("")
        self.refresh(recompose=True)

    def action_start(self) -> None:
        missing = self._missing()
        if missing:
            self._set_status(f"The {' and '.join(missing)} not downloaded yet - press d to download.", error=True)
            return
        if not self._port_ok:
            self._set_status("Port must be a number from 1 to 65535", error=True)
            return
        s = dict(self.settings)
        api_key = None
        if s["mode"] == "server" and s["expose"] == "lan":
            api_key = self.app.state.get("api_key") or "sk-" + secrets.token_urlsafe(24)
            self.app.state["api_key"] = api_key
        self.app.state[self.model_id] = dict(s)
        save_state(self.app.state_path, self.app.state)
        s["model_id"], s["model_name"] = self.model_id, self.model["name"]
        self.post_message(self.StartRequested(self.model_id, s, api_key))

    def action_download(self) -> None:
        if self._downloading:
            return
        if not self._missing():
            self._set_status("Everything is already downloaded.")
            return
        self._downloading = True
        self.query_one("#dl", ProgressBar).display = True
        self._set_status("Downloading…")
        self.run_worker(self._download_thread, thread=True, exclusive=True, name="download")

    def _download_thread(self) -> None:
        app, root, m, rt = self.app, self.app.root, self.model, self.runtime

        def progress(done: int, total: int | None) -> None:
            app.call_from_thread(self._download_progress, done, total)

        error = None
        try:
            if not runtime_bin(root, rt, "llama-server").exists():
                downloads.install_runtime(root, m["runtime"], rt, progress)
            if not model_path(root, m).exists():
                downloads.fetch_model(root, m, progress)
        except Exception as e:  # any failure must reach the status line, not kill the app
            error = str(e) or type(e).__name__
        app.call_from_thread(self._download_finished, error)

    def _download_progress(self, done: int, total: int | None) -> None:
        bar = self.query_one("#dl", ProgressBar)
        bar.update(total=total or None, progress=done)

    def _download_finished(self, error: str | None) -> None:
        self._downloading = False
        self._status_text, self._status_error = (error, True) if error else ("Download complete.", False)
        self.refresh(recompose=True)


def memory_human(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)
