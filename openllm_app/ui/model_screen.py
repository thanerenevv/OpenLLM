"""Model screen: one model's settings as `< value >` selectors, a live memory gauge, download and start."""
import secrets

from textual import on
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, ProgressBar, Static

from ..core import downloads, hf, memory
from ..core.registry import model_path, runtime_bin
from ..core.state import merge_defaults, save_state
from .widgets import Cycler, MemoryGauge

CTX_PRESETS = (8192, 16384, 32768, 65536, 131072)
OUT_PRESETS = (4096, 8192, 16384, 32768, 65536)
KV_OPTIONS = [("f16", "f16  (best quality)"), ("q8_0", "q8_0  (half the memory)"), ("q4_0", "q4_0  (quarter)")]
MODE_OPTIONS = [("server", "API server"), ("chat", "Terminal chat")]
EXPOSE_OPTIONS = [("local", "This Mac only"), ("lan", "My network (API key)")]


def memory_human(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def _sizes(presets, current: int, limit: int | None = None) -> list[tuple[str, str]]:
    vals = sorted({p for p in presets if limit is None or p <= limit} | {current})
    return [(str(v), memory_human(v)) for v in vals]


class ModelScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "start", "Start"),
        Binding("d", "download", "Download"),
        Binding("up", "app.focus_previous", "", show=False),
        Binding("down", "app.focus_next", "", show=False),
    ]

    DEFAULT_CSS = """
    ModelScreen #card { border: round ansi_bright_black; margin: 1 2; padding: 1 3; height: 1fr; }
    ModelScreen #m-title { text-style: bold; color: $accent; }
    ModelScreen #m-sub { color: ansi_bright_black; margin-bottom: 1; }
    ModelScreen #content { height: 1fr; }
    ModelScreen #grid { grid-size: 3; grid-gutter: 0 2; grid-rows: 4; height: auto; }
    ModelScreen #port-row { height: 3; margin-top: 1; }
    ModelScreen #port-row Label { width: 8; content-align: left middle; height: 3; }
    ModelScreen #port { width: 14; }
    ModelScreen #ctx-custom { width: 36; margin-top: 1; }
    ModelScreen #status { margin-top: 1; color: ansi_bright_black; }
    ModelScreen #status.error { color: $error; text-style: bold; }
    ModelScreen #dl { display: none; }
    ModelScreen #bottom { height: 4; margin-top: 1; }
    ModelScreen #start { width: 16; height: 3; margin-top: 1; margin-right: 3; border: round ansi_default;
                         background: transparent; color: ansi_default; text-style: bold; }
    ModelScreen #start:hover, ModelScreen #start:focus { border: round $accent; color: $accent; }
    ModelScreen #mode { width: 30; }
    """

    class StartRequested(Message):
        def __init__(self, model_id: str, settings: dict, api_key: str | None) -> None:
            super().__init__()
            self.model_id, self.settings, self.api_key = model_id, settings, api_key

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.settings: dict = {}
        self._status_text = ""
        self._status_error = False
        self._port_ok = True
        self._downloading = False
        self._budget: float | None = None

    def _ensure_init(self) -> None:
        if not self.settings:
            self.settings = merge_defaults(self.model, self.app.state.get(self.model_id, {}))

    @property
    def model(self) -> dict:
        return self.app.registry["models"][self.model_id]

    @property
    def runtime(self) -> dict:
        return self.app.registry["runtimes"][self.settings["runtime"]]

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def status_is_error(self) -> bool:
        return self._status_error

    def _missing(self) -> list[str]:
        out = []
        if not runtime_bin(self.app.root, self.runtime, "llama-server").exists():
            out.append("runtime")
        if not model_path(self.app.root, self.model).exists():
            out.append("model")
        return out

    def _ctx_options(self) -> list[tuple[str, str]]:
        return _sizes(CTX_PRESETS, self.settings["ctx"], self.model["max_ctx"])

    # --------------------------------------------------------------- layout
    def compose(self):
        self._ensure_init()
        s, m = self.settings, self.model
        rts = self.app.registry["runtimes"]
        quant = hf.quant_of(m["file"])
        with Vertical(id="card"):
            yield Static(m["name"], id="m-title", markup=False)
            yield Static(f"{quant}  ·  {m['file']}" if quant else m["file"], id="m-sub", markup=False)
            with VerticalScroll(id="content"):
                with Grid(id="grid"):
                    yield Cycler("Effort", [(e, e) for e in m["reasoning_efforts"]], s["effort"], id="effort")
                    yield Cycler("Runtime", [(r, r) for r in m["runtimes"]], s["runtime"], id="runtime")
                    yield Cycler("KV Cache", KV_OPTIONS, s["kv"], id="kv")
                    yield Cycler("Context", self._ctx_options(), str(s["ctx"]), id="ctx")
                    yield Cycler("Max Output", _sizes(OUT_PRESETS, s["max_tokens"]), str(s["max_tokens"]), id="max-out")
                    yield Cycler("API access", EXPOSE_OPTIONS, s["expose"], id="expose")
                yield Input(placeholder="custom context, e.g. 48k or 49152", id="ctx-custom")
                with Horizontal(id="port-row"):
                    yield Label("Port")
                    yield Input(value=str(s["port"]), type="integer", id="port")
                gauge = MemoryGauge(id="gauge")
                gauge.update_estimate(*self._estimate())
                yield gauge
                yield ProgressBar(total=100, show_eta=False, id="dl")
                yield Static(self._status_text, id="status", classes="error" if self._status_error else "")
            with Horizontal(id="bottom"):
                yield Button("Start", id="start", flat=False)
                yield Cycler("Mode", MODE_OPTIONS, s["mode"], id="mode")
        yield Footer()

    def on_mount(self) -> None:
        self._apply_mode_visibility()
        self.query_one("#effort", Cycler).focus()

    def _estimate(self) -> tuple[float, float]:
        used = memory.estimate_gb(self.model, self.settings["ctx"], self.settings["kv"])
        if self._budget is None:  # RAM doesn't change; asking sysctl on every keypress spawned a process each time
            self._budget = memory.gpu_budget_gb(memory.total_ram_gb())
        return used, self._budget

    def _apply_mode_visibility(self) -> None:
        server = self.settings["mode"] == "server"
        self.query_one("#expose").display = server
        self.query_one("#port-row").display = server

    def _changed(self) -> None:
        self.query_one(MemoryGauge).update_estimate(*self._estimate())
        self._apply_mode_visibility()

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_text, self._status_error = text, error
        status = self.query_one("#status", Static)
        status.update(text)
        status.set_class(error, "error")

    # ------------------------------------------------------------ settings events
    @on(Cycler.Changed)
    def _cycler_changed(self, ev: Cycler.Changed) -> None:
        ev.stop()
        key, val = ev.cycler.id, ev.value
        if key in ("effort", "runtime", "kv", "expose", "mode"):
            self.settings[key] = val
        elif key == "ctx":
            self.settings["ctx"] = int(val)
        elif key == "max-out":
            self.settings["max_tokens"] = int(val)
        self._changed()
        if key == "runtime":
            self._set_status("The runtime is not downloaded yet - press d to download."
                             if "runtime" in self._missing() else "")

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
        self.query_one("#ctx", Cycler).set_options(self._ctx_options(), str(ctx))
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

    @on(Button.Pressed, "#start")
    def _start_pressed(self) -> None:
        self.action_start()

    # ------------------------------------------------------------------ actions
    def action_back(self) -> None:
        if not self._downloading:
            self.app.pop_screen()

    def action_start(self) -> None:
        missing = self._missing()
        if missing:
            self._set_status(f"The {' and '.join(missing)} not downloaded yet - press d to download.", error=True)
            return
        if self.settings["mode"] == "server" and not self._port_ok:
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
        app, root, m, rt, rt_id = self.app, self.app.root, self.model, self.runtime, self.settings["runtime"]

        def progress(done: int, total: int | None) -> None:
            app.call_from_thread(self._download_progress, done, total)

        error = None
        try:
            if not runtime_bin(root, rt, "llama-server").exists():
                downloads.install_runtime(root, rt_id, rt, progress)
            if not model_path(root, m).exists():
                downloads.fetch_model(root, m, progress)
        except Exception as e:  # any failure must reach the status line, not kill the app
            error = str(e) or type(e).__name__
        app.call_from_thread(self._download_finished, error)

    def _download_progress(self, done: int, total: int | None) -> None:
        self.query_one("#dl", ProgressBar).update(total=total or None, progress=done)

    def _download_finished(self, error: str | None) -> None:
        self._downloading = False
        self.query_one("#dl", ProgressBar).display = False
        self._set_status(error, True) if error else self._set_status("Download complete.")
