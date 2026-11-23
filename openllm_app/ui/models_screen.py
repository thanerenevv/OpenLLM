"""Models screen: the list of models. Enter opens one model's settings; add and edit live here too."""
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, ListItem, ListView, Static

from ..core import hf, user_models
from ..core.registry import allowed_runtimes, load_registry, model_path, runtime_bin
from ..core.state import merge_defaults
from .add_model_screen import AddModelScreen
from .edit_model_screen import RESET, EditModelScreen
from .model_screen import ModelScreen, memory_human  # noqa: F401  (memory_human re-exported)


class ModelsScreen(Screen):
    BINDINGS = [
        Binding("enter", "open_model", "Open"),
        Binding("a", "add_model", "Add model"),
        Binding("e", "edit_model", "Edit model"),
        Binding("q", "quit_app", "Quit"),
    ]

    DEFAULT_CSS = """
    ModelsScreen #title { text-style: bold; color: $accent; padding: 1 3 0 3; }
    ModelsScreen #list-card { border: round ansi_bright_black; margin: 1 2; padding: 0 1; height: 1fr; }
    ModelsScreen #models { height: 1fr; background: transparent; }
    ModelsScreen #notes { height: auto; color: ansi_bright_black; padding: 1 1 0 1; }
    ModelsScreen #buttons { height: 3; margin-top: 1; }
    ModelsScreen #buttons Button { margin-right: 2; }
    ModelsScreen #status { padding: 0 1; }
    ModelsScreen #status.error { color: $error; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_id = ""
        self._status_text = ""
        self._status_error = False

    @property
    def model(self) -> dict:
        return self.app.registry["models"][self.model_id]

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def status_is_error(self) -> bool:
        return self._status_error

    # ------------------------------------------------------------------ layout
    def _missing(self, model_id: str) -> list[str]:
        m = self.app.registry["models"][model_id]
        rid = merge_defaults(m, self.app.state.get(model_id, {}))["runtime"]
        rt = self.app.registry["runtimes"][rid]
        out = []
        if not runtime_bin(self.app.root, rt, "llama-server").exists():
            out.append("runtime")
        if not model_path(self.app.root, m).exists():
            out.append("model")
        return out

    def _model_label(self, model_id: str) -> str:
        m = self.app.registry["models"][model_id]
        quant = hf.quant_of(m["file"])
        size = m.get("size_gb") or m["weights_gib"]
        state = "not downloaded" if self._missing(model_id) else "✓ ready"
        return f"{m['name']}   {quant or ''}   {size:g} GB   {state}"

    def compose(self):
        models = self.app.registry["models"]
        if self.model_id not in models:
            self.model_id = next(iter(models))
        yield Static("OpenLLM - local model launcher", id="title")
        with Vertical(id="list-card"):
            yield ListView(*[ListItem(Label(self._model_label(i), markup=False), name=i) for i in models],
                           initial_index=list(models).index(self.model_id), id="models")
            yield Static(self.model.get("notes", ""), id="notes", markup=False)
            with Horizontal(id="buttons"):
                yield Button("Open (Enter)", id="open")
                yield Button("+ Add model (a)", id="add-model")
                yield Button("Edit model (e)", id="edit-model")
            yield Static(self._status_text, id="status", classes="error" if self._status_error else "")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#models", ListView).focus()

    def on_screen_resume(self) -> None:
        """Back from a model's screen: its download state may have changed."""
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        for item in self.query_one("#models", ListView).children:
            item.query_one(Label).update(self._model_label(item.name))

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_text, self._status_error = text, error
        status = self.query_one("#status", Static)
        status.update(text)
        status.set_class(error, "error")

    # ------------------------------------------------------------------ list events
    @on(ListView.Highlighted, "#models")
    def _model_highlighted(self, ev: ListView.Highlighted) -> None:
        ev.stop()
        if ev.item and ev.item.name:
            self.model_id = ev.item.name
            self.query_one("#notes", Static).update(self.model.get("notes", ""))

    @on(ListView.Selected, "#models")
    def _model_selected(self, ev: ListView.Selected) -> None:
        ev.stop()
        self.action_open_model()

    @on(Button.Pressed, "#open")
    def _open_pressed(self) -> None:
        self.action_open_model()

    @on(Button.Pressed, "#add-model")
    def _add_pressed(self) -> None:
        self.action_add_model()

    @on(Button.Pressed, "#edit-model")
    def _edit_pressed(self) -> None:
        self.action_edit_model()

    # ------------------------------------------------------------------ actions
    def action_quit_app(self) -> None:
        self.app.exit()

    def action_open_model(self) -> None:
        self.app.push_screen(ModelScreen(self.model_id))

    def action_add_model(self) -> None:
        runtime = next(iter(self.app.registry["runtimes"]))
        self.app.push_screen(AddModelScreen(self.app.hf, self.app.registry["models"], runtime), self._model_added)

    def _model_added(self, result: tuple[str, dict | None] | None) -> None:
        if result is None:
            return
        model_id, entry = result
        if entry is not None:
            user_models.add(self.app.root / user_models.FILENAME, model_id, entry)
            reg = self.app.registry
            reg["models"][model_id] = {**entry, "runtimes": allowed_runtimes(reg, entry)}  # not persisted: stays "all"
        self.model_id = model_id
        self._status_error = False
        self._status_text = (f"Added {self.model['name']}. Open it and press d to download." if entry is not None
                             else f"{self.model['name']} is already in your list.")
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_models)

    def _focus_models(self) -> None:
        self.query_one("#models", ListView).focus()

    def action_edit_model(self) -> None:
        self.app.push_screen(EditModelScreen(self.model), self._model_edited)

    def _model_edited(self, result) -> None:
        if result is None:
            return
        path = self.app.root / user_models.OVERRIDES_FILENAME
        if result == RESET:
            user_models.clear_override(path, self.model_id)
        else:
            user_models.set_override(path, self.model_id, result)
        self.app.registry = load_registry(self.app.root / "models.json")
        self._status_error = False
        self._status_text = "Model reset to its defaults." if result == RESET else f"Saved changes to {self.model['name']}."
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_models)
