"""Dialog for editing a model's registry entry (saved as an override; the curated list is untouched)."""
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..core.user_models import SAMPLING_KEYS, clean_overrides

RESET = "reset"


class EditModelScreen(ModalScreen):
    """Dismisses with a dict of edited fields, the string 'reset', or None (cancelled)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    EditModelScreen { align: center middle; }
    EditModelScreen #dialog { width: 76; height: 90%; border: thick $primary;
                              background: $surface; padding: 1 2; }
    EditModelScreen #fields { height: 1fr; }
    EditModelScreen .h { text-style: bold; }
    EditModelScreen Label { margin-top: 1; }
    EditModelScreen #edit-status { margin-top: 1; color: $error; }
    EditModelScreen Horizontal { height: 3; margin-top: 1; }
    EditModelScreen Horizontal Button { margin-right: 2; }
    """

    def __init__(self, model: dict) -> None:
        super().__init__()
        self._m = model
        self._status_text = ""

    @property
    def status_text(self) -> str:
        return self._status_text

    def compose(self):
        m = self._m
        with Vertical(id="dialog"):
            yield Static(f"Edit model: {m['name']}", classes="h")
            with VerticalScroll(id="fields"):
                yield from self._fields(m)
            yield Static("", id="edit-status")
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Reset to defaults", id="reset")
                yield Button("Cancel", id="cancel")

    def _fields(self, m: dict):
        yield Label("Name")
        yield Input(value=m["name"], id="f-name")
        yield Label("Hugging Face repo (owner/name)")
        yield Input(value=m["repo"], id="f-repo")
        yield Label("File (.gguf)")
        yield Input(value=m["file"], id="f-file")
        yield Label("Max context (tokens, e.g. 131072 or 128k)")
        yield Input(value=str(m["max_ctx"]), id="f-max_ctx")
        for k in SAMPLING_KEYS:
            yield Label(f"Sampling: {k}")
            yield Input(value=str(m["sampling"].get(k, "")), id=f"f-{k}")
        yield Label("Reasoning efforts (comma separated, e.g. off,low,high)")
        yield Input(value=",".join(m["reasoning_efforts"]), id="f-efforts")

    def _val(self, key: str) -> str:
        return self.query_one(f"#f-{key}", Input).value.strip()

    def _fail(self, text: str) -> None:
        self._status_text = text
        self.query_one("#edit-status", Static).update(text)

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        from ..core.memory import parse_tokens
        try:
            max_ctx = parse_tokens(self._val("max_ctx"))
            sampling = {k: (int(self._val(k)) if k == "top_k" else float(self._val(k))) for k in SAMPLING_KEYS}
        except ValueError:
            self._fail("Max context and sampling values must be numbers")
            return
        fields = clean_overrides({
            "name": self._val("name"), "repo": self._val("repo"), "file": self._val("file"),
            "max_ctx": max_ctx, "sampling": sampling,
            "reasoning_efforts": [e for e in self._val("efforts").split(",") if e.strip()],
        })
        for key, label in (("name", "Name"), ("repo", "Repo (owner/name)"), ("file", "File (a top-level .gguf)"),
                           ("max_ctx", "Max context (at least 1024)"), ("reasoning_efforts", "Reasoning efforts")):
            if key not in fields:
                self._fail(f"{label} is missing or invalid")
                return
        self.dismiss(fields)

    @on(Button.Pressed, "#reset")
    def _reset(self) -> None:
        self.dismiss(RESET)

    @on(Button.Pressed, "#cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
