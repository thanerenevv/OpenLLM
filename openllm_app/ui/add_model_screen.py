"""Dialog that turns a Hugging Face model id into a registry entry."""
from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ..core import hf


class AddModelScreen(ModalScreen):
    """Dismisses with (model_id, entry), (model_id, None) if the file is already listed, or None."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    AddModelScreen { align: center middle; }
    AddModelScreen #dialog { width: 76; height: auto; max-height: 90%; border: thick $primary;
                             background: $surface; padding: 1 2; }
    AddModelScreen .h { text-style: bold; }
    AddModelScreen #files { display: none; height: auto; max-height: 12; margin-top: 1; }
    AddModelScreen #add-status { margin-top: 1; }
    AddModelScreen #add-status.error { color: $error; }
    AddModelScreen #cancel { margin-top: 1; }
    """

    def __init__(self, client, models: dict, runtime: str) -> None:
        super().__init__()
        self._client, self._models, self._runtime = client, models, runtime
        self._repo = ""
        self._files: list[dict] = []
        self._busy = False
        self._mode = "files"
        self._status_text = ""
        self._status_error = False

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def status_is_error(self) -> bool:
        return self._status_error

    def compose(self):
        with Vertical(id="dialog"):
            yield Static("Add a model from Hugging Face", classes="h")
            yield Static("Search by name (e.g. qwen3 8b), or paste an owner/name id or URL. Press Enter.")
            yield Input(placeholder="e.g. qwen3 8b  or  unsloth/Qwen3-8B-GGUF", id="repo")
            yield ListView(id="files")
            yield Static("", id="add-status")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#repo", Input).focus()

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status_text, self._status_error = text, error
        status = self.query_one("#add-status", Static)
        status.update(text)
        status.set_class(error, "error")

    # ------------------------------------------------------------------ flow
    @on(Input.Submitted, "#repo")
    def _submitted(self, ev: Input.Submitted) -> None:
        ev.stop()
        if self._busy:
            return
        text = ev.value.strip()
        if not text:
            self._set_status("Type a model name to search, or an owner/name id", error=True)
            return
        self._busy = True
        self._mode = "files"
        self.query_one("#files", ListView).display = False
        try:
            repo = hf.parse_repo_id(text)
        except hf.HFError:
            self._set_status(f"Searching Hugging Face for {text!r}…")
            self.run_worker(lambda: self._search(text), thread=True, exclusive=True, name="search")
            return
        self._set_status(f"Looking up {repo}…")
        self.run_worker(lambda: self._lookup(repo), thread=True, exclusive=True, name="lookup")

    def _search(self, query: str) -> None:
        try:
            results = self._client.search(query)
        except Exception as e:
            self.app.call_from_thread(self._failed, str(e) or type(e).__name__)
            return
        self.app.call_from_thread(self._got_results, results)

    def _show_picker(self, items: list[ListItem], hint: str) -> None:
        self._busy = False
        picker = self.query_one("#files", ListView)
        picker.clear()
        picker.extend(items)
        picker.display = True
        picker.index = 0
        picker.focus()
        self._set_status(hint)

    def _got_results(self, results: list[dict]) -> None:
        self._mode = "repos"
        self._show_picker([ListItem(Label(f"{r['repo']}  ({r['downloads']:,} downloads)"), name=r["repo"])
                           for r in results], "Pick a model (Enter).")

    def _lookup(self, repo: str) -> None:
        try:
            files = self._client.list_files(repo)
        except Exception as e:  # any failure must reach the status line, not kill the app
            self.app.call_from_thread(self._failed, str(e) or type(e).__name__)
            return
        self.app.call_from_thread(self._got_files, repo, files)

    def _got_files(self, repo: str, files: list[dict]) -> None:
        self._repo, self._files = repo, files
        if len(files) == 1:
            self._choose(files[0])
            return
        self._mode = "files"
        self._show_picker([ListItem(Label(f"{f['file']}  ({f['size'] / 1e9:.1f} GB)"), name=f["file"]) for f in files],
                          "Pick a file (Enter). Smaller files use less memory.")

    @on(ListView.Selected, "#files")
    def _picked(self, ev: ListView.Selected) -> None:
        ev.stop()
        if self._busy:
            return
        if self._mode == "repos":
            self._busy = True
            self._set_status(f"Looking up {ev.item.name}…")
            repo = ev.item.name
            self.run_worker(lambda: self._lookup(repo), thread=True, exclusive=True, name="lookup")
            return
        chosen = next((f for f in self._files if f["file"] == ev.item.name), None)
        if chosen:
            self._choose(chosen)

    def _choose(self, f: dict) -> None:
        for mid, m in self._models.items():
            if m["repo"] == self._repo and m["file"] == f["file"]:
                self.dismiss((mid, None))
                return
        self._busy = True
        self._set_status("Reading the model's metadata…")
        self.run_worker(lambda: self._build(self._repo, f), thread=True, exclusive=True, name="build")

    def _build(self, repo: str, f: dict) -> None:
        try:
            profile = self._client.read_profile(repo, f["file"])
            entry = hf.build_entry(repo, f["file"], f["size"], profile, self._runtime)
            mid = hf.make_model_id(repo, f["file"], self._models)
        except Exception as e:
            self.app.call_from_thread(self._failed, str(e) or type(e).__name__)
            return
        self.app.call_from_thread(self.dismiss, (mid, entry))

    def _failed(self, message: str) -> None:
        self._busy = False
        self._set_status(message, error=True)

    # ------------------------------------------------------------------ cancel
    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)
