"""Textual application shell."""
from pathlib import Path

from textual.app import App

from ..core.registry import load_registry
from ..core.state import load_state
from .models_screen import ModelsScreen
from .run_screen import RunScreen


class OpenOllamaApp(App):
    TITLE = "OpenOllama"

    def __init__(self, root: Path, run_kwargs: dict | None = None):
        super().__init__()
        self.root = root
        self.run_kwargs = run_kwargs or {}  # server_factory / chat_fn seams for RunScreen
        self.registry = load_registry(root / "models.json")
        self.state_path = root / "state.json"
        self.state = load_state(self.state_path)

    def on_mount(self) -> None:
        self.push_screen(ModelsScreen())

    def on_models_screen_start_requested(self, msg: ModelsScreen.StartRequested) -> None:
        self.push_screen(RunScreen(msg.model_id, msg.settings, msg.api_key, **self.run_kwargs))
