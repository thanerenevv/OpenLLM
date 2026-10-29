"""Textual application shell."""
from pathlib import Path

from textual.app import App

from ..core.registry import load_registry
from ..core.state import load_state
from .models_screen import ModelsScreen


class OpenOllamaApp(App):
    TITLE = "OpenOllama"

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.registry = load_registry(root / "models.json")
        self.state_path = root / "state.json"
        self.state = load_state(self.state_path)

    def on_mount(self) -> None:
        self.push_screen(ModelsScreen())

    def on_models_screen_start_requested(self, msg: ModelsScreen.StartRequested) -> None:
        self.notify(f"Starting {msg.model_id}… (Run screen is added in the next step)")
