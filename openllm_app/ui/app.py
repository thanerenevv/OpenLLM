"""Textual application shell."""
from pathlib import Path

from textual.app import App

from ..core.hf import HFClient
from ..core.registry import load_registry
from ..core.state import load_state
from .model_screen import ModelScreen
from .models_screen import ModelsScreen
from .run_screen import RunScreen


class OpenLLMApp(App):
    TITLE = "OpenLLM"
    CSS = """
    Screen { background: ansi_default; }
    Footer, Footer FooterKey, Footer FooterLabel { background: ansi_default; }
    """

    def __init__(self, root: Path, run_kwargs: dict | None = None, hf_client=None):
        super().__init__(ansi_color=True)  # use the terminal's own colours and background
        self.root = root
        self.run_kwargs = run_kwargs or {}  # server_factory / chat_fn seams for RunScreen
        self.hf = hf_client or HFClient()  # seam for the add-model dialog
        self.registry = load_registry(root / "models.json")
        self.state_path = root / "state.json"
        self.state = load_state(self.state_path)

    def on_mount(self) -> None:
        self.push_screen(ModelsScreen())

    def on_model_screen_start_requested(self, msg: ModelScreen.StartRequested) -> None:
        self.push_screen(RunScreen(msg.model_id, msg.settings, msg.api_key, **self.run_kwargs))
