import json

import pytest
from textual.widgets import Input

from openllm_app.core import memory, user_models
from openllm_app.ui.app import OpenLLMApp
from openllm_app.ui.edit_model_screen import EditModelScreen
from openllm_app.ui.models_screen import ModelsScreen

SIZE = (120, 50)


@pytest.fixture(autouse=True)
def big_mac(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 64.0)


@pytest.fixture
def app_root(root, registry):
    (root / "models.json").write_text(json.dumps(registry))
    return root


async def test_edit_saves_an_override_and_reset_clears_it(app_root):
    app = OpenLLMApp(app_root)
    async with app.run_test(size=SIZE) as pilot:
        screen = app.screen
        mid = screen.model_id
        original = screen.model["name"]
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(app.screen, EditModelScreen)
        app.screen.query_one("#f-name", Input).value = "Heretic Tuned"
        app.screen.query_one("#f-temp", Input).value = "0.3"
        app.screen.query_one("#f-max_ctx", Input).value = "16k"
        await pilot.click("#save")
        await pilot.pause()
        assert isinstance(app.screen, ModelsScreen)
        assert app.screen.model["name"] == "Heretic Tuned"
        assert app.screen.model["sampling"]["temp"] == 0.3 and app.screen.model["max_ctx"] == 16384
        saved = user_models.load_overrides(app_root / user_models.OVERRIDES_FILENAME)
        assert saved[mid]["name"] == "Heretic Tuned"
        # the next launch sees it too
        from openllm_app.core.registry import load_registry
        assert load_registry(app_root / "models.json")["models"][mid]["name"] == "Heretic Tuned"

        await pilot.press("e")
        await pilot.pause()
        await pilot.click("#reset")
        await pilot.pause()
        assert app.screen.model["name"] == original
        assert user_models.load_overrides(app_root / user_models.OVERRIDES_FILENAME) == {}


async def test_invalid_input_keeps_the_dialog_open(app_root):
    app = OpenLLMApp(app_root)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one("#f-file", Input).value = "../evil.gguf"
        await pilot.click("#save")
        await pilot.pause()
        assert isinstance(app.screen, EditModelScreen) and "File" in app.screen.status_text
        await pilot.pause(0.5)  # a Button ignores presses while its press animation runs
        app.screen.query_one("#f-file", Input).value = "ok.gguf"
        app.screen.query_one("#f-temp", Input).value = "hot"
        await pilot.pause()
        await pilot.click("#save")
        await pilot.pause()
        assert isinstance(app.screen, EditModelScreen) and "numbers" in app.screen.status_text


async def test_cancel_changes_nothing(app_root):
    app = OpenLLMApp(app_root)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("e")
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ModelsScreen)
        assert not (app_root / user_models.OVERRIDES_FILENAME).exists()
