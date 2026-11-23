import pytest
from textual.widgets import ListView

from openllm_app.core import downloads, memory, state as state_mod
from openllm_app.ui.app import OpenLLMApp
from openllm_app.ui.models_screen import ModelsScreen
from openllm_app.ui.run_screen import RunScreen
from openllm_app.ui.widgets import Cycler

from .fakes import Factory, scripted_chat
from .test_models_screen import MID, RecordingApp, SIZE, open_model, setup_root

GEN = "generic-8b"


@pytest.fixture(autouse=True)
def big_mac(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 64.0)


@pytest.fixture
def two_models(registry, model):
    """Bonsai (prismml only) plus a model that may use either runtime."""
    registry["models"][GEN] = {**model, "name": "Generic 8B", "file": "generic-Q4_K_M.gguf",
                               "runtimes": ["prismml", "llamacpp"], "reasoning_efforts": ["off"]}
    return registry


def install_llamacpp(root, registry):
    rt = registry["runtimes"]["llamacpp"]
    d = root / rt["dir"].format(tag=rt["tag"])
    d.mkdir(parents=True)
    (d / "llama-server").touch()


async def select(pilot, app, model_id):
    """Open a model's screen from the list."""
    await open_model(app, pilot, model_id)
    assert app.screen.model_id == model_id


def runtime_choices(app):
    return [k for k, _ in app.screen.query_one("#runtime", Cycler).options]


async def test_bonsai_only_offers_the_prismml_runtime(root, two_models):
    setup_root(root, two_models)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, MID)
        assert runtime_choices(app) == ["prismml"]


async def test_generic_model_offers_both_and_defaults_to_its_own(root, two_models):
    setup_root(root, two_models)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, GEN)
        assert runtime_choices(app) == ["prismml", "llamacpp"]
        assert app.screen.settings["runtime"] == "prismml"


async def test_choosing_an_uninstalled_runtime_blocks_start_until_it_is_installed(root, two_models):
    setup_root(root, two_models)
    (root / "models" / "generic-Q4_K_M.gguf").touch()
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, GEN)
        app.screen.query_one("#runtime", Cycler).select("llamacpp")
        await pilot.pause()
        assert app.screen.settings["runtime"] == "llamacpp"
        assert app.screen.status_is_error is False and "runtime" in app.screen.status_text.lower()
        await pilot.click("#start")
        await pilot.pause()
        assert app.started == [] and app.screen.status_is_error

        install_llamacpp(root, two_models)
        await pilot.pause(0.4)  # Button ignores a second press while its "active" flash is running
        await pilot.click("#start")
        await pilot.pause()
        assert app.started[0].settings["runtime"] == "llamacpp"
    assert state_mod.load_state(root / "state.json")[GEN]["runtime"] == "llamacpp"


async def test_download_installs_the_chosen_runtime(root, two_models, monkeypatch):
    setup_root(root, two_models)
    (root / "models" / "generic-Q4_K_M.gguf").touch()
    installed = []
    monkeypatch.setattr(downloads, "install_runtime", lambda r, rt_id, rt, p: installed.append((rt_id, rt["tag"])))
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, GEN)
        app.screen.query_one("#runtime", Cycler).select("llamacpp")
        await pilot.pause()
        await pilot.press("d")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert installed == [("llamacpp", "b-test")]


async def test_saved_runtime_choice_is_restored(root, two_models):
    setup_root(root, two_models, saved={GEN: {"runtime": "llamacpp"}})
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, GEN)
        assert app.screen.settings["runtime"] == "llamacpp"
        assert app.screen.query_one("#runtime", Cycler).value == "llamacpp"


async def test_model_list_checkmark_follows_the_saved_runtime(root, two_models):
    setup_root(root, two_models, saved={GEN: {"runtime": "llamacpp"}})
    (root / "models" / "generic-Q4_K_M.gguf").touch()
    app = RecordingApp(root)
    async with app.run_test(size=SIZE):
        assert isinstance(app.screen, ModelsScreen)
        assert app.screen._missing(GEN) == ["runtime"]
        assert app.screen._missing(MID) == []


async def test_server_launches_the_chosen_runtimes_binary(root, two_models):
    setup_root(root, two_models)
    (root / "models" / "generic-Q4_K_M.gguf").touch()
    install_llamacpp(root, two_models)
    factory = Factory()
    app = OpenLLMApp(root, run_kwargs={"server_factory": factory, "chat_fn": scripted_chat([])})
    async with app.run_test(size=SIZE) as pilot:
        await select(pilot, app, GEN)
        app.screen.query_one("#runtime", Cycler).select("llamacpp")
        await pilot.pause()
        await pilot.click("#start")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        assert "/runtimes/llamacpp/" in factory.server.argv[0]
