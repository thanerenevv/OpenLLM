import json

import pytest
from textual.widgets import Button, Input, ListView

from openllm_app.core import downloads, memory, state as state_mod
from openllm_app.ui.app import OpenLLMApp
from openllm_app.ui.model_screen import ModelScreen
from openllm_app.ui.models_screen import ModelsScreen
from openllm_app.ui.widgets import Cycler, MemoryGauge

MID = "ternary-bonsai-2-27b"
SIZE = (120, 40)


def setup_root(root, registry, installed=True, saved=None):
    (root / "models.json").write_text(json.dumps(registry))
    if installed:
        rt = registry["runtimes"]["prismml"]
        bindir = root / rt["dir"].format(tag=rt["tag"])
        bindir.mkdir(parents=True)
        (bindir / "llama-server").touch()
        (root / "models").mkdir()
        (root / "models" / registry["models"][MID]["file"]).touch()
    if saved is not None:
        state_mod.save_state(root / "state.json", saved)


class RecordingApp(OpenLLMApp):
    def __init__(self, root):
        super().__init__(root)
        self.started = []

    def on_model_screen_start_requested(self, msg):
        self.started.append(msg)
        msg.prevent_default()  # don't let the real handler push a RunScreen


async def open_model(app, pilot, model_id=None):
    """From the list screen into one model's screen (the first model by default)."""
    if model_id:
        app.screen.query_one("#models", ListView).index = list(app.registry["models"]).index(model_id)
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    assert isinstance(app.screen, ModelScreen)


@pytest.fixture(autouse=True)
def big_mac(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 64.0)


async def test_gauge_updates_when_kv_changes(root, registry, model):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        screen = app.screen
        screen.query_one("#kv", Cycler).select("q4_0")
        await pilot.pause()
        gauge = screen.query_one(MemoryGauge)
        assert gauge.used == pytest.approx(memory.estimate_gb(model, 32768, "q4_0"))


async def test_gauge_red_when_over_budget(root, registry, monkeypatch):
    setup_root(root, registry)
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 8.0)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        app.screen.query_one("#ctx", Cycler).action_step(1)
        app.screen.query_one("#ctx", Cycler).action_step(1)
        await pilot.pause()
        gauge = app.screen.query_one(MemoryGauge)
        assert gauge.has_class("over")


async def test_gauge_not_red_when_it_fits(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        assert not app.screen.query_one(MemoryGauge).has_class("over")


async def test_context_stepper_moves_through_presets(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        ctx = app.screen.query_one("#ctx", Cycler)
        ctx.action_step(1)
        await pilot.pause()
        assert app.screen.settings["ctx"] == 65536
        ctx.action_step(-1)
        ctx.action_step(-1)
        await pilot.pause()
        assert app.screen.settings["ctx"] == 16384
        await pilot.press("right")  # keys work too: the focused selector moves
        await pilot.pause()


@pytest.mark.parametrize("text", ["abc", "0", "-5k", "infk"])
async def test_custom_context_garbage_is_rejected(root, registry, text):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        app.screen.query_one("#ctx-custom", Input).focus()
        await pilot.press(*text, "enter")
        await pilot.pause()
        assert app.screen.settings["ctx"] == 32768
        assert app.screen.status_is_error


async def test_custom_context_huge_is_clamped_and_valid_is_accepted(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        box = app.screen.query_one("#ctx-custom", Input)
        box.focus()
        await pilot.press(*"9999999k", "enter")
        assert app.screen.settings["ctx"] == 262144
        box.value = ""
        await pilot.press(*"48k", "enter")
        assert app.screen.settings["ctx"] == 49152
        assert not app.screen.status_is_error


async def test_old_chat_entry_fills_defaults(root, registry):
    old = {"mode": "chat", "kv": "q8_0", "ctx": 32768, "effort": "xhigh", "max_tokens": 16384}
    setup_root(root, registry, saved={MID: old})
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        s = app.screen.settings
        assert s["mode"] == "chat" and s["port"] == 8080 and s["expose"] == "local"
        assert app.screen.query_one("#mode", Cycler).value == "chat"
        assert not app.screen.query_one("#port-row").display  # server-only options hide in chat mode


async def test_not_downloaded_model_blocks_start_and_offers_download(root, registry):
    setup_root(root, registry, installed=False)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        await pilot.click("#start")
        await pilot.pause()
        assert app.started == []
        assert "download" in app.screen.status_text.lower()


async def test_download_key_fetches_runtime_and_model(root, registry, monkeypatch):
    setup_root(root, registry, installed=False)
    calls = []

    def fake_runtime(r, rt_id, rt, on_progress):
        calls.append("runtime")
        on_progress(5, 10)
        d = r / rt["dir"].format(tag=rt["tag"])
        d.mkdir(parents=True)
        (d / "llama-server").touch()

    def fake_model(r, m, on_progress):
        calls.append("model")
        (r / "models").mkdir(exist_ok=True)
        (r / "models" / m["file"]).touch()

    monkeypatch.setattr(downloads, "install_runtime", fake_runtime)
    monkeypatch.setattr(downloads, "fetch_model", fake_model)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        await pilot.press("d")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert calls == ["runtime", "model"]
        await pilot.click("#start")
        await pilot.pause()
        assert len(app.started) == 1


async def test_download_failure_is_reported_not_raised(root, registry, monkeypatch):
    setup_root(root, registry, installed=False)

    def boom(*a, **k):
        raise downloads.DownloadError("download failed: boom")

    monkeypatch.setattr(downloads, "install_runtime", boom)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        await pilot.press("d")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.status_is_error and "boom" in app.screen.status_text


async def test_start_posts_message_and_saves_state(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        app.screen.query_one("#kv", Cycler).select("q4_0")
        await pilot.pause()
        await pilot.click("#start")
        await pilot.pause()
        assert len(app.started) == 1
        msg = app.started[0]
        assert msg.model_id == MID and msg.settings["kv"] == "q4_0" and msg.api_key is None
        assert msg.settings["model_id"] == MID
    saved = state_mod.load_state(root / "state.json")
    assert saved[MID]["kv"] == "q4_0" and "model_id" not in saved[MID]
    assert (root / "state.json").stat().st_mode & 0o777 == 0o600


async def test_lan_start_generates_and_reuses_api_key(root, registry):
    setup_root(root, registry, saved={"api_key": "sk-existing"})
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        app.screen.query_one("#expose", Cycler).select("lan")
        await pilot.pause()
        await pilot.click("#start")
        await pilot.pause()
        assert app.started[0].api_key == "sk-existing"


async def test_enter_on_list_opens_the_model_and_enter_there_starts(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        app.screen.query_one(ListView).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelScreen) and app.started == []
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.started) == 1


async def test_escape_goes_back_to_the_list(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ModelsScreen)


async def test_mode_selector_toggles_server_options(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        assert app.screen.query_one("#port-row").display and app.screen.query_one("#expose").display
        app.screen.query_one("#mode", Cycler).select("chat")
        await pilot.pause()
        assert not app.screen.query_one("#port-row").display and not app.screen.query_one("#expose").display
        assert app.screen.settings["mode"] == "chat"


async def test_invalid_port_blocks_start(root, registry):
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        box = app.screen.query_one("#port", Input)
        box.value = "99999"
        await pilot.pause()
        await pilot.click("#start")
        await pilot.pause()
        assert app.started == [] and app.screen.status_is_error


async def test_download_unexpected_error_is_reported_not_fatal(root, registry, monkeypatch):
    import tarfile
    setup_root(root, registry, installed=False)
    monkeypatch.setattr(downloads, "install_runtime", lambda *a, **k: (_ for _ in ()).throw(tarfile.ReadError("not a tar")))
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        await open_model(app, pilot)
        await pilot.press("d")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.is_running and app.screen.status_is_error and "not a tar" in app.screen.status_text
        assert not app.screen._downloading


async def test_arrow_keys_switch_models_and_keep_list_focus(root, registry):
    second = dict(registry["models"][MID], name="Second Model")
    registry["models"]["second-model"] = second
    setup_root(root, registry)
    app = RecordingApp(root)
    async with app.run_test(size=SIZE) as pilot:
        app.screen.query_one(ListView).focus()
        await pilot.press("down")
        await pilot.pause()
        assert app.screen.model_id == "second-model"
        assert app.screen.focused is not None and app.screen.focused.id == "models"
        await pilot.press("up")
        await pilot.pause()
        assert app.screen.model_id == MID
