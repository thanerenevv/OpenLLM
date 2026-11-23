import json

import pytest
from textual.widgets import Input, ListView

from openllm_app.core import hf, memory, user_models
from openllm_app.ui.add_model_screen import AddModelScreen
from openllm_app.ui.app import OpenLLMApp
from openllm_app.ui.models_screen import ModelsScreen

SIZE = (120, 40)
REPO = "RavichandranJ/Dolphin3-Cyber-8B-GGUF"
Q6 = {"file": "Dolphin3.0-Llama3.1-8B-abliterated.Q6_K.gguf", "size": 6596020288}
Q8 = {"file": "Dolphin3.0-Llama3.1-8B-abliterated.Q8_0.gguf", "size": 8540788672}
PROFILE = {"max_ctx": 131072, "f16": 128}


class FakeHF:
    def __init__(self, files=None, profile=PROFILE, error=None):
        self.files, self.profile, self.error = files if files is not None else [Q6], profile, error
        self.listed, self.read = [], []

    def list_files(self, repo):
        self.listed.append(repo)
        if self.error:
            raise self.error
        return self.files

    def search(self, query):
        self.searched = [*getattr(self, "searched", []), query]
        return [{"repo": REPO, "downloads": 1234}, {"repo": "o/other-GGUF", "downloads": 5}]

    def read_profile(self, repo, file):
        self.read.append((repo, file))
        return self.profile


class App(OpenLLMApp):
    def __init__(self, root, fake):
        super().__init__(root, hf_client=fake)


@pytest.fixture(autouse=True)
def big_mac(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 64.0)


@pytest.fixture
def app_root(root, registry):
    (root / "models.json").write_text(json.dumps(registry))
    return root


async def settle(app, pilot):
    for _ in range(3):
        await app.workers.wait_for_complete()
        await pilot.pause()


async def submit(app, pilot, text):
    app.screen.query_one("#repo", Input).value = text
    await pilot.press("enter")
    await settle(app, pilot)


async def test_a_key_opens_the_dialog(app_root):
    app = App(app_root, FakeHF())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, AddModelScreen)
        assert app.screen.query_one("#repo", Input).has_focus


async def test_button_opens_the_dialog(app_root):
    app = App(app_root, FakeHF())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.click("#add-model")
        await pilot.pause()
        assert isinstance(app.screen, AddModelScreen)


async def test_escape_cancels_without_changes(app_root):
    app = App(app_root, FakeHF())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ModelsScreen)
        assert not (app_root / "user_models.json").exists()


async def test_single_gguf_repo_is_added_selected_and_saved(app_root):
    fake = FakeHF()
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        screen = app.screen
        assert isinstance(screen, ModelsScreen)
        assert fake.listed == [REPO] and fake.read == [(REPO, Q6["file"])]
        mid = "dolphin3-cyber-8b-q6-k"
        assert screen.model_id == mid
        assert app.registry["models"][mid]["kv_mib_per_1k"]["f16"] == 128
        assert screen.model["reasoning_efforts"] == ["off"]
        assert "download" in screen.status_text.lower()
        assert list(screen.query_one("#models", ListView).children)[-1].name == mid
    assert list(user_models.load(app_root / "user_models.json")) == [mid]


async def test_url_is_accepted(app_root):
    fake = FakeHF()
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, f"https://huggingface.co/{REPO}/tree/main")
        assert fake.listed == [REPO]


async def test_multiple_files_show_a_picker_and_the_chosen_one_is_added(app_root):
    fake = FakeHF(files=[Q6, Q8])
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        assert isinstance(app.screen, AddModelScreen)
        files = app.screen.query_one("#files", ListView)
        assert files.display and len(files.children) == 2
        files.focus()
        await pilot.press("down", "enter")
        await settle(app, pilot)
        assert isinstance(app.screen, ModelsScreen)
        assert fake.read == [(REPO, Q8["file"])]
        assert app.screen.model["file"] == Q8["file"]


async def test_unreadable_header_still_adds_with_a_conservative_estimate(app_root):
    app = App(app_root, FakeHF(profile=None))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        assert "estimated" in app.screen.model["notes"].lower()


@pytest.mark.parametrize("text", ["", "   "])
async def test_empty_input_is_reported_and_dialog_stays_open(app_root, text):
    fake = FakeHF()
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, text)
        assert isinstance(app.screen, AddModelScreen)
        assert app.screen.status_is_error and fake.listed == []


async def test_lookup_failure_is_reported_and_can_be_retried(app_root):
    fake = FakeHF(error=hf.HFError("o/x not found, or it is private or gated"))
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, "o/x")
        assert isinstance(app.screen, AddModelScreen)
        assert app.screen.status_is_error and "not found" in app.screen.status_text
        fake.error = None
        await submit(app, pilot, REPO)
        assert isinstance(app.screen, ModelsScreen)


async def test_unexpected_error_is_reported_not_fatal(app_root):
    app = App(app_root, FakeHF(error=RuntimeError("boom")))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        assert isinstance(app.screen, AddModelScreen) and "boom" in app.screen.status_text


async def test_already_listed_model_is_selected_not_duplicated(app_root, registry):
    existing = hf.build_entry(REPO, Q6["file"], Q6["size"], PROFILE, "prismml")
    user_models.add(app_root / "user_models.json", "mine", existing)
    app = App(app_root, FakeHF())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        assert app.screen.model_id == "mine"
        assert len(app.registry["models"]) == 2
    assert list(user_models.load(app_root / "user_models.json")) == ["mine"]


async def test_added_model_offers_every_runtime_without_freezing_the_list_on_disk(app_root):
    from openllm_app.ui.widgets import Cycler

    app = App(app_root, FakeHF())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, REPO)
        await pilot.press("enter")  # open the freshly added model
        await pilot.pause()
        names = [k for k, _ in app.screen.query_one("#runtime", Cycler).options]
        assert names == ["prismml", "llamacpp"]
    saved = json.loads((app_root / "user_models.json").read_text())
    assert "runtimes" not in saved["dolphin3-cyber-8b-q6-k"]


async def test_plain_text_searches_then_pick_repo_then_add(app_root):
    fake = FakeHF()
    app = App(app_root, fake)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        await submit(app, pilot, "dolphin cyber")
        assert fake.searched == ["dolphin cyber"] and fake.listed == []
        assert isinstance(app.screen, AddModelScreen)
        results = app.screen.query_one("#files", ListView)
        assert results.display and len(results.children) == 2
        results.focus()
        await pilot.press("enter")
        await settle(app, pilot)
        assert fake.listed == [REPO]
        assert isinstance(app.screen, ModelsScreen)
        assert app.screen.model["repo"] == REPO
