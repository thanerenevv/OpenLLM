import json

import pytest

from openllm_app.core import user_models
from openllm_app.core.registry import load_registry


def entry(**over):
    e = {
        "name": "Mine (Q4_K_M)", "runtime": "prismml", "repo": "o/m-GGUF", "file": "m-Q4_K_M.gguf",
        "size_gb": 4.0, "max_ctx": 32768, "sampling": {"temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05},
        "reasoning_efforts": ["off"], "weights_gib": 3.7, "kv_mib_per_1k": {"f16": 128, "q8_0": 68, "q4_0": 36},
        "notes": "x",
    }
    e.update(over)
    return e


@pytest.fixture
def reg_path(root, registry):
    p = root / "models.json"
    p.write_text(json.dumps(registry))
    return p


def test_add_creates_the_file_and_load_reads_it_back(root):
    path = root / "user_models.json"
    user_models.add(path, "mine", entry())
    assert user_models.load(path) == {"mine": entry()}


def test_add_keeps_existing_models(root):
    path = root / "user_models.json"
    user_models.add(path, "a", entry())
    user_models.add(path, "b", entry(name="B"))
    assert list(user_models.load(path)) == ["a", "b"]


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"a": 1}'])
def test_load_tolerates_a_broken_file(root, content):
    path = root / "user_models.json"
    path.write_text(content)
    assert user_models.load(path) == {}


def test_load_missing_file_is_empty(root):
    assert user_models.load(root / "nope.json") == {}


def test_load_drops_entries_that_would_break_the_app(root):
    path = root / "user_models.json"
    path.write_text(json.dumps({
        "ok": entry(),
        "no-efforts": entry(reasoning_efforts=[]),
        "missing-kv": entry(kv_mib_per_1k={"f16": 1}),
        "traversal": entry(file="../../etc/passwd"),
        "subdir": entry(file="a/b.gguf"),
        "bad-repo": entry(repo="not a repo"),
        "bad-runtimes": entry(runtimes="prismml"),
        "bad-runtimes-2": entry(runtimes=[1]),
        "no-name": {k: v for k, v in entry().items() if k != "name"},
    }))
    assert list(user_models.load(path)) == ["ok"]


def test_registry_merges_user_models_after_curated_ones(root, reg_path):
    user_models.add(root / "user_models.json", "mine", entry())
    reg = load_registry(reg_path)
    assert list(reg["models"]) == ["ternary-bonsai-2-27b", "mine"]


def test_curated_model_wins_on_id_clash(root, reg_path):
    user_models.add(root / "user_models.json", "ternary-bonsai-2-27b", entry(name="Impostor"))
    assert load_registry(reg_path)["models"]["ternary-bonsai-2-27b"]["name"] == "Ternary Bonsai 2 27B"


def test_user_model_with_unknown_runtime_is_dropped(root, reg_path):
    user_models.add(root / "user_models.json", "mine", entry(runtime="nonesuch"))
    assert "mine" not in load_registry(reg_path)["models"]


def test_registry_without_user_file_is_unchanged(reg_path, registry):
    assert load_registry(reg_path) == registry


def test_valid_runtimes_list_is_accepted(root):
    path = root / "user_models.json"
    user_models.add(path, "mine", entry(runtimes=["prismml", "llamacpp"]))
    assert "mine" in user_models.load(path)


# ------------------------------------------------------------------ overrides
def test_overrides_round_trip_and_clear(tmp_path):
    p = tmp_path / user_models.OVERRIDES_FILENAME
    user_models.set_override(p, "m", {"name": " New ", "max_ctx": 4096, "sampling": {"temp": 0.2}, "junk": 1})
    assert user_models.load_overrides(p) == {"m": {"name": "New", "max_ctx": 4096, "sampling": {"temp": 0.2}}}
    user_models.clear_override(p, "m")
    assert user_models.load_overrides(p) == {}


def test_bad_override_fields_are_dropped():
    bad = {"name": "", "repo": "nope", "file": "../x.gguf", "max_ctx": 5, "sampling": {"temp": "hot"},
           "reasoning_efforts": []}
    assert user_models.clean_overrides(bad) == {}
    assert user_models.clean_overrides("x") == {}


def test_apply_overrides_merges_sampling():
    models = {"m": {"name": "A", "sampling": {"temp": 1, "top_p": 0.9}}}
    user_models.apply_overrides(models, {"m": {"name": "B", "sampling": {"temp": 0.1}}, "gone": {"name": "x"}})
    assert models == {"m": {"name": "B", "sampling": {"temp": 0.1, "top_p": 0.9}}}
