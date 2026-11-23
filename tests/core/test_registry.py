import json

import pytest

from openllm_app.core import registry as reg


def test_load_registry_reads_file(tmp_path, registry):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(registry))
    assert reg.load_registry(p)["models"]["ternary-bonsai-2-27b"]["name"] == "Ternary Bonsai 2 27B"


def test_load_registry_missing_raises(tmp_path):
    with pytest.raises(reg.RegistryError):
        reg.load_registry(tmp_path / "nope.json")


def test_load_registry_invalid_raises(tmp_path):
    p = tmp_path / "models.json"
    p.write_text("{not json")
    with pytest.raises(reg.RegistryError):
        reg.load_registry(p)


def test_runtime_paths(root, runtime, model):
    assert reg.runtime_dir(root, runtime) == root / "runtimes/prismml/prism-test"
    assert reg.runtime_bin(root, runtime, "llama-server") == root / "runtimes/prismml/prism-test/llama-server"
    assert reg.model_path(root, model) == root / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"


def _load(tmp_path, registry):
    p = tmp_path / "models.json"
    p.write_text(json.dumps(registry))
    return reg.load_registry(p)["models"]["ternary-bonsai-2-27b"]


def test_model_without_runtimes_allows_every_runtime(tmp_path, registry):
    del registry["models"]["ternary-bonsai-2-27b"]["runtimes"]
    assert _load(tmp_path, registry)["runtimes"] == ["prismml", "llamacpp"]


def test_explicit_runtimes_are_kept(tmp_path, registry):
    assert _load(tmp_path, registry)["runtimes"] == ["prismml"]


def test_unknown_runtimes_are_dropped_and_the_default_is_always_allowed(tmp_path, registry):
    registry["models"]["ternary-bonsai-2-27b"]["runtimes"] = ["nonesuch", "llamacpp"]
    assert _load(tmp_path, registry)["runtimes"] == ["prismml", "llamacpp"]
