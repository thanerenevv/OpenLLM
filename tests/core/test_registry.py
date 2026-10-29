import json

import pytest

from openollama_app.core import registry as reg


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
