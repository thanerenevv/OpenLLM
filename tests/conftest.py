import json
from pathlib import Path

import pytest

REGISTRY = {
    "runtimes": {
        "prismml": {
            "adapter": "llama.cpp",
            "description": "PrismML llama.cpp fork",
            "tag": "prism-test",
            "url": "https://example.invalid/llama-{tag}.tar.gz",
            "dir": "runtimes/prismml/{tag}",
        },
        "llamacpp": {
            "adapter": "llama.cpp",
            "description": "Upstream llama.cpp",
            "tag": "b-test",
            "url": "https://example.invalid/llama-{tag}.tar.gz",
            "dir": "runtimes/llamacpp/{tag}",
        },
    },
    "models": {
        "ternary-bonsai-2-27b": {
            "name": "Ternary Bonsai 2 27B",
            "runtime": "prismml",
            "runtimes": ["prismml"],
            "repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
            "file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            "size_gb": 5.95,
            "max_ctx": 262144,
            "sampling": {"temp": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05},
            "reasoning_efforts": ["medium", "xhigh"],
            "weights_gib": 5.54,
            "kv_mib_per_1k": {"f16": 64, "q8_0": 34, "q4_0": 18},
            "notes": "Thinking model.",
        }
    },
}


@pytest.fixture
def registry():
    return json.loads(json.dumps(REGISTRY))


@pytest.fixture
def model(registry):
    return registry["models"]["ternary-bonsai-2-27b"]


@pytest.fixture
def runtime(registry):
    return registry["runtimes"]["prismml"]


@pytest.fixture
def root(tmp_path):
    return tmp_path
