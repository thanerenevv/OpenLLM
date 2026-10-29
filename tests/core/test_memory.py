import pytest

from openollama_app.core import memory


def test_estimate_gb_matches_registry_values(model):
    assert memory.estimate_gb(model, 32768, "q8_0") == pytest.approx(5.54 + 34 * 32 / 1024 + 0.55)


def test_gpu_budget():
    assert memory.gpu_budget_gb(16) == 11.0


@pytest.mark.parametrize("text,expected", [("48k", 49152), ("49152", 49152), ("1,024", 1024), (" 8K ", 8192)])
def test_parse_tokens_ok(text, expected):
    assert memory.parse_tokens(text) == expected


@pytest.mark.parametrize("text", ["abc", "0", "-5k", "", "k", "infk", "nan", "1e999"])
def test_parse_tokens_rejects(text):
    with pytest.raises(ValueError):
        memory.parse_tokens(text)


def test_clamp_ctx():
    assert memory.clamp_ctx(100, 262144) == 1024
    assert memory.clamp_ctx(10**9, 262144) == 262144
    assert memory.clamp_ctx(32768, 262144) == 32768


def test_total_ram_is_positive_on_this_machine():
    assert memory.total_ram_gb() > 0
