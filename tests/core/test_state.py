from openollama_app.core import state


def test_state_roundtrip_and_mode(tmp_path):
    p = tmp_path / "state.json"
    state.save_state(p, {"api_key": "sk-x", "m": {"ctx": 8192}})
    assert state.load_state(p) == {"api_key": "sk-x", "m": {"ctx": 8192}}
    assert p.stat().st_mode & 0o777 == 0o600


def test_load_state_missing_returns_empty(tmp_path):
    assert state.load_state(tmp_path / "nope.json") == {}


def test_load_state_corrupt_returns_empty(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{oops")
    assert state.load_state(p) == {}


def test_merge_defaults_empty(model):
    s = state.merge_defaults(model, {})
    assert s == {"mode": "server", "kv": "q8_0", "ctx": 32768, "effort": "xhigh",
                 "expose": "local", "port": 8080, "max_tokens": 16384}


def test_merge_defaults_old_chat_entry(model):
    saved = {"mode": "chat", "kv": "q8_0", "ctx": 32768, "effort": "xhigh", "max_tokens": 16384}
    s = state.merge_defaults(model, saved)
    s["mode"] = "server"
    assert s["expose"] == "local" and s["port"] == 8080


def test_merge_defaults_replaces_bad_values_and_clamps(model):
    s = state.merge_defaults(model, {"kv": "q2", "effort": "turbo", "ctx": 10**9})
    assert s["kv"] == "q8_0" and s["effort"] == "xhigh" and s["ctx"] == 262144
