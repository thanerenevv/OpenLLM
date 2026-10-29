import json

from openollama_app.core import snippets

S = {"expose": "local", "port": 8080, "ctx": 32768, "model_id": "ternary-bonsai-2-27b",
     "model_name": "Ternary Bonsai 2 27B"}


def test_local_snippets_use_loopback_and_fallback_key():
    out = snippets.client_snippets(S, None)
    assert out["base_url"] == "http://127.0.0.1:8080/v1"
    assert out["api_key"] == "sk-local"
    assert "ternary-bonsai-2-27b" in out["curl"] and "Bearer sk-local" in out["curl"]


def test_lan_uses_given_ip(monkeypatch):
    monkeypatch.setattr(snippets, "lan_ip", lambda: "192.168.1.9")
    out = snippets.client_snippets({**S, "expose": "lan"}, "sk-abc")
    assert out["base_url"] == "http://192.168.1.9:8080/v1" and out["api_key"] == "sk-abc"


def test_opencode_is_valid_json():
    cfg = json.loads(snippets.client_snippets(S, None)["opencode"])
    prov = cfg["provider"]["openollama"]
    assert prov["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert prov["models"]["ternary-bonsai-2-27b"]["limit"] == {"context": 32768, "output": 16384}
    assert cfg["model"] == "openollama/ternary-bonsai-2-27b"


def test_cline_snippet_mentions_context():
    assert "32768" in snippets.client_snippets(S, None)["cline"]
