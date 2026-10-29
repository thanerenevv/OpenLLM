"""state.json: last-used settings per model, plus the shared LAN API key."""
import json
from pathlib import Path

from .memory import clamp_ctx

KV_CHOICES = ("f16", "q8_0", "q4_0")


def load_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))
    path.chmod(0o600)  # holds the API key


def merge_defaults(m: dict, saved: dict) -> dict:
    """A complete settings dict for model m: defaults, overlaid with whatever was saved, sanitised."""
    s = {"mode": "server", "kv": "q8_0", "ctx": 32768, "effort": m["reasoning_efforts"][-1],
         "expose": "local", "port": 8080, "max_tokens": 16384}
    s.update({k: v for k, v in saved.items() if k in s})
    if s["kv"] not in KV_CHOICES:
        s["kv"] = "q8_0"
    if s["effort"] not in m["reasoning_efforts"]:
        s["effort"] = m["reasoning_efforts"][-1]
    if s["mode"] not in ("server", "chat"):
        s["mode"] = "server"
    if s["expose"] not in ("local", "lan"):
        s["expose"] = "local"
    try:
        s["ctx"] = clamp_ctx(int(s["ctx"]), m["max_ctx"])
    except (TypeError, ValueError):
        s["ctx"] = 32768
    return s
