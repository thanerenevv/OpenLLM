"""Turn chosen settings into a runtime command line. One adapter per runtime type."""
import json
from pathlib import Path

from .registry import model_path, runtime_bin


def build_llama_cmd(root: Path, rt: dict, m: dict, s: dict, api_key: str | None) -> list[str]:
    smp = m["sampling"]
    base = [
        "-m", str(model_path(root, m)), "-ngl", "99", "-fa", "on", "-c", str(s["ctx"]),
        "--cache-type-k", s["kv"], "--cache-type-v", s["kv"],
        "--temp", str(smp["temp"]), "--top-p", str(smp["top_p"]),
        "--top-k", str(smp["top_k"]), "--min-p", str(smp["min_p"]),
        "--jinja",
        "--chat-template-kwargs", json.dumps({"reasoning_effort": s["effort"]}),
    ]
    if s["mode"] == "chat":
        return [str(runtime_bin(root, rt, "llama-cli")), *base, "-n", str(s["max_tokens"])]
    host = "0.0.0.0" if s["expose"] == "lan" else "127.0.0.1"
    cmd = [str(runtime_bin(root, rt, "llama-server")), *base,
           "--host", host, "--port", str(s["port"]),
           "-np", "1",  # one slot, so the whole context belongs to the one client
           "--alias", s["model_id"]]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


ADAPTERS = {"llama.cpp": build_llama_cmd}


def build_command(root: Path, rt: dict, m: dict, s: dict, api_key: str | None) -> list[str]:
    return ADAPTERS[rt["adapter"]](root, rt, m, s, api_key)
