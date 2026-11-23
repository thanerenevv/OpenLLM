"""Entry logic: interactive TUI by default; `--last` starts the last-used model without any UI."""
import os
import secrets
import sys
import time
from pathlib import Path

from .core import cmdline, snippets
from .core.registry import RegistryError, load_registry, model_path, runtime_bin
from .core.state import load_state, merge_defaults, save_state
from .server import ServerProcess, install_cleanup


def err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def run_last(root: Path, reg: dict) -> int:
    state_path = root / "state.json"
    state = load_state(state_path)
    saved = [mid for mid in state if mid in reg["models"]]
    if not saved:
        err("no saved settings yet - run ./openllm once and start a model")
        return 1
    model_id = saved[0]  # state.json keeps insertion order; with several models this is the earliest saved
    m = reg["models"][model_id]
    s = merge_defaults(m, state[model_id])
    rt = reg["runtimes"][s["runtime"]]
    if not runtime_bin(root, rt, "llama-server").exists() or not model_path(root, m).exists():
        err("the runtime or model is not downloaded - run ./openllm and press d to download")
        return 1

    s["model_id"], s["model_name"] = model_id, m["name"]
    api_key = None
    if s["mode"] == "server" and s["expose"] == "lan":
        api_key = state.get("api_key") or "sk-" + secrets.token_urlsafe(24)
        state["api_key"] = api_key
        save_state(state_path, state)

    argv = cmdline.build_command(root, rt, m, s, api_key)
    if s["mode"] == "chat":
        print("\nStarting chat. Ctrl+C to stop.\n", flush=True)
        os.execv(argv[0], argv)
        return 0

    server = ServerProcess(argv, "127.0.0.1", s["port"], lambda line: print(line, flush=True), lambda st: None)
    install_cleanup(server)
    server.start()
    try:
        while server.state in ("idle", "starting"):
            time.sleep(0.2)
        if server.state == "failed":
            err(server.failure_reason or "server failed to start")
            return 1
        snip = snippets.client_snippets(s, api_key)
        print(f"""
================ OpenAI-compatible API ================
 Base URL : {snip['base_url']}
 API key  : {snip['api_key']}
 Model ID : {model_id}
 Context  : {s['ctx']} tokens (one request at a time)

 Cline  : {snip['cline'].replace(chr(10), chr(10) + '          ')}
 opencode: put this in opencode.json (project) or ~/.config/opencode/opencode.json:
{snip['opencode']}
 curl   : {snip['curl']}

 Tip: it is a thinking model. Clients must allow large max_tokens (>=16384)
 or replies can come back empty. Ctrl+C here stops the server.
=======================================================
""", flush=True)
        while server.state == "ready":
            time.sleep(0.5)
        if server.state == "failed":
            err(server.failure_reason or "server stopped unexpectedly")
            return 1
        return 0
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        server.stop()


def main(argv: list[str], root: Path | None = None) -> int:
    root = root or Path(os.path.realpath(__file__)).parent.parent
    try:
        reg = load_registry(root / "models.json")
    except RegistryError as e:
        err(str(e))
        return 1
    if "--last" in argv:
        return run_last(root, reg)
    from .ui.app import OpenLLMApp  # imported late: needs the UI dependencies
    OpenLLMApp(root).run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
