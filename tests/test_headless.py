import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from openollama_app import __main__ as entry
from openollama_app.core import state as state_mod

MID = "ternary-bonsai-2-27b"
FAKE_SRC = (Path(__file__).parent / "fake_llama_server.py").read_text()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_root(root, registry, saved=None, mode="ok", model_file=True, runtime=True):
    (root / "models.json").write_text(json.dumps(registry))
    rt = registry["runtimes"]["prismml"]
    if runtime:
        d = root / rt["dir"].format(tag=rt["tag"])
        d.mkdir(parents=True)
        script = d / "llama-server"
        script.write_text(f"#!{sys.executable}\n" + FAKE_SRC.replace('default="ok"', f'default="{mode}"'))
        script.chmod(0o755)
        (d / "llama-cli").write_text("#!/bin/sh\n")
        (d / "llama-cli").chmod(0o755)
    if model_file:
        (root / "models").mkdir()
        (root / "models" / registry["models"][MID]["file"]).touch()
    if saved is not None:
        state_mod.save_state(root / "state.json", saved)


def saved_settings(**over):
    s = {"mode": "server", "kv": "q8_0", "ctx": 8192, "effort": "medium", "expose": "local",
         "port": free_port(), "max_tokens": 16384}
    s.update(over)
    return s


def test_last_without_state_exits_1(root, registry, capsys):
    make_root(root, registry)
    assert entry.main(["--last"], root) == 1
    assert "no saved settings" in capsys.readouterr().err.lower()


def test_last_without_downloaded_files_exits_1(root, registry, capsys):
    make_root(root, registry, saved={MID: saved_settings()}, model_file=False, runtime=False)
    assert entry.main(["--last"], root) == 1
    assert "download" in capsys.readouterr().err.lower()


def test_last_chat_mode_execs_llama_cli(root, registry, monkeypatch):
    make_root(root, registry, saved={MID: saved_settings(mode="chat")})
    calls = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
    entry.main(["--last"], root)
    assert calls and calls[0][0].endswith("llama-cli") and "-n" in calls[0][1]


def test_last_server_failure_exits_1_with_reason(root, registry, capsys):
    port = free_port()
    make_root(root, registry, saved={MID: saved_settings(port=port)}, mode="bindfail")
    assert entry.main(["--last"], root) == 1
    assert f"Port {port} is already in use" in capsys.readouterr().err


def test_last_starts_server_prints_base_url_and_stops_on_sigint(root, registry):
    port = free_port()
    make_root(root, registry, saved={MID: saved_settings(port=port)})
    code = ("import sys; from pathlib import Path; from openollama_app.__main__ import main; "
            "sys.exit(main(['--last'], Path(sys.argv[1])))")
    p = subprocess.Popen([sys.executable, "-c", code, str(root)], stdout=subprocess.PIPE, text=True,
                         cwd=Path(__file__).parent.parent)
    seen = []
    deadline = time.time() + 20
    while time.time() < deadline:
        line = p.stdout.readline()
        seen.append(line)
        if f"http://127.0.0.1:{port}/v1" in line:
            break
    assert any(f"http://127.0.0.1:{port}/v1" in l for l in seen), "".join(seen)
    assert urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=15) == 0
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)


def test_last_lan_reuses_saved_api_key(root, registry, capsys):
    make_root(root, registry, saved={"api_key": "sk-keep", MID: saved_settings(expose="lan", port=free_port())},
              mode="exit1")
    entry.main(["--last"], root)
    assert state_mod.load_state(root / "state.json")["api_key"] == "sk-keep"
