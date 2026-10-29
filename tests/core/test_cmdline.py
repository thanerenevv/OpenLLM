import json

from openollama_app.core import cmdline

BASE = {"mode": "server", "kv": "q8_0", "ctx": 32768, "effort": "xhigh",
        "expose": "local", "port": 8080, "max_tokens": 16384, "model_id": "ternary-bonsai-2-27b"}


def pair(argv, flag):
    return argv[argv.index(flag) + 1]


def test_build_command_server_local(root, runtime, model):
    argv = cmdline.build_command(root, runtime, model, dict(BASE), None)
    assert argv[0].endswith("llama-server")
    assert pair(argv, "-m").endswith("Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    assert pair(argv, "-ngl") == "99" and pair(argv, "-fa") == "on" and pair(argv, "-c") == "32768"
    assert pair(argv, "--cache-type-k") == "q8_0" and pair(argv, "--cache-type-v") == "q8_0"
    assert pair(argv, "--temp") == "1.0" and pair(argv, "--top-k") == "20" and pair(argv, "--min-p") == "0.05"
    assert "--jinja" in argv
    assert json.loads(pair(argv, "--chat-template-kwargs")) == {"reasoning_effort": "xhigh"}
    assert pair(argv, "--host") == "127.0.0.1" and pair(argv, "--port") == "8080"
    assert pair(argv, "-np") == "1" and pair(argv, "--alias") == "ternary-bonsai-2-27b"
    assert "--api-key" not in argv


def test_build_command_lan_has_key(root, runtime, model):
    s = {**BASE, "expose": "lan"}
    argv = cmdline.build_command(root, runtime, model, s, "sk-x")
    assert pair(argv, "--host") == "0.0.0.0" and pair(argv, "--api-key") == "sk-x"


def test_build_command_chat_uses_llama_cli_with_n(root, runtime, model):
    s = {**BASE, "mode": "chat"}
    argv = cmdline.build_command(root, runtime, model, s, None)
    assert argv[0].endswith("llama-cli") and pair(argv, "-n") == "16384"
    assert "--host" not in argv


def test_adapters_keyed_by_adapter_name():
    assert cmdline.ADAPTERS["llama.cpp"] is cmdline.build_llama_cmd
