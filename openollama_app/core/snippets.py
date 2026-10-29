"""Copy-paste client configuration for a running server."""
import json
import subprocess

FALLBACK_KEY = "sk-local"


def lan_ip() -> str:
    for iface in ("en0", "en1"):
        try:
            ip = subprocess.check_output(["ipconfig", "getifaddr", iface], text=True, stderr=subprocess.DEVNULL).strip()
            if ip:
                return ip
        except (OSError, subprocess.CalledProcessError):
            pass
    return "<your-mac-ip>"


def client_snippets(s: dict, api_key: str | None) -> dict[str, str]:
    host = lan_ip() if s["expose"] == "lan" else "127.0.0.1"
    base = f"http://{host}:{s['port']}/v1"
    key = api_key or FALLBACK_KEY
    ctx, mid = s["ctx"], s["model_id"]
    oc = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"openollama": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "OpenOllama",
            "options": {"baseURL": base, "apiKey": key},
            "models": {mid: {"name": s["model_name"], "limit": {"context": ctx, "output": min(16384, ctx // 2)}}},
        }},
        "model": f"openollama/{mid}",
    }
    body = json.dumps({"model": mid, "max_tokens": 16384, "messages": [{"role": "user", "content": "hi"}]})
    return {
        "base_url": base,
        "api_key": key,
        "cline": (f'Settings -> API Provider "OpenAI Compatible"\n'
                  f"Base URL: {base}\nAPI Key: {key}\nModel ID: {mid}\nContext Window: {ctx}"),
        "opencode": json.dumps(oc, indent=2),
        "curl": (f"curl {base}/chat/completions -H \"Authorization: Bearer {key}\" \\\n"
                 f"  -H \"Content-Type: application/json\" \\\n  -d '{body}'"),
    }
