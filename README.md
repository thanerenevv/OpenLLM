# OpenLLM

A small terminal app for running GGUF models on a Mac with llama.cpp. You pick a model from a list, tweak a few settings, press enter, and it either drops you into a chat or starts an OpenAI-compatible server you can point Cline, opencode or curl at.

It's not trying to be Ollama or LM Studio. It's a thin wrapper around llama.cpp that builds the long `llama-server` command for you, remembers your settings per model, and tells you whether a given context size will fit in memory.

Apple Silicon only for now. The runtime download URLs in `models.json` point at the macOS arm64 builds.

## Running it

```
git clone https://github.com/thanerenevv/OpenLLM
cd OpenLLM
./openllm
```

The launcher is plain Python with no dependencies. The first time you start the UI it offers to create a `.venv` with [uv](https://docs.astral.sh/uv/) and install [Textual](https://textual.textualize.io/) and httpx. If you don't have uv it prints the three commands to do it by hand.

Press `d` on a model to download the runtime and the weights. They go into `runtimes/` and `models/` next to the script, so deleting the folder removes everything.

## Using it

The model list shows what's downloaded and how much memory each model will need at the context size you've chosen. The estimate is weights plus KV cache plus a bit of overhead, checked against the roughly 75% of RAM that Metal will let the GPU use. It's an estimate, not a promise.

Keys on the list: `enter` opens a model, `a` adds one, `e` edits one, `q` quits.

On a model's page you can set the context length, KV cache precision (f16, q8_0, q4_0), mode (chat or server), and for server mode whether to listen on localhost only or on your LAN. LAN mode generates an API key for you. Start it with `enter`.

The run screen has tabs for chat, the server log, and a connect tab with ready-to-paste config for Cline, opencode and curl.

### Adding your own model

Press `a`, type a Hugging Face repo like `user/Some-Model-GGUF`, and pick one of its `.gguf` files. It reads the file header to find the model's maximum context and works out the KV cache cost, so the memory estimate works for models you add too. Your additions are saved in `user_models.json` and are not committed.

### Skipping the UI

```
./openllm --last
```

Starts the last model you configured with the same settings and no UI, which is handy for scripts. It needs nothing beyond the standard library.

## A note on thinking models

Several of the bundled models reason before they answer. If your client caps `max_tokens` low, the whole budget can get spent on thinking and you get an empty reply. Allow 16384 or more.

## Files worth knowing about

- `models.json` is the registry of runtimes and models. Edit it to add things by hand.
- `state.json` holds your per-model settings (ignored by git).
- `openllm_app/core/` is the UI-free logic: command building, downloads, memory estimates, the Hugging Face lookup.
- `openllm_app/ui/` is the Textual screens.

## Tests

```
uv venv && uv pip install textual httpx pytest pytest-asyncio
.venv/bin/python -m pytest
```

The suite uses a fake llama-server (`tests/fake_llama_server.py`), so it doesn't download or load any real models.

## License

Apache 2.0. See [LICENSE](LICENSE).
