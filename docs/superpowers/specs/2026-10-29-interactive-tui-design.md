# OpenLLM interactive TUI — design

Date: 2026-10-29
Status: draft, awaiting review

## Goal

Replace the numbered `input()` menus in `openllm` with a full-screen interactive
terminal app (a dashboard that stays open while the model runs): pick a model, tune
settings with live memory feedback, start/stop the server, chat with it, watch its log,
and copy client connection snippets, all in one place.

## Decisions (agreed)

- **Scope:** full dashboard (setup, run, log, built-in chat), not just a nicer wizard.
- **Library:** Textual, installed into a local `.venv` next to the script via `uv`.
- **Structure:** a small package with a UI-free `core`, and a thin `openllm` entry point.
- **Data formats unchanged:** `models.json` and `state.json` keep their current shape, so the
  old settings and API key carry over.

## Non-goals

- Multiple models running at once, multi-user chat, chat history persistence.
- Remote or cloud models, non-llama.cpp runtimes (the adapter dict stays as the extension point).
- Windows or Linux support (the code stays macOS-first: `sysctl`, `ipconfig`).

## Layout

```
openllm                      entry point: bootstrap venv, then start the app
openllm_app/
  core/
    registry.py                 load models.json, runtime paths
    state.py                    load/save state.json (0600, holds the API key)
    memory.py                   RAM / GPU-budget estimate (current estimate_gb logic)
    cmdline.py                  settings -> llama.cpp argv (current build_llama_cmd), adapter dict
    downloads.py                curl-based resumable downloads that report progress
    snippets.py                 Cline / opencode / curl client snippets (current print_server_info)
  server.py                     owns the llama-server child process
  client.py                     streaming chat against /v1/chat/completions
  ui/
    app.py                      Textual App, screen routing
    models_screen.py            model list + settings panel + memory gauge
    run_screen.py               header + Chat / Log / Connect tabs
    widgets.py                  MemoryGauge, LogPane, ChatPane, ThinkingBlock
tests/                          core + server + client tests (no terminal needed)
```

`core` imports nothing from `ui`, `server` or `client`. The UI imports `core`, `server`
and `client`.

## Components

**core/memory.py.** `estimate_gb(model, ctx, kv)` and `gpu_budget_gb()` exactly as today
(weights + KV per 1K tokens + 0.55 GiB; budget = 75% of RAM − 1 GiB). Returns numbers; the
gauge decides colour (red above budget).

**core/cmdline.py.** `build_command(runtime, model, settings, api_key)` returns the argv list,
unchanged semantics: `-ngl 99 -fa on`, KV types, sampling flags, `--jinja`, the
`reasoning_effort` template kwarg; server mode adds host, port, `-np 1`, `--alias`,
`--api-key`. Chat mode (`llama-cli`) remains available for the headless path only.

**core/downloads.py.** Same `curl -fL -C - --retry 5`, but progress is read from curl's output
and reported through a callback so the UI can draw a progress bar. The runtime tarball keeps
its path-traversal check, extended to reject symlink and hardlink members. The runtime size
shown to the user comes from the `Content-Length` header instead of the hardcoded "~12 MB".

**server.py.** `ServerProcess` spawns `llama-server` as a child (no `execv`), merges
stdout and stderr into a line stream for the log pane, and polls `GET /health` until ready.
States: `starting → ready → stopping → stopped | failed`. Stop sends SIGTERM, then SIGKILL
after a timeout. An `atexit` handler plus signal handlers guarantee the child never outlives
the app.

**client.py.** `stream_chat(base_url, key, model_id, messages, max_tokens)` yields typed events:
`reasoning_delta`, `content_delta`, `usage`, `timings`, `done`, `error`. Uses `httpx` streaming
and reads `reasoning_content` and `content` deltas from the SSE chunks. Tokens/sec comes from
the `timings` object in the final chunk; context used comes from `usage`
(prompt + completion tokens).

## Screens

**Models screen.** Left: model list (✓ downloaded / not downloaded). Right: settings widgets,
all keyboard- and mouse-operable: Mode (server/chat), KV cache, Context (stepper over the
presets 8K–128K, capped at `max_ctx`, plus a typed custom value), Effort, and, for server
mode, Expose (local/lan) and Port. A memory gauge under the settings updates on every
change; it turns red when the estimate exceeds the GPU budget (a warning, not a block).
Keys: `↵` start, `d` download missing runtime/model, `l` apply last saved settings, `q` quit.
Settings pre-fill from `state.json`.

**Run screen.** Header: model name, status dot, port, context used / total. Tabs:
- *Chat:* message list, input box, streamed reply. Reasoning is shown in a collapsible block
  (collapsed once the answer starts). `Ctrl+C` stops the current generation. Footer shows
  tokens/sec.
- *Log:* scrolling `llama-server` output.
- *Connect:* base URL, API key and the Cline / opencode / curl snippets, with a key binding
  to copy each one.

`s` or `Esc` stops the server and returns to the Models screen (with confirmation if a
reply is generating).

## Data flow

```
Models screen --settings--> state.json (save on Start)
      |
      v
core.cmdline.build_command --> ServerProcess.start()
      |                              |-- log lines --> Log tab
      |                              `-- /health   --> header status
      v
Chat tab --> client.stream_chat --> llama-server --> events --> ChatPane / header stats
```

LAN mode generates (or reuses) the `sk-…` key in `state.json`, as today.

## Bootstrap

`openllm` is stdlib-only until it needs Textual:
1. If `import textual` works in the current interpreter, run the app.
2. Otherwise, if `.venv/` exists, re-exec with `.venv/bin/python`.
3. Otherwise, ask once ("Install the UI dependencies into ./.venv with uv? [Y/n]"), run
   `uv venv .venv` and `uv pip install textual httpx`, then re-exec.
If `uv` is missing, print the exact manual commands and exit.

`openllm --last` runs headless: loads saved settings, starts the server via the same
`core` code with no UI, prints the connection info, and stays in the foreground until Ctrl+C
(same behaviour as the current script, minus `execv`).

## Error handling

- **Start failures** (port in use, out of memory, bad model file): the Run screen opens in
  the `failed` state, shows the last ~30 log lines and the detected reason when recognisable,
  with "back to settings".
- **Server dies mid-session:** header turns red, chat input is disabled, log stays readable.
- **Chat request errors:** shown inline as an error message in the transcript; the
  conversation continues.
- **Download failure:** progress bar shows the error; re-running resumes (`-C -`).
- **Corrupt or missing `state.json`:** fall back to defaults (current `load_json` behaviour).

## Testing

- `core`: unit tests for memory estimates, argv generation (server/lan/api-key cases),
  state round-trip and file mode, tar member rejection, snippet content.
- `server.py`: tests against a tiny fake `llama-server` script (serves `/health`, prints
  log lines, can exit non-zero) to cover start, ready, failure and cleanup-on-exit.
- `client.py`: tests against a local fake SSE endpoint, including reasoning deltas,
  usage, timings and mid-stream disconnects.
- UI: Textual's `run_test()` pilot for the settings → gauge interaction and Start navigation.
- Manual: one real run with the Bonsai model for chat streaming and tokens/sec.

## Risks

- **Python 3.15.1 is very new.** Textual or its dependencies may not support it yet.
  Mitigation: the bootstrap tells `uv` to use a supported interpreter if the default fails.
- **llama.cpp fork response shape.** The reasoning field name and `timings` object are
  assumed from upstream llama-server and must be confirmed against the PrismML build
  before the client is finalised.
- **Quitting uncleanly** could orphan a 5.9 GB model process; covered by the cleanup
  handlers and the server tests above.
