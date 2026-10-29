# OpenOllama Interactive TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the numbered-menu launcher with a full-screen Textual dashboard (model picker, live memory gauge, server control, chat, log, client snippets).

**Architecture:** The logic in the current `openollama` script moves into a UI-free `openollama_app/core` package. A `ServerProcess` (child process, not `execv`) and a streaming `client` sit on top of it, and two Textual screens (Models, Run) sit on top of those. `openollama` becomes a stdlib-only entry point that bootstraps a local `.venv` with `uv` and starts the app.

**Tech Stack:** Python 3, Textual, httpx, pytest + pytest-asyncio, uv, curl, llama.cpp (PrismML fork) `llama-server`.

**Spec:** `docs/superpowers/specs/2026-10-29-interactive-tui-design.md`

## Global Constraints

- Textual and httpx are installed only into `./.venv` next to the script via `uv`; `openollama` itself uses only the stdlib until it re-execs into the venv.
- `models.json` and `state.json` keep their current formats; existing `state.json` content (including `api_key`) must keep working.
- `state.json` is written with mode `0600`.
- `core` imports nothing from `ui`, `server` or `client`.
- Memory estimate: `weights_gib + kv_mib_per_1k[kv] * (ctx/1024) / 1024 + 0.55`; GPU budget: `ram_gb * 0.75 - 1.0`. Exceeding the budget is a warning (red gauge), never a block.
- Server mode argv keeps: `-ngl 99 -fa on`, `--cache-type-k/-v`, sampling flags, `--jinja`, `--chat-template-kwargs {"reasoning_effort": ...}`, `-np 1`, `--alias <model_id>`, `--api-key` only for `lan`.
- `llama-server` runs as a child process (no `execv`); it must never outlive the app (`atexit` + signal handlers).
- Runtime tarball extraction rejects absolute paths, `..`, symlink and hardlink members.
- macOS-first (`sysctl`, `ipconfig`); no other platforms.
- Context range: presets 8K, 16K, 32K, 64K, 128K, capped at the model's `max_ctx`; custom values clamp to `[1024, max_ctx]`.

## Review Focus

- Port already in use when starting: server goes to `failed` with a readable reason, and the UI offers "back to settings" instead of hanging on "starting".
- Saved `state.json` entry from the old script that lacks keys (e.g. a chat-mode entry with no `expose`/`port`) loaded via `l` while in server mode: missing keys fall back to defaults, no `KeyError`.
- Custom context entered as garbage (`"abc"`, `"0"`, `"-5k"`, `"9999999k"`): rejected or clamped, never crashes the settings panel.
- Thinking model spends the entire `max_tokens` on reasoning and returns empty `content`: Chat shows a visible "no answer, reasoning used all tokens" notice rather than a blank reply.
- Server killed externally (e.g. `kill`) while a reply is streaming: chat shows an error line, input is disabled, header turns red, app stays alive.

---

### Task 1: Scaffold, git, venv bootstrap

**Files:**
- Create: `openollama_app/__init__.py`, `openollama_app/bootstrap.py`, `tests/test_bootstrap.py`, `pyproject.toml` (pytest config only: `asyncio_mode = "auto"`, `testpaths = ["tests"]`), `.gitignore` (`.venv/`, `.dl/`, `models/`, `runtimes/`, `state.json`, `__pycache__/`)
- Modify: `openollama` (becomes the thin entry point; the old logic stays in `openollama.legacy` until Task 8 so the tool keeps working)

**Interfaces:**
- Produces: `bootstrap.venv_python(root: Path) -> Path` (`root/.venv/bin/python`); `bootstrap.plan(root: Path, have_textual: bool, venv_exists: bool, have_uv: bool) -> Literal["run", "reexec", "install", "manual"]`; `bootstrap.install(root: Path) -> None` (runs `uv venv .venv`, then `uv pip install -p .venv/bin/python textual httpx`; on failure retries with `--python 3.13`; adds dev deps `pytest pytest-asyncio` only when `OPENOLLAMA_DEV=1`).

- [ ] **Step 1: Write failing tests** in `tests/test_bootstrap.py`: `test_plan_runs_when_textual_importable` (`plan(r, True, False, False) == "run"`), `test_plan_reexecs_into_existing_venv` (`plan(r, False, True, True) == "reexec"`), `test_plan_installs_when_uv_present` (`plan(r, False, False, True) == "install"`), `test_plan_manual_when_no_uv` (`plan(r, False, False, False) == "manual"`).
- [ ] **Step 2:** `git init`; run `python3 -m pytest tests/test_bootstrap.py -v` (system pytest or `uvx pytest`); expect FAIL (module missing).
- [ ] **Step 3:** Implement `bootstrap.py` per the Interfaces block. The "manual" case prints the exact `uv venv` / `uv pip install` commands.
- [ ] **Step 4:** Make `openollama` call `bootstrap.plan(...)` and act: `run` → import and start the app (a placeholder `main()` that prints "TUI not built yet" and falls back to `openollama.legacy.main()` for now), `reexec` → `os.execv(venv_python, [venv_python, __file__, *argv])`, `install` → prompt "Install the UI dependencies into ./.venv with uv? [Y/n]" then `install` and re-exec.
- [ ] **Step 5: Verify Python compatibility (spec risk).** Run `OPENOLLAMA_DEV=1 ./openollama`, then `.venv/bin/python -c "import textual, httpx; print(textual.__version__)"`. Expected: prints a version. If install fails on the default interpreter and the `--python 3.13` retry succeeds, keep the retry; record which interpreter worked in a one-line comment in `bootstrap.py`.
- [ ] **Step 6:** Run `.venv/bin/python -m pytest tests/test_bootstrap.py -v`. Expected: PASS. Commit: `git add -A && git commit -m "feat: scaffold package and uv venv bootstrap"`.

### Task 2: Core (registry, state, memory, cmdline, snippets)

**Files:**
- Create: `openollama_app/core/{__init__,registry,state,memory,cmdline,snippets}.py`, `tests/core/test_{registry,state,memory,cmdline,snippets}.py`

**Interfaces:**
- Produces:
  - `registry.load_registry(path: Path) -> dict` (raises `RegistryError` on unreadable/invalid); `registry.runtime_dir(root, rt) -> Path`; `registry.runtime_bin(root, rt, name) -> Path`; `registry.model_path(root, m) -> Path`
  - `state.load_state(path: Path) -> dict` (missing/corrupt → `{}`); `state.save_state(path: Path, state: dict) -> None` (mode `0600`); `state.merge_defaults(m: dict, saved: dict) -> dict` (returns a complete settings dict: `mode="server"`, `kv="q8_0"`, `ctx=32768`, `effort=m["reasoning_efforts"][-1]`, `expose="local"`, `port=8080`, `max_tokens=16384`, overlaid with `saved`, with `ctx` clamped and unknown `kv`/`effort` values replaced by the defaults)
  - `memory.estimate_gb(m, ctx, kv) -> float`; `memory.total_ram_gb() -> float`; `memory.gpu_budget_gb(ram_gb: float) -> float`; `memory.clamp_ctx(value: int, max_ctx: int) -> int`; `memory.parse_tokens(text: str) -> int` (accepts `48k`, `49152`, `1,024`; raises `ValueError` otherwise, including values ≤ 0)
  - `cmdline.build_command(root, rt, m, s, api_key: str | None) -> list[str]`; `cmdline.ADAPTERS: dict[str, Callable]` keyed by `rt["adapter"]`
  - `snippets.lan_ip() -> str`; `snippets.client_snippets(s: dict, api_key: str | None) -> dict[str, str]` with keys `base_url`, `api_key`, `cline`, `opencode`, `curl`

- [ ] **Step 1: Write failing tests** (port the behaviour from the current script). `test_estimate_gb_matches_registry_values`: Bonsai at 32768 / `q8_0` → `5.54 + 34*32/1024 + 0.55`. `test_gpu_budget`: `gpu_budget_gb(16) == 11.0`. `test_parse_tokens_cases`: `"48k"→49152`, `"49152"→49152`, and `ValueError` for `"abc"`, `"0"`, `"-5k"`. `test_clamp_ctx`: `clamp_ctx(100, 262144) == 1024`, `clamp_ctx(10**9, 262144) == 262144`. `test_state_roundtrip_and_mode`: saved file has `stat().st_mode & 0o777 == 0o600`. `test_load_state_corrupt_returns_empty`. `test_merge_defaults_old_chat_entry`: a saved `{"mode": "chat", "kv": "q8_0", "ctx": 32768, "effort": "xhigh", "max_tokens": 16384}` merged then switched to `mode="server"` has `expose == "local"` and `port == 8080`. `test_build_command_server_local` (host `127.0.0.1`, `-np 1`, `--alias`, no `--api-key`), `test_build_command_lan_has_key` (host `0.0.0.0`, `--api-key sk-x`), `test_build_command_chat_uses_llama_cli_with_n`. `test_snippets_lan_uses_given_ip` (monkeypatch `lan_ip`), `test_snippets_opencode_is_valid_json`.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/core -v`. Expected: FAIL (imports).
- [ ] **Step 3:** Implement the modules by moving the equivalent functions out of `openollama.legacy` (same formulas and flags); `snippets` returns the strings instead of printing them.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/core -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `git add -A && git commit -m "feat: extract UI-free core from launcher script"`.

### Task 3: Downloads with progress

**Files:**
- Create: `openollama_app/core/downloads.py`, `tests/core/test_downloads.py`

**Interfaces:**
- Consumes: `registry.runtime_dir`, `registry.model_path`.
- Produces: `class DownloadError(Exception)`; `class UnsafeArchive(Exception)`; `remote_size(url: str) -> int | None` (HEAD via `curl -sIL`, parses the last `content-length`); `download(url: str, dest: Path, on_progress: Callable[[int, int | None], None]) -> None` (runs `curl -fL -C - --retry 5 -sS -o dest url` in a subprocess and polls `dest` size every 0.25 s from a thread, calling `on_progress(done, total)`); `safe_extract(archive: Path, dest_dir: Path) -> None`; `install_runtime(root, rt_id, rt, on_progress) -> None`; `fetch_model(root, m, on_progress) -> None` (downloads to `.part`, renames on success).

- [ ] **Step 1: Write failing tests.** `test_safe_extract_rejects_parent_traversal`, `test_safe_extract_rejects_absolute`, `test_safe_extract_rejects_symlink_member`, `test_safe_extract_rejects_hardlink_member` (each builds a tar in `tmp_path` and expects `UnsafeArchive`, with nothing extracted), `test_safe_extract_strips_first_component` (a clean tar extracts with `--strip-components=1` semantics), `test_download_reports_progress_and_resumes` (serve a 2 MB file from `http.server` in a thread; assert `on_progress` called with increasing `done`, final `done == total`; pre-create a half-size `dest` and assert the final file is byte-identical), `test_download_failure_raises` (404 → `DownloadError`).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/core/test_downloads.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. `safe_extract` inspects every member (`isabs`, `..` in parts, `issym()`, `islnk()`) before extracting with the `tarfile` module's `filter="data"`; no shelling out to `tar`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `git add -A && git commit -m "feat: downloads with progress and stricter archive checks"`.

### Task 4: ServerProcess

**Files:**
- Create: `openollama_app/server.py`, `tests/fake_llama_server.py` (stdlib script: argv `--port N --mode ok|exit1|bindfail|slowstart`; serves `GET /health`; prints a few log lines), `tests/test_server.py`

**Interfaces:**
- Consumes: argv from `cmdline.build_command`.
- Produces: `class ServerProcess` with `__init__(argv: list[str], host: str, port: int, on_line: Callable[[str], None], on_state: Callable[[str], None])`; `start() -> None` (non-blocking; a reader thread forwards each stdout/stderr line to `on_line`, a health thread polls `GET /health` every 0.5 s); `stop(timeout: float = 10.0) -> None` (SIGTERM, then SIGKILL); property `state` in `{"idle","starting","ready","stopping","stopped","failed"}`; property `failure_reason: str | None` (set from a known-pattern scan of the last 30 log lines: `address already in use` → "Port N is already in use", `out of memory`/`failed to allocate` → "Not enough memory for this context size", otherwise "Server exited with code N"); `last_lines(n: int = 30) -> list[str]`; module-level `install_cleanup(proc)` registering `atexit` and SIGTERM/SIGHUP handlers that call `stop`.

- [ ] **Step 1: Write failing tests.** `test_start_reaches_ready` (waits ≤ 5 s for `state == "ready"`; log lines were received), `test_stop_terminates_child` (child pid gone after `stop`), `test_nonzero_exit_goes_failed` (`exit1` → `failed`, reason contains "exited with code 1"), `test_port_in_use_reason` (`bindfail` prints "address already in use" and exits → reason "Port N is already in use"), `test_child_killed_externally_goes_failed` (kill pid while `ready`), `test_cleanup_kills_child_when_parent_exits` (spawn a Python subprocess that starts a `ServerProcess` with `install_cleanup`, then exits via `sys.exit`; assert the fake server pid is gone).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_server.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement with `subprocess.Popen(..., start_new_session=True)` so `stop` can signal the process group; state changes call `on_state` from the worker threads (the UI marshals to its own loop in Task 6/7).
- [ ] **Step 4:** Run the tests. Expected: PASS. Commit: `git add -A && git commit -m "feat: ServerProcess with health polling and guaranteed cleanup"`.

### Task 5: Streaming chat client

**Files:**
- Create: `openollama_app/client.py`, `tests/test_client.py`

**Interfaces:**
- Produces: `@dataclass ChatEvent(kind: Literal["reasoning","content","usage","timings","done","error"], text: str = "", data: dict | None = None)`; `async stream_chat(base_url: str, api_key: str | None, model_id: str, messages: list[dict], max_tokens: int) -> AsyncIterator[ChatEvent]`. It POSTs `{"model", "messages", "max_tokens", "stream": true}` with `Authorization: Bearer <key>` when a key is given. `delta.reasoning_content` → `reasoning`; `delta.content` → `content`; the chunk's `usage` → `usage` (`data={"prompt_tokens", "completion_tokens"}`); the chunk's `timings` → `timings` (`data` has `predicted_per_second`); `[DONE]` → `done`. A transport error or non-200 → one `error` event, then the generator ends.

- [ ] **Step 1: Write failing tests** against a local fake SSE server (httpx `MockTransport` is not enough for streaming disconnects, so use a thread-based `http.server`): `test_reasoning_then_content_order`, `test_usage_and_timings_events`, `test_auth_header_sent_only_with_key`, `test_http_401_yields_error`, `test_connection_refused_yields_error`, `test_mid_stream_disconnect_yields_error_not_exception`, `test_empty_content_with_length_finish_reports_notice` (a stream with only reasoning deltas and `finish_reason: "length"` ends with a `done` event whose `data == {"finish_reason": "length", "empty_answer": True}`).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_client.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement with `httpx.AsyncClient.stream` and manual SSE line parsing; the `done` event carries `finish_reason` and `empty_answer` (no `content` text seen).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Verify against the real PrismML build (spec risk).** Start the real server (`.venv/bin/python -c` with `build_command` using saved settings), then `curl -N localhost:8080/v1/chat/completions -d '{"model":"ternary-bonsai-2-27b","stream":true,"max_tokens":512,"messages":[{"role":"user","content":"hi"}]}'`. Confirm the reasoning field name and where `timings` and `usage` appear; if they differ from the assumptions above, fix `client.py` and the test fixtures, and note the actual field names in a comment at the top of `client.py`.
- [ ] **Step 6:** Commit: `git add -A && git commit -m "feat: streaming chat client"`.

### Task 6: App shell and Models screen

**Files:**
- Create: `openollama_app/ui/{__init__,app,models_screen,widgets}.py`, `tests/ui/test_models_screen.py`

**Interfaces:**
- Consumes: `registry`, `state`, `memory`, `downloads` (Task 2–3).
- Produces: `class OpenOllamaApp(App)` with `__init__(root: Path)`, loading registry and state; `class ModelsScreen(Screen)`; `widgets.MemoryGauge(Static)` with `update_estimate(used_gb: float, budget_gb: float)` (adds CSS class `over` when `used > budget`); message `ModelsScreen.StartRequested(model_id: str, settings: dict)`. Settings widgets: `RadioSet` for mode/KV/effort/expose, a context stepper over the presets plus an `Input` for a custom value, an `Input` for port. Keys: `enter` start, `d` download, `l` apply last, `q` quit.

- [ ] **Step 1: Write failing pilot tests** using `app.run_test()` with a `tmp_path` root containing a small `models.json`: `test_gauge_updates_when_kv_changes` (select `q4_0` → gauge value equals `estimate_gb(m, ctx, "q4_0")`), `test_gauge_red_when_over_budget` (monkeypatch `total_ram_gb` to 8 → `over` class present at 128K), `test_custom_context_garbage_is_rejected` (type `abc`, `0`, `-5k`, `9999999k`: the settings panel keeps its previous valid ctx and shows an inline error for the first three, clamps the last to `max_ctx`), `test_last_settings_old_chat_entry_fills_defaults` (state has an old chat-mode entry; press `l` with server mode → no exception, `port == 8080`), `test_not_downloaded_model_blocks_start_and_offers_download` (Enter shows "press d to download" instead of starting), `test_enter_posts_start_requested_and_saves_state` (state file updated with the chosen settings).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/ui -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Settings changes call `memory.estimate_gb` and update the gauge on every change. `d` runs `downloads.install_runtime` / `fetch_model` in a Textual worker thread and shows a `ProgressBar` fed through `call_from_thread`. Wire `openollama` so `run` starts `OpenOllamaApp(root).run()`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Manual check: `./openollama` shows the Models screen and the gauge reacts to arrow keys and mouse clicks.
- [ ] **Step 5:** Commit: `git add -A && git commit -m "feat: Textual app shell and Models screen"`.

### Task 7: Run screen (Chat, Log, Connect)

**Files:**
- Create: `openollama_app/ui/run_screen.py`; extend `openollama_app/ui/widgets.py` (`LogPane`, `ChatPane`, `ThinkingBlock`); `tests/ui/test_run_screen.py`

**Interfaces:**
- Consumes: `ServerProcess` (Task 4), `stream_chat` (Task 5), `build_command`, `client_snippets`.
- Produces: `class RunScreen(Screen)` with `__init__(model_id: str, settings: dict, api_key: str | None)`. The header shows model name, status dot (starting/ready/failed/stopped), port, `ctx used/total` (from the last `usage` event) and tokens/sec (from `timings`). Tabs: `Chat`, `Log`, `Connect`. Keys: `enter` send, `ctrl+c` cancels the current generation, `s`/`escape` stop the server and pop the screen (with a confirm modal while a reply is generating), `y` on the Connect tab copies the focused snippet. `ThinkingBlock` is collapsible and auto-collapses when the first `content` event arrives.

- [ ] **Step 1: Write failing pilot tests** with `ServerProcess` and `stream_chat` replaced by fakes injected via `RunScreen` constructor kwargs `server_factory` and `chat_fn` (default to the real ones): `test_status_follows_server_state` (starting → ready → header text changes), `test_log_lines_appear_in_log_tab`, `test_chat_streams_reasoning_then_content` (reasoning goes into the `ThinkingBlock`, which collapses when content starts), `test_header_shows_tokens_per_sec_and_ctx_used`, `test_empty_answer_shows_notice` (a `done` event with `empty_answer` → a transcript line containing "reasoning used all tokens"), `test_failed_start_shows_reason_and_back_action` (server `failed` with reason "Port 8080 is already in use" → the reason is visible, the "Back to settings" action pops to `ModelsScreen`), `test_server_dies_mid_stream_disables_input` (the fake server flips to `failed` while the chat is streaming → an error line appears, the input is disabled, the app is still running), `test_stop_key_confirms_when_generating`, `test_connect_tab_shows_snippets_and_copy` (monkeypatch the clipboard call; the lan key is shown when `expose == "lan"`).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/ui/test_run_screen.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Server callbacks arrive on threads and are marshalled with `app.call_from_thread`; chat runs in an `asyncio` worker that is cancellable; the screen calls `install_cleanup` on the real server in `on_mount`. Connect `ModelsScreen.StartRequested` to push `RunScreen`; for `chat` mode, push `RunScreen` with an in-app server as well (the UI replaces `llama-cli`).
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests -v`. Expected: PASS.
- [ ] **Step 5: Manual check with the real model.** `./openollama` → Start → wait for `ready` → send "explain ternary weights" → thinking streams and collapses, tokens/sec appears, `s` stops the server; confirm with `pgrep llama-server` that no process remains, including after quitting with `q` mid-load.
- [ ] **Step 6:** Commit: `git add -A && git commit -m "feat: Run screen with chat, log and connect tabs"`.

### Task 8: Headless `--last` and cleanup of the old launcher

**Files:**
- Modify: `openollama` (remove the `legacy` fallback), `openollama_app/__main__.py` (new: `main(argv)`); Delete: `openollama.legacy`
- Create: `tests/test_headless.py`

**Interfaces:**
- Consumes: `core.*`, `ServerProcess`, `client_snippets`.
- Produces: `openollama_app.__main__.main(argv: list[str]) -> int`; `--last` starts the saved model's server without the UI, prints the connection info (same fields as the old banner) and blocks until Ctrl+C, then stops the server. `--last` with no saved state exits 1 with a message. When the saved mode is `chat`, `--last` keeps the old behaviour and `execv`s `llama-cli` (the spec keeps `llama-cli` for the headless path only).

- [ ] **Step 1: Write failing tests.** `test_last_without_state_exits_1`, `test_last_chat_mode_execs_llama_cli` (monkeypatch `os.execv`, assert argv[0] ends in `llama-cli`), `test_last_starts_server_prints_base_url_and_stops_on_sigint` (use the `fake_llama_server` through a test registry; send SIGINT; assert exit code 0 and the child is gone).
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_headless.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `main`; delete `openollama.legacy`; make `openollama` always bootstrap then call `main(sys.argv[1:])`.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests -v`. Expected: all PASS. Manual: `./openollama --last` serves on the saved port and Ctrl+C leaves no `llama-server` process.
- [ ] **Step 5:** Commit: `git add -A && git commit -m "feat: headless --last and remove legacy launcher"`.
