import asyncio

import pytest
from textual.widgets import Input, TabbedContent

from openollama_app.core import memory
from openollama_app.ui.app import OpenOllamaApp
from openollama_app.ui.models_screen import ModelsScreen
from openollama_app.ui.run_screen import ConfirmStop, RunScreen
from openollama_app.ui.widgets import ChatPane, LogPane, ThinkingBlock

from .fakes import Factory, ev, scripted_chat
from .test_models_screen import MID, SIZE, setup_root

FINISH = ev("done", finish_reason="stop", empty_answer=False)


@pytest.fixture(autouse=True)
def big_mac(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_gb", lambda: 64.0)


async def launch(root, registry, pilot_fn, chat=None, saved=None):
    setup_root(root, registry, saved=saved)
    factory = Factory()
    app = OpenOllamaApp(root, run_kwargs={"server_factory": factory, "chat_fn": chat or scripted_chat([FINISH])})
    async with app.run_test(size=SIZE) as pilot:
        await pilot.click("#start")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        await pilot_fn(app, pilot, factory.server)


async def send(app, pilot, text):
    app.screen.query_one("#chat-input", Input).focus()
    await pilot.press(*text, "enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_status_follows_server_state(root, registry):
    async def run(app, pilot, server):
        screen = app.screen
        assert "loading" in screen.header_text and ":8080" in screen.header_text
        assert screen.query_one("#chat-input", Input).disabled
        server.set_state("ready")
        await pilot.pause()
        assert "ready" in screen.header_text
        assert not screen.query_one("#chat-input", Input).disabled
    await launch(root, registry, run)


async def test_server_argv_and_chat_mode_uses_llama_server(root, registry):
    async def run(app, pilot, server):
        assert server.argv[0].endswith("llama-server") and "--port" in server.argv
    await launch(root, registry, run, saved={MID: {"mode": "chat", "kv": "q8_0", "ctx": 32768,
                                                   "effort": "xhigh", "max_tokens": 16384}})


async def test_log_lines_appear_in_log_tab(root, registry):
    async def run(app, pilot, server):
        server.emit_line("loading tensors")
        await pilot.pause()
        assert "loading tensors" in app.screen.query_one(LogPane).texts
    await launch(root, registry, run)


async def test_chat_streams_reasoning_then_content(root, registry):
    chat = scripted_chat([ev("reasoning", "let me "), ev("reasoning", "think"),
                          ev("content", "Hello"), ev("content", " there"), FINISH])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "hi")
        pane = app.screen.query_one(ChatPane)
        block = pane.query_one(ThinkingBlock)
        assert block.text == "let me think" and block.collapsed
        text = pane.transcript_text()
        assert "you ▸ hi" in text and "Hello there" in text
        assert chat.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
        assert chat.calls[0]["base_url"] == "http://127.0.0.1:8080/v1"
    await launch(root, registry, run, chat=chat)


async def test_history_is_sent_on_second_turn(root, registry):
    chat = scripted_chat([ev("content", "A1"), FINISH])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "q1")
        await send(app, pilot, "q2")
        assert chat.calls[1]["messages"] == [{"role": "user", "content": "q1"},
                                              {"role": "assistant", "content": "A1"},
                                              {"role": "user", "content": "q2"}]
    await launch(root, registry, run, chat=chat)


async def test_header_shows_tokens_per_sec_and_ctx_used(root, registry):
    chat = scripted_chat([ev("content", "x"), ev("usage", prompt_tokens=4096, completion_tokens=100),
                          ev("timings", predicted_per_second=18.4), FINISH])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "hi")
        h = app.screen.header_text
        assert "4.1K/32K" in h and "18.4 tok/s" in h
    await launch(root, registry, run, chat=chat)


async def test_model_text_with_markup_characters_does_not_crash(root, registry):
    chat = scripted_chat([ev("content", "[/bold] [red]x[/] [unclosed"), FINISH])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "[b]hi")
        assert "[/bold] [red]x[/] [unclosed" in app.screen.query_one(ChatPane).transcript_text()
    await launch(root, registry, run, chat=chat)


async def test_empty_answer_shows_notice(root, registry):
    chat = scripted_chat([ev("reasoning", "thinking forever"),
                          ev("done", finish_reason="length", empty_answer=True)])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "hi")
        assert "reasoning used all tokens" in app.screen.query_one(ChatPane).transcript_text()
    await launch(root, registry, run, chat=chat)


async def test_chat_error_event_is_shown_and_chat_continues(root, registry):
    chat = scripted_chat([ev("error", "HTTP 401: bad key")])

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await send(app, pilot, "hi")
        assert "HTTP 401: bad key" in app.screen.query_one(ChatPane).transcript_text()
        assert not app.screen.query_one("#chat-input", Input).disabled
    await launch(root, registry, run, chat=chat)


async def test_failed_start_shows_reason_and_back_action(root, registry):
    async def run(app, pilot, server):
        server.emit_line("bind: address already in use")
        server.set_state("failed", reason="Port 8080 is already in use")
        await pilot.pause()
        assert "Port 8080 is already in use" in app.screen.fail_text
        assert "address already in use" in app.screen.fail_text
        await pilot.click("#back")
        await pilot.pause()
        assert isinstance(app.screen, ModelsScreen)
    await launch(root, registry, run)


async def test_server_dies_mid_stream_disables_input(root, registry):
    gate = asyncio.Event()
    chat = scripted_chat([ev("reasoning", "hmm"), "WAIT", ev("content", "never")], gate)

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        app.screen.query_one("#chat-input", Input).focus()
        await pilot.press(*"hi", "enter")
        await pilot.pause(0.2)
        server.set_state("failed", reason="Server exited with code -9")
        await pilot.pause(0.2)
        screen = app.screen
        assert screen.query_one("#chat-input", Input).disabled
        assert "Server stopped" in screen.query_one(ChatPane).transcript_text()
        assert "failed" in screen.header_text
        assert app.is_running
    await launch(root, registry, run, chat=chat)


async def test_stop_key_confirms_when_generating(root, registry):
    gate = asyncio.Event()
    chat = scripted_chat([ev("reasoning", "hmm"), "WAIT"], gate)

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        app.screen.query_one("#chat-input", Input).focus()
        await pilot.press(*"hi", "enter")
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmStop) and not server.stopped
        await pilot.press("y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert server.stopped and isinstance(app.screen, ModelsScreen)
    await launch(root, registry, run, chat=chat)


async def test_stop_when_idle_goes_back_without_confirmation(root, registry):
    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert server.stopped and isinstance(app.screen, ModelsScreen)
    await launch(root, registry, run)


async def test_ctrl_c_cancels_generation_but_keeps_server(root, registry):
    gate = asyncio.Event()
    chat = scripted_chat([ev("reasoning", "hmm"), "WAIT"], gate)

    async def run(app, pilot, server):
        server.set_state("ready")
        await pilot.pause()
        app.screen.query_one("#chat-input", Input).focus()
        await pilot.press(*"hi", "enter")
        await pilot.pause(0.2)
        assert app.screen.generating
        await pilot.press("ctrl+c")
        await pilot.pause(0.2)
        assert not app.screen.generating and not server.stopped
        assert "(stopped)" in app.screen.query_one(ChatPane).transcript_text()
    await launch(root, registry, run, chat=chat)


async def test_connect_tab_shows_snippets_and_copy(root, registry, monkeypatch):
    async def run(app, pilot, server):
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        screen = app.screen
        assert "http://127.0.0.1:8080/v1" in screen.connect_text
        screen.query_one(TabbedContent).active = "connect-tab"
        await pilot.pause()
        screen.query_one("#snip-curl").focus()
        await pilot.pause()
        screen.action_copy_snippet()
        assert copied and copied[0].startswith("curl http://127.0.0.1:8080/v1/chat/completions")
    await launch(root, registry, run)


async def test_connect_tab_lan_shows_generated_key(root, registry, monkeypatch):
    from openollama_app.core import snippets
    monkeypatch.setattr(snippets, "lan_ip", lambda: "192.168.1.9")

    async def run(app, pilot, server):
        assert "192.168.1.9:8080" in app.screen.connect_text and "sk-lan-key" in app.screen.connect_text
    await launch(root, registry, run, saved={"api_key": "sk-lan-key",
                                              MID: {"mode": "server", "expose": "lan", "kv": "q8_0",
                                                    "ctx": 32768, "effort": "xhigh", "port": 8080}})
