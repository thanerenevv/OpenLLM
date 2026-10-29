import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from openollama_app.server import ServerProcess

FAKE = str(Path(__file__).parent / "fake_llama_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def make():
    made = []

    def _make(mode="ok"):
        port = free_port()
        lines, states = [], []
        sp = ServerProcess([sys.executable, FAKE, "--port", str(port), "--mode", mode],
                           "127.0.0.1", port, lines.append, states.append)
        made.append(sp)
        return sp, lines, states, port

    yield _make
    for sp in made:
        sp.stop(timeout=2)


def test_start_reaches_ready(make):
    sp, lines, states, _ = make()
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    assert "fake server booting" in lines
    assert states[:2] == ["starting", "ready"]


def test_slow_start_stays_starting_until_health_ok(make):
    sp, *_ = make("slowstart")
    sp.start()
    time.sleep(0.3)
    assert sp.state == "starting"
    assert wait_for(lambda: sp.state == "ready")


def test_stop_terminates_child(make):
    sp, *_ = make()
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    pid = sp.pid
    sp.stop()
    assert sp.state == "stopped"
    assert not alive(pid)


def test_nonzero_exit_goes_failed(make):
    sp, *_ = make("exit1")
    sp.start()
    assert wait_for(lambda: sp.state == "failed")
    assert "exited with code 1" in sp.failure_reason


def test_port_in_use_reason(make):
    sp, _, _, port = make("bindfail")
    sp.start()
    assert wait_for(lambda: sp.state == "failed")
    assert sp.failure_reason == f"Port {port} is already in use"


def test_out_of_memory_reason(make):
    sp, *_ = make("oom")
    sp.start()
    assert wait_for(lambda: sp.state == "failed")
    assert sp.failure_reason == "Not enough memory for this context size"


def test_child_killed_externally_goes_failed(make):
    sp, *_ = make()
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    os.kill(sp.pid, 9)
    assert wait_for(lambda: sp.state == "failed")


def test_last_lines_keeps_recent_output(make):
    sp, *_ = make()
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    assert "listening" in sp.last_lines(5)


def test_cleanup_kills_child_when_parent_exits(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
        from openollama_app.server import ServerProcess, install_cleanup
        sp = ServerProcess([sys.executable, {FAKE!r}, "--port", sys.argv[1]], "127.0.0.1", int(sys.argv[1]), lambda l: None, lambda s: None)
        install_cleanup(sp)
        sp.start()
        while sp.state != "ready":
            time.sleep(0.05)
        print(sp.pid, flush=True)
        sys.exit(0)
    """))
    out = subprocess.run([sys.executable, str(script), str(free_port())], capture_output=True, text=True, timeout=20)
    pid = int(out.stdout.strip())
    assert wait_for(lambda: not alive(pid), timeout=5)


def test_cleanup_kills_child_on_sigterm(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
        from openollama_app.server import ServerProcess, install_cleanup
        sp = ServerProcess([sys.executable, {FAKE!r}, "--port", sys.argv[1]], "127.0.0.1", int(sys.argv[1]), lambda l: None, lambda s: None)
        install_cleanup(sp)
        sp.start()
        while sp.state != "ready":
            time.sleep(0.05)
        print(sp.pid, flush=True)
        time.sleep(60)
    """))
    p = subprocess.Popen([sys.executable, str(script), str(free_port())], stdout=subprocess.PIPE, text=True)
    pid = int(p.stdout.readline())
    p.terminate()
    p.wait(timeout=10)
    assert wait_for(lambda: not alive(pid), timeout=5)


def test_port_already_in_use_fails_before_launching_anything(make):
    sp, _, states, port = make()
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", port))
        squatter.listen()
        sp.start()
        assert sp.state == "failed" and sp.failure_reason == f"Port {port} is already in use"
        assert sp.pid is None and states[-1] == "failed"


def test_invalid_utf8_output_does_not_kill_the_reader(make):
    sp, lines, *_ = make("badbytes")
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    assert wait_for(lambda: "listening" in lines)


def test_raising_line_callback_does_not_kill_the_reader():
    port = free_port()
    seen = []

    def bad_line(line):
        seen.append(line)
        raise RuntimeError("UI went away")

    sp = ServerProcess([sys.executable, FAKE, "--port", str(port)], "127.0.0.1", port, bad_line, lambda s: None)
    try:
        sp.start()
        assert wait_for(lambda: "listening" in seen)
    finally:
        sp.stop(timeout=2)


def test_exit_after_a_raising_callback_still_reports_failed():
    port = free_port()

    def bad_line(line):
        raise RuntimeError("UI went away")

    sp = ServerProcess([sys.executable, FAKE, "--port", str(port), "--mode", "exit1"], "127.0.0.1", port,
                       bad_line, lambda s: None)
    sp.start()
    assert wait_for(lambda: sp.state == "failed")


def test_stop_does_not_deadlock_when_state_lock_is_held_by_the_same_thread(make):
    # A SIGTERM handler runs stop() on the main thread, possibly while it is inside _set_state.
    sp, *_ = make()
    sp.start()
    assert wait_for(lambda: sp.state == "ready")
    done = threading.Event()

    def run():
        with sp._lock:
            sp.stop(timeout=2)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(5)


def test_unlaunchable_binary_goes_failed_with_a_reason():
    states = []
    sp = ServerProcess(["/nonexistent/llama-server", "--port", "1"], "127.0.0.1", free_port(), lambda l: None, states.append)
    sp.start()
    assert sp.state == "failed" and "could not start" in sp.failure_reason.lower()
