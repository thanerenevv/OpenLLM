"""Owns the llama-server child process: output stream, readiness, failure reason, guaranteed cleanup."""
import atexit
import errno
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Callable

STATES = ("idle", "starting", "ready", "stopping", "stopped", "failed")


class ServerProcess:
    def __init__(self, argv: list[str], host: str, port: int,
                 on_line: Callable[[str], None], on_state: Callable[[str], None]):
        self.argv, self.host, self.port = argv, host, port
        self._on_line, self._on_state = on_line, on_state
        self._state = "idle"
        self._lock = threading.RLock()  # stop() may run inside a signal handler that interrupted _set_state
        self._tail: deque[str] = deque(maxlen=200)
        self._proc: subprocess.Popen | None = None
        self.failure_reason: str | None = None

    # ---- observable state
    @property
    def state(self) -> str:
        return self._state

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def last_lines(self, n: int = 30) -> list[str]:
        return list(self._tail)[-n:]

    def _set_state(self, new: str, only_from: tuple[str, ...] | None = None) -> bool:
        with self._lock:
            if only_from is not None and self._state not in only_from:
                return False
            if self._state == new:
                return False
            self._state = new
        self._notify_state(new)
        return True

    def _notify_state(self, state: str) -> None:
        try:
            self._on_state(state)
        except Exception:
            pass  # a broken listener must not take the server thread down with it

    # ---- lifecycle
    def _port_in_use(self) -> bool:
        host = self.argv[self.argv.index("--host") + 1] if "--host" in self.argv else "127.0.0.1"
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ignore TIME_WAIT leftovers
            try:
                probe.bind((host, self.port))
            except OSError as e:
                return e.errno == errno.EADDRINUSE
        return False

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self._set_state("failed")

    def start(self) -> None:
        self._set_state("starting")
        if self._port_in_use():  # otherwise a foreign server's /health would look like ours
            self._fail(f"Port {self.port} is already in use")
            return
        try:
            self._proc = subprocess.Popen(
                self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                bufsize=1, start_new_session=True)  # own process group, so stop() reaches any helpers
        except OSError as e:
            self._fail(f"Could not start the server: {e}")
            return
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        threading.Thread(target=self._poll_health, daemon=True).start()

    def _read_output(self) -> None:
        proc = self._proc
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                self._tail.append(line)
                try:
                    self._on_line(line)
                except Exception:
                    pass  # a broken listener must not stop us draining the pipe (the child would block on write)
        except (OSError, ValueError):
            pass
        code = proc.wait()
        if self._state in ("stopping", "stopped"):
            return
        self.failure_reason = self._explain(code)
        self._set_state("failed")

    def _explain(self, code: int) -> str:
        text = "\n".join(self.last_lines(30)).lower()
        if "address already in use" in text or "couldn't bind" in text:
            return f"Port {self.port} is already in use"
        if "out of memory" in text or "failed to allocate" in text:
            return "Not enough memory for this context size"
        return f"Server exited with code {code}"

    def _poll_health(self) -> None:
        url = f"http://{self.host}:{self.port}/health"
        while self._state == "starting" and self._proc.poll() is None:
            try:
                with urllib.request.urlopen(url, timeout=1) as r:
                    if r.status == 200:
                        self._set_state("ready", only_from=("starting",))
                        return
            except (urllib.error.URLError, OSError):
                pass  # not listening yet, or 503 while the model loads
            time.sleep(0.2)

    def stop(self, timeout: float = 10.0) -> None:
        proc = self._proc
        if proc is None or self._state in ("idle", "stopped"):
            return
        if proc.poll() is None:
            self._set_state("stopping")
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._signal(proc, signal.SIGKILL)
                proc.wait()
        self._reader.join(timeout=2)
        self._state = "stopped" if self._state != "failed" else "failed"
        if self._state == "stopped":
            self._on_state("stopped")

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass


def install_cleanup(server: ServerProcess) -> None:
    """Make sure the child dies with us: on normal exit, SIGTERM and SIGHUP."""
    atexit.register(server.stop, 3.0)

    def handler(signum, _frame):
        server.stop(3.0)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, handler)
