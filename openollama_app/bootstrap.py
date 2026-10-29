"""Stdlib-only bootstrap: make sure Textual is importable, installing a local .venv with uv if needed."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Verified: uv's default interpreter (CPython 3.14.0) installs textual 8.2.8 / httpx 0.28.1; the 3.13 retry is a fallback.
RUNTIME_DEPS = ["textual", "httpx"]
DEV_DEPS = ["pytest", "pytest-asyncio"]


def venv_python(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def plan(root: Path, have_textual: bool, venv_exists: bool, have_uv: bool, in_venv: bool = False) -> str:
    """What the entry point should do: run | reexec | install | manual.

    in_venv: we are already running the venv's interpreter and still lack textual, so re-exec'ing
    would loop forever; reinstall instead.
    """
    if have_textual:
        return "run"
    if venv_exists and not in_venv:
        return "reexec"
    return "install" if have_uv else "manual"


def manual_instructions(root: Path) -> str:
    deps = " ".join(RUNTIME_DEPS)
    return (
        "uv is not installed. Set up the UI dependencies by hand:\n"
        f"  python3 -m venv {root / '.venv'}\n"
        f"  {root / '.venv' / 'bin' / 'pip'} install {deps}\n"
    )


def install(root: Path) -> None:
    """Create ./.venv and install the UI dependencies with uv.

    If the default interpreter can't install them, retry with Python 3.13.
    """
    deps = RUNTIME_DEPS + (DEV_DEPS if os.environ.get("OPENOLLAMA_DEV") == "1" else [])
    py = str(venv_python(root))
    for extra in ([], ["--python", "3.13"]):
        try:
            subprocess.check_call(["uv", "venv", str(root / ".venv"), "--clear", *extra])
        except subprocess.CalledProcessError:
            continue
        if subprocess.call(["uv", "pip", "install", "-p", py, *deps]) == 0:
            return
    shutil.rmtree(root / ".venv", ignore_errors=True)  # a half-built venv would be re-exec'd into forever
    raise SystemExit("could not install UI dependencies with uv")


def have_textual() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def shutil_which_uv() -> bool:
    from shutil import which
    return which("uv") is not None


def reexec(root: Path, argv: list[str]) -> None:
    py = str(venv_python(root))
    os.execv(py, [py, *argv])


def ensure(root: Path, argv: list[str]) -> None:
    """Return when Textual is importable in this interpreter; otherwise install/re-exec/exit."""
    in_venv = Path(sys.prefix).resolve() == (root / ".venv").resolve()
    action = plan(root, have_textual(), venv_python(root).exists(), shutil_which_uv(), in_venv)
    if action == "run":
        return
    if action == "manual":
        print(manual_instructions(root), file=sys.stderr)
        raise SystemExit(1)
    if action == "install":
        answer = input("Install the UI dependencies into ./.venv with uv? [Y/n] ").strip().lower()
        if answer == "n":
            raise SystemExit("UI dependencies required")
        install(root)
    reexec(root, argv)
