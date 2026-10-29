"""Resumable downloads (curl) with progress callbacks, and safe runtime-archive extraction."""
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Callable

from .registry import model_path, runtime_dir

Progress = Callable[[int, "int | None"], None]


class DownloadError(Exception):
    pass


class UnsafeArchive(Exception):
    pass


def remote_size(url: str) -> int | None:
    """Content-Length after redirects, or None if it can't be determined."""
    try:
        out = subprocess.run(["curl", "-sIL", "--max-time", "15", url], capture_output=True, text=True).stdout
    except OSError:
        return None
    size = None
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1])
            except ValueError:
                pass
    return size


def download(url: str, dest: Path, on_progress: Progress) -> None:
    """Download url to dest, resuming a partial file. Calls on_progress(done_bytes, total_bytes_or_None)."""
    total = remote_size(url)
    proc = subprocess.Popen(
        ["curl", "-fL", "-C", "-", "--retry", "5", "-sS", "-o", str(dest), url],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    stop = threading.Event()

    def size() -> int:
        try:
            return dest.stat().st_size
        except OSError:
            return 0

    def poll():
        while not stop.wait(0.25):
            on_progress(size(), total)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    err = proc.communicate()[1]
    stop.set()
    t.join()
    if proc.returncode != 0:
        raise DownloadError(f"download failed: {url}: {err.strip() or 'curl exit ' + str(proc.returncode)} (re-run to resume)")
    done = size()
    on_progress(done, total if total is not None else done)


def safe_extract(archive: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz, dropping the first path component. Rejects absolute paths, '..', and links."""
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts:
                raise UnsafeArchive(f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk():
                raise UnsafeArchive(f"link in archive not allowed: {m.name}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for m in members:
            parts = Path(m.name).parts[1:]
            if not parts:
                continue
            tf.extract(m.replace(name=str(Path(*parts))), dest_dir, filter="data")


def install_runtime(root: Path, rt_id: str, rt: dict, on_progress: Progress) -> None:
    tmp = root / ".dl"
    tmp.mkdir(exist_ok=True)
    archive = tmp / f"{rt_id}-{rt['tag']}.tar.gz"
    download(rt["url"].format(tag=rt["tag"]), archive, on_progress)
    safe_extract(archive, runtime_dir(root, rt))
    archive.unlink()


def fetch_model(root: Path, m: dict, on_progress: Progress) -> None:
    path = model_path(root, m)
    path.parent.mkdir(exist_ok=True)
    part = path.with_name(path.name + ".part")
    download(f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}", part, on_progress)
    part.rename(path)
