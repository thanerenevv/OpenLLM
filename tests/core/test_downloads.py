import http.server
import io
import tarfile
import threading

import pytest

from openllm_app.core import downloads

PAYLOAD = bytes(range(256)) * 8192  # 2 MiB


class RangeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        if self.path == "/missing":
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD) - start))
        self.end_headers()
        if not head:
            chunk = 64 * 1024
            for i in range(start, len(PAYLOAD), chunk):
                try:
                    self.wfile.write(PAYLOAD[i:i + chunk])
                except BrokenPipeError:
                    return


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def make_tar(path, members):
    """members: list of (name, kind, data_or_linkname)."""
    with tarfile.open(path, "w:gz") as tf:
        for name, kind, extra in members:
            ti = tarfile.TarInfo(name)
            if kind == "dir":
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            elif kind == "file":
                ti.size = len(extra)
                tf.addfile(ti, io.BytesIO(extra))
            elif kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, extra
                tf.addfile(ti)
            elif kind == "hard":
                ti.type, ti.linkname = tarfile.LNKTYPE, extra
                tf.addfile(ti)


@pytest.mark.parametrize("members", [
    [("top/", "dir", None), ("top/../evil", "file", b"x")],
    [("top/", "dir", None), ("/abs/evil", "file", b"x")],
    [("top/", "dir", None), ("top/link", "sym", "/etc/passwd")],
    [("top/", "dir", None), ("top/sub/", "dir", None), ("top/sub/link", "sym", "../../outside")],
    [("top/", "dir", None), ("top/hard", "hard", "top/other")],
], ids=["parent-traversal", "absolute", "symlink-absolute", "symlink-escapes", "hardlink"])
def test_safe_extract_rejects_unsafe_members(tmp_path, members):
    archive, dest = tmp_path / "a.tar.gz", tmp_path / "out"
    make_tar(archive, members)
    with pytest.raises(downloads.UnsafeArchive):
        downloads.safe_extract(archive, dest)
    assert not dest.exists() or not any(dest.iterdir())


def test_safe_extract_strips_first_component(tmp_path):
    archive, dest = tmp_path / "a.tar.gz", tmp_path / "out"
    make_tar(archive, [("top/", "dir", None), ("top/llama-server", "file", b"bin"), ("top/sub/x", "file", b"y")])
    downloads.safe_extract(archive, dest)
    assert (dest / "llama-server").read_bytes() == b"bin"
    assert (dest / "sub" / "x").read_bytes() == b"y"


def test_download_reports_progress_and_finishes(tmp_path, server):
    seen = []
    dest = tmp_path / "f.bin"
    downloads.download(f"{server}/f", dest, lambda done, total: seen.append((done, total)))
    assert dest.read_bytes() == PAYLOAD
    assert seen and seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_download_does_not_wait_for_the_size_lookup(tmp_path, server, monkeypatch):
    import time

    def slow_size(url):
        time.sleep(3)  # a slow HEAD through redirects
        return len(PAYLOAD)

    monkeypatch.setattr(downloads, "remote_size", slow_size)
    dest, seen = tmp_path / "f.bin", []
    t = time.perf_counter()
    downloads.download(f"{server}/f", dest, lambda done, total: seen.append((done, total)))
    assert time.perf_counter() - t < 2.5
    assert dest.read_bytes() == PAYLOAD and seen[-1] == (len(PAYLOAD), len(PAYLOAD))


def test_download_resumes_partial_file(tmp_path, server):
    dest = tmp_path / "f.bin"
    dest.write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
    downloads.download(f"{server}/f", dest, lambda d, t: None)
    assert dest.read_bytes() == PAYLOAD


def test_download_failure_raises(tmp_path, server):
    with pytest.raises(downloads.DownloadError):
        downloads.download(f"{server}/missing", tmp_path / "f.bin", lambda d, t: None)


def test_remote_size_reads_content_length(server):
    assert downloads.remote_size(f"{server}/f") == len(PAYLOAD)


def test_remote_size_unreachable_is_none():
    assert downloads.remote_size("http://127.0.0.1:1/nope") is None


def test_safe_extract_allows_relative_symlinks_inside_the_archive(tmp_path):
    # The real llama.cpp runtime ships libggml.dylib -> libggml.0.dylib style links.
    archive, dest = tmp_path / "a.tar.gz", tmp_path / "out"
    make_tar(archive, [("top/", "dir", None), ("top/libx.0.21.0.dylib", "file", b"lib"),
                       ("top/libx.0.dylib", "sym", "libx.0.21.0.dylib"), ("top/libx.dylib", "sym", "libx.0.dylib")])
    downloads.safe_extract(archive, dest)
    assert (dest / "libx.dylib").is_symlink() and (dest / "libx.dylib").read_bytes() == b"lib"


def test_install_runtime_with_corrupt_archive_raises_download_error_and_removes_it(tmp_path, server):
    rt = {"tag": "t", "url": f"{server}/f", "dir": "runtimes/x/{tag}"}
    with pytest.raises(downloads.DownloadError):
        downloads.install_runtime(tmp_path, "x", rt, lambda d, t: None)
    assert not (tmp_path / ".dl" / "x-t.tar.gz").exists()
