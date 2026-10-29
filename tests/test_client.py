import http.server
import json
import threading

import pytest

from openollama_app.client import stream_chat

MSGS = [{"role": "user", "content": "hi"}]


def chunk(delta=None, finish=None, **extra):
    c = {"choices": [{"delta": delta or {}, "finish_reason": finish}], **extra}
    return f"data: {json.dumps(c)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class Fake:
    """Local HTTP server replaying a canned response; records the last request."""

    def __init__(self, body=b"", status=200):
        self.body, self.status = body, status
        self.headers, self.payload = None, None
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # close-delimited body, so cutting it short is a plain EOF

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.payload = json.loads(self.rfile.read(n))
                outer.headers = dict(self.headers)
                self.send_response(outer.status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(outer.body)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    made = []

    def _fake(body=b"", status=200):
        f = Fake(body, status)
        made.append(f)
        return f

    yield _fake
    for f in made:
        f.close()


async def collect(url, key=None, max_tokens=512):
    return [e async for e in stream_chat(url, key, "m", MSGS, max_tokens)]


async def test_reasoning_then_content_order(fake):
    f = fake(chunk({"reasoning_content": "think "}) + chunk({"reasoning_content": "more"})
             + chunk({"content": "Hel"}) + chunk({"content": "lo"}) + chunk(finish="stop") + DONE)
    ev = await collect(f.url)
    assert [(e.kind, e.text) for e in ev if e.kind in ("reasoning", "content")] == [
        ("reasoning", "think "), ("reasoning", "more"), ("content", "Hel"), ("content", "lo")]
    assert ev[-1].kind == "done" and ev[-1].data == {"finish_reason": "stop", "empty_answer": False}


async def test_request_body(fake):
    f = fake(chunk({"content": "x"}) + chunk(finish="stop") + DONE)
    await collect(f.url, max_tokens=777)
    assert f.payload == {"model": "m", "messages": MSGS, "max_tokens": 777, "stream": True}


async def test_usage_and_timings_events(fake):
    last = chunk(finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 7},
                 timings={"predicted_per_second": 18.4})
    f = fake(chunk({"content": "x"}) + last + DONE)
    ev = await collect(f.url)
    usage = next(e for e in ev if e.kind == "usage")
    timings = next(e for e in ev if e.kind == "timings")
    assert usage.data == {"prompt_tokens": 5, "completion_tokens": 7}
    assert timings.data["predicted_per_second"] == 18.4


async def test_auth_header_sent_only_with_key(fake):
    f = fake(chunk(finish="stop") + DONE)
    await collect(f.url, key="sk-abc")
    assert f.headers["Authorization"] == "Bearer sk-abc"
    g = fake(chunk(finish="stop") + DONE)
    await collect(g.url)
    assert "Authorization" not in g.headers


async def test_http_401_yields_error(fake):
    f = fake(b'{"error":"bad key"}', status=401)
    ev = await collect(f.url)
    assert [e.kind for e in ev] == ["error"] and "401" in ev[0].text


async def test_connection_refused_yields_error():
    ev = await collect("http://127.0.0.1:1/v1")
    assert [e.kind for e in ev] == ["error"]


async def test_mid_stream_disconnect_yields_error_not_exception(fake):
    f = fake(chunk({"content": "partial"}))  # then EOF: no finish_reason, no [DONE]
    ev = await collect(f.url)
    assert ev[0].kind == "content" and ev[-1].kind == "error"


async def test_empty_content_with_length_finish_reports_notice(fake):
    f = fake(chunk({"reasoning_content": "thinking forever"}) + chunk(finish="length") + DONE)
    ev = await collect(f.url)
    assert ev[-1].kind == "done" and ev[-1].data == {"finish_reason": "length", "empty_answer": True}


async def test_usage_derived_from_timings_when_server_sends_no_usage(fake):
    # Real PrismML llama-server (b10809) streams timings but no usage object.
    last = chunk(finish="stop", timings={"cache_n": 3, "prompt_n": 11, "predicted_n": 55,
                                         "predicted_per_second": 10.9})
    f = fake(chunk({"content": "x"}) + last + DONE)
    ev = await collect(f.url)
    usage = [e for e in ev if e.kind == "usage"]
    assert len(usage) == 1 and usage[0].data == {"prompt_tokens": 14, "completion_tokens": 55}
