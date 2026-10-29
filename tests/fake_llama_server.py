"""Stand-in for llama-server: --port N --mode ok|exit1|bindfail|slowstart."""
import argparse
import http.server
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", default="ok")
args = ap.parse_args()

print("fake server booting", flush=True)
if args.mode == "exit1":
    print("something went wrong", flush=True)
    sys.exit(1)
if args.mode == "bindfail":
    print(f"couldn't bind HTTP server socket, hostname: 127.0.0.1, port: {args.port}: address already in use", flush=True)
    sys.exit(1)
if args.mode == "oom":
    print("ggml_metal: failed to allocate buffer", flush=True)
    sys.exit(1)

started = time.time()
delay = 1.0 if args.mode == "slowstart" else 0.0


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        ready = time.time() - started >= delay
        self.send_response(200 if ready else 503)
        self.send_header("Content-Length", "15")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}' if ready else b'{"status":"no"}'.ljust(15))


print("listening", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
