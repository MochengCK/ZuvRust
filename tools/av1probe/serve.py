#!/usr/bin/env python3
"""收 Chrome 页面 POST 回来的编码结果。

用法：python3 serve.py <输出目录> [端口]

页面本体也从这里发（同源，省掉 CORS）。
"""
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/av1probe"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8799
os.makedirs(OUT, exist_ok=True)


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        name = self.path.strip("/").replace("/", "_") or "out"
        with open(os.path.join(OUT, name + ".bin"), "wb") as f:
            f.write(body)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {ROOT} -> POST 写到 {OUT}，端口 {PORT}", flush=True)
    http.server.HTTPServer(("127.0.0.1", PORT), H).serve_forever()
