"""A fake ComfyUI, in-process, for the `render` tests.

The real service lives on the other Docker engine, needs 35 GB of weights and a
GPU, and is off by default. None of that is what the render stage's code is
about: the render stage is about templating a workflow, uploading images, and
driving `/prompt` -> `/history` -> `/view` correctly. So this serves canned
answers on an ephemeral port and records what it was sent, and the tests assert
against the recording.

Endpoints, matching ComfyUI's real shapes:
    GET  /system_stats      version + device list
    POST /upload/image      {"name": ..., "subfolder": "", "type": "input"}
    POST /prompt            {"prompt_id": ...}
    GET  /history/<id>      {"<id>": {"status": ..., "outputs": {...}}}
    GET  /view?filename=... the canned PNG bytes
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image


def canned_png(size: tuple[int, int] = (1216, 320), *,
               prompt_chunk: dict[str, Any] | None = None) -> bytes:
    """A PNG the size of the strip, optionally carrying a `prompt` text chunk.

    The chunk matters: core SaveImage embeds the graph there, and `render`
    records whether it came back, so the fixture has to be able to produce one.
    """
    from PIL.PngImagePlugin import PngInfo

    img = Image.new("RGB", size, (255, 255, 255))
    info = PngInfo()
    if prompt_chunk is not None:
        info.add_text("prompt", json.dumps(prompt_chunk))
    buffer = BytesIO()
    img.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


class FakeComfy:
    """Start with `with FakeComfy() as fake:` and read `fake.url`."""

    def __init__(self, *, png: bytes | None = None, fail_prompt: bool = False,
                 error_job: bool = False, batch: list[bytes] | None = None,
                 version: str = "0.35.0-fake") -> None:
        self.png = png if png is not None else canned_png()
        # The clip render feature's SaveImage returns one file per frame; when
        # `batch` is given, /history lists that many and /view serves them by
        # the frame number in the filename.
        self.batch = batch
        self.fail_prompt = fail_prompt
        self.error_job = error_job
        self.version = version
        self.uploads: list[str] = []
        self.prompts: list[dict[str, Any]] = []
        self.views: list[dict[str, str]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "FakeComfy":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        assert self._server is not None, "FakeComfy is not running"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # --- handler -----------------------------------------------------------
    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: Any) -> None:
                self._send(status, json.dumps(payload).encode(),
                           "application/json")

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/system_stats":
                    self._json(200, {
                        "system": {"comfyui_version": fake.version,
                                   "python_version": "3.12.14 (fake)"},
                        "devices": [{"name": "NVIDIA RTX A5000 (fake)",
                                     "type": "cuda", "vram_total": 25757220864}],
                    })
                    return
                if parsed.path.startswith("/history/"):
                    prompt_id = parsed.path.rsplit("/", 1)[-1]
                    if fake.error_job:
                        self._json(200, {prompt_id: {
                            "status": {"status_str": "error",
                                       "messages": [["execution_error",
                                                     {"exception_message":
                                                      "fake failure"}]]},
                            "outputs": {}}})
                        return
                    count = len(fake.batch) if fake.batch else 1
                    self._json(200, {prompt_id: {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {"13": {"images": [
                            {"filename": f"{prompt_id}_{i + 1:05d}_.png",
                             "subfolder": "anim_pipeline", "type": "output"}
                            for i in range(count)]}},
                    }})
                    return
                if parsed.path == "/view":
                    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                    fake.views.append(query)
                    body = fake.png
                    if fake.batch:
                        number = int(query["filename"].rsplit("_", 2)[-2])
                        body = fake.batch[number - 1]
                    self._send(200, body, "image/png")
                    return
                self._send(404, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if parsed.path == "/upload/image":
                    name = _multipart_filename(raw) or "upload.png"
                    fake.uploads.append(name)
                    self._json(200, {"name": name, "subfolder": "",
                                     "type": "input"})
                    return
                if parsed.path == "/prompt":
                    if fake.fail_prompt:
                        self._json(400, {"error": {"type": "prompt_outputs_failed_validation",
                                                   "message": "fake rejection"}})
                        return
                    body = json.loads(raw or b"{}")
                    fake.prompts.append(body.get("prompt") or {})
                    self._json(200, {"prompt_id": f"fake-{len(fake.prompts)}",
                                     "number": len(fake.prompts)})
                    return
                self._send(404, b"not found", "text/plain")

            def log_message(self, *args: object) -> None:
                pass  # silence; pytest output is the signal here

        return Handler


def _multipart_filename(body: bytes) -> str | None:
    """Pull `filename="..."` out of a multipart body without a parser.

    The upload is one part with one filename; a full multipart parse would be
    more machinery than the assertion is worth.
    """
    marker = b'filename="'
    start = body.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = body.find(b'"', start)
    return body[start:end].decode("utf-8", "replace") if end > start else None
