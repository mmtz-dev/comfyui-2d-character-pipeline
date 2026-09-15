"""Stage: `anim review` - the click-to-pick page, on the LAN

A contact sheet of every extracted frame with the current picks highlighted.
Click a frame to add or remove it; picks.json is rewritten on the spot.

    anim review knight_strike
    # then, from the Mac:  http://192.168.1.158:8710/

Bound to 0.0.0.0 because the owner opens it from the Mac, not from the Fedora
host. (A curl from the host to its own LAN IP routes via lo and proves nothing
about LAN reach.)

THE REGENERATE BUTTON DOES NOT TOUCH THE GPU. It sets `rerender_requested` in
job.json and stops. A web page that could start a 24 GB model load is a web page
that can be triggered by a stray browser prefetch, and the GPU service is on the
other Docker engine and usually not even running. The owner re-runs `anim strip`
and `anim render` when they are ready; the flag is just a note to self that
survives closing the tab.

Everything the page does lives in `ReviewApp`, which knows nothing about HTTP.
The handler below is a thin adapter, so the behaviour is unit-testable without a
browser and without a socket.
"""

from __future__ import annotations

import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from anim import config
from anim.job import Job
from anim.stages.pick import PICKS_SCHEMA
from anim.util import read_json, write_json_in_place

NAME = "review"
HELP = "click-to-pick page on the LAN"


class ReviewApp:
    """The review page's behaviour. No HTTP, no sockets, no globals."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.manifest = job.read_manifest()
        self.frames = job.frames()
        if not self.frames:
            raise FileNotFoundError(
                f"no frames in {job.frames_dir} - run `anim extract` first")

    # --- picks.json --------------------------------------------------------
    def _picks_record(self) -> dict[str, Any]:
        if self.job.picks_path.exists():
            return read_json(self.job.picks_path)
        # Reviewing before picking is legitimate: the owner may want to choose
        # every pose by hand. Start from nothing rather than refusing.
        return {"schema": PICKS_SCHEMA, "n": 0,
                "action": self.manifest.get("action", ""),
                "dedup": config.PICK_DEDUP, "picks": [], "auto": [],
                # No shape until `pick` or a hand edit gives one; `toggle` and
                # `set_picks` rewrite the record they read, so a hand `timing`
                # survives every click.
                "timing": None, "auto_timing": None,
                "signals": {}}

    def state(self) -> dict[str, Any]:
        record = self._picks_record()
        picks = list(record.get("picks") or [])
        return {
            "job": self.job.name,
            "action": self.manifest.get("action", ""),
            "slug": self.manifest.get("slug", ""),
            "fps": self.manifest.get("fps"),
            "n": record.get("n", len(picks)),
            "picks": picks,
            "auto": list(record.get("auto") or []),
            "edited": picks != list(record.get("auto") or []),
            "rerender_requested": bool(self.manifest.get("rerender_requested")),
            "frames": [
                {"index": i, "file": path.name, "picked": i in picks,
                 "order": picks.index(i) if i in picks else None,
                 "skeleton": (self.job.pose_dir / f"skel_{i + 1:04d}.png").exists()}
                for i, path in enumerate(self.frames)
            ],
        }

    def toggle(self, index: int) -> dict[str, Any]:
        """Add or remove one frame from the picks. Idempotent per click.

        The other picks keep their frame numbers - they are frame indices, not
        slots - so removing the second of four does not renumber the rest, and
        `strip` reads them back in ascending frame order either way.
        """
        if not 0 <= index < len(self.frames):
            raise IndexError(f"frame {index} is outside 0..{len(self.frames) - 1}")
        record = self._picks_record()
        picks = list(record.get("picks") or [])
        if index in picks:
            picks.remove(index)
        else:
            picks.append(index)
        record["picks"] = sorted(set(picks))
        record["n"] = len(record["picks"])
        write_json_in_place(self.job.picks_path, record)
        return self.state()

    def set_picks(self, indices: list[int]) -> dict[str, Any]:
        record = self._picks_record()
        clean = sorted({int(i) for i in indices if 0 <= int(i) < len(self.frames)})
        record["picks"] = clean
        record["n"] = len(clean)
        write_json_in_place(self.job.picks_path, record)
        return self.state()

    def regenerate(self) -> dict[str, Any]:
        """Flag job.json and nothing else. See this module's docstring."""
        self.manifest = self.job.update_manifest(rerender_requested=True)
        return self.state()

    # --- static-ish content ------------------------------------------------
    def image(self, kind: str, name: str) -> bytes:
        """Serve one frame or skeleton PNG, by exact name, from inside the job.

        Names are matched against the job's own listing rather than joined onto
        a directory, so no amount of ../ in a URL reaches outside work/<job>/.
        """
        if kind == "frames":
            allowed = {p.name: p for p in self.frames}
        elif kind == "skel":
            allowed = {p.name: p for p in self.job.skeletons()}
        else:
            raise KeyError(kind)
        path = allowed.get(name)
        if path is None:
            raise KeyError(name)
        return path.read_bytes()

    def page(self) -> bytes:
        return _render_page(self.state()).encode("utf-8")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def _render_page(state: dict[str, Any]) -> str:
    """One self-contained HTML page. No CDN, no build step, no framework.

    Dark ground because the sprites are judged against a near-black board, and
    `image-rendering: pixelated` because a browser's default smoothing makes a
    64px sprite look like a decision it is not.
    """
    cards = []
    for frame in state["frames"]:
        classes = "card picked" if frame["picked"] else "card"
        badge = (f'<span class="badge">{frame["order"] + 1}</span>'
                 if frame["order"] is not None else "")
        cards.append(
            f'<button class="{classes}" data-index="{frame["index"]}" '
            f'onclick="toggle({frame["index"]})">'
            f'{badge}'
            f'<img src="/frames/{html.escape(frame["file"])}" alt="frame '
            f'{frame["index"]}" loading="lazy">'
            f'<span class="n">{frame["index"]}</span></button>')
    edited = ('<span class="tag warn">hand-edited</span>'
              if state["edited"] else '<span class="tag">auto</span>')
    flagged = ('<span class="tag warn">regenerate requested</span>'
               if state["rerender_requested"] else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>anim review - {html.escape(state['job'])}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#15171c; color:#d7dae0;
         font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  header {{ position:sticky; top:0; background:#1d2027; border-bottom:1px solid #2c3038;
            padding:12px 16px; display:flex; gap:12px; align-items:center;
            flex-wrap:wrap; z-index:2; }}
  h1 {{ font-size:15px; margin:0; font-weight:600; }}
  .muted {{ color:#8b929e; }}
  .tag {{ font-size:12px; padding:2px 8px; border-radius:999px;
          background:#2c3038; color:#a9b0bc; }}
  .tag.warn {{ background:#4a3312; color:#f0c070; }}
  button.act {{ font:inherit; background:#2c3038; color:#d7dae0; border:1px solid #3a3f49;
                border-radius:6px; padding:6px 12px; cursor:pointer; }}
  button.act:hover {{ background:#363b45; }}
  main {{ display:grid; gap:10px; padding:16px;
          grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); }}
  .card {{ position:relative; padding:0; border:2px solid #2c3038; border-radius:6px;
           background:#0e1013; cursor:pointer; overflow:hidden; }}
  .card img {{ display:block; width:100%; height:auto;
               image-rendering:pixelated; image-rendering:crisp-edges; }}
  .card.picked {{ border-color:#f0c070; box-shadow:0 0 0 2px #f0c07044; }}
  .card .n {{ position:absolute; left:4px; bottom:2px; font-size:11px;
              color:#8b929e; background:#000a; padding:0 4px; border-radius:3px; }}
  .badge {{ position:absolute; right:4px; top:4px; background:#f0c070; color:#15171c;
            font-size:11px; font-weight:700; border-radius:999px;
            width:18px; height:18px; line-height:18px; text-align:center; }}
  footer {{ padding:0 16px 24px; }}
  @media (max-width:420px) {{ main {{ grid-template-columns:repeat(auto-fill,minmax(90px,1fr)); }} }}
</style></head><body>
<header>
  <h1>{html.escape(state['job'])}</h1>
  <span class="muted">{html.escape(state['slug'])} / {html.escape(state['action'])}
    &middot; {len(state['frames'])} frames @ {state['fps']} fps</span>
  {edited}{flagged}
  <span class="muted" id="picks">picks: {', '.join(str(p) for p in state['picks']) or '(none)'}</span>
  <button class="act" onclick="regen()">Request regenerate</button>
</header>
<main id="sheet">{''.join(cards)}</main>
<footer class="muted">Click a frame to keep or drop it. Saved straight to
picks.json. "Request regenerate" only flags job.json - it never starts a render.
</footer>
<script>
async function post(path, body) {{
  const r = await fetch(path, {{method:'POST', headers:{{'Content-Type':'application/json'}},
                               body: JSON.stringify(body || {{}})}});
  if (!r.ok) {{ alert(await r.text()); return null; }}
  return r.json();
}}
async function toggle(i) {{ const s = await post('/api/toggle', {{frame:i}}); if (s) apply(s); }}
async function regen() {{ const s = await post('/api/regenerate'); if (s) location.reload(); }}
function apply(s) {{
  document.getElementById('picks').textContent =
    'picks: ' + (s.picks.join(', ') || '(none)');
  for (const f of s.frames) {{
    const el = document.querySelector('.card[data-index="' + f.index + '"]');
    if (!el) continue;
    el.classList.toggle('picked', f.picked);
    const old = el.querySelector('.badge');
    if (old) old.remove();
    if (f.order !== null) {{
      const b = document.createElement('span');
      b.className = 'badge'; b.textContent = f.order + 1;
      el.prepend(b);
    }}
  }}
}}
</script></body></html>
"""


# ---------------------------------------------------------------------------
# The HTTP adapter
# ---------------------------------------------------------------------------
def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "anim-review/1"
        protocol_version = "HTTP/1.1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, app.page(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._json(HTTPStatus.OK, app.state())
                return
            parts = path.strip("/").split("/")
            if len(parts) == 2 and parts[0] in ("frames", "skel"):
                try:
                    self._send(HTTPStatus.OK, app.image(parts[0], parts[1]),
                               "image/png")
                except KeyError:
                    self._send(HTTPStatus.NOT_FOUND, b"no such image",
                               "text/plain; charset=utf-8")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found",
                       "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "body is not JSON"})
                return
            try:
                if path == "/api/toggle":
                    self._json(HTTPStatus.OK, app.toggle(int(body["frame"])))
                elif path == "/api/picks":
                    self._json(HTTPStatus.OK, app.set_picks(body.get("picks") or []))
                elif path == "/api/regenerate":
                    self._json(HTTPStatus.OK, app.regenerate())
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            # One tidy line per request instead of stderr noise from favicon.ico.
            print(f"  review {self.address_string()} {fmt % args}")

    return Handler


def serve(app: ReviewApp, host: str, port: int) -> ThreadingHTTPServer:
    """Build (but do not start) the server. Port 0 asks the OS for a free one."""
    return ThreadingHTTPServer((host, port), make_handler(app))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--port", type=int, default=config.REVIEW_PORT,
                        help=f"LAN port (default {config.REVIEW_PORT}; "
                             f"compose.yml publishes the same number)")
    parser.add_argument("--bind", default=config.REVIEW_BIND,
                        help="interface to bind (default 0.0.0.0 - the Mac has "
                             "to reach it)")


def run_stage(args: argparse.Namespace) -> int:
    app = ReviewApp(Job.open(args.job))
    server = serve(app, args.bind, args.port)
    port = server.server_address[1]
    print(f"review: {len(app.frames)} frames of {app.job.name}")
    print(f"  open  http://{config.LAN_HOST}:{port}/   (from the Mac)")
    print("  Ctrl-C to stop; picks.json is written on every click.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nreview: stopped")
    finally:
        server.server_close()
    return 0


run = run_stage
