"""Stage: `anim review` - the click-to-pick behaviour, and where its page went

`ReviewApp` is the review page's behaviour: a contact sheet of every extracted
frame with the current picks highlighted; toggle a frame and picks.json is
rewritten on the spot. It knows nothing about HTTP.

THE PAGE ITSELF MOVED TO THE PIPELINE API (the pipeline API feature, #279).
This stage used to serve it from a stdlib http.server on :8710; that server is
gone, and `anim review <job>` now only prints where the page is:

    anim review knight_strike
    # -> http://192.168.1.158:8720/jobs/knight_strike/review

The API (anim/api/) drives the same `ReviewApp` through GET/PUT
/jobs/<job>/picks, so picks.json is written byte for byte as before. The
module stays because `ReviewApp` is what the API and the rig tests import.

THE REGENERATE FLAG DOES NOT TOUCH THE GPU. `regenerate()` sets
`rerender_requested` in job.json and stops. A web page that could start a 24 GB
model load is a web page that can be triggered by a stray browser prefetch, and
the GPU service is on the other Docker engine and usually not even running. The
owner re-runs `anim strip` and `anim render` when they are ready (or POSTs a
run to the API on purpose); the flag is just a note to self that survives
closing the tab.
"""

from __future__ import annotations

import argparse
from typing import Any

from anim import config
from anim.job import Job
from anim.stages.pick import PICKS_SCHEMA
from anim.util import read_json, write_json_in_place

NAME = "review"
HELP = "where the click-to-pick page is (it lives on the API now)"


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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")


def review_url(job_name: str, *, host: str | None = None,
               port: int | None = None) -> str:
    """The API page for one job's picks, as the owner reaches it from the Mac."""
    host = config.LAN_HOST if host is None else host
    port = config.API_PORT if port is None else port
    return f"http://{host}:{port}/jobs/{job_name}/review"


def run_stage(args: argparse.Namespace) -> int:
    job = Job.open(args.job)
    frames = job.frames()
    print(f"review: the page moved to the pipeline API (port {config.API_PORT})")
    print(f"  open  {review_url(job.name)}   (from the Mac)")
    print("  start it with  docker compose up -d anim-api  if it is not running")
    if not frames:
        print(f"  ({job.name} has no frames yet - run `anim extract` or "
              f"`anim rig` first)")
    else:
        print(f"  {len(frames)} frame(s) of {job.name}; picks.json is written "
              f"on every click, as before")
    return 0


run = run_stage
