"""Stage: `anim publish` - frames -> game-02 assets + one sidecar, then the audit

    anim publish knight_strike

    <assets>/2d/units/knight_strike_0.png .. _3.png   the frames
    <assets>/2d/units/knight_strike.json              ONE sidecar for all four

ONE SIDECAR PER ACTION, not per frame. `contract.frames: [n, 1]` with n > 1 is
the audit's frame-group spelling: the sidecar covers `<stem>_0.png` through
`<stem>_{n-1}.png` and there is no bare `<stem>.png`. Godot finds the same files
by counting up from `_0` (render/unit_sprites.gd `frames_for`), so publishing
numbered PNGs plus this one file IS the registration step - no .tres, no atlas.

FRAMES ARE COPIED IN PLACE. Never written to a temp name and renamed. Docker
Desktop's file share caches by inode, so a rename leaves a new inode behind the
same path and a container holding the old one keeps serving the old bytes -
the "the Mac saved it but the game still shows the old sprite" quirk. See
`anim.util.write_bytes_in_place`.

STALE FRAMES ARE DELETED. Re-publishing a 4-frame action over a 6-frame one
would otherwise leave `_4` and `_5` behind, and the audit correctly calls an
unclaimed numbered PNG an error - it is silently unused art.

THE AUDIT IS NOT OPTIONAL. `tools/asset_audit.py` runs afterwards and a non-zero
exit fails the stage loudly. It is stdlib-only, so it runs in this container
against the read-only /game02 mount; if that mount is absent the stage says so
and tells the owner the one command to run on the host.

ON `generation.motion_source`
-----------------------------
When the job's motion came from a rig (`anim rig`, or `anim base --rig`), the
manifest's `motion_source` block is copied into the sidecar's `generation` so
the audit can see two provenances side by side: the MOTION is a CC0 rig clip
(pack, file, sha256, clip name, licence, url) and the PIXELS are a local
generation (seed, workflow, model hashes). A filmed job has no such block and
`generation.motion_source` is null; `generation.clip`/`clip_sha256` then name
the footage as before.

ON `source.license`
-------------------
It says **CC0-1.0**, the licence of the OUTPUT - and it has to: the audit
rejects any other value (`ALLOWED_LICENCES` in asset_audit.py), and its own
tests write CC0-1.0 for an ai-generated sidecar. The MODEL weights are
Apache-2.0 and that is recorded per model in `source.models[].license` and
`generation.models[].license`, plus `source.model_license` for the human
reading the file. Apache-2.0 weights place no restriction on generated output,
so the frames are commercially usable - which is the thing the repo's CC0-only
rule is actually protecting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

from anim import __version__, config
from anim.job import Job
from anim.util import read_json, write_bytes_in_place, write_json_in_place
from anim.util import run as run_tool

NAME = "publish"
HELP = "frames -> game-02 assets + sidecar"

UNITS_SUBDIR = ("2d", "units")
TOOL = f"tools/anim-pipeline (anim {__version__}) - the pose-to-pixel animation pipeline feature"
FORK_URL = "https://github.com/mmtz-dev/comfyui-2d-character-pipeline"
MODEL_LICENSE = "Apache-2.0"
FRAME_RE = re.compile(r"^_\d+$")


def sidecar_for(*, slug: str, action: str, frames: int, manifest: dict[str, Any],
                provenance: dict[str, Any] | None, picks: list[int],
                report: dict[str, Any] | None,
                retrieved: str | None = None) -> dict[str, Any]:
    """Build the sidecar. Pure - takes what the stages wrote, returns the dict."""
    stem = f"{slug}_{action}" if action else slug
    passes = (provenance or {}).get("passes") or []
    final = passes[-1] if passes else {}
    models = (provenance or {}).get("models") or []
    steps: list[str] = []
    for record in (report or {}).get("frames") or []:
        if record.get("steps"):
            steps = list(record["steps"])
            break
    motion = manifest.get("motion_source")
    if motion and motion.get("kind") == "rig":
        author = (f"generated locally from a {motion.get('license', '?')} rig "
                  f"clip ({motion.get('pack', '?')} / {motion.get('clip', '?')})")
    else:
        author = "generated locally from a filmed reference clip"
    return {
        "id": f"unit/{stem}",
        "provenance": "ai-generated",
        "status": "placeholder",
        "source": {
            "pack": "the pose-to-pixel animation pipeline",
            "tool": TOOL,
            "url": FORK_URL,
            "author": author,
            # The OUTPUT's licence. The audit allows CC0 only; see the module
            # docstring for why that is the right value and where the weights'
            # own licence is recorded.
            "license": "CC0-1.0",
            "model_license": MODEL_LICENSE,
            "license_note": (
                "Frames are generated locally and released CC0 with the rest of "
                "the assets. The model weights are Apache-2.0, which places no "
                "restriction on generated output, so these are commercially "
                "usable."),
            "models": models,
            "retrieved": retrieved or dt.date.today().isoformat(),
        },
        "contract": {
            "canvas": list(config.CANVAS),
            "frames": [frames, 1],
            "pivot": "bottom-center",
            # True 64px detail authored at 64 (art bible section 1), not a 32px
            # tile doubled - which is what every placeholder on disk is.
            "scale_from_source": 1,
        },
        "generation": {
            "seed": (provenance or {}).get("seed"),
            "workflow": final.get("workflow"),
            "workflow_sha256": final.get("workflow_sha256"),
            "models": models,
            "clip_sha256": manifest.get("clip_sha256"),
            "clip": Path(manifest.get("clip", "")).name or None,
            "picks": list(picks),
            "fps": manifest.get("fps"),
            "comfyui": (provenance or {}).get("comfyui", {}).get("version"),
            # Where the MOTION came from, distinct from where the pixels came
            # from. A rig clip's pack/file/sha256/licence when `anim rig` made
            # the job; null for filmed footage.
            "motion_source": dict(motion) if motion else None,
        },
        "postprocess": steps,
    }


def frame_names(stem: str, count: int) -> list[str]:
    """`<stem>_0.png ... _{n-1}.png`, or `<stem>.png` when there is only one."""
    if count == 1:
        return [f"{stem}.png"]
    return [f"{stem}_{i}.png" for i in range(count)]


def clear_stale(units_dir: Path, stem: str, keep: set[str]) -> list[Path]:
    """Remove numbered PNGs of this stem that this publish does not claim."""
    removed: list[Path] = []
    for path in sorted(units_dir.glob(f"{stem}_*.png")):
        if path.name in keep:
            continue
        # Only ever the reserved `_<number>` spelling: `knight_strike_sneak.png`
        # is a different asset, not a stale frame of this one.
        if FRAME_RE.fullmatch(path.stem[len(stem):]):
            path.unlink()
            removed.append(path)
    return removed


def audit(script: Path | None = None, *, python: str | None = None) -> tuple[int, str]:
    """Run game-02's asset audit. Returns (exit code, combined output)."""
    script = Path(script) if script else config.AUDIT_SCRIPT
    if not script.exists():
        raise FileNotFoundError(
            f"no asset audit at {script}. Either mount the game-02 repo into "
            f"this container (compose.yml maps it read-only at /game02) or run "
            f"it yourself on the host - it is stdlib-only:\n"
            f"    python3 game-02-tactical-rpg/tools/asset_audit.py\n"
            f"Pass --no-audit to publish without it (and then run it).")
    proc = run_tool([python or sys.executable, str(script)], check=False, timeout=600)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def publish(job_name: str, *, assets_dir: Path | None = None,
            slug: str | None = None, action: str | None = None,
            run_audit: bool = True, audit_script: Path | None = None,
            dry_run: bool = False, work_dir: Path | None = None) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    slug = slug or manifest.get("slug") or "unit"
    action = action if action is not None else (manifest.get("action") or "")
    stem = f"{slug}_{action}" if action else slug

    frames = sorted(job.post_dir.glob(f"{stem}_*.png"))
    if not frames:
        frames = sorted(p for p in job.post_dir.glob("*.png")
                        if p.name != "palette.png")
    if not frames:
        raise FileNotFoundError(
            f"no finished frames in {job.post_dir} - run `anim post` first")

    report = read_json(job.report_path) if job.report_path.exists() else None
    provenance = (read_json(job.provenance_path)
                  if job.provenance_path.exists() else None)
    picks = list(read_json(job.picks_path).get("picks") or []) \
        if job.picks_path.exists() else []

    base = Path(assets_dir) if assets_dir is not None else config.ASSETS_DIR
    units_dir = base.joinpath(*UNITS_SUBDIR)
    names = frame_names(stem, len(frames))
    meta = sidecar_for(slug=slug, action=action, frames=len(frames),
                       manifest=manifest, provenance=provenance, picks=picks,
                       report=report)

    result: dict[str, Any] = {"stem": stem, "dir": str(units_dir),
                              "frames": names, "sidecar": f"{stem}.json",
                              "removed": [], "audit": None, "dry_run": dry_run}
    if dry_run:
        return result

    units_dir.mkdir(parents=True, exist_ok=True)
    for source, name in zip(frames, names):
        write_bytes_in_place(units_dir / name, source.read_bytes())
    result["removed"] = [p.name for p in clear_stale(units_dir, stem, set(names))]
    write_json_in_place(units_dir / f"{stem}.json", meta)

    if run_audit:
        code, output = audit(audit_script)
        result["audit"] = {"exit": code, "output": output.strip()}
        if code != 0:
            raise RuntimeError(
                f"asset_audit.py exited {code} after publishing {stem}:\n"
                f"{output.strip()}")
    job.update_manifest(publish={"dir": str(units_dir), "frames": names,
                                 "sidecar": f"{stem}.json"})
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--slug", default=None, help="override the unit slug")
    parser.add_argument("--action", default=None, help="override the action")
    parser.add_argument("--assets", default=None,
                        help=f"assets root (default {config.ASSETS_DIR})")
    parser.add_argument("--no-audit", dest="audit", action="store_false",
                        help="skip the asset audit (then run it yourself)")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be written and stop")
    parser.set_defaults(audit=True)


def run_stage(args: argparse.Namespace) -> int:
    result = publish(args.job, assets_dir=Path(args.assets) if args.assets else None,
                     slug=args.slug, action=args.action, run_audit=args.audit,
                     dry_run=args.dry_run)
    verb = "would write" if result["dry_run"] else "wrote"
    print(f"publish: {verb} {len(result['frames'])} frame(s) + "
          f"{result['sidecar']} to {result['dir']}")
    for name in result["frames"]:
        print(f"  {name}")
    for name in result["removed"]:
        print(f"  removed stale {name}")
    if result["audit"]:
        print(f"  asset_audit.py: {result['audit']['output'].splitlines()[-1]}")
    return 0


run = run_stage
