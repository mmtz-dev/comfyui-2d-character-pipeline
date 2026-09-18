"""Stage: `anim publish` - frames -> game-02 assets + one sidecar, then the audit

    anim publish knight_strike

    <assets>/2d/units/knight_strike_0.png .. _3.png   the frames
    <assets>/2d/units/knight_strike.json              ONE sidecar for all four

ONE SIDECAR PER ACTION, not per frame. `contract.frames: [n, 1]` with n > 1 is
the audit's frame-group spelling: the sidecar covers `<stem>_0.png` through
`<stem>_{n-1}.png` and there is no bare `<stem>.png`. Godot finds the same files
by counting up from `_0` (render/unit_sprites.gd `frames_for`), so publishing
numbered PNGs plus this one file IS the registration step - no .tres, no atlas.

ON `contract.timing` (the strip timing feature)
-----------------------------------------------
Beside `canvas` / `frames` / `pivot` the sidecar may carry

    "timing": {"weights": [2, 3, 4, 3], "anchor": 2}

one dimensionless weight per frame - FE-style ticks, NOT seconds, because the
game's AttackChoreography already scales every duration by the speed setting
and seconds here would fight it - and the index of the frame whose START lands
on the plan's anchor instant: impact for a strike, the launch for a shot or a
cast, frame 0 for a reaction. render/board_fx.gd `schedule` turns the weights
and the plan's windows into per-frame durations and holds the anchor frame
through the hit-stop. game-02's audit checks the block against
`contract.frames` (exactly n positive weights, anchor inside the strip).

Where the shape comes from, in order: a HAND edit of picks.json's `timing`
(it differs from `auto_timing`), else whatever `anim pick` recorded (the
per-verb template in config.ACTION_TIMING, or the clip's frame gaps with
`--timing clip`), else the template row for (action, n) when picks.json
predates the feature, else UNIFORM - in which case no block is written at all
and the game supplies both the even weights and the default anchor, so that
rule has exactly one home. `generation.timing_source` records which:
"hand", "clip", "template:<action>" or "uniform". A hand `timing` whose length
no longer matches the frames (picks toggled after it was written) is dropped
with a warning rather than published broken.

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

ON `generation.parts` (the part palette lock feature)
-----------------------------------------------------
When `post` ran with `--parts`, the sidecar carries what it found: the part
names, the palettes (on a base sprite, the palettes it EXTRACTED - which is
where every action of that unit then reads them from; on an action, the
palettes it LOCKED to and where they came from), the SAM 2.1 weights that
drew the masks (also appended to `models`, since they touched the frame), and
per frame, per part: pixels, colours, whether the part was found, and `drift`
- the pixels that were outside the part's palette before the lock. The audit
ignores keys it does not know, so this is additive.

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
                retrieved: str | None = None,
                timing: dict[str, Any] | None = None,
                timing_source: str = "uniform") -> dict[str, Any]:
    """Build the sidecar. Pure - takes what the stages wrote, returns the dict."""
    stem = f"{slug}_{action}" if action else slug
    contract: dict[str, Any] = {
        "canvas": list(config.CANVAS),
        "frames": [frames, 1],
        "pivot": "bottom-center",
        # True 64px detail authored at 64 (art bible section 1), not a 32px
        # tile doubled - which is what every placeholder on disk is.
        "scale_from_source": 1,
    }
    if timing:
        # Only the two fields the game reads; `source` lives in generation.
        contract["timing"] = {"weights": [w for w in timing["weights"]],
                              "anchor": int(timing["anchor"])}
    passes = (provenance or {}).get("passes") or []
    final = passes[-1] if passes else {}
    models = (provenance or {}).get("models") or []
    steps: list[str] = []
    for record in (report or {}).get("frames") or []:
        if record.get("steps"):
            steps = list(record["steps"])
            break
    parts_block = None
    parts_meta = (report or {}).get("parts")
    if parts_meta:
        parts_block = {
            "mode": parts_meta.get("mode"),
            "parts": list(parts_meta.get("parts") or []),
            "palettes": parts_meta.get("palettes"),
            "palettes_from": parts_meta.get("palettes_from"),
            "model": parts_meta.get("model"),
            "frames": [
                {"index": r.get("index"),
                 "parts": {name: {k: v for k, v in info.items()
                                  if k in ("pixels", "drift", "colours", "found",
                                           "locked", "reason")}
                           for name, info in (r.get("parts") or {}).items()}}
                for r in (report or {}).get("frames") or []],
        }
        model = parts_meta.get("model")
        if model and model not in models:
            models = list(models) + [model]
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
        "contract": contract,
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
            # Where the timing SHAPE came from (see the module docstring).
            "timing_source": timing_source,
            # The part palette lock feature's record, or null when it did not run.
            "parts": parts_block,
        },
        "postprocess": steps,
    }


def resolve_timing(action: str, frames: int,
                   picks_record: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    """(timing block or None, source) - hand > recorded > template > uniform.

    Pure, so the precedence is testable without a job on disk. The block that
    comes back has only `weights` and `anchor`; the source is the second value.
    """
    record = picks_record or {}
    timing = record.get("timing")
    if isinstance(timing, dict) and timing.get("weights") is not None:
        weights = list(timing.get("weights") or [])
        if len(weights) == frames:
            block = {"weights": weights, "anchor": timing.get("anchor")}
            if timing != record.get("auto_timing"):
                return block, "hand"
            return block, str(timing.get("source") or "hand")
        print(f"publish: picks.json timing has {len(weights)} weight(s) for "
              f"{frames} frame(s) - ignoring it and falling back to the template",
              file=sys.stderr)
    template = config.timing_for_action(action, frames) if action else None
    if template:
        return {"weights": template[0], "anchor": template[1]}, f"template:{action}"
    return None, "uniform"


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
    picks_record = read_json(job.picks_path) if job.picks_path.exists() else None
    picks = list((picks_record or {}).get("picks") or [])
    timing, timing_source = resolve_timing(action, len(frames), picks_record)

    base = Path(assets_dir) if assets_dir is not None else config.ASSETS_DIR
    units_dir = base.joinpath(*UNITS_SUBDIR)
    names = frame_names(stem, len(frames))
    meta = sidecar_for(slug=slug, action=action, frames=len(frames),
                       manifest=manifest, provenance=provenance, picks=picks,
                       report=report, timing=timing, timing_source=timing_source)

    result: dict[str, Any] = {"stem": stem, "dir": str(units_dir),
                              "frames": names, "sidecar": f"{stem}.json",
                              "timing": timing, "timing_source": timing_source,
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
    if result["timing"]:
        print(f"  timing: {result['timing']['weights']} anchor "
              f"{result['timing']['anchor']} ({result['timing_source']})")
    else:
        print("  timing: uniform (no contract.timing written)")
    for name in result["removed"]:
        print(f"  removed stale {name}")
    if result["audit"]:
        print(f"  asset_audit.py: {result['audit']['output'].splitlines()[-1]}")
    return 0


run = run_stage
