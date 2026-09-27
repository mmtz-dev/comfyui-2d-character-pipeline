"""Stage: `anim bakeoff` - two routes x two rigs on one action, one page (a GPU stage)

The frame route bake-off feature (#274, the third feature of the sprite studio
epic). One character x one action - the knight's attack - through both
candidate routes for the in-between frames, on both candidate rigs, rendered
for real, judged side by side on one page with one metric table. The owner
rules; this stage produces the evidence and picks nothing.

    anim bakeoff --unit knight --action attack
    anim bakeoff --unit knight --action attack --cells V-kaykit,K-quaternius
    anim bakeoff --unit knight --action attack --force          # redo every cell
    anim bakeoff --unit knight --action attack --page-only      # rebuild the page

    work/bakeoff_knight_attack/bakeoff.json      the plan, one record per cell
    work/bakeoff_knight_attack/page/index.html   the page (+ img/); cp to the visuals dir
    work/bakeoff_knight_attack/report.md         the metric table, in markdown
    work/bakeoff_knight_attack_<cell>/           an ordinary job per cell

THE FOUR CELLS. `V-kaykit`, `V-quaternius`, `K-kaykit`, `K-quaternius`:

  Route V - rig -> Wan video.   What `anim clip` does today: the rig clip at
        the extract fps, skeleton over full-body depth at gain 0.35 (the second
        gate's variant (b), the only one with one sword in the sword hand on
        every strike frame), Wan 2.2 Fun-Control 5B conditioned on the unit's
        base render, then the generated clip DOWNSAMPLED to the action's frame
        budget: `pick --phases` with the even spread (`even_spread`: 12 over
        the rig frames, tagged by phase), then `post --input clip --parts`.

  Route K - keyframe stills -> tween.   The strip route's pose pass (Z-Image-
        Turbo + the Fun ControlNet Union on a 4 x 1 skeleton strip, the
        published knight's sprite as the identity pass's reference, exactly as
        `strip` and `render` do it today) with the RIG-SPACE TWEEN as the
        in-betweening: the rig is sampled at exactly 12 instants across the
        clip (`phase_times`; `rig(..., times=)`), so the five phase poses and
        the seven in-betweens are all stills from the rig, and "tween" means
        denser rig sampling rather than pixels invented between two renders.
        The 12 cells go through the pose pass as THREE 4-cell strips with one
        seed, at the anchor gate's control strength and unit prompt
        (`K_STRENGTH`, `POSE_PROMPTS` - the workflow's own prompt returns
        silhouettes) (`K_CHUNK`): a 12-cell 3648 x 320 strip was tried first on
        2026-09-20 and the pose pass returned eight silhouettes and four cells
        of noise (work/_exp/bo_k12q) - the 4-cell strip is the unit the anchor
        gate verified. The three pose-pass strips are joined into one 12-cell
        `render/pose_pass.png` for the page, and ALSO cut into
        `clip/frames/%04d.png` (1:1 with the rig frames, as `anim clip` lays
        them out) so `post --input clip --parts` mattes each cell on its own -
        the very same call Route V's frames go through (`rembg` on a 3648 px
        strip returned an empty matte for a cell; twelve cells wide is past
        what isnet-anime resolves). The identity
        pass runs too (it is what `render` does) and its strip is on the page
        for the owner to see; the frames are the POSE PASS, as every published
        strip has been (DEVLOG 2026-09-15, finding 4).

        Why not RIFE / FILM: no interpolation node is pinned in
        gpu/custom_nodes.txt and the rule for this feature is no new weights.
        The pixel-space crossfade - the honest baseline - is on the page as a
        display-only strip under each K row (blends of the five phase stills,
        no palette snap, no metrics), because it costs nothing and shows what
        "tween" would mean without the rig.

  Both rigs. KayKit Adventurers Knight (`1H_Melee_Attack_Chop`) and the
        Quaternius UAL (`Sword_Attack`, the rig the second gate ran on). The
        same seed in every cell - the second gate's, so `V-quaternius` is
        comparable with that page. The identity reference is the current
        accepted `work/<unit>_base` render (Route V) and the published sprite
        (Route K), which are the same knight.

THE METRIC TABLE, per cell, from the part palette lock's report plus two
proxies computed on the HD cells before the downscale:

  frames        frames that passed `post`'s verification, against the budget
  per part      pixels (mean over the frames the part was found in), drift
                [min, max] - pixels the lock had to move - and the frames the
                part was NOT found in (reported, never invented)
  one sword     frames where the sword mask is one blob (islands under 15 % of
                the largest ignored) whose nearest pixel to the RIGHT wrist is
                within `SWORD_REACH` of the cell height and no nearer to the
                left wrist - the second gate's check 3, as a proxy
  one figure    frames where the matte is one blob (islands under 25 % of the
                largest ignored) and the helmet mask is at most one - the
                second gate's check 1, as a proxy

Proxies, and said so on the page: the second gate counted these by eye; the
owner's eye still overrules the number. The masks are SAM 2.1's, prompted
from the rig exactly as `post --parts` prompts them - the stage records what
`post` asked the segmenter and what came back (`RecordingSegmenter`) rather
than segmenting twice.

RESUMABLE, PER CELL. A cell whose job has `bakeoff.status == "done"` in its
job.json is skipped; `--force` redoes it; `--cells` names a subset. Seconds
per stage per cell (rig / gpu / post) and the service's VRAM after the GPU
stage are in bakeoff.json and on the page. A cell that fails is recorded with
its error and the others still run - one bad cell must not cost the page.

AND SO THE STAGE REPORTS ITS TALLY (2026-09-24). Carrying on past a failed cell
used to mean exiting 0 whatever came of the run, which read as `done` at 100%
in the studio - the bug `anim.outcome` documents, found on one batch stage and
fixed for all of them at once. `bakeoff.json` now carries `summary`,
the last line of the run is `outcome: done=1 failed=3 blocked=0 of=4`, and the
exit code is 0 only when every selected cell is done (3 when some are, 4 when
none are). `--page-only` rebuilds the page from disk without running a cell, so
it reports no tally and exits 0.

WHAT IT DOES NOT DO. It does not publish, does not touch OpenProject, does not
change `pick`, `clip`, `strip`, `render` or `post` - every cell is those
stages called as a library, and every cell job can be re-run by hand with
the same commands. The phase labels on the page are the attack row of the
keyframe phase feature's template (config.ACTION_PHASES, by column); the
frames themselves are still the even spread the owner judged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from anim import comfy, config, outcome, partkit, posekit, rigkit
from anim.job import Job
from anim.stages import clip as clip_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import render as render_stage
from anim.stages import rig as rig_stage
from anim.stages import strip as strip_stage
from anim.util import read_json, write_bytes_in_place, write_json_in_place, write_text_in_place

NAME = "bakeoff"
HELP = "two routes x two rigs on one action -> one page + metric table (a GPU stage)"

BAKEOFF_SCHEMA = "anim.bakeoff/1"
REPO = Path(__file__).resolve().parents[2]

# The attack's phase plan from the keyframe phase feature's table
# (config.ACTION_PHASES: cap 12, attack 12): the budget, the phase label per
# column and the five key columns the crossfade blends between all come from
# it. The two routes still sample the clip EVENLY (`spread="even"`) - the
# owner judged those frames and a re-run must produce the same numbers; the
# labels are the template's.
PLAN = pick_stage.phase_plan("attack")
BUDGET = PLAN.budget
# The second gate's seed (2026-09-17), so V-quaternius is the same render as
# that page's variant (b) and the two pages can be read against each other.
SEED = 3139000483690707
ROUTES: dict[str, str] = {"V": "rig -> Wan video", "K": "keyframe stills -> rig-space tween"}
RIGS: dict[str, dict[str, Any]] = {
    "kaykit": {"glb": REPO / "rigs" / "kaykit-adventurers" / "Knight.glb",
               "label": "KayKit Adventurers Knight (chibi)"},
    "quaternius": {"glb": REPO / "rigs" / "quaternius" / "UAL1_Standard.glb",
                   "label": "Quaternius UAL (realistic proportions)"},
}
# The studio's action names against the pipeline's verbs and each rig's clip
# (config.STUDIO_ACTIONS); the bake-off runs the one the epic named.
ACTIONS: dict[str, dict[str, Any]] = {"attack": config.STUDIO_ACTIONS["attack"]}
# Route V: the second gate's variant (b).
V_CONTROL = "blend"
V_DEPTH_GAIN = 0.35
V_CELL = config.CLIP_CELL
# Route K: the strip route's verified unit, and the pose pass as the anchor
# gate ran it (2026-09-15): control strength 1.0 (the workflow's 0.7 default
# predates the 2 px line-width fix) and the unit's description in the prompt.
# `pose_pass.json`'s own prompt names no unit and returns black silhouettes -
# measured on this feature's first K run, 2026-09-21. The prompt below is the
# one embedded in work/knight_strike/render/pose_pass.png (the gate's strip);
# it lived on the command line until now.
K_CHUNK = config.STRIP_CELLS
K_CELL = config.STRIP_CELL
K_STRENGTH = 1.0
POSE_PROMPTS: dict[str, str] = {
    "knight": ("a side-view pixel art sprite strip of a plate-armoured knight in a closed "
               "steel great helm with a one-handed sword and a kite shield, four key poses "
               "of one sword strike left to right, facing right, full body in every pose, "
               "bold dark outline, flat shaded, plain flat white background, no ground "
               "shadow, no floor"),
}
# Which column is which phase, and the five phase keys the crossfade
# baseline blends between - the attack template's, per column.
PHASES: tuple[str, ...] = tuple(PLAN.tags())
PHASE_KEYS: tuple[int, ...] = tuple(PLAN.keys())
# Proxy thresholds (fractions of the cell height / of the largest blob).
SWORD_REACH = 0.12
SWORD_ISLAND = 0.15
FIGURE_ISLAND = 0.25
PAGE_DIR_NAME = "page"
VISUALS_TARGET = "~/.cache/claude/visuals/sprite-studio/bake-off"
VISUALS_URL = "http://192.168.1.158:8090/sprite-studio/bake-off/"


# ---------------------------------------------------------------------------
# The plan (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    name: str
    route: str
    rig: str
    glb: str
    clip: str
    job: str


def bakeoff_name(unit: str, action: str) -> str:
    return f"bakeoff_{unit}_{action}"


def plan(unit: str, action: str, *, rigs: dict[str, dict[str, Any]] | None = None,
         clips: dict[str, str] | None = None) -> list[Cell]:
    """The four cells, V before K, rigs in table order. `rigs` / `clips` are
    for tests (a fixture rig with its own clip name)."""
    if action not in ACTIONS:
        raise ValueError(f"no bake-off action {action!r}; known: {sorted(ACTIONS)}")
    rigs = rigs if rigs is not None else RIGS
    clips = clips if clips is not None else ACTIONS[action]["clips"]
    prefix = bakeoff_name(unit, action)
    cells: list[Cell] = []
    for route in ROUTES:
        for rig_name, rig in rigs.items():
            if rig_name not in clips:
                raise ValueError(f"no {action!r} clip named for rig {rig_name!r}")
            name = f"{route}-{rig_name}"
            cells.append(Cell(name=name, route=route, rig=rig_name, glb=str(rig["glb"]),
                              clip=clips[rig_name], job=f"{prefix}_{name}"))
    return cells


def _plan(budget: int) -> pick_stage.PhasePlan:
    return PLAN if budget == PLAN.budget else pick_stage.phase_plan("attack", budget=budget)


def even_spread(frames: int, budget: int = BUDGET) -> list[int]:
    """`budget` frame indices spread evenly over `frames`, both ends included
    (all of them when there are no more than `budget`) - the phase picker's
    even spread."""
    if frames <= 0:
        raise ValueError("a clip needs at least one frame")
    return pick_stage.phase_frames(frames, _plan(budget), spread="even")


def phase_times(duration: float, budget: int = BUDGET) -> list[float]:
    """`budget` sample instants across [0, duration], both ends included -
    the phase picker's even instants."""
    return pick_stage.phase_instants(duration, _plan(budget), spread="even")


def frame_files(job: Job, manifest: dict[str, Any] | None = None) -> list[Path]:
    """The cell's finished frames, `<slug>_<verb>_<n>.png` (post/ also holds
    palette.png), in index order."""
    manifest = manifest if manifest is not None else job.read_manifest()
    stem = f"{manifest.get('slug', 'unit')}_{manifest.get('action', '')}"
    files = [p for p in job.post_dir.glob(f"{stem}_*.png")
             if p.stem[len(stem) + 1:].isdigit()]
    return sorted(files, key=lambda p: int(p.stem[len(stem) + 1:]))


def is_done(job: Job, budget: int = BUDGET) -> bool:
    """A cell is done when its job says so AND its frames are on disk."""
    if not job.manifest_path.exists():
        return False
    manifest = job.read_manifest()
    block = manifest.get("bakeoff") or {}
    return block.get("status") == "done" and job.report_path.exists() \
        and len(frame_files(job, manifest)) >= 1


# ---------------------------------------------------------------------------
# Recording what `post --parts` asked the segmenter
# ---------------------------------------------------------------------------
class RecordingSegmenter:
    """Wraps a `partkit.Segmenter`; keeps every (prompts, results) pair in
    call order, which is `post`'s cell order, so the proxies below read the
    same masks the lock used without a second SAM pass."""

    def __init__(self, inner: partkit.Segmenter) -> None:
        self.inner = inner
        self.calls: list[tuple[list[partkit.Prompt], list[tuple[np.ndarray, float] | None]]] = []

    def segment(self, rgb: np.ndarray,
                prompts: Sequence[partkit.Prompt]) -> list[tuple[np.ndarray, float] | None]:
        results = self.inner.segment(rgb, prompts)
        self.calls.append((list(prompts), list(results)))
        return results


# ---------------------------------------------------------------------------
# The proxies (pure; numpy + cv2)
# ---------------------------------------------------------------------------
def blobs(mask: np.ndarray, island: float) -> int:
    """Connected components of `mask` at least `island` of the largest."""
    import cv2

    arr = np.asarray(mask, dtype=bool).astype(np.uint8)
    if not arr.any():
        return 0
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas >= areas.max() * island).sum())


def nearest_distance(mask: np.ndarray, point: tuple[float, float] | None) -> float:
    """Distance from `point` to the nearest set pixel, or inf."""
    if point is None:
        return float("inf")
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(ys):
        return float("inf")
    return float(np.sqrt((xs - point[0]) ** 2 + (ys - point[1]) ** 2).min())


def wrist_points(entry: dict[str, Any], names: Sequence[str],
                 source_size: tuple[int, int], cell: tuple[int, int],
                 kpt_thr: float) -> dict[str, tuple[float, float] | None]:
    """The two wrists in cell pixels (the letterbox `post --parts` uses)."""
    scale, off_x, off_y = posekit.letterbox(source_size, cell)
    kp, sc = entry.get("keypoints") or [], entry.get("scores") or []
    out: dict[str, tuple[float, float] | None] = {}
    for side in ("right_wrist", "left_wrist"):
        i = list(names).index(side) if side in names else -1
        if 0 <= i < len(kp) and i < len(sc) and sc[i] >= kpt_thr:
            out[side] = (kp[i][0] * scale + off_x, kp[i][1] * scale + off_y)
        else:
            out[side] = None
    return out


def sword_check(prompts: Sequence[partkit.Prompt],
                results: Sequence[tuple[np.ndarray, float] | None],
                alpha: np.ndarray, wrists: dict[str, tuple[float, float] | None],
                cell_height: int) -> dict[str, Any]:
    """One sword, in the sword hand: one blob, near the right wrist, not
    nearer the left. Returns the verdict and the numbers behind it."""
    index = next((i for i, p in enumerate(prompts) if p.part == "sword"), -1)
    result = results[index] if 0 <= index < len(results) else None
    if result is None:
        return {"ok": False, "blobs": 0, "right": None, "left": None,
                "reason": "no sword mask"}
    mask = np.asarray(result[0], dtype=bool) & (np.asarray(alpha) > 0)
    count = blobs(mask, SWORD_ISLAND)
    right = nearest_distance(mask, wrists.get("right_wrist"))
    left = nearest_distance(mask, wrists.get("left_wrist"))
    reach = SWORD_REACH * cell_height
    ok = count == 1 and right <= reach and right <= left
    reason = ("" if ok else
              "no sword pixels" if count == 0 else
              f"{count} sword blobs" if count != 1 else
              "sword nearer the shield hand" if right > left else
              f"sword {right:.0f} px from the sword hand (reach {reach:.0f})")
    return {"ok": bool(ok), "blobs": count,
            "right": None if not np.isfinite(right) else round(right, 1),
            "left": None if not np.isfinite(left) else round(left, 1),
            "reason": reason}


def figure_check(prompts: Sequence[partkit.Prompt],
                 results: Sequence[tuple[np.ndarray, float] | None],
                 alpha: np.ndarray) -> dict[str, Any]:
    """One figure: the matte is one blob and there is at most one helmet."""
    bodies = blobs(np.asarray(alpha) > 0, FIGURE_ISLAND)
    index = next((i for i, p in enumerate(prompts) if p.part == "helmet"), -1)
    result = results[index] if 0 <= index < len(results) else None
    helmets = blobs(np.asarray(result[0], dtype=bool) & (np.asarray(alpha) > 0),
                    FIGURE_ISLAND) if result is not None else 0
    ok = bodies == 1 and helmets <= 1
    return {"ok": bool(ok), "bodies": bodies, "helmets": helmets,
            "reason": "" if ok else f"{bodies} bodies, {helmets} helmets"}


def cell_metrics(job: Job, *, route: str, matte: str,
                 recorder: RecordingSegmenter, budget: int = BUDGET) -> dict[str, Any]:
    """The metric block for one cell from post/report.json + the recorded masks."""
    report = read_json(job.report_path) if job.report_path.exists() else {}
    frames = report.get("frames") or []
    part_names = list((report.get("parts") or {}).get("parts") or [])
    good = [f for f in frames if not f.get("problems")]

    per_part: dict[str, dict[str, Any]] = {}
    for name in part_names:
        pixels = [int(f["parts"][name].get("pixels") or 0) for f in frames
                  if f.get("parts", {}).get(name, {}).get("found")]
        drifts = [int(f["parts"][name]["drift"]) for f in frames
                  if f.get("parts", {}).get(name, {}).get("drift") is not None]
        missing = [f["index"] for f in frames
                   if not f.get("parts", {}).get(name, {}).get("found")]
        per_part[name] = {
            "pixels": round(sum(pixels) / len(pixels)) if pixels else 0,
            "drift": [min(drifts), max(drifts)] if drifts else None,
            "drift_total": sum(drifts) if drifts else 0,
            "missing": missing,
        }

    # The HD cells again (matte + keypoints), against the recorded masks.
    swords: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    if recorder.calls and job.keypoints_path.exists():
        cut = post_stage.MATTES[matte]
        cells, _ = post_stage.clip_cells(job, cut)   # both routes: one HD cell per pick
        payload = read_json(job.keypoints_path)
        names = list(payload.get("names") or posekit.rig_names(payload["rig"]))
        source_size = tuple(payload["source_size"])
        thr = float(payload.get("kpt_thr", config.POSE_KPT_THR))
        by_index = {f.get("index", i): f for i, f in enumerate(payload["frames"])}
        for (meta, matted, _hd), (prompts, results) in zip(cells, recorder.calls):
            alpha = np.asarray(matted)[:, :, 3]
            wrists = wrist_points(by_index.get(meta["frame"]) or {}, names,
                                  source_size, matted.size, thr)
            swords.append({"frame": meta["index"],
                           **sword_check(prompts, results, alpha, wrists, matted.height)})
            figures.append({"frame": meta["index"],
                            **figure_check(prompts, results, alpha)})

    return {
        "budget": budget,
        "frames": len(good),
        "frames_attempted": len(frames),
        "parts": per_part,
        "one_sword": {"count": sum(1 for s in swords if s["ok"]), "of": len(swords),
                      "frames": swords},
        "one_figure": {"count": sum(1 for f in figures if f["ok"]), "of": len(figures),
                       "frames": figures},
    }


# ---------------------------------------------------------------------------
# Running one cell
# ---------------------------------------------------------------------------
def _vram_used_gb(client: comfy.ComfyClient) -> float | None:
    try:
        devices = client.system_stats().get("devices") or []
    except comfy.ComfyError:
        return None
    if not devices:
        return None
    d = devices[0]
    total, free = d.get("vram_total") or 0, d.get("vram_free") or 0
    return round((total - free) / 2 ** 30, 1) if total else None


def _phased_picks(job: Job, *, verb: str, budget: int) -> list[int]:
    """`pick` against the attack template with the even spread: the phase
    tag per column and the phase timing land in picks.json, the frames are
    the ones the routes have always used."""
    record = pick_stage.pick(job.name, n=budget, action=verb, force=True,
                             phases=True, spread="even", work_dir=job.root.parent)
    return list(record["picks"])


def _hand_set_picks(job: Job, picks: list[int], *, verb: str) -> None:
    """`pick` as usual, then `picks` in force with the auto picks kept
    beside it - a hand-set pick, exactly as the review page leaves one. The
    K route's 4-cell chunks are set this way."""
    record = pick_stage.pick(job.name, n=len(picks), action=verb, force=True,
                             phases=False, work_dir=job.root.parent)
    record["picks"] = list(picks)
    write_json_in_place(job.picks_path, record)
    job.update_manifest(picks=list(picks))


def run_v_cell(cell: Cell, *, unit: str, verb: str, seed: int, matte: str,
               parts_from: Path | None, client: comfy.ComfyClient,
               segmenter: partkit.Segmenter | None, work_dir: Path | None,
               assets_dir: Path | None, workflow_dir: Path | None,
               lock_path: Path | None, reference: Path | None,
               budget: int = BUDGET, force: bool = False) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    job = rig_stage.rig(Path(cell.glb), clip=cell.clip, action=verb, slug=unit,
                        job_name=cell.job, cell=tuple(V_CELL), depth="full",
                        work_dir=work_dir)
    timings["rig"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    record = clip_stage.clip(job.name, seed=seed, control=V_CONTROL,
                             depth_gain=V_DEPTH_GAIN, reference=reference,
                             client=client, workflow_dir=workflow_dir,
                             lock_path=lock_path, work_dir=work_dir,
                             assets_dir=assets_dir)
    timings["gpu"] = round(time.monotonic() - t0, 1)
    gpu_seconds = float(record.get("seconds") or 0.0)
    vram = _vram_used_gb(client)

    picks = _phased_picks(job, verb=verb, budget=budget)

    t0 = time.monotonic()
    recorder = RecordingSegmenter(segmenter or partkit.Sam2Segmenter())
    post_error: str | None = None
    try:
        post_stage.post(job.name, matte=matte, source=post_stage.CLIP_SOURCE,
                        parts=True, parts_from=parts_from, segmenter=recorder,
                        assets_dir=assets_dir, work_dir=work_dir)
    except RuntimeError as exc:   # verification failures: frames that passed are on disk
        post_error = str(exc).splitlines()[0]
    timings["post"] = round(time.monotonic() - t0, 1)

    metrics = cell_metrics(job, route="V", matte=matte, recorder=recorder, budget=budget)
    return {"job": job.name, "timings": timings, "gpu_seconds": round(gpu_seconds, 1),
            "vram_used_gb": vram, "picks": picks, "rig_frames": int(record["frames"]),
            "control": record["control"], "depth_gain": record["depth_gain"],
            "seed": seed, "post_error": post_error, "metrics": metrics}


def _copy_rig_job(src: Job, dest: Job) -> None:
    """The skeletons, keypoints and manifest of a rig job under a new name -
    what `strip` and `render` read; frames/ and depth/ stay behind."""
    if dest.root.exists():
        shutil.rmtree(dest.root)
    dest.pose_dir.mkdir(parents=True, exist_ok=True)
    for path in src.pose_dir.iterdir():
        shutil.copy2(path, dest.pose_dir / path.name)
    manifest = src.read_manifest()
    manifest["bakeoff_chunk_of"] = src.name
    dest.write_manifest(manifest)


def _join_strips(paths: Sequence[Path], dest: Path) -> Path:
    images = [Image.open(p).convert("RGB") for p in paths]
    width, height = sum(i.width for i in images), images[0].height
    joined = Image.new("RGB", (width, height), strip_stage.REFERENCE_BG)
    x = 0
    for image in images:
        joined.paste(image, (x, 0))
        x += image.width
    from io import BytesIO

    buffer = BytesIO()
    joined.save(buffer, format="PNG")
    return write_bytes_in_place(dest, buffer.getvalue())


def _service_seconds(client: comfy.ComfyClient, provenance: dict[str, Any]) -> dict[str, float | None]:
    """Each pass's seconds by ComfyUI's own execution stamps (the clock
    `anim clip` records), read back from /history - which is also how a
    reused chunk from an earlier attempt gets its number."""
    out: dict[str, float | None] = {}
    for record in provenance.get("passes") or []:
        pid = record.get("prompt_id")
        entry = client.history(pid).get(pid) if pid else None
        seconds = clip_stage.execution_seconds(entry) if entry else None
        out[record.get("name", pid)] = round(seconds, 1) if seconds is not None else None
    return out


def _chunk_rendered(chunk: Job) -> bool:
    return all((chunk.render_dir / f).exists()
               for f in ("pose_pass.png", "identity_pass.png", "provenance.json"))


def run_k_cell(cell: Cell, *, unit: str, verb: str, seed: int, matte: str,
               parts_from: Path | None, client: comfy.ComfyClient,
               segmenter: partkit.Segmenter | None, work_dir: Path | None,
               assets_dir: Path | None, workflow_dir: Path | None,
               lock_path: Path | None, reference: Path | None,
               budget: int = BUDGET, force: bool = False) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    duration = rigkit.load_rig(Path(cell.glb)).clip(cell.clip).duration
    times = phase_times(duration, budget)
    job = rig_stage.rig(Path(cell.glb), clip=cell.clip, action=verb, slug=unit,
                        job_name=cell.job, cell=tuple(K_CELL), depth="full",
                        times=times, work_dir=work_dir)
    frames = len(job.skeletons())
    _phased_picks(job, verb=verb, budget=budget)     # every sampled instant, tagged
    # The 12-cell layout the page and the metrics slice by; its skeleton strip
    # is not rendered.
    layout = strip_stage.strip(job.name, reference=reference, work_dir=work_dir,
                               assets_dir=assets_dir)
    timings["rig"] = round(time.monotonic() - t0, 1)

    # Three 4-cell strips through the strip route, one seed. A chunk that was
    # rendered by an earlier attempt at this cell is reused unless `force` -
    # the GPU minutes are the expensive part and the render is deterministic.
    t0 = time.monotonic()
    chunks: list[dict[str, Any]] = []
    pose_strips: list[Path] = []
    identity_strips: list[Path] = []
    for start in range(0, frames, K_CHUNK):
        picks = list(range(start, min(start + K_CHUNK, frames)))
        chunk = Job.open(f"{job.name}_c{len(chunks)}", work_dir)
        reused = not force and _chunk_rendered(chunk) and \
            list((read_json(chunk.picks_path) if chunk.picks_path.exists() else {})
                 .get("picks") or []) == picks
        seconds = 0.0
        if not reused:
            _copy_rig_job(job, chunk)
            _hand_set_picks(chunk, picks, verb=verb)
            strip_stage.strip(chunk.name, reference=reference, work_dir=work_dir,
                              assets_dir=assets_dir)
            t_chunk = time.monotonic()
            render_stage.render(chunk.name, seed=seed, strength=K_STRENGTH,
                                pose_prompt=POSE_PROMPTS.get(unit), client=client,
                                workflow_dir=workflow_dir, lock_path=lock_path,
                                work_dir=work_dir)
            seconds = round(time.monotonic() - t_chunk, 1)
        provenance = read_json(chunk.provenance_path)
        service = _service_seconds(client, provenance)
        chunks.append({"job": chunk.name, "picks": picks, "seconds": seconds,
                       "service_seconds": service, "reused": reused,
                       "passes": [p["name"] for p in provenance["passes"]]})
        pose_strips.append(chunk.render_dir / "pose_pass.png")
        identity_strips.append(chunk.render_dir / "identity_pass.png")
    timings["gpu"] = round(time.monotonic() - t0, 1)
    vram = _vram_used_gb(client)

    job.render_dir.mkdir(parents=True, exist_ok=True)
    joined = _join_strips(pose_strips, job.render_dir / "pose_pass.png")
    _join_strips(identity_strips, job.render_dir / "identity_pass.png")
    merged = [read_json(Job.open(c["job"], work_dir).provenance_path) for c in chunks]
    provenance = render_stage.build_provenance(
        client.base_url, {}, seed, [p for m in merged for p in m["passes"]])
    provenance["final"] = "pose_pass.png"
    provenance["chunks"] = [c["job"] for c in chunks]
    write_json_in_place(job.provenance_path, provenance)
    # The frames are the pose pass (DEVLOG 2026-09-15, finding 4); the identity
    # strip is kept beside it for the page. The 12 cells are ALSO written as
    # clip/frames/%04d.png, 1:1 with the rig frames like `anim clip` writes
    # them, so `post --input clip` mattes each cell on its own: `rembg` on the
    # whole 3648 px strip returned an empty matte for a cell (2026-09-21) -
    # isnet-anime works at ~1024 px and twelve cells wide leaves each knight
    # ~85 px tall. The clip route's post is what Route V runs, so both routes'
    # frames now go through the very same call.
    job.update_manifest(render={"seed": seed, "passes": ["pose_pass", "identity_pass"],
                                "final": "pose_pass.png",
                                "chunks": [c["job"] for c in chunks]})
    job.clip_frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in job.clip_frames_dir.glob("*.png"):
        stale.unlink()
    with Image.open(joined) as strip_img:
        for cell_meta in layout["cells"]:
            x, y, w, h = cell_meta["rect"]
            _save(strip_img.crop((x, y, x + w, y + h)).convert("RGB"),
                  job.clip_frames_dir / f"{cell_meta['frame'] + 1:04d}.png")

    t0 = time.monotonic()
    recorder = RecordingSegmenter(segmenter or partkit.Sam2Segmenter())
    post_error: str | None = None
    try:
        post_stage.post(job.name, matte=matte, source=post_stage.CLIP_SOURCE,
                        parts=True, parts_from=parts_from, segmenter=recorder,
                        assets_dir=assets_dir, work_dir=work_dir)
    except RuntimeError as exc:
        post_error = str(exc).splitlines()[0]
    timings["post"] = round(time.monotonic() - t0, 1)

    metrics = cell_metrics(job, route="K", matte=matte, recorder=recorder, budget=budget)
    service_total = sum(v for c in chunks for v in c["service_seconds"].values() if v)
    return {"job": job.name, "timings": timings,
            # The service's clock when it answers (every pass, reused or not);
            # the wall clock around the render calls otherwise.
            "gpu_seconds": round(service_total or sum(c["seconds"] for c in chunks), 1),
            "vram_used_gb": vram, "picks": list(range(frames)), "rig_frames": frames,
            "times": times, "chunks": chunks, "seed": seed,
            "strength": K_STRENGTH, "pose_prompt": POSE_PROMPTS.get(unit),
            "post_error": post_error, "metrics": metrics}


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {"V": run_v_cell, "K": run_k_cell}


def run_cell(cell: Cell, **kw: Any) -> dict[str, Any]:
    """One cell end to end; the record goes into bakeoff.json and the cell's
    job.json (`bakeoff` block) so a re-run can tell it is done."""
    started = time.monotonic()
    record: dict[str, Any] = {**asdict(cell), "status": "running",
                              "started": _now()}
    try:
        record.update(RUNNERS[cell.route](cell, **kw))
        record["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - one bad cell must not cost the page
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.monotonic() - started, 1)
    job = Job.open(cell.job, kw.get("work_dir"))
    if job.manifest_path.exists():
        job.update_manifest(bakeoff={"status": record["status"], "cell": cell.name,
                                     "seconds": record["seconds"],
                                     "error": record.get("error")})
    return record


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
def bakeoff(unit: str, action: str, *, cells: Sequence[str] | None = None,
            force: bool = False, page_only: bool = False, seed: int = SEED,
            matte: str = "rembg", parts_from: Path | None = None,
            client: comfy.ComfyClient | None = None,
            segmenter: partkit.Segmenter | None = None,
            rigs: dict[str, dict[str, Any]] | None = None,
            clips: dict[str, str] | None = None,
            reference: Path | None = None, work_dir: Path | None = None,
            assets_dir: Path | None = None, workflow_dir: Path | None = None,
            lock_path: Path | None = None, budget: int = BUDGET,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    verb = ACTIONS[action]["verb"] if action in ACTIONS else action
    planned = plan(unit, action, rigs=rigs, clips=clips)
    wanted = set(cells) if cells else {c.name for c in planned}
    unknown = wanted - {c.name for c in planned}
    if unknown:
        raise ValueError(f"unknown cell(s) {sorted(unknown)}; the plan is "
                         f"{[c.name for c in planned]}")
    # `plan` stays all four cells - the page draws four rows whatever this call
    # asked for - so the tally counts `selected`, the subset `--cells` named.
    selected = [c.name for c in planned if c.name in wanted]

    root = Job.open(bakeoff_name(unit, action), work_dir)
    root.root.mkdir(parents=True, exist_ok=True)
    state_path = root.root / "bakeoff.json"
    state = read_json(state_path) if state_path.exists() else {}
    records: dict[str, Any] = dict(state.get("cells") or {})
    if page_only and state:
        # A page rebuild describes the run that produced the cells, not the
        # flags this call happened to be given.
        seed = int(state.get("seed", seed))
        matte = str(state.get("matte", matte))
        parts_from = Path(state["parts_from"]) if state.get("parts_from") else parts_from

    if not page_only:
        client = client or comfy.ComfyClient()
        # SAM once for every cell (a 184 MB checkpoint, loaded lazily).
        segmenter = segmenter or partkit.Sam2Segmenter()
        for cell in planned:
            job = Job.open(cell.job, work_dir)
            if cell.name not in wanted:
                continue
            if is_done(job, budget) and not force:
                log(f"bakeoff: {cell.name} done -> skipped (--force redoes it)")
                records.setdefault(cell.name, {**asdict(cell), "status": "done"})
                continue
            log(f"bakeoff: {cell.name} ({ROUTES[cell.route]}, {cell.clip}) ...")
            record = run_cell(cell, unit=unit, verb=verb, seed=seed, matte=matte,
                              parts_from=parts_from, client=client,
                              segmenter=segmenter, work_dir=work_dir,
                              assets_dir=assets_dir, workflow_dir=workflow_dir,
                              lock_path=lock_path, reference=reference,
                              budget=budget, force=force)
            records[cell.name] = record
            if record["status"] == "done":
                m = record["metrics"]
                log(f"  {cell.name}: {m['frames']}/{budget} frame(s), gpu "
                    f"{record['gpu_seconds']} s, total {record['seconds']} s, "
                    f"one sword {m['one_sword']['count']}/{m['one_sword']['of']}, "
                    f"one figure {m['one_figure']['count']}/{m['one_figure']['of']}")
            else:
                log(f"  {cell.name}: FAILED after {record['seconds']} s - {record['error']}")
            state = _state(unit, action, verb, budget, seed, matte, parts_from,
                           planned, selected, records)
            write_json_in_place(state_path, state)

    state = _state(unit, action, verb, budget, seed, matte, parts_from,
                   planned, selected, records)
    write_json_in_place(state_path, state)
    page = build_page(state, root.root / PAGE_DIR_NAME, work_dir=work_dir)
    report = write_text_in_place(root.root / "report.md", report_markdown(state))
    state["page"] = str(page)
    state["report"] = str(report)
    return state


def _state(unit: str, action: str, verb: str, budget: int, seed: int, matte: str,
           parts_from: Path | None, planned: Sequence[Cell],
           selected: Sequence[str], records: dict[str, Any]) -> dict[str, Any]:
    """`anim.bakeoff/1`, written after every cell and again at the end.

    `summary` is the shared tally (`anim.outcome`): what the run accomplished in
    cells, which is what the exit code and the studio's run status are made of.
    """
    return {"schema": BAKEOFF_SCHEMA, "unit": unit, "action": action, "verb": verb,
            "budget": budget, "seed": seed, "matte": matte,
            "parts_from": str(parts_from) if parts_from else None,
            "routes": ROUTES, "phases": list(PHASES), "phase_keys": list(PHASE_KEYS),
            "updated": _now(), "plan": [c.name for c in planned],
            "selected": list(selected), "cells": records,
            "summary": outcome.count(records, selected).counts}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def _sheet(frames: Sequence[Image.Image], dest: Path) -> Path:
    """The 12 frames side by side, one 64 px cell each, for the CSS animation."""
    canvas = tuple(config.CANVAS)
    sheet = Image.new("RGBA", (canvas[0] * max(1, len(frames)), canvas[1]), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame.convert("RGBA"), (i * canvas[0], 0))
    from io import BytesIO

    buffer = BytesIO()
    sheet.save(buffer, format="PNG")
    return write_bytes_in_place(dest, buffer.getvalue())


def crossfade(keys: dict[int, Image.Image], budget: int = BUDGET) -> list[Image.Image]:
    """The pixel-space baseline: the phase-key frames held, every other frame a
    linear RGBA blend of the two keys around it. Display only - no palette
    snap, no metrics; it is here to show what a tween without the rig means."""
    canvas = tuple(config.CANVAS)
    blank = Image.new("RGBA", canvas, (0, 0, 0, 0))
    order = sorted(keys)
    out: list[Image.Image] = []
    for i in range(budget):
        if i in keys:
            out.append(keys[i].convert("RGBA"))
            continue
        before = max((k for k in order if k < i), default=None)
        after = min((k for k in order if k > i), default=None)
        if before is None or after is None:
            out.append((keys[after] if before is None and after is not None else
                        keys[before] if before is not None else blank).convert("RGBA"))
            continue
        t = (i - before) / (after - before)
        out.append(Image.blend(keys[before].convert("RGBA"), keys[after].convert("RGBA"), t))
    return out


def _save(img: Image.Image, dest: Path) -> Path:
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return write_bytes_in_place(dest, buffer.getvalue())


def collect_cell_images(record: dict[str, Any], img_dir: Path, *,
                        work_dir: Path | None, budget: int) -> dict[str, Any]:
    """Copy one cell's frames into the page: post frames, raw HD cells, part
    overlays, the animation sheet and (K) the crossfade sheet + identity strip."""
    job = Job.open(record["job"], work_dir)
    name = record["name"]
    out: dict[str, Any] = {"post": {}, "raw": {}, "parts": {}, "sheet": None,
                           "xfade": None, "identity": None, "pose_strip": None}
    if not job.manifest_path.exists():
        return out
    manifest = job.read_manifest()
    stem = f"{manifest.get('slug', 'unit')}_{manifest.get('action', '')}"
    report = read_json(job.report_path) if job.report_path.exists() else {}
    by_index = {f["index"]: f for f in report.get("frames") or []}
    post_frames: dict[int, Image.Image] = {}
    for index in range(budget):
        src = job.post_dir / f"{stem}_{index}.png"
        if src.exists():
            dest = img_dir / f"{name}_post_{index:02d}.png"
            shutil.copyfile(src, dest)
            out["post"][index] = dest.name
            with Image.open(src) as img:
                post_frames[index] = img.convert("RGBA").copy()
        overlay = job.post_dir / "parts" / f"{stem}_{index}_hd.png"
        if overlay.exists():
            dest = img_dir / f"{name}_parts_{index:02d}.png"
            shutil.copyfile(overlay, dest)
            out["parts"][index] = dest.name
        # The raw HD cell: the generated clip frame (V) or the pose-pass cell
        # (K) - both live in clip/frames/, 1:1 with the rig frames.
        meta = by_index.get(index)
        if meta is not None:
            src = job.clip_frames_dir / f"{meta['frame'] + 1:04d}.png"
            if src.exists():
                dest = img_dir / f"{name}_raw_{index:02d}.png"
                shutil.copyfile(src, dest)
                out["raw"][index] = dest.name
    if post_frames:
        frames = [post_frames.get(i, Image.new("RGBA", tuple(config.CANVAS), (0, 0, 0, 0)))
                  for i in range(budget)]
        out["sheet"] = _sheet(frames, img_dir / f"{name}_sheet.png").name
        if record["route"] == "K":
            keys = {k: post_frames[k] for k in PHASE_KEYS if k in post_frames}
            if len(keys) >= 2:
                out["xfade"] = _sheet(crossfade(keys, budget),
                                      img_dir / f"{name}_xfade_sheet.png").name
    if record["route"] == "K":
        for key, filename in (("identity", "identity_pass.png"), ("pose_strip", "pose_pass.png")):
            src = job.render_dir / filename
            if src.exists():
                dest = img_dir / f"{name}_{key}.png"
                shutil.copyfile(src, dest)
                out[key] = dest.name
    return out


_STYLE = """
:root { color-scheme: dark; }
body { background: #14161b; color: #d8dee9; font: 14px/1.45 -apple-system, BlinkMacSystemFont,
       "Segoe UI", sans-serif; margin: 0; padding: 24px; max-width: 1800px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 36px 0 8px; color: #9fb0c9; border-top: 1px solid #2a2f3a;
     padding-top: 14px; }
h3 { font-size: 14px; margin: 18px 0 6px; color: #bfc8d6; }
p.caption, .note { color: #8b96a5; }
code { color: #cdd; }
.cell { margin: 18px 0 28px; }
.cell .title { font-weight: 600; color: #e8ecf1; font-size: 15px; }
.cell .meta { color: #8b96a5; font-size: 12px; margin: 2px 0 8px; }
.strip { display: grid; grid-template-columns: 110px repeat(12, 1fr); gap: 4px;
         align-items: end; }
.strip .lbl { color: #8b96a5; font-size: 11px; align-self: center; }
.strip .ph { font-size: 10px; color: #6b7686; text-align: center; }
.strip .ph.key { color: #d6a94a; }
.strip .col { text-align: center; }
.strip img.px { width: 100%; max-width: 128px; image-rendering: pixelated;
                background: #2a2d36; border: 1px solid #333; display: block; margin: 0 auto; }
.strip img.hd { width: 100%; max-width: 128px; background: #fff; border: 1px solid #333;
                display: block; margin: 0 auto; }
.strip .miss { width: 100%; max-width: 128px; aspect-ratio: 1; border: 1px dashed #444;
               color: #666; font-size: 10px; display: flex; align-items: center;
               justify-content: center; margin: 0 auto; }
.anim { width: 192px; height: 192px; image-rendering: pixelated; background-repeat: no-repeat;
        background-size: 2304px 192px; background-color: #2a2d36; border: 1px solid #333;
        animation: play12 1s steps(12) infinite; }
@keyframes play12 { from { background-position: 0 0; } to { background-position: -2304px 0; } }
.anims { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; margin: 10px 0; }
.anims figure { margin: 0; max-width: 260px; }
.anims figcaption { color: #8b96a5; font-size: 11px; margin-top: 4px; }
table { border-collapse: collapse; font-size: 12px; margin: 8px 0 14px; }
th { color: #999; text-align: left; padding: 3px 8px; border-bottom: 1px solid #333;
     font-weight: 600; }
td { padding: 3px 8px; border-bottom: 1px solid #222; vertical-align: top; }
.ok { color: #8fd18f; } .bad { color: #f08080; } .warn { color: #e0b060; }
.wide img { max-width: 100%; background: #fff; border: 1px solid #333; display: block; }
.toggle { font-size: 12px; color: #8b96a5; margin: 6px 0; }
body.parts .strip img.hd.raw { display: none; }
body:not(.parts) .strip img.hd.ov { display: none; }
footer { margin-top: 32px; color: #6b7686; font-size: 12px; }
"""


def _fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def _cell_html(record: dict[str, Any], images: dict[str, Any], budget: int,
               phases: Sequence[str], keys: Sequence[int]) -> str:
    name = record["name"]
    if record.get("status") != "done":
        return (f'<div class="cell" id="{name}"><div class="title">{name}</div>'
                f'<div class="meta">{record.get("status", "not run")}'
                f'{" — " + record["error"] if record.get("error") else ""}</div></div>')
    m = record.get("metrics") or {}
    t = record.get("timings") or {}
    head = "".join(f'<div class="ph{" key" if i in keys else ""}">{i}<br>{phases[i]}</div>'
                   for i in range(budget))

    def row(label: str, kind: str) -> str:
        cols = []
        for i in range(budget):
            src = images[kind].get(i)
            if kind == "post":
                cols.append(f'<div class="col"><img class="px" src="img/{src}" alt="{name} {i}"></div>'
                            if src else '<div class="col"><div class="miss">no frame</div></div>')
            else:
                raw, ov = images["raw"].get(i), images["parts"].get(i)
                inner = (f'<img class="hd raw" src="img/{raw}" alt="{name} raw {i}">' if raw else
                         '<div class="miss">—</div>')
                if ov:
                    inner += f'<img class="hd ov" src="img/{ov}" alt="{name} parts {i}">'
                cols.append(f'<div class="col">{inner}</div>')
        return f'<div class="strip"><div class="lbl">{label}</div>{"".join(cols)}</div>'

    anims = ""
    if images.get("sheet"):
        anims += (f'<figure><div class="anim" style="background-image:url(img/{images["sheet"]})"></div>'
                  f'<figcaption>12 fps — post frames</figcaption></figure>')
    if images.get("xfade"):
        anims += (f'<figure><div class="anim" style="background-image:url(img/{images["xfade"]})"></div>'
                  f'<figcaption>12 fps — pixel crossfade baseline (display only: keys '
                  f'{list(keys)} held, blends between; no palette snap, no metrics)</figcaption></figure>')

    parts_rows = "".join(
        f"<tr><td>{p}</td><td>{v['pixels']}</td>"
        f"<td>{_fmt(v['drift'][0]) + '–' + _fmt(v['drift'][1]) if v['drift'] else '—'}</td>"
        f"<td>{v['drift_total']}</td>"
        f"<td class=\"{'bad' if v['missing'] else 'ok'}\">{len(v['missing'])}"
        f"{' (' + ', '.join(map(str, v['missing'])) + ')' if v['missing'] else ''}</td></tr>"
        for p, v in (m.get("parts") or {}).items())
    sword, figure = m.get("one_sword") or {}, m.get("one_figure") or {}
    sword_fail = ", ".join(f"{s['frame']}: {s['reason']}" for s in sword.get("frames", [])
                           if not s["ok"]) or "—"
    figure_fail = ", ".join(f"{f['frame']}: {f['reason']}" for f in figure.get("frames", [])
                            if not f["ok"]) or "—"
    extra = ""
    if record["route"] == "K":
        extra = (f'<h3>The three 4-cell pose-pass strips joined (the frames above) and the '
                 f'identity pass on the same strips (not used for the frames)</h3>'
                 f'<div class="wide">'
                 + (f'<img src="img/{images["pose_strip"]}" alt="{name} pose pass">' if images.get("pose_strip") else "")
                 + (f'<img src="img/{images["identity"]}" alt="{name} identity pass" style="margin-top:4px">' if images.get("identity") else "")
                 + '</div>')
    return f"""
<div class="cell" id="{name}">
  <div class="title">{name} — {ROUTES[record['route']]}</div>
  <div class="meta">{record['rig']} · {record['clip']} · {record.get('rig_frames')} rig frame(s)
    {'sampled at ' + str(len(record.get('times') or [])) + ' instants' if record['route'] == 'K' else '→ picks ' + str(record.get('picks'))}
    · seed {record.get('seed')} · rig {t.get('rig')} s · GPU {record.get('gpu_seconds')} s by the
    service's clock ({t.get('gpu')} s wall this attempt) · post {t.get('post')} s
    · VRAM after GPU {_fmt(record.get('vram_used_gb'), ' GB')}
    {'· <span class="warn">post: ' + record['post_error'] + '</span>' if record.get('post_error') else ''}</div>
  <div class="strip"><div class="lbl"></div>{head}</div>
  {row("64 px, post", "post")}
  {row("raw (pre-post)", "raw")}
  <div class="anims">{anims}</div>
  <table>
    <tr><th>frames</th><th>one sword, sword hand</th><th>one figure</th></tr>
    <tr><td class="{'ok' if m.get('frames') == budget else 'bad'}">{m.get('frames')} / {budget}</td>
        <td class="{'ok' if sword.get('count') == sword.get('of') else 'warn'}">{sword.get('count')} / {sword.get('of')}<br><span class="note">{sword_fail}</span></td>
        <td class="{'ok' if figure.get('count') == figure.get('of') else 'warn'}">{figure.get('count')} / {figure.get('of')}<br><span class="note">{figure_fail}</span></td></tr>
  </table>
  <table>
    <tr><th>part</th><th>pixels (mean, found frames)</th><th>drift px/frame (min–max)</th><th>drift total</th><th>not found (frames)</th></tr>
    {parts_rows}
  </table>
  {extra}
</div>"""


def _summary_table(state: dict[str, Any]) -> str:
    rows = []
    for name in state["plan"]:
        r = state["cells"].get(name)
        if not r or r.get("status") != "done":
            rows.append(f"<tr><td><b>{name}</b></td><td colspan=8 class=\"bad\">"
                        f"{(r or {}).get('status', 'not run')} {(r or {}).get('error', '')}</td></tr>")
            continue
        m, t = r["metrics"], r.get("timings") or {}
        drift = sum(v["drift_total"] for v in m["parts"].values())
        missing = sum(len(v["missing"]) for v in m["parts"].values())
        rows.append(
            f"<tr><td><b><a href=\"#{name}\" style=\"color:#d8dee9\">{name}</a></b></td>"
            f"<td>{r['rig']}</td><td>{ROUTES[r['route']]}</td>"
            f"<td>{m['frames']} / {m['budget']}</td>"
            f"<td>{m['one_sword']['count']} / {m['one_sword']['of']}</td>"
            f"<td>{m['one_figure']['count']} / {m['one_figure']['of']}</td>"
            f"<td>{drift}</td><td>{missing}</td>"
            f"<td>GPU {r['gpu_seconds']} s · rig {t.get('rig')} s · post {t.get('post')} s · "
            f"{_fmt(r.get('vram_used_gb'), ' GB')}</td></tr>")
    return ("<table><tr><th>cell</th><th>rig</th><th>route</th><th>frames</th>"
            "<th>one sword, sword hand</th><th>one figure</th><th>drift px, all parts, 12 frames</th>"
            "<th>part-not-found cells</th><th>seconds (GPU by the service's clock) · VRAM</th></tr>"
            + "".join(rows) + "</table>")


def build_page(state: dict[str, Any], out_dir: Path, *, work_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    budget = int(state.get("budget") or BUDGET)
    phases, keys = list(state.get("phases") or PHASES), list(state.get("phase_keys") or PHASE_KEYS)
    sections = []
    for name in state["plan"]:
        record = state["cells"].get(name) or {"name": name, "status": "not run"}
        record.setdefault("name", name)
        images = (collect_cell_images(record, img_dir, work_dir=work_dir, budget=budget)
                  if record.get("status") == "done" else {})
        sections.append(_cell_html(record, images, budget, phases, keys))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>the frame route bake-off — {state['unit']} / {state['action']}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>The frame route bake-off — {state['unit']} × {state['action']}, two routes × two rigs</h1>
<p class="caption">{state.get('updated', '')} · seed <b>{state['seed']}</b> in every cell · budget {budget} frames ·
matte <code>{state['matte']}</code> · part lock <code>--parts-from {state.get('parts_from')}</code> ·
<a href="http://192.168.1.158:8700/work_packages/274" style="color:#7aa2d6">#274</a>.
<b>V</b> = {ROUTES['V']} (<code>anim clip</code>, skeleton over full-body depth at gain {V_DEPTH_GAIN}, the second gate's variant (b), the clip downsampled to {budget} by an even spread of picks).
<b>K</b> = {ROUTES['K']} (the rig sampled at {budget} instants, three 4-cell pose-pass strips with one seed, joined; the frames are the pose pass).
The ruling is the owner's — this page is evidence, and the two counts are proxies (see the footer).</p>
<div class="toggle"><label><input type="checkbox" id="parts"> show the part masks (SAM 2.1 overlays) in the raw row</label></div>
<h2>Summary</h2>
{_summary_table(state)}
<h2>The cells</h2>
{''.join(sections)}
<h2>What the numbers are</h2>
<p class="note"><b>frames</b>: frames that passed <code>post</code>'s verification, against the budget.
<b>drift</b>: pixels the part palette lock had to move to a member of that part's palette — 0 means the render already agreed with the base sprite.
<b>not found</b>: frames where SAM 2.1, prompted from the rig, found no pixels for the part (reported, never invented).
<b>one sword, sword hand</b> (proxy for the second gate's check 3): the sword mask is one blob (islands under {int(SWORD_ISLAND * 100)} % of the largest ignored), its nearest pixel to the right wrist is within {int(SWORD_REACH * 100)} % of the cell height, and it is no nearer the left wrist.
<b>one figure</b> (proxy for check 1): the matte is one blob (islands under {int(FIGURE_ISLAND * 100)} % ignored) and at most one helmet mask.
Both proxies are computed on the HD cells before the downscale, from the same masks the lock used. The phase labels are the attack row of the keyframe phase template (start / wind-up / action / wind-down / end, <code>config.ACTION_PHASES</code>); the frames are an even spread of the clip, not the template's shaped one.</p>
<footer>Generated by <code>anim bakeoff --unit {state['unit']} --action {state['action']}</code>. Every cell is
<code>rig → clip → pick → post --input clip --parts</code> (V) or <code>rig → pick → strip → render → post --parts</code> (K), called as library functions; the jobs are under <code>work/{bakeoff_name(state['unit'], state['action'])}_*</code>.</footer>
<script>
document.getElementById('parts').addEventListener('change', e => document.body.classList.toggle('parts', e.target.checked));
</script>
</body>
</html>
"""
    return write_text_in_place(out_dir / "index.html", html)


# ---------------------------------------------------------------------------
# The markdown report
# ---------------------------------------------------------------------------
QUESTIONS = """\
## The owner's questions

1. **Route.** V (rig → Wan video) or K (keyframe stills → rig-space tween) for the
   action matrix's in-between frames?
2. **Rig.** KayKit (chibi, the art bible's motion source) or Quaternius UAL (realistic
   proportions, what the anchor gate and the second gate ran on)? Open since #261.
3. **Mid-swing drift.** Does the drift on the contact frames need a second reference at
   the contact pose, or a Wan-side LoRA? The two Wan 2.2 pixel LoRAs found target the
   14B i2v, not the 5B Fun-Control in the container (`docs/research/2026-09-20-pixel-art-loras.md`).
4. **Exaggeration.** Squash/stretch, smears and secondary motion on the rig before
   projection (#277 as planned), or as a 2D warp in `post` if the model corrects a
   stretched skeleton (sprite-studio.md §5)?
"""


def report_markdown(state: dict[str, Any]) -> str:
    budget = int(state.get("budget") or BUDGET)
    lines = [f"# The frame route bake-off — {state['unit']} × {state['action']}",
             "",
             f"Generated {state.get('updated', '')} by `anim bakeoff --unit {state['unit']} "
             f"--action {state['action']}`; the page is `{VISUALS_URL}` (source "
             f"`work/{bakeoff_name(state['unit'], state['action'])}/page/`). Seed "
             f"{state['seed']} in every cell, budget {budget}, matte `{state['matte']}`, "
             f"part lock `--parts-from {state.get('parts_from')}`. No verdict here: the "
             f"ruling is the owner's.",
             "",
             "| cell | rig | route | frames | one sword, sword hand | one figure | "
             "drift px (all parts, all frames) | part-not-found cells | GPU s (service clock) | "
             "rig + post s | VRAM GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in state["plan"]:
        r = state["cells"].get(name)
        if not r or r.get("status") != "done":
            lines.append(f"| {name} | | | {(r or {}).get('status', 'not run')} "
                         f"{(r or {}).get('error', '')} | | | | | | | |")
            continue
        m = r["metrics"]
        drift = sum(v["drift_total"] for v in m["parts"].values())
        missing = sum(len(v["missing"]) for v in m["parts"].values())
        lines.append(f"| {name} | {r['rig']} | {ROUTES[r['route']]} | {m['frames']} / {budget} | "
                     f"{m['one_sword']['count']} / {m['one_sword']['of']} | "
                     f"{m['one_figure']['count']} / {m['one_figure']['of']} | {drift} | "
                     f"{missing} | {r['gpu_seconds']} | "
                     f"{(r.get('timings') or {}).get('rig')} + {(r.get('timings') or {}).get('post')} | "
                     f"{_fmt(r.get('vram_used_gb'))} |")
    lines += ["", "## Per part, per cell", ""]
    for name in state["plan"]:
        r = state["cells"].get(name)
        if not r or r.get("status") != "done":
            continue
        lines += [f"### {name}", "",
                  "| part | pixels (mean, found frames) | drift px/frame (min–max) | "
                  "drift total | not found (frames) |", "|---|---|---|---|---|"]
        for part, v in r["metrics"]["parts"].items():
            drift = f"{v['drift'][0]}–{v['drift'][1]}" if v["drift"] else "—"
            lines.append(f"| {part} | {v['pixels']} | {drift} | {v['drift_total']} | "
                         f"{len(v['missing'])} {v['missing'] if v['missing'] else ''} |")
        sword = r["metrics"]["one_sword"]
        fails = [f"{s['frame']}: {s['reason']}" for s in sword["frames"] if not s["ok"]]
        lines += ["", f"one sword, sword hand: {sword['count']} / {sword['of']}"
                  + (f" — {'; '.join(fails)}" if fails else ""), ""]
    lines += ["", QUESTIONS]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit", "--slug", dest="unit", required=True,
                        help="unit slug, e.g. knight")
    parser.add_argument("--action", required=True, choices=sorted(ACTIONS),
                        help="the studio's action name (attack -> the pipeline's strike)")
    parser.add_argument("--cells", default=None,
                        help="comma-separated subset of the four cells, e.g. "
                             "V-kaykit,K-quaternius (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="redo cells whose outputs exist")
    parser.add_argument("--page-only", action="store_true",
                        help="rebuild the page and report from what is on disk; no GPU")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"one seed for every cell (default {SEED}, the second gate's)")
    parser.add_argument("--matte", default="rembg", choices=sorted(post_stage.MATTES))
    parser.add_argument("--parts-from", dest="parts_from", default=None, metavar="FILE",
                        help="part palettes to lock to (default: the unit's sidecar, "
                             "then work/<unit>_base/post/parts.json - see anim post)")
    parser.add_argument("--reference", default=None,
                        help="identity image for both routes (default: the base render "
                             "for V, the published sprite for K)")


def run_stage(args: argparse.Namespace) -> int:
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()] if args.cells else None
    state = bakeoff(args.unit, args.action, cells=cells, force=args.force,
                    page_only=args.page_only, seed=args.seed, matte=args.matte,
                    parts_from=Path(args.parts_from) if args.parts_from else None,
                    reference=Path(args.reference) if args.reference else None,
                    client=client)
    tally = outcome.count(state["cells"], state["selected"])
    done = [n for n in state["selected"]
            if (state["cells"].get(n) or {}).get("status") == "done"]
    print(f"bakeoff: {tally.done}/{tally.planned} cell(s) done "
          f"({', '.join(done) or 'none'})")
    print(f"  page   -> {state['page']}")
    print(f"  report -> {state['report']}")
    page_dir = Path(state["page"]).parent
    print(f"  publish: mkdir -p {VISUALS_TARGET} && cp -r {page_dir}/. {VISUALS_TARGET}/")
    print(f"  then:    {VISUALS_URL}")
    if args.page_only:
        # A page rebuild ran no cell; it must not inherit the verdict of the
        # run that produced them.
        return 0
    return tally.report()


run = run_stage
