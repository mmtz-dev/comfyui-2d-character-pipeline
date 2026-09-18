"""`work/<job>/` - the only state the pipeline has.

Every stage reads files and writes files. There is no in-memory pipeline object
handed from stage to stage, and that is deliberate: any stage can be re-run on
its own, a stage can be re-run after the owner hand-edits what the previous one
wrote, and a job that failed at `post` does not cost another GPU render.

    work/<job>/
      job.json              the manifest: clip hash, fps, action, slug, ...
      frames/0001.png ...   `extract`
      pose/keypoints.json   `pose`
      pose/skel_0001.png    `pose`
      picks.json            `pick`, rewritten by `review`
      strip/                `strip`
      depth/0001.png ...    `rig` (the rig props feature; optional)
      clip/frames/0001.png  `clip` (the clip render feature; optional)
      render/               `render`
      post/                 `post`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anim import config
from anim.util import read_json, write_json_in_place

JOB_SCHEMA = "anim.job/1"


@dataclass(frozen=True)
class Job:
    """Where one job's files live. Pure path arithmetic - touches no disk."""

    name: str
    root: Path

    @classmethod
    def open(cls, name: str, work_dir: Path | None = None) -> "Job":
        base = Path(work_dir) if work_dir is not None else config.WORK_DIR
        return cls(name=name, root=base / name)

    # --- directories -------------------------------------------------------
    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def pose_dir(self) -> Path:
        return self.root / "pose"

    @property
    def strip_dir(self) -> Path:
        return self.root / "strip"

    @property
    def depth_dir(self) -> Path:
        """Per-frame depth PNGs from `rig` (the rig props feature). Optional."""
        return self.root / "depth"

    @property
    def clip_dir(self) -> Path:
        return self.root / "clip"

    @property
    def clip_frames_dir(self) -> Path:
        """Generated frames, 1:1 with frames/ (the clip render feature)."""
        return self.clip_dir / "frames"

    @property
    def render_dir(self) -> Path:
        return self.root / "render"

    @property
    def post_dir(self) -> Path:
        return self.root / "post"

    @property
    def depth_dir(self) -> Path:
        # The rig props feature: depth/%04d.png, one per frame, same index as
        # frames/ and pose/skel_. Optional - only a rig job writes it.
        return self.root / "depth"

    # --- files -------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "job.json"

    @property
    def keypoints_path(self) -> Path:
        return self.pose_dir / "keypoints.json"

    @property
    def picks_path(self) -> Path:
        return self.root / "picks.json"

    @property
    def layout_path(self) -> Path:
        return self.strip_dir / "layout.json"

    @property
    def provenance_path(self) -> Path:
        return self.render_dir / "provenance.json"

    @property
    def clip_path(self) -> Path:
        return self.clip_dir / "clip.json"

    @property
    def clip_provenance_path(self) -> Path:
        return self.clip_dir / "provenance.json"

    @property
    def report_path(self) -> Path:
        return self.post_dir / "report.json"

    # --- listings ----------------------------------------------------------
    def frames(self) -> list[Path]:
        """Extracted frames in numeric order (they are zero-padded, so sorted)."""
        if not self.frames_dir.is_dir():
            return []
        return sorted(self.frames_dir.glob("*.png"))

    def skeletons(self) -> list[Path]:
        if not self.pose_dir.is_dir():
            return []
        return sorted(self.pose_dir.glob("skel_*.png"))

    def depths(self) -> list[Path]:
        """`depth/%04d.png` in order, or [] when `rig` wrote no depth."""
        if not self.depth_dir.is_dir():
            return []
        return sorted(self.depth_dir.glob("[0-9][0-9][0-9][0-9].png"))

    def clip_frames(self) -> list[Path]:
        if not self.clip_frames_dir.is_dir():
            return []
        return sorted(self.clip_frames_dir.glob("*.png"))

    def post_frames(self) -> list[Path]:
        if not self.post_dir.is_dir():
            return []
        return sorted(self.post_dir.glob("*.png"))

    # --- manifest ----------------------------------------------------------
    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"no job.json in {self.root} - run `anim extract` first")
        return read_json(self.manifest_path)

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return write_json_in_place(self.manifest_path, manifest)

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        """Merge fields into job.json. Every stage records what it did here."""
        manifest = self.read_manifest()
        manifest.update(fields)
        self.write_manifest(manifest)
        return manifest


def new_manifest(*, clip: str, clip_sha256: str, fps: int, frames: int,
                 action: str, slug: str, frame_size: list[int],
                 crop: list[int] | None, source_fps: float | None,
                 created: str) -> dict[str, Any]:
    """The manifest `extract` writes. Everything downstream reads from here."""
    return {
        "schema": JOB_SCHEMA,
        "clip": clip,
        "clip_sha256": clip_sha256,
        "fps": fps,
        "source_fps": source_fps,
        "frames": frames,
        "frame_size": frame_size,
        "crop": crop,
        "action": action,
        "slug": slug,
        "created": created,
        # Set true by the review page's "regenerate" button and by nothing else.
        # It is a FLAG, not a trigger: the page must never reach the GPU.
        "rerender_requested": False,
    }
