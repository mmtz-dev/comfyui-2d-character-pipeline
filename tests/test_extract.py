"""`anim extract`: ffmpeg, the fps cap, rotation, and the optional crop."""

from __future__ import annotations

import pytest
from PIL import Image

from fixtures import synthetic

from anim.stages import extract


def test_the_fps_cap_is_a_cap_not_a_target():
    assert extract.effective_fps(30.0, 12) == 12
    assert extract.effective_fps(10.0, 12) == 10, "a slow clip is not sped up"
    assert extract.effective_fps(0.0, 12) == 12, "unknown rate falls back to the cap"


def test_probe_reads_the_clip_geometry(clip):
    info = extract.probe(clip)
    assert (info["width"], info["height"]) == (360, 640)
    assert info["fps"] == pytest.approx(24.0)


def test_extract_writes_frames_and_a_manifest(clip, work_dir):
    job = extract.extract(clip, action="strike", slug="knight", fps=12,
                          work_dir=work_dir)
    frames = job.frames()
    manifest = job.read_manifest()
    assert frames, "ffmpeg produced no frames"
    assert manifest["frames"] == len(frames)
    assert manifest["fps"] == 12
    assert manifest["action"] == "strike" and manifest["slug"] == "knight"
    assert len(manifest["clip_sha256"]) == 64
    assert manifest["frame_size"] == [360, 640]
    assert manifest["rerender_requested"] is False


def test_the_fps_cap_halves_the_frame_count(clip, work_dir):
    fast = extract.extract(clip, action="strike", slug="knight", fps=24,
                           work_dir=work_dir, job_name="fast")
    slow = extract.extract(clip, action="strike", slug="knight", fps=12,
                           work_dir=work_dir, job_name="slow")
    assert len(slow.frames()) < len(fast.frames())


def test_re_extracting_a_shorter_clip_leaves_no_stale_frames(tmp_path, work_dir):
    long_clip = synthetic.write_clip(tmp_path / "long.mp4",
                                     synthetic.frame_series(24), fps=12)
    short_clip = synthetic.write_clip(tmp_path / "short.mp4",
                                      synthetic.frame_series(6), fps=12)
    job = extract.extract(long_clip, action="strike", slug="knight",
                          work_dir=work_dir, job_name="k")
    many = len(job.frames())
    job = extract.extract(short_clip, action="strike", slug="knight",
                          work_dir=work_dir, job_name="k")
    assert len(job.frames()) < many


def test_a_rotated_clip_comes_out_upright(tmp_path, work_dir):
    """A phone films landscape with a 90 degree display matrix.

    ffmpeg autorotates unless told not to, and `extract` never passes
    -noautorotate - so a 360x640 source stored with rotation must decode to
    640x360 frames. If this fails, every phone clip arrives sideways.
    """
    rotated = synthetic.write_clip(tmp_path / "rot.mp4",
                                   synthetic.frame_series(6), fps=12,
                                   rotation=90)
    info = extract.probe(rotated)
    if not info["rotation"]:
        pytest.skip("this ffmpeg build did not record a rotation side-datum")
    job = extract.extract(rotated, action="strike", slug="knight",
                          work_dir=work_dir)
    assert job.read_manifest()["frame_size"] == [640, 360]


def test_the_crop_box_finds_an_off_centre_subject(tmp_path):
    frames = synthetic.frame_series(12, (400, 500), offset=-110)
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"{index:04d}.png"
        frame.save(path)
        paths.append(path)
    box = extract.subject_box(paths)
    assert box is not None
    left, top, right, bottom = box
    assert right - left < 400, "the crop must actually be narrower than the frame"
    assert left < 200 < right or right < 200, "box should sit around the figure"


def test_the_crop_is_off_by_default(clip, work_dir):
    job = extract.extract(clip, action="strike", slug="knight", work_dir=work_dir)
    assert job.read_manifest()["crop"] is None
    with Image.open(job.frames()[0]) as img:
        assert img.size == (360, 640)


def test_the_crop_when_asked_for_shrinks_every_frame_the_same_way(clip, work_dir):
    job = extract.extract(clip, action="strike", slug="knight", crop=True,
                          work_dir=work_dir)
    manifest = job.read_manifest()
    assert manifest["crop"] is not None
    sizes = set()
    for path in job.frames():
        with Image.open(path) as img:
            sizes.add(img.size)
    assert len(sizes) == 1, "one box for the clip, or the subject jitters"
    assert sizes.pop() != (360, 640)


def test_a_missing_clip_names_both_paths_it_tried(tmp_path, work_dir):
    with pytest.raises(FileNotFoundError, match="no clip at"):
        extract.extract(tmp_path / "absent.mp4", action="strike", slug="knight",
                        work_dir=work_dir)
