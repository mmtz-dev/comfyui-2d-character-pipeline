"""`anim run`: the whole chain, and the polite stop when the GPU is off."""

from __future__ import annotations

import argparse

import pytest

from fixtures.fakecomfy import FakeComfy

from anim import comfy, config
from anim.stages import run as run_stage


def _args(clip, **overrides):
    defaults = dict(clip=str(clip), slug="knight", action="strike", fps=12,
                    crop=False, n=None, seed=4242, palette=None, matte="chroma",
                    reference=None, job=None, force=False, require_gpu=False,
                    audit=False, pixelfixer=True, comfyui_url=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def patched_pose(monkeypatch, keypoints_payload):
    """Swap DWPose inference for the canned keypoints - no 200 MB download.

    Everything else in the chain is the real code; only the model is stubbed.
    """
    from anim.stages import pose as pose_stage

    def fake_estimate(frames, **kwargs):
        payload = dict(keypoints_payload)
        entries = []
        for index, path in enumerate(frames):
            source = keypoints_payload["frames"][index % len(
                keypoints_payload["frames"])]
            entries.append({**source, "index": index, "file": path.name})
        payload["frames"] = entries
        from PIL import Image

        with Image.open(frames[0]) as img:
            payload["source_size"] = [img.width, img.height]
        return payload

    monkeypatch.setattr(pose_stage, "estimate", fake_estimate)
    return fake_estimate


def test_run_stops_before_render_when_the_gpu_is_off(clip, work_dir,
                                                     patched_pose, monkeypatch,
                                                     reference_sprite, capsys):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    code = run_stage.run_stage(_args(clip, comfyui_url="http://127.0.0.1:1",
                                     reference=str(reference_sprite)))
    out = capsys.readouterr().out
    assert code == 0, "the CPU half succeeded; this is the normal case"
    assert "STOPPED BEFORE RENDER" in out
    assert "sudo systemctl start docker" in out
    assert "docker-default" in out
    assert "[4/7] strip" in out and "[5/7]" not in out
    assert not (work_dir / "knight_strike" / "render").exists()


def test_require_gpu_turns_the_polite_stop_into_a_failure(clip, work_dir,
                                                          patched_pose,
                                                          monkeypatch,
                                                          reference_sprite):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    code = run_stage.run_stage(_args(clip, comfyui_url="http://127.0.0.1:1",
                                     require_gpu=True,
                                     reference=str(reference_sprite)))
    assert code == 1


def test_run_leaves_every_cpu_artefact_behind_for_a_re_run(clip, work_dir,
                                                            patched_pose,
                                                            monkeypatch,
                                                            reference_sprite):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    run_stage.run_stage(_args(clip, comfyui_url="http://127.0.0.1:1",
                              reference=str(reference_sprite)))
    job = work_dir / "knight_strike"
    assert (job / "job.json").exists()
    assert (job / "picks.json").exists()
    assert (job / "pose" / "keypoints.json").exists()
    assert (job / "strip" / "skeleton_strip.png").exists()
    assert (job / "strip" / "layout.json").exists()


def test_run_goes_all_the_way_when_the_service_answers(clip, work_dir,
                                                       patched_pose, monkeypatch,
                                                       reference_sprite, palette,
                                                       assets_tree, tmp_path,
                                                       capsys):
    from fixtures import synthetic
    from fixtures.fakecomfy import canned_png

    monkeypatch.setattr(config, "WORK_DIR", work_dir)

    # The fake returns a drawn strip, so `post` has real pixels to work on.
    strip_bytes = synthetic.write_strip(tmp_path / "gen.png").read_bytes()
    with FakeComfy(png=strip_bytes) as fake:
        monkeypatch.setattr(config, "ASSETS_DIR", assets_tree / "assets")
        code = run_stage.run_stage(_args(
            clip, comfyui_url=fake.url, palette=str(palette),
            reference=str(reference_sprite)))
    out = capsys.readouterr().out
    assert code == 0
    assert "[7/7] publish" in out
    units = assets_tree / "assets" / "2d" / "units"
    assert sorted(p.name for p in units.glob("*.png")) == [
        f"knight_strike_{i}.png" for i in range(4)]
    assert (units / "knight_strike.json").exists()
    assert canned_png  # imported for the fixture's sake


def test_run_prints_the_review_url(clip, work_dir, patched_pose, monkeypatch,
                                   reference_sprite, capsys):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    run_stage.run_stage(_args(clip, comfyui_url="http://127.0.0.1:1",
                              reference=str(reference_sprite)))
    out = capsys.readouterr().out
    assert f"http://{config.LAN_HOST}:{config.REVIEW_PORT}/" in out
    assert "anim review knight_strike" in out


def test_review_is_deliberately_not_in_the_chain():
    """A server cannot be a step: it would block forever or serve nobody."""
    import inspect

    source = inspect.getsource(run_stage)
    assert "review_stage" not in source
    assert "anim review" in source, "but the command must be printed"
