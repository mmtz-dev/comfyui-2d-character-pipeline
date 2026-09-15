"""`anim publish`: the sidecar, the in-place copy, and game-02's real audit.

The audit is not mocked. `tests/conftest.py` copies `tools/asset_audit.py` out
of the mounted game-02 repo into a temp tree, so the sidecar this stage writes
is judged by the code that will actually judge it - which is the only way to
know the schema is right.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from PIL import Image

from anim import config
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import strip as strip_stage


@pytest.fixture
def posted_job(posed_job, reference_sprite, strip_png, palette):
    """A job with four finished frames in post/, ready to publish."""
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    posed_job.render_dir.mkdir(parents=True, exist_ok=True)
    (posed_job.render_dir / "identity_pass.png").write_bytes(strip_png.read_bytes())
    posed_job.update_manifest(render={"seed": 4242, "final": "identity_pass.png"})
    (posed_job.provenance_path).write_text(json.dumps({
        "schema": "anim.provenance/1",
        "seed": 4242,
        "comfyui": {"version": "0.35.0-fake", "devices": ["A5000"]},
        "models": [{"repo": "Comfy-Org/z_image_turbo",
                    "file": "z_image_turbo_int8_convrot.safetensors",
                    "sha256": "c" * 64, "license": "apache-2.0",
                    "revision": "deadbeef"}],
        "passes": [{"name": "identity_pass", "workflow": "identity_pass.json",
                    "workflow_sha256": "b" * 64, "file": "identity_pass.png"}],
        "final": "identity_pass.png",
    }))
    post_stage.post(posed_job.name, palette=palette, matte="chroma",
                    work_dir=posed_job.root.parent)
    return posed_job


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------
def test_the_sidecar_declares_a_frame_group():
    from anim.stages.publish import sidecar_for

    meta = sidecar_for(slug="knight", action="strike", frames=4,
                       manifest={"clip_sha256": "a" * 64, "clip": "x.mp4",
                                 "fps": 12},
                       provenance={"seed": 1, "models": [],
                                   "passes": [{"workflow": "identity_pass.json",
                                               "workflow_sha256": "b" * 64}]},
                       picks=[0, 4, 6, 12], report=None)
    assert meta["id"] == "unit/knight_strike"
    assert meta["provenance"] == "ai-generated"
    assert meta["status"] == "placeholder"
    assert meta["contract"]["frames"] == [4, 1]
    assert meta["contract"]["canvas"] == list(config.CANVAS)
    assert meta["contract"]["pivot"] == "bottom-center"
    assert meta["generation"]["seed"] == 1
    assert meta["generation"]["workflow"] == "identity_pass.json"
    assert meta["generation"]["picks"] == [0, 4, 6, 12]
    assert meta["generation"]["clip_sha256"] == "a" * 64


def test_the_output_licence_is_cc0_and_the_weights_licence_is_recorded():
    """asset_audit.py's ALLOWED_LICENCES is CC0 only - see the module docstring."""
    from anim.stages.publish import sidecar_for

    meta = sidecar_for(slug="knight", action="strike", frames=2, manifest={},
                       provenance={"models": [{"repo": "r", "file": "f",
                                               "license": "apache-2.0"}]},
                       picks=[], report=None)
    assert meta["source"]["license"] == "CC0-1.0"
    assert meta["source"]["model_license"] == "Apache-2.0"
    assert meta["source"]["models"][0]["license"] == "apache-2.0"
    assert "commercially usable" in meta["source"]["license_note"]


def test_frame_names_use_the_reserved_underscore_number_spelling():
    from anim.stages.publish import frame_names

    assert frame_names("knight_strike", 4) == [f"knight_strike_{i}.png"
                                               for i in range(4)]
    assert frame_names("knight", 1) == ["knight.png"], \
        "a single asset keeps its bare name - _0 would be an orphan frame"


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def test_publish_writes_the_frames_and_one_sidecar(posted_job, assets_tree,
                                                   audit_script):
    from anim.stages.publish import publish

    result = publish(posted_job.name, assets_dir=assets_tree / "assets",
                     audit_script=audit_script,
                     work_dir=posted_job.root.parent)
    units = assets_tree / "assets" / "2d" / "units"
    assert sorted(p.name for p in units.glob("*.png")) == [
        f"knight_strike_{i}.png" for i in range(4)]
    assert (units / "knight_strike.json").exists()
    assert not (units / "knight_strike.png").exists(), \
        "a frame group has no bare PNG"
    assert result["audit"]["exit"] == 0


def test_the_published_tree_passes_game_02s_own_audit(posted_job, assets_tree,
                                                      audit_script):
    from anim.stages.publish import publish

    publish(posted_job.name, assets_dir=assets_tree / "assets",
            audit_script=audit_script, work_dir=posted_job.root.parent)
    proc = subprocess.run([sys.executable, str(audit_script)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "frame group" in proc.stdout


def test_the_audit_sees_the_frames_as_one_group(posted_job, assets_tree,
                                                audit_script):
    from anim.stages.publish import publish

    publish(posted_job.name, assets_dir=assets_tree / "assets",
            audit_script=audit_script, work_dir=posted_job.root.parent)
    proc = subprocess.run([sys.executable, str(audit_script), "--list",
                           "--provenance", "ai-generated"],
                          capture_output=True, text=True)
    assert "unit/knight_strike" in proc.stdout
    assert proc.stdout.strip().endswith("4"), "four files, one record"


def test_credits_regenerates_cleanly_for_generated_art(posted_job, assets_tree,
                                                       audit_script):
    from anim.stages.publish import publish

    publish(posted_job.name, assets_dir=assets_tree / "assets",
            audit_script=audit_script, work_dir=posted_job.root.parent)
    proc = subprocess.run([sys.executable, str(audit_script), "--credits"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    credits = (assets_tree / "CREDITS.md").read_text()
    assert "Generated with" in credits
    assert "Comfy-Org/z_image_turbo" in credits


def test_publishing_a_shorter_take_clears_the_stale_frames(posted_job,
                                                           assets_tree,
                                                           audit_script):
    """Leftover _4/_5 would be an orphan frame, which the audit rightly fails."""
    from anim.stages.publish import publish

    units = assets_tree / "assets" / "2d" / "units"
    units.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64)).save(units / "knight_strike_4.png")
    Image.new("RGBA", (64, 64)).save(units / "knight_strike_5.png")
    result = publish(posted_job.name, assets_dir=assets_tree / "assets",
                     audit_script=audit_script, work_dir=posted_job.root.parent)
    assert sorted(result["removed"]) == ["knight_strike_4.png",
                                         "knight_strike_5.png"]
    assert result["audit"]["exit"] == 0


def test_a_pose_variant_is_not_mistaken_for_a_stale_frame(posted_job,
                                                          assets_tree,
                                                          audit_script):
    from anim.stages.publish import publish

    units = assets_tree / "assets" / "2d" / "units"
    units.mkdir(parents=True, exist_ok=True)
    keep = units / "knight_strike_sneak.png"
    Image.new("RGBA", (64, 64)).save(keep)
    publish(posted_job.name, assets_dir=assets_tree / "assets",
            audit_script=audit_script, run_audit=False,
            work_dir=posted_job.root.parent)
    assert keep.exists(), "_sneak is a different asset, not frame number 'sneak'"


def test_frames_are_written_in_place_so_the_inode_survives(posted_job,
                                                           assets_tree,
                                                           audit_script):
    """Docker Desktop's share caches by inode; a rename serves stale bytes."""
    from anim.stages.publish import publish

    units = assets_tree / "assets" / "2d" / "units"
    units.mkdir(parents=True, exist_ok=True)
    target = units / "knight_strike_0.png"
    Image.new("RGBA", (64, 64)).save(target)
    before = target.stat().st_ino
    publish(posted_job.name, assets_dir=assets_tree / "assets",
            audit_script=audit_script, run_audit=False,
            work_dir=posted_job.root.parent)
    assert target.stat().st_ino == before
    assert target.read_bytes() != b""


def test_a_dry_run_writes_nothing(posted_job, assets_tree):
    from anim.stages.publish import publish

    result = publish(posted_job.name, assets_dir=assets_tree / "assets",
                     dry_run=True, work_dir=posted_job.root.parent)
    assert result["dry_run"] is True
    assert not list((assets_tree / "assets" / "2d" / "units").glob("*"))


def test_a_failing_audit_fails_the_stage_loudly(posted_job, assets_tree,
                                                audit_script):
    from anim.stages.publish import publish

    units = assets_tree / "assets" / "2d" / "units"
    units.mkdir(parents=True, exist_ok=True)
    # An asset with no sidecar - exactly what the audit exists to catch.
    Image.new("RGBA", (64, 64)).save(units / "orphan.png")
    with pytest.raises(RuntimeError, match="asset_audit.py exited 1"):
        publish(posted_job.name, assets_dir=assets_tree / "assets",
                audit_script=audit_script, work_dir=posted_job.root.parent)


def test_a_missing_audit_script_tells_the_owner_the_host_command(posted_job,
                                                                 assets_tree,
                                                                 tmp_path):
    from anim.stages.publish import publish

    with pytest.raises(FileNotFoundError, match="stdlib-only"):
        publish(posted_job.name, assets_dir=assets_tree / "assets",
                audit_script=tmp_path / "nope.py",
                work_dir=posted_job.root.parent)


def test_publish_without_finished_frames_says_to_run_post(posed_job,
                                                          assets_tree):
    from anim.stages.publish import publish

    with pytest.raises(FileNotFoundError, match="anim post"):
        publish(posed_job.name, assets_dir=assets_tree / "assets",
                run_audit=False, work_dir=posed_job.root.parent)
