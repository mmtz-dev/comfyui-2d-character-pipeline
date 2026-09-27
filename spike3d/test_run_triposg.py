"""Tests for run_triposg.py's GPU gatekeeper wiring, with fakes: no GPU, no TripoSG, no weights.

Runs with the other spike3d tests in the CPU pipeline container:

    docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
      run --rm --entrypoint pytest anim spike3d/ -q
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_triposg as rt  # noqa: E402


class FakeLease:
    guarded = True
    id = "L1"
    gb = 10.0

    def __init__(self, log):
        self.log = log

    def mark_loaded(self):
        self.log.append("loaded")

    def release(self):
        self.log.append("released")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class FakeMesh:
    vertices = [0, 1, 2]
    faces = [0]

    def __init__(self, log):
        self.log = log

    def export(self, path):
        self.log.append(f"export {Path(path).suffix}")


@pytest.fixture()
def rgba(tmp_path) -> Path:
    a = np.zeros((32, 32, 4), np.uint8)
    a[8:24, 8:24] = (200, 100, 50, 255)  # opaque square on a transparent ground
    p = tmp_path / "sprite.png"
    Image.fromarray(a, "RGBA").save(p)
    return p


@pytest.fixture()
def fakes(monkeypatch):
    log: list[str] = []
    calls: list[tuple] = []

    def acquire(profile, **kw):
        calls.append((profile, kw))
        log.append("granted")
        return FakeLease(log)

    monkeypatch.setattr(rt.gpu_lease, "acquire", acquire)
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda i: "Fake GPU"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return log, calls


def test_run_books_lease_after_weights_marks_loaded_and_releases(fakes, rgba, tmp_path, monkeypatch):
    log, calls = fakes
    weights = tmp_path / "w"
    weights.mkdir()

    def generate(image, wdir, seed, steps, guidance, faces, on_loaded):
        log.append("model load")
        on_loaded()
        log.append("infer")
        return FakeMesh(log)

    monkeypatch.setattr(rt, "generate", generate)
    rc = rt.main(["--image-input", str(rgba), "--output-path", str(tmp_path / "o.glb"),
                  "--weights", str(weights), "--skip-download"])
    assert rc == 0
    assert log == ["granted", "model load", "loaded", "infer", "released", "export .glb"]
    profile, kw = calls[0]
    assert profile == "anim-triposg"
    assert kw["holder"] == "anim-pipeline spike3d triposg"
    assert kw["cap_torch"] is False  # the seed isn't a measured peak; don't cap to it


def test_missing_weights_books_nothing(fakes, rgba, tmp_path):
    log, calls = fakes
    rc = rt.main(["--image-input", str(rgba), "--weights", str(tmp_path / "none"), "--skip-download"])
    assert rc == 2 and calls == []


def test_dry_run_skips_weights_and_model_and_never_marks_loaded(fakes, rgba, monkeypatch):
    log, calls = fakes
    torch = sys.modules["torch"]
    torch.float16 = "f16"
    torch.ones = lambda shape, device, dtype: types.SimpleNamespace(numel=lambda: 64 * 2**20)
    monkeypatch.setattr(rt, "generate", lambda *a, **k: pytest.fail("dry run loaded the model"))
    assert rt.main(["--image-input", str(rgba), "--dry-run"]) == 0
    # not "loaded": a dry run's tiny peak must never teach the gatekeeper the profile's size
    assert log == ["granted", "released"]
    assert calls[0][0] == "anim-triposg"


def test_non_rgba_input_is_refused_before_any_booking(fakes, tmp_path):
    log, calls = fakes
    p = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8)).save(p)
    with pytest.raises(SystemExit):
        rt.main(["--image-input", str(p), "--dry-run"])
    assert calls == []
