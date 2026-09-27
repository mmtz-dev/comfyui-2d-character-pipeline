"""The GPU stages book their memory from the shared GPU gatekeeper (anim/gpu.py).

The gatekeeper is faked by the autouse `gatekeeper` fixture in conftest.py; ComfyUI by a
requests-shaped session. A GPU stage books one lease on its first prompt, marks it loaded
once a prompt has finished, and releases it when the stage ends."""

from __future__ import annotations

from anim import cli, comfy, gpu
from anim.api import runs


class FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code, self._payload, self.text = status, payload, ""

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """POST /prompt -> a prompt id; GET /history/<id> -> finished."""

    def __init__(self) -> None:
        self.posts = 0

    def post(self, url, json=None, timeout=None, **kw):
        self.posts += 1
        return FakeResponse(200, {"prompt_id": f"p{self.posts}"})

    def get(self, url, timeout=None, **kw):
        pid = url.rsplit("/", 1)[-1]
        return FakeResponse(200, {pid: {"status": {"completed": True, "status_str": "success"}, "outputs": {}}})


def _prompt(client: comfy.ComfyClient) -> None:
    pid = client.post_prompt({"1": {"class_type": "X", "inputs": {}}})
    client.wait(pid, sleep=lambda s: None)


def test_every_gpu_stage_has_a_profile():
    assert set(gpu.STAGE_PROFILES) == set(runs.GPU_STAGES)
    assert set(gpu.STAGE_PROFILES.values()) == {"anim-zimage", "anim-qwen-edit", "anim-concept", "anim-wan"}


def test_gpu_stage_books_one_lease_marks_it_loaded_and_releases_it(gatekeeper):
    client = comfy.ComfyClient("http://comfy.test", session=FakeSession())
    with gpu.stage("clip"):
        assert gatekeeper.leases == []  # nothing booked until the first prompt
        _prompt(client)
        _prompt(client)
        assert len(gatekeeper.leases) == 1
        lease = gatekeeper.leases[0]
        assert lease.profile == "anim-wan" and not lease.released
        assert lease.kw["service"] == "anim-pipeline-comfyui"
        assert lease.kw["holder"] == "anim-pipeline clip" and lease.kw["cap_torch"] is False
        assert lease.loaded == 1  # once, not once per prompt
    assert lease.released


def test_lease_released_when_the_stage_raises(gatekeeper):
    client = comfy.ComfyClient("http://comfy.test", session=FakeSession())
    try:
        with gpu.stage("render"):
            _prompt(client)
            raise SystemExit(143)  # Cancel's SIGTERM, as anim.cli turns it into SystemExit
    except SystemExit:
        pass
    assert gatekeeper.leases[0].profile == "anim-qwen-edit" and gatekeeper.leases[0].released


def test_cpu_stages_and_code_outside_a_stage_book_nothing(gatekeeper):
    client = comfy.ComfyClient("http://comfy.test", session=FakeSession())
    _prompt(client)  # a script or test using ComfyClient directly
    with gpu.stage("pick"):
        _prompt(client)
    assert gatekeeper.leases == []


def test_cli_wraps_the_stage(gatekeeper, monkeypatch):
    from anim import curation
    from anim.stages import base

    seen = {}

    def fake_run(args):
        _prompt(comfy.ComfyClient("http://comfy.test", session=FakeSession()))
        seen["lease"] = gatekeeper.leases[-1]
        seen["released_during"] = seen["lease"].released
        return 0

    monkeypatch.setattr(base, "run", fake_run)
    monkeypatch.setattr(curation, "gate_for_namespace", lambda stage, args: None)
    assert cli.main(["base", "--unit", "knight"]) == 0
    assert seen["lease"].profile == "anim-zimage"
    assert seen["released_during"] is False and seen["lease"].released
