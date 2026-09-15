"""`anim review`: the click-to-pick behaviour, and the HTTP skin over it."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from anim.stages import pick as pick_stage
from anim.stages import review as review_stage


@pytest.fixture
def app(posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    return review_stage.ReviewApp(posed_job)


def test_state_marks_the_picked_frames(app):
    state = app.state()
    assert len(state["frames"]) == 13
    picked = [f["index"] for f in state["frames"] if f["picked"]]
    assert picked == state["picks"]
    assert state["edited"] is False
    assert state["action"] == "strike"


def test_clicking_an_unpicked_frame_adds_it(app):
    before = app.state()["picks"]
    spare = next(i for i in range(13) if i not in before)
    after = app.toggle(spare)["picks"]
    assert spare in after
    assert set(before) < set(after)


def test_clicking_a_picked_frame_removes_it_and_renumbers_nothing(app):
    picks = app.state()["picks"]
    dropped = picks[1]
    after = app.toggle(dropped)["picks"]
    assert dropped not in after
    assert after == [p for p in picks if p != dropped], \
        "the surviving picks keep their own frame numbers"


def test_toggling_twice_is_a_no_op(app):
    before = app.state()["picks"]
    app.toggle(5)
    assert app.toggle(5)["picks"] == before


def test_a_click_reaches_picks_json_immediately(app):
    spare = next(i for i in range(13) if i not in app.state()["picks"])
    app.toggle(spare)
    on_disk = json.loads(app.job.picks_path.read_text())
    assert spare in on_disk["picks"]
    assert on_disk["n"] == len(on_disk["picks"])


def test_a_hand_edit_is_visible_as_such(app):
    spare = next(i for i in range(13) if i not in app.state()["picks"])
    state = app.toggle(spare)
    assert state["edited"] is True, "so `pick` knows not to clobber it"


def test_toggling_a_frame_that_does_not_exist_is_refused(app):
    with pytest.raises(IndexError):
        app.toggle(999)


def test_regenerate_only_flags_the_job(app):
    """It must NEVER reach the GPU - see the module docstring."""
    state = app.regenerate()
    assert state["rerender_requested"] is True
    assert json.loads(app.job.manifest_path.read_text())["rerender_requested"]
    assert not (app.job.render_dir / "pose_pass.png").exists()


def test_set_picks_replaces_the_whole_list(app):
    state = app.set_picks([2, 0, 2, 11])
    assert state["picks"] == [0, 2, 11], "sorted and de-duplicated"


def test_set_picks_drops_out_of_range_indices(app):
    assert app.set_picks([0, 99, -3])["picks"] == [0]


def test_images_are_served_by_name_and_nothing_else(app):
    name = app.state()["frames"][0]["file"]
    assert app.image("frames", name).startswith(b"\x89PNG")
    for attack in ("../../job.json", "/etc/passwd", "0001.png.bak"):
        with pytest.raises(KeyError):
            app.image("frames", attack)
    with pytest.raises(KeyError):
        app.image("secrets", name)


def test_the_page_is_dark_and_pixelated_and_lists_every_frame(app):
    page = app.page().decode()
    assert "image-rendering:pixelated" in page.replace(" ", "")
    assert "#15171c" in page, "dark ground"
    assert page.count('class="card') == 13
    assert "Request regenerate" in page


def test_a_hand_timing_survives_a_click(app):
    """The strip timing feature: the page rewrites the record it read, so the
    owner's weights are not lost to a pick toggle - and a review started before
    `pick` ran carries empty timing keys rather than none at all."""
    record = json.loads(app.job.picks_path.read_text())
    record["timing"]["weights"] = [1, 2, 6, 3]
    app.job.picks_path.write_text(json.dumps(record))
    app.toggle(3)
    on_disk = json.loads(app.job.picks_path.read_text())
    assert on_disk["timing"]["weights"] == [1, 2, 6, 3]
    assert on_disk["auto_timing"]["weights"] == [2, 3, 4, 3]
    app.job.picks_path.unlink()
    assert app._picks_record()["timing"] is None
    assert "auto_timing" in app._picks_record()


def test_a_review_without_frames_says_to_run_extract(posed_job):
    for frame in posed_job.frames():
        frame.unlink()
    with pytest.raises(FileNotFoundError, match="anim extract"):
        review_stage.ReviewApp(posed_job)


# ---------------------------------------------------------------------------
# The HTTP skin, over a real socket on an ephemeral port. Still no browser.
# ---------------------------------------------------------------------------
@pytest.fixture
def served(app):
    server = review_stage.serve(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", app
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_the_index_page_is_served(served):
    url, _ = served
    with urllib.request.urlopen(f"{url}/", timeout=10) as response:
        assert response.status == 200
        assert b"anim review" in response.read()


def test_a_posted_toggle_changes_picks_json(served):
    url, app = served
    spare = next(i for i in range(13) if i not in app.state()["picks"])
    state = _post(f"{url}/api/toggle", {"frame": spare})
    assert spare in state["picks"]
    assert spare in json.loads(app.job.picks_path.read_text())["picks"]


def test_a_posted_regenerate_flags_the_job(served):
    url, app = served
    assert _post(f"{url}/api/regenerate", {})["rerender_requested"] is True


def test_a_bad_toggle_is_a_400_not_a_traceback(served):
    url, _ = served
    with pytest.raises(urllib.error.HTTPError) as raised:
        _post(f"{url}/api/toggle", {"frame": 999})
    assert raised.value.code == 400


def test_an_unknown_path_is_a_404(served):
    url, _ = served
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(f"{url}/nope", timeout=10)
    assert raised.value.code == 404


def test_frames_are_served_over_http(served):
    url, app = served
    name = app.state()["frames"][0]["file"]
    with urllib.request.urlopen(f"{url}/frames/{name}", timeout=10) as response:
        assert response.headers["Content-Type"] == "image/png"
        assert response.read().startswith(b"\x89PNG")
