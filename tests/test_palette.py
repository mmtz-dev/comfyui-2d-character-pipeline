"""The palette feature: one project palette from Lospec, sub-palettes per unit,
snapped at export.

No network anywhere: the Lospec fetch is exercised through a fake `urlopen`
(200, 404, unreachable) and, for the API's subprocess, through
`ANIM_LOSPEC_URL` pointed at a file:// directory. `post` runs on the drawn
fixture strip with the chroma matte, as test_post.py does.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from anim import cli, imaging, palettekit
from anim.stages import palette as palette_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import publish as publish_stage
from anim.stages import strip as strip_stage

ENDESGA_32 = {
    "name": "Endesga 32", "author": "ENDESGA",
    "colors": ["be4a2f", "d77643", "ead4aa", "e4a672", "b86f50", "733e39", "3e2731",
               "a22633", "e43b44", "f77622", "feae34", "fee761", "63c74d", "3e8948",
               "265c42", "193c3e", "124e89", "0099db", "2ce8f5", "ffffff", "c0cbdc",
               "8b9bb4", "5a6988", "3a4466", "262b44", "181425", "ff0044", "68386c",
               "b55088", "f6757a", "e8b796", "c28569"],
}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def opener_for(payload: dict | None, *, status: int = 200, unreachable: bool = False):
    """A stand-in for urllib's urlopen: one canned answer, and it records the URL."""
    seen: list[str] = []

    def opener(request, timeout=None):
        seen.append(request.full_url)
        if unreachable:
            raise urllib.error.URLError("Name or service not known")
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "Not Found", {},
                                         io.BytesIO(b'{"error":"palette not found"}'))
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    opener.seen = seen  # type: ignore[attr-defined]
    return opener


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """Where the project palette set is written: a throwaway palette.gpl path."""
    return tmp_path / "assets_src" / "palette.gpl"


@pytest.fixture
def installed(target: Path) -> dict:
    return palettekit.install_lospec("endesga-32", target=target,
                                     opener=opener_for(ENDESGA_32))


# ---------------------------------------------------------------------------
# Parsing and writing
# ---------------------------------------------------------------------------
def test_lospec_hex_is_one_rrggbb_per_line_without_a_hash(tmp_path):
    path = tmp_path / "endesga-32.hex"
    path.write_text("\n".join(ENDESGA_32["colors"]) + "\n")
    colours, meta = palettekit.read_palette_file(path)
    assert colours[0] == (0xBE, 0x4A, 0x2F) and len(colours) == 32
    assert meta == {}
    # tolerant of the other spellings a pasted list arrives in
    assert palettekit.parse_hex_text("#ffffff 0x000000\nBE4A2F, ffffff") == \
        [(255, 255, 255), (0, 0, 0), (0xBE, 0x4A, 0x2F)]


def test_the_gpl_writer_round_trips_and_carries_the_name(tmp_path):
    colours = [palettekit.rgb_of(c) for c in ENDESGA_32["colors"]]
    text = palettekit.gpl_text("Endesga 32", colours, author="ENDESGA",
                               url="https://lospec.com/palette-list/endesga-32")
    back, meta = palettekit.parse_gpl_text(text)
    assert back == colours
    assert meta["name"] == "Endesga 32" and meta["author"] == "ENDESGA"
    assert meta["url"] == "https://lospec.com/palette-list/endesga-32"
    # and it is the .gpl `post` reads, row for row
    (tmp_path / "p.gpl").write_text(text, encoding="utf-8")
    assert imaging.read_palette(tmp_path / "p.gpl") == colours
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text), \
        "no date line - the same palette must be the same bytes"


def test_the_gpl_reader_reads_install_palette_pys_header():
    text = ("GIMP Palette\nName: Lospec 2000\nColumns: 8\n"
            "# Lospec 2000 — 182 colours — Lospec community\n"
            "# Author: Lospec community\n# Pasted hex list from lospec-2000.hex.\n#\n"
            "  0   0   0\t#000000\n 20  20  21\t#141415\n")
    colours, meta = palettekit.parse_gpl_text(text)
    assert colours == [(0, 0, 0), (20, 20, 21)]
    assert meta == {"name": "Lospec 2000", "author": "Lospec community"}


def test_install_from_a_file_writes_the_whole_set_idempotently(tmp_path, target):
    source = tmp_path / "endesga-32.hex"
    source.write_text("\n".join(ENDESGA_32["colors"]) + "\n")
    record = palettekit.install_file(source, target=target)
    paths = palettekit.palette_paths(target)
    assert all(paths[k].exists() for k in ("gpl", "hex", "png", "json"))
    assert record["name"] == "endesga-32", "no header in a .hex: the file's stem"
    assert record["source"] == {"kind": "file", "file": str(source)}
    assert record["count"] == 32 and record["colours"][0] == "#be4a2f"
    assert imaging.read_palette(paths["gpl"]) == imaging.read_palette(paths["png"])
    assert paths["hex"].read_text() == "\n".join(ENDESGA_32["colors"]) + "\n"
    before = {k: paths[k].read_bytes() for k in ("gpl", "hex", "png")}
    again = palettekit.install_file(source, target=target)
    assert {k: paths[k].read_bytes() for k in before} == before
    assert {k: v for k, v in again.items() if k != "fetched_at"} == \
        {k: v for k, v in record.items() if k != "fetched_at"}
    assert json.loads(paths["json"].read_text()) == again


def test_a_gpl_source_keeps_its_own_name_and_author(tmp_path, target):
    source = tmp_path / "grim.gpl"
    source.write_text(palettekit.gpl_text("Grim 32", [(1, 2, 3), (4, 5, 6)], author="me"))
    record = palettekit.install_file(source, target=target)
    assert record["name"] == "Grim 32" and record["author"] == "me"
    assert record["colours"] == ["#010203", "#040506"]
    assert palettekit.install_file(source, name="Other", target=target)["name"] == "Other"


def test_an_empty_or_missing_file_is_a_clear_error(tmp_path, target):
    empty = tmp_path / "empty.hex"
    empty.write_text("nothing here\n")
    with pytest.raises(palettekit.PaletteError, match="no colours"):
        palettekit.install_file(empty, target=target)
    with pytest.raises(palettekit.PaletteError, match="not found"):
        palettekit.install_file(tmp_path / "nope.hex", target=target)


# ---------------------------------------------------------------------------
# Lospec
# ---------------------------------------------------------------------------
def test_fetch_lospec_reads_the_documented_endpoint():
    opener = opener_for(ENDESGA_32)
    fetched = palettekit.fetch_lospec("Endesga-32", opener=opener)
    assert opener.seen == ["https://lospec.com/palette-list/endesga-32.json"]
    assert fetched["name"] == "Endesga 32" and fetched["author"] == "ENDESGA"
    assert fetched["colors"] == ENDESGA_32["colors"] and fetched["slug"] == "endesga-32"


def test_a_404_says_the_slug_is_wrong_and_an_outage_says_no_network():
    with pytest.raises(palettekit.PaletteError, match="no Lospec palette called 'nope'"):
        palettekit.fetch_lospec("nope", opener=opener_for(None, status=404))
    with pytest.raises(palettekit.PaletteError, match="could not reach Lospec"):
        palettekit.fetch_lospec("endesga-32", opener=opener_for(None, unreachable=True))
    with pytest.raises(palettekit.PaletteError, match="not a Lospec slug"):
        palettekit.fetch_lospec("../etc", opener=opener_for(ENDESGA_32))
    with pytest.raises(palettekit.PaletteError, match="no `colors`"):
        palettekit.fetch_lospec("odd", opener=opener_for({"name": "odd"}))


def test_install_lospec_records_the_slug_and_page(installed, target):
    assert installed["name"] == "Endesga 32" and installed["author"] == "ENDESGA"
    assert installed["source"]["kind"] == "lospec"
    assert installed["source"]["slug"] == "endesga-32"
    assert installed["source"]["url"] == "https://lospec.com/palette-list/endesga-32"
    assert installed["fetched_at"].endswith("+00:00")
    assert imaging.read_palette(target) == [palettekit.rgb_of(c) for c in ENDESGA_32["colors"]]
    gpl = target.read_text()
    assert "Name: Endesga 32" in gpl and "# Author: ENDESGA" in gpl
    assert palettekit.load(target) == installed


def test_load_derives_a_record_from_a_bare_gpl_and_none_from_nothing(tmp_path, target):
    assert palettekit.load(target) is None
    target.parent.mkdir(parents=True)
    target.write_text(palettekit.gpl_text("Hand Made", [(9, 9, 9)], author="owner"))
    record = palettekit.load(target)
    assert record["name"] == "Hand Made" and record["colours"] == ["#090909"]
    assert record["source"]["kind"] == "file" and record["fetched_at"] is None
    assert palettekit.summary(target) == {
        "path": str(target), "name": "Hand Made", "colours": ["#090909"],
        "source": {"kind": "file", "file": str(target)}}


# ---------------------------------------------------------------------------
# Unit sub-palettes
# ---------------------------------------------------------------------------
def test_a_typed_sub_palette_must_be_a_subset(installed, target):
    record = palettekit.write_unit_palette(
        "knight", palettekit.parse_colour_list("#be4a2f, #ffffff,181425"),
        source={"kind": "colours"}, target=target)
    path = palettekit.unit_palette_path("knight", target)
    assert path == target.parent / "palette_units" / "knight.json"
    assert record["colours"] == ["#be4a2f", "#ffffff", "#181425"]
    assert record["project"] == {"name": "Endesga 32", "count": 32}
    assert json.loads(path.read_text()) == record
    assert palettekit.read_unit_palette("knight", target) == record
    assert palettekit.list_unit_palettes(target) == ["knight"]
    with pytest.raises(palettekit.PaletteError, match="not in the project palette"):
        palettekit.write_unit_palette("knight", [(1, 2, 3)], source={"kind": "colours"},
                                      target=target)
    assert palettekit.read_unit_palette("hound", target) is None
    assert palettekit.remove_unit_palette("knight", target) is True
    assert palettekit.remove_unit_palette("knight", target) is False


def test_a_sub_palette_from_a_sprite_takes_the_colours_it_uses(installed, target, tmp_path):
    sprite = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    sprite.putpixel((1, 1), (0xBE, 0x4A, 0x2F, 255))
    sprite.putpixel((2, 1), (0xFF, 0xFF, 0xFF, 255))
    sprite.putpixel((3, 1), (0xFE, 0xFE, 0xFE, 255))      # nearly white: not a member
    sprite.putpixel((4, 4), (0x18, 0x14, 0x25, 0))        # transparent: ignored
    path = tmp_path / "knight.png"
    sprite.save(path)
    record = palette_stage.set_unit("knight", from_sprite=path, target=target)
    assert record["colours"] == ["#be4a2f", "#ffffff"]
    assert record["snapped"] == ["#fefefe"], "the stray moved to its nearest member"
    assert record["source"] == {"kind": "sprite", "file": str(path)}


def test_unit_colours_refuses_a_sub_palette_the_palette_no_longer_has(installed, target):
    palettekit.write_unit_palette("knight", [(0xBE, 0x4A, 0x2F)],
                                  source={"kind": "colours"}, target=target)
    project = imaging.read_palette(target)
    assert palettekit.unit_colours("knight", project, target) == \
        ([(0xBE, 0x4A, 0x2F)], palettekit.read_unit_palette("knight", target))
    assert palettekit.unit_colours("hound", project, target) == (project, None)
    palettekit.install(project[1:], name="Smaller", source={"kind": "file", "file": "x"},
                       target=target)
    with pytest.raises(palettekit.PaletteError, match="changed since"):
        palettekit.unit_colours("knight", imaging.read_palette(target), target)


def test_without_a_project_palette_there_is_nothing_to_subset(target):
    with pytest.raises(palettekit.PaletteError, match="no project palette"):
        palettekit.write_unit_palette("knight", [(0, 0, 0)], source={}, target=target)


# ---------------------------------------------------------------------------
# post: snap at export, to the unit's colours
# ---------------------------------------------------------------------------
@pytest.fixture
def rendered_job(posed_job, reference_sprite, strip_png):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    posed_job.render_dir.mkdir(parents=True, exist_ok=True)
    (posed_job.render_dir / "identity_pass.png").write_bytes(strip_png.read_bytes())
    posed_job.update_manifest(render={"seed": 4242, "final": "identity_pass.png"})
    return posed_job


def frames_of(job) -> list[bytes]:
    return [p.read_bytes() for p in sorted(job.post_dir.glob("knight_strike_*.png"))]


def colours_of(job) -> set:
    out: set = set()
    for path in sorted(job.post_dir.glob("knight_strike_*.png")):
        with Image.open(path) as img:
            out |= imaging.opaque_colours(img.convert("RGBA"))
    return out


def test_post_snaps_to_the_unit_sub_palette_when_there_is_one(rendered_job, installed,
                                                              target):
    wd = rendered_job.root.parent
    post_stage.post(rendered_job.name, palette=target, matte="chroma", work_dir=wd)
    whole = frames_of(rendered_job)
    whole_colours = colours_of(rendered_job)
    assert len(whole_colours) > 3
    report = json.loads(rendered_job.report_path.read_text())
    assert report["palette"]["unit_palette"] is None and report["palette"]["snap"] is True
    assert report["palette"]["colours"] == report["palette"]["project_colours"] == 32

    subset = [(0x18, 0x14, 0x25), (0xFF, 0xFF, 0xFF), (0x3A, 0x44, 0x66)]
    palettekit.write_unit_palette("knight", subset, source={"kind": "colours"},
                                  target=target)
    report = post_stage.post(rendered_job.name, palette=target, matte="chroma",
                             work_dir=wd)
    assert colours_of(rendered_job) <= set(subset)
    assert frames_of(rendered_job) != whole
    assert report["palette"]["colours"] == 3 and report["palette"]["project_colours"] == 32
    assert report["palette"]["unit_palette"]["path"].endswith("palette_units/knight.json")
    assert rendered_job.read_manifest()["post"]["unit_palette"].endswith("knight.json")
    assert imaging.read_palette(rendered_job.post_dir / "palette.png") == subset, \
        "the -remap source `post` wrote is the subset"

    # the whole project palette again once the file is gone - the same bytes
    palettekit.remove_unit_palette("knight", target)
    post_stage.post(rendered_job.name, palette=target, matte="chroma", work_dir=wd)
    assert frames_of(rendered_job) == whole


def test_a_stale_sub_palette_stops_post_with_the_fix(rendered_job, installed, target):
    palettekit.write_unit_palette("knight", [(0xBE, 0x4A, 0x2F)],
                                  source={"kind": "colours"}, target=target)
    project = imaging.read_palette(target)
    palettekit.install(project[1:], name="Smaller", source={"kind": "file", "file": "x"},
                       target=target)
    with pytest.raises(palettekit.PaletteError, match="anim palette --unit knight"):
        post_stage.post(rendered_job.name, palette=target, matte="chroma",
                        work_dir=rendered_job.root.parent)


def test_no_snap_keeps_the_raw_colours_and_cannot_be_published(rendered_job, installed,
                                                              target, assets_tree):
    wd = rendered_job.root.parent
    report = post_stage.post(rendered_job.name, palette=target, matte="chroma",
                             work_dir=wd, snap=False)
    members = set(imaging.read_palette(target))
    raw = colours_of(rendered_job)
    assert raw - members, "the fixture's colours are not palette members"
    assert report["palette"]["snap"] is False
    assert all(r["problems"] == [] for r in report["frames"]), "still verified for size + alpha"
    assert any("snap off" in s for s in report["frames"][0]["steps"])
    assert not any("remap" in s for s in report["frames"][0]["steps"])
    assert rendered_job.read_manifest()["post"]["snap"] is False
    for path in sorted(rendered_job.post_dir.glob("knight_strike_*.png")):
        with Image.open(path) as img:
            assert img.size == (64, 64) and imaging.alpha_is_binary(img)
    with pytest.raises(RuntimeError, match="--no-snap"):
        publish_stage.publish(rendered_job.name, assets_dir=assets_tree / "assets",
                              run_audit=False, work_dir=wd)
    with pytest.raises(ValueError, match="no palette to lock"):
        post_stage.post(rendered_job.name, palette=target, matte="chroma",
                        work_dir=wd, snap=False, parts=True)


def test_narrowing_part_palettes_moves_outsiders_to_the_nearest_member():
    allowed = [(0, 0, 0), (255, 255, 255), (60, 78, 128)]
    palettes = {"torso": [[60, 78, 128], [62, 80, 130]], "legs": [[250, 250, 250]],
                "sword": []}
    narrowed, moved = post_stage.narrow_part_palettes(palettes, allowed)
    assert moved == 2
    assert narrowed == {"torso": [[60, 78, 128]], "legs": [[255, 255, 255]], "sword": []}
    assert post_stage.narrow_part_palettes(narrowed, allowed) == (narrowed, 0)


def test_generation_never_quantises_only_post_does():
    """The one `-remap` in the package is post's palette_snap; no stage before
    it names a palette PNG or a quantiser."""
    root = Path(post_stage.__file__).resolve().parent
    for name in ("concept", "base", "render", "clip", "strip", "rig"):
        source = (root / f"{name}.py").read_text(encoding="utf-8")
        assert "-remap" not in source and ".quantize(" not in source, name
    assert "-remap" in (root / "post.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def test_the_cli_installs_lists_and_sets_a_unit(tmp_path, target, capsys, monkeypatch):
    source = tmp_path / "endesga-32.hex"
    source.write_text("\n".join(ENDESGA_32["colors"]) + "\n")
    assert cli.main(["palette", "--file", str(source), "--name", "Endesga 32",
                     "--into", str(target)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("palette: Endesga 32: 32 colour(s), from")
    assert f"wrote {target}" in out and "re-run `anim post`" in out

    assert cli.main(["palette", "--list", "--into", str(target)]) == 0
    assert "#be4a2f" in capsys.readouterr().out

    assert cli.main(["palette", "--unit", "Knight", "--colours", "#be4a2f,#ffffff",
                     "--into", str(target)]) == 0
    assert "knight: 2 colour(s) -> " in capsys.readouterr().out
    assert cli.main(["palette", "--unit", "knight", "--into", str(target)]) == 0
    assert "2 of the project palette's 32" in capsys.readouterr().out
    assert cli.main(["palette", "--unit", "knight", "--colours", "#123456",
                     "--into", str(target)]) == 1
    assert "not in the project palette" in capsys.readouterr().err
    assert cli.main(["palette", "--unit", "knight", "--clear", "--into", str(target)]) == 0
    assert palettekit.read_unit_palette("knight", target) is None

    monkeypatch.setattr(palettekit, "fetch_lospec",
                        lambda slug, **kw: (_ for _ in ()).throw(
                            palettekit.PaletteError("no Lospec palette called 'x'")))
    assert cli.main(["palette", "--lospec", "x", "--into", str(target)]) == 1
    assert "no Lospec palette" in capsys.readouterr().err


def test_the_cli_with_nothing_installed_says_so(target, capsys):
    assert cli.main(["palette", "--into", str(target)]) == 1
    assert "none installed" in capsys.readouterr().out
    assert cli.main(["palette", "--into", str(target), "--lospec", "a", "--file", "b"]) == 1
    assert "one of them" in capsys.readouterr().err


def test_post_has_a_no_snap_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["post", "x", "--no-snap"])
    assert args.snap is False
    assert parser.parse_args(["post", "x"]).snap is True
