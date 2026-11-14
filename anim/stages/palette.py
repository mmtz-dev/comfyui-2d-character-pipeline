"""Stage: `anim palette` - the project palette and the per-unit sub-palettes

The palette feature (sprite-studio.md S2): ONE project palette from Lospec,
per-character sub-palette picks, snapped at export - `post` - so generation
stays free. Two ways in, one set of files out:

    anim palette --lospec endesga-32              # fetch by slug
    anim palette --file ~/Downloads/lospec-2000.hex   # a .hex, .gpl or one-row .png
    anim palette --list                           # what is installed

    -> palette.gpl  palette.hex  palette.png  palette.json
       beside whatever ANIM_PALETTE names (/assets_src/palette.gpl); the .png is
       what `post`'s `magick -remap` reads, the .json is what the API serves.

    anim palette --unit knight --colours '#141415,#444443,#c9c2a3,...'
    anim palette --unit knight --from-sprite assets/2d/units/knight.png
    anim palette --unit knight --list | --clear
    anim palette --unit wanderer --from-art wanderer_walk [--max-colours 16]

    `--from-art` (the pixel grid downscale feature) takes the colours the
    unit's own art drew - a job's frames sampled on its art grid, before any
    snap - merged down to at most --max-colours. Unlike the other two it may
    leave the project palette: the unit becomes **off-palette**, the file says
    so and every post report repeats it (docs/adr/0001). Default job: the
    unit's base, `<slug>_base`.

    -> palette_units/knight.json, the subset of the project palette the knight
       may use. `post` then remaps the knight's frames to that subset and
       narrows the part palette lock to it; a unit with no file gets the whole
       project palette.

Idempotent: the same input writes the same bytes (only palette.json's
`fetched_at` moves). No stage before `post` reads any of this - `concept`,
`base`, `render` and `clip` generate in free colour on purpose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from anim import config, palettekit

NAME = "palette"
HELP = "the project palette (Lospec slug or file) and per-unit sub-palettes"


def install(*, lospec: str | None = None, file: Path | None = None,
            name: str | None = None, author: str | None = None,
            target: Path | None = None) -> dict[str, Any]:
    """Install the project palette from a Lospec slug or a file."""
    if bool(lospec) == bool(file):
        raise palettekit.PaletteError("give --lospec <slug> or --file <path>, one of them")
    if lospec:
        return palettekit.install_lospec(lospec, target=target)
    return palettekit.install_file(Path(file), name=name, author=author, target=target)


def set_unit(slug: str, *, colours: str | None = None, from_sprite: Path | None = None,
             target: Path | None = None) -> dict[str, Any]:
    """Write a unit's sub-palette from a typed list or from a sprite's pixels."""
    if bool(colours) == bool(from_sprite):
        raise palettekit.PaletteError(
            "give --colours '#hex,#hex,...' or --from-sprite <png>, one of them")
    if colours:
        return palettekit.write_unit_palette(
            slug, palettekit.parse_colour_list(colours),
            source={"kind": "colours"}, target=target)
    # A sprite drawn by hand may hold a colour the palette does not; the
    # nearest member is what the owner meant, and the record says it moved.
    return palettekit.write_unit_palette(
        slug, palettekit.colours_of_sprite(Path(from_sprite)),
        source={"kind": "sprite", "file": str(from_sprite)}, target=target,
        snap_strays=True)


def from_art(slug: str, job_name: str | None = None, *,
             max_colours: int = config.ART_PALETTE_COLOURS, matte: str = "rembg",
             target: Path | None = None, work_dir: Path | None = None,
             grid_measure: Any = None) -> dict[str, Any]:
    """A unit palette from the colours the unit's own art drew: the job's
    frames (the clip's picks when it has a clip, else its strip) matted,
    sampled on the job's art grid, merged to at most `max_colours`. Written
    off-palette - see the module docstring."""
    from anim import gridkit
    from anim.job import Job
    from anim.stages import post as post_stage

    slug = palettekit.unit_slug(slug)
    name = job_name or f"{slug}_base"
    job = Job.open(name, work_dir)
    if matte not in post_stage.MATTES:
        raise palettekit.PaletteError(f"unknown matte {matte!r}; known: "
                                      f"{sorted(post_stage.MATTES)}")
    source = post_stage.CLIP_SOURCE if (job.clip_frames() and job.picks_path.exists()) \
        else None
    cells, _label, route = post_stage.job_cells(job, job.read_manifest(), source,
                                                post_stage.MATTES[matte])
    grid = (grid_measure or gridkit.measure)([hd for _, _, hd in cells])
    if grid is None:
        raise palettekit.PaletteError(
            f"no art grid found in {name}'s frames - --from-art samples one colour "
            f"per art pixel, so it needs pixel art; use --from-sprite instead")
    import numpy as np

    pixels = []
    for _meta, cell, _hd in cells:
        small = np.asarray(gridkit.sample(cell, grid))
        pixels.append(small[small[..., 3] > 0][:, :3])
    colours = palettekit.reduce_colours(np.concatenate(pixels), max_colours)
    return palettekit.write_unit_palette(
        slug, colours, target=target, off_palette=True,
        source={"kind": "art", "job": name, "route": route, "frames": len(cells),
                "pitch": round(grid.pitch, 3), "max_colours": int(max_colours)})


def _print_palette(record: dict[str, Any], *, indent: str = "") -> None:
    source = record.get("source") or {}
    where = (f"lospec {source.get('slug')}" if source.get("kind") == "lospec"
             else source.get("file") or "?")
    author = f" — {record['author']}" if record.get("author") else ""
    print(f"{indent}{record['name']}{author}: {record['count']} colour(s), from {where}")
    colours = list(record["colours"])
    for start in range(0, len(colours), 8):
        print(f"{indent}  " + " ".join(colours[start:start + 8]))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lospec", metavar="SLUG",
                        help="fetch https://lospec.com/palette-list/<slug>.json and "
                             "install it as the project palette")
    parser.add_argument("--file", metavar="PATH",
                        help="install a .hex (Lospec's: one rrggbb per line), a "
                             "GIMP .gpl, or a one-row .png as the project palette")
    parser.add_argument("--name", default=None,
                        help="the palette's name for --file (default: the .gpl "
                             "header, else the file's stem)")
    parser.add_argument("--author", default=None, help="the author, for --file")
    parser.add_argument("--into", dest="target", default=None, metavar="PATH",
                        help=f"the palette .gpl to write (its .hex/.png/.json and "
                             f"palette_units/ land beside it; default "
                             f"{config.PALETTE}, i.e. ANIM_PALETTE)")
    parser.add_argument("--unit", metavar="SLUG",
                        help="act on this unit's sub-palette instead of the "
                             "project palette")
    parser.add_argument("--colours", "--colors", dest="colours", metavar="LIST",
                        help="with --unit: the unit's colours, '#rrggbb,#rrggbb,...' "
                             "(every one a member of the project palette)")
    parser.add_argument("--from-sprite", dest="from_sprite", metavar="PNG",
                        help="with --unit: take the colours the sprite actually "
                             "uses (a colour outside the palette snaps to its "
                             "nearest member, and the record says so)")
    parser.add_argument("--from-art", dest="from_art", metavar="JOB", nargs="?",
                        const="", default=None,
                        help="with --unit: take the colours the unit's own art "
                             "drew in JOB (default <unit>_base), one per art "
                             "pixel. May leave the project palette: the unit "
                             "becomes off-palette, and every post report says so")
    parser.add_argument("--max-colours", "--max-colors", dest="max_colours", type=int,
                        default=config.ART_PALETTE_COLOURS, metavar="N",
                        help=f"with --from-art: at most N colours (default "
                             f"{config.ART_PALETTE_COLOURS})")
    parser.add_argument("--matte", default="rembg",
                        help="with --from-art: how the frames are cut, as for "
                             "`anim post` (rembg, chroma, none)")
    parser.add_argument("--clear", action="store_true",
                        help="with --unit: remove the sub-palette; the whole "
                             "project palette applies again")
    parser.add_argument("--list", action="store_true",
                        help="print the project palette (or, with --unit, that "
                             "unit's sub-palette)")


def run_stage(args: argparse.Namespace) -> int:
    target = Path(args.target) if args.target else None
    try:
        if args.unit:
            slug = palettekit.unit_slug(args.unit)
            if args.clear:
                gone = palettekit.remove_unit_palette(slug, target)
                print(f"palette: {slug}: " + ("sub-palette removed; the whole project "
                                             "palette applies" if gone else
                                             "had no sub-palette"))
                return 0
            if args.from_art is not None:
                record = from_art(slug, args.from_art or None,
                                  max_colours=args.max_colours, matte=args.matte,
                                  target=target)
                path = palettekit.unit_palette_path(slug, target)
                print(f"palette: {slug}: {record['count']} colour(s) from "
                      f"{record['source']['job']}'s art (pitch "
                      f"{record['source']['pitch']}) -> {path}")
                print("  off-palette: these need not be in the project palette, "
                      "and every `post` of this unit will say so")
                print("  " + " ".join(record["colours"]))
                return 0
            if args.list or not (args.colours or args.from_sprite):
                record = palettekit.read_unit_palette(slug, target)
                if record is None:
                    print(f"palette: {slug} has no sub-palette (the whole project "
                          f"palette applies). Set one with --colours or --from-sprite.")
                    return 0
                if record.get("off_palette"):
                    print(f"palette: {slug}: {record['count']} colour(s), OFF-PALETTE "
                          f"(from {record['source'].get('job', 'its art')})")
                else:
                    print(f"palette: {slug}: {record['count']} of the project palette's "
                          f"{record['project']['count']} ({record['project']['name']})")
                colours = list(record["colours"])
                for start in range(0, len(colours), 8):
                    print("  " + " ".join(colours[start:start + 8]))
                return 0
            record = set_unit(slug, colours=args.colours,
                              from_sprite=Path(args.from_sprite) if args.from_sprite else None,
                              target=target)
            path = palettekit.unit_palette_path(slug, target)
            print(f"palette: {slug}: {record['count']} colour(s) -> {path}")
            if record.get("snapped"):
                print(f"  {len(record['snapped'])} colour(s) outside the palette snapped "
                      f"to the nearest member: " + " ".join(record["snapped"][:8]))
            return 0

        if args.lospec or args.file:
            record = install(lospec=args.lospec,
                             file=Path(args.file) if args.file else None,
                             name=args.name, author=args.author, target=target)
            _print_palette(record, indent="palette: ")
            for key in ("gpl", "hex", "png", "json"):
                print(f"  wrote {record['files'][key]}")
            units = palettekit.list_unit_palettes(target)
            if units:
                print(f"  sub-palettes on file: {', '.join(units)} - `post` refuses "
                      f"one whose colours the new palette no longer has")
            print("  reminder: frames do not re-snap themselves; re-run `anim post` "
                  "on every job that should follow the new palette.")
            return 0

        record = palettekit.load(target)
        if record is None:
            shown = target if target is not None else config.PALETTE
            print(f"palette: none installed at {shown}. `anim palette --lospec "
                  f"<slug>` or `--file <path>`.")
            return 1
        _print_palette(record, indent="palette: ")
        units = palettekit.list_unit_palettes(target)
        if units:
            print(f"  sub-palettes: {', '.join(units)}")
        return 0
    except palettekit.PaletteError as exc:
        print(f"palette: {exc}", file=sys.stderr)
        return 1


run = run_stage
