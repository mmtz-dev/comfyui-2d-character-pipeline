"""The palette feature: one project palette from Lospec, sub-palettes per unit.

sprite-studio.md S2 "Curation, palette, export": ONE project palette, taken
from Lospec (a `.hex` / `.gpl` file, or fetched by slug), per-character
sub-palette picks, and the snap happens AT EXPORT - `post` - so generation
stays free. This module is the library: parse, write, fetch, and the per-unit
files. `anim palette` (anim/stages/palette.py) is the CLI over it, the API's
`/palette` routes read it, and `post` asks it which colours a unit may use.

THE FILES
---------
The project palette is the set `install_palette.py` in game-02's assets_src/
already writes, so the two tools never disagree about what a palette is:

    palette.gpl    the human-readable source (GIMP palette, RGB triples)
    palette.hex    Lospec's own format: one `rrggbb` per line, no `#`
    palette.png    the one-row PNG `magick -dither None -remap` consumes
    palette.json   name, where it came from (slug / file), the colours as
                   hex, when it was fetched - the record the API serves

all beside whatever `ANIM_PALETTE` names (its stem), and the unit sub-palettes
one file each under `palette_units/<slug>.json` next to them. They are written
by truncating in place (anim.util.write_bytes_in_place) for the same reason
`publish` does: the Mac holds these paths open over the share.

WHY THE .GPL CARRIES NO DATE
----------------------------
`install_palette.py` stamps its .gpl with the day it ran; this writer does not,
so the same input gives the same bytes every time and a re-run is a no-op in
git. The date lives in palette.json (`fetched_at`) where it is data, not a
diff.

WHERE A SUB-PALETTE LIVES, AND WHY NOT IN parts.json
----------------------------------------------------
`post --parts` reads PART palettes (shield greys, torso blues) from the base
sprite's `post/parts.json` or the published sidecar - derived data, per part,
regenerated whenever the base is re-posted. A unit's sub-palette is a DESIGN
DECISION - "the knight is these 14 of the project's 32" - made before any base
exists and outliving every re-render, so it lives with the other art decisions
(the palette, timing.json) in the project's palette directory, not in scratch.
`post` applies it to the unit-level `-remap` AND narrows the part palettes to
it, so the lock can only ever snap to the unit's own colours.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from anim import config, imaging
from anim.util import read_json, write_bytes_in_place, write_json_in_place

RGB = tuple[int, int, int]

PALETTE_SCHEMA = "anim.palette/1"
UNIT_PALETTE_SCHEMA = "anim.unit-palette/1"
UNITS_DIRNAME = "palette_units"

# Lospec's JSON endpoint for one palette: https://lospec.com/palette-list/<slug>.json
# -> {"name": "Endesga 32", "author": "ENDESGA", "colors": ["be4a2f", ...]}
# (lowercase rrggbb, no '#'; a missing slug is a 404 with {"error": "palette
# not found"}; checked 2026-09-21). config.LOSPEC_URL is the template so a test
# can point it at a file:// directory and the API's subprocess never touches
# the network.
_HEX6 = re.compile(r"(?<![0-9a-fA-F])(?:#|0x)?([0-9a-fA-F]{6})(?![0-9a-fA-F])")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class PaletteError(ValueError):
    """A palette that cannot be read, fetched or applied - the message says why."""


# ---------------------------------------------------------------------------
# Colours as text
# ---------------------------------------------------------------------------
def hex_of(colour: Sequence[int]) -> str:
    r, g, b = (int(v) for v in colour[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_of(text: str) -> RGB:
    """`#rrggbb`, `rrggbb` or `0xrrggbb` -> (r, g, b); anything else is a PaletteError."""
    match = _HEX6.fullmatch(text.strip())
    if not match:
        raise PaletteError(f"not a colour: {text!r} (want rrggbb or #rrggbb)")
    h = match.group(1)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def dedupe(colours: Sequence[Sequence[int]]) -> list[RGB]:
    """Order kept, duplicates dropped - the order IS the palette's identity
    on disk (imaging.read_palette keeps it for the same reason)."""
    seen: dict[RGB, None] = {}
    for colour in colours:
        seen.setdefault(tuple(int(v) for v in colour[:3]), None)   # type: ignore[arg-type]
    return list(seen)


def parse_hex_text(text: str) -> list[RGB]:
    """Lospec's `.hex`: one `rrggbb` per line, no `#`. Also tolerates `#`,
    `0x`, and any separator, so a list pasted from a web page parses too."""
    return dedupe([rgb_of(m.group(1)) for m in _HEX6.finditer(text)])


def parse_colour_list(text: str) -> list[RGB]:
    """`--colours '#1a1a1a,#ffffff, 3e8948'` -> colours; every item must parse."""
    items = [item for item in re.split(r"[,\s]+", text.strip()) if item]
    if not items:
        raise PaletteError("no colours given")
    return dedupe([rgb_of(item) for item in items])


def parse_gpl_text(text: str) -> tuple[list[RGB], dict[str, str]]:
    """Colours and the header metadata of a GIMP palette.

    The colour rows are read exactly as imaging._read_gpl reads them (so
    `post` and this module can never disagree about a .gpl); the metadata is
    what install_palette.py writes: `Name:`, a `# Author:` comment, a `# http`
    URL, and the `# <name> - N colours - <author>` line.
    """
    colours: list[RGB] = []
    meta: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("gimp palette") or low.startswith("columns:"):
            continue
        if low.startswith("name:"):
            meta.setdefault("name", line.split(":", 1)[1].strip())
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if body.lower().startswith("author:"):
                meta.setdefault("author", body.split(":", 1)[1].strip())
            elif body.lower().startswith("source:"):
                meta.setdefault("source", body.split(":", 1)[1].strip())
            elif body.startswith("http"):
                meta.setdefault("url", body.split()[0])
            else:
                m = re.match(r"(.+?)\s+[-—]\s+\d+\s+colours?\s+[-—]\s+(.+)$", body)
                if m:
                    meta.setdefault("name", m.group(1).strip())
                    meta.setdefault("author", m.group(2).strip())
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[i]) for i in range(3))
        except ValueError:
            continue
        if all(0 <= v <= 255 for v in (r, g, b)):
            colours.append((r, g, b))
    return dedupe(colours), meta


def read_palette_file(path: Path) -> tuple[list[RGB], dict[str, str]]:
    """(colours, metadata) from a `.hex`, `.gpl` or one-row `.png`."""
    path = Path(path)
    if not path.exists():
        raise PaletteError(f"palette file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".gpl":
        colours, meta = parse_gpl_text(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".png":
        colours, meta = dedupe(imaging.read_palette(path)), {}
    else:
        # .hex, .txt, or anything else that is a list of colours
        colours, meta = parse_hex_text(path.read_text(encoding="utf-8", errors="replace")), {}
    if not colours:
        raise PaletteError(f"no colours found in {path}")
    return colours, meta


# ---------------------------------------------------------------------------
# Writing the set
# ---------------------------------------------------------------------------
def gpl_text(name: str, colours: Sequence[RGB], *, author: str | None = None,
             url: str | None = None, source: str | None = None) -> str:
    """The .gpl in install_palette.py's layout, minus its date line, so the
    same palette is the same bytes however many times it is written."""
    author = author or "unknown"
    lines = ["GIMP Palette", f"Name: {name}", "Columns: 8",
             f"# {name} — {len(colours)} colours — {author}",
             f"# Author: {author}"]
    if url:
        lines.append(f"# {url}")
    if source:
        lines.append(f"# Source: {source}")
    lines.append("# Written by `anim palette` (the palette feature); "
                 "palette.json beside it is the record.")
    lines.append("#")
    for r, g, b in colours:
        lines.append(f"{r:3d} {g:3d} {b:3d}\t{hex_of((r, g, b))}")
    return "\n".join(lines) + "\n"


def hex_text(colours: Sequence[RGB]) -> str:
    return "\n".join(hex_of(c)[1:] for c in colours) + "\n"


def palette_paths(target: Path | None = None) -> dict[str, Path]:
    """Every file of the set for the palette `target` names (default
    config.PALETTE): the stem's .gpl / .hex / .png / .json and the units dir."""
    target = Path(target) if target is not None else config.PALETTE
    stem = target.with_suffix("")
    return {"gpl": stem.with_suffix(".gpl"), "hex": stem.with_suffix(".hex"),
            "png": stem.with_suffix(".png"), "json": stem.with_suffix(".json"),
            "units": target.parent / UNITS_DIRNAME}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def install(colours: Sequence[RGB], *, name: str, source: dict[str, Any],
            author: str | None = None, target: Path | None = None,
            fetched_at: str | None = None) -> dict[str, Any]:
    """Write the whole set in place and return the palette.json record.

    `source` says where the colours came from - `{"kind": "lospec", "slug":
    ..., "url": ...}` or `{"kind": "file", "file": ...}`. Everything but
    palette.json's `fetched_at` is a pure function of the input.
    """
    colours = dedupe(colours)
    if not colours:
        raise PaletteError("a palette needs at least one colour")
    paths = palette_paths(target)
    url = source.get("url")
    origin = (f"lospec {source.get('slug')}" if source.get("kind") == "lospec"
              else f"file {Path(source.get('file', '')).name}" if source.get("file")
              else None)
    write_bytes_in_place(paths["gpl"], gpl_text(name, colours, author=author, url=url,
                                                source=origin).encode("utf-8"))
    write_bytes_in_place(paths["hex"], hex_text(colours).encode("ascii"))
    imaging.palette_png(colours, paths["png"])
    record = {
        "schema": PALETTE_SCHEMA,
        "name": name,
        "author": author,
        "count": len(colours),
        "colours": [hex_of(c) for c in colours],
        "source": dict(source),
        "fetched_at": fetched_at or now_iso(),
        "files": {k: str(v) for k, v in paths.items() if k != "units"},
    }
    write_json_in_place(paths["json"], record)
    return record


# ---------------------------------------------------------------------------
# Lospec
# ---------------------------------------------------------------------------
def lospec_url(slug: str) -> str:
    slug = slug.strip().lower()
    if not _SLUG.match(slug):
        raise PaletteError(f"{slug!r} is not a Lospec slug (the last part of "
                           f"https://lospec.com/palette-list/<slug>, e.g. endesga-32)")
    return config.LOSPEC_URL.format(slug=slug)


def fetch_lospec(slug: str, *, timeout: float = 20.0,
                 opener: Any = None) -> dict[str, Any]:
    """`{"name", "author", "colors": [rrggbb, ...], "slug", "url"}` from Lospec.

    A 404 and an unreachable host are both PaletteError with a message that
    says which; nothing else about the failure is hidden. `opener` is
    urllib's `urlopen` unless a test hands in another callable.
    """
    url = lospec_url(slug)
    slug = slug.strip().lower()
    request = urllib.request.Request(url, headers={
        "User-Agent": "anim-pipeline/palette (+the pose-to-pixel animation pipeline)",
        "Accept": "application/json"})
    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise PaletteError(
                f"no Lospec palette called {slug!r} ({url} is a 404). The slug is "
                f"the last part of the palette's lospec.com URL, e.g. endesga-32.") from None
        raise PaletteError(f"Lospec answered {exc.code} for {url}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise PaletteError(
            f"could not reach Lospec at {url} ({reason}). No network from this "
            f"container? Download the palette's .hex from lospec.com on a machine "
            f"that has one and run `anim palette --file <it>.hex` instead.") from None
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise PaletteError(f"{url} did not return JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("colors"), list) \
            or not data["colors"]:
        raise PaletteError(f"{url} has no `colors` list: {str(data)[:200]}")
    return {"name": str(data.get("name") or slug), "author": str(data.get("author") or ""),
            "colors": [str(c) for c in data["colors"]], "slug": slug, "url": url}


def install_lospec(slug: str, *, target: Path | None = None,
                   opener: Any = None) -> dict[str, Any]:
    fetched = fetch_lospec(slug, opener=opener)
    colours = dedupe([rgb_of(c) for c in fetched["colors"]])
    page = f"https://lospec.com/palette-list/{fetched['slug']}"
    return install(colours, name=fetched["name"], author=fetched["author"] or None,
                   source={"kind": "lospec", "slug": fetched["slug"], "url": page,
                           "json": fetched["url"]},
                   target=target)


def install_file(path: Path, *, name: str | None = None, author: str | None = None,
                 target: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    colours, meta = read_palette_file(path)
    name = name or meta.get("name") or path.stem
    source: dict[str, Any] = {"kind": "file", "file": str(path)}
    if meta.get("url"):
        source["url"] = meta["url"]
    return install(colours, name=name, author=author or meta.get("author"),
                   source=source, target=target)


# ---------------------------------------------------------------------------
# Reading what is installed
# ---------------------------------------------------------------------------
def load(target: Path | None = None) -> dict[str, Any] | None:
    """The palette.json record, or one derived from the .gpl / .png when only
    the palette file exists (a palette installed by install_palette.py or by
    hand); None when there is no palette at all."""
    paths = palette_paths(target)
    target = Path(target) if target is not None else config.PALETTE
    if paths["json"].exists():
        record = read_json(paths["json"])
        if isinstance(record, dict) and record.get("schema") == PALETTE_SCHEMA:
            return record
    source_file = target if target.exists() else (
        paths["gpl"] if paths["gpl"].exists() else
        paths["png"] if paths["png"].exists() else None)
    if source_file is None:
        return None
    colours, meta = read_palette_file(source_file)
    return {
        "schema": PALETTE_SCHEMA,
        "name": meta.get("name") or source_file.stem,
        "author": meta.get("author"),
        "count": len(colours),
        "colours": [hex_of(c) for c in colours],
        "source": {"kind": "file", "file": str(source_file),
                   **({"url": meta["url"]} if meta.get("url") else {})},
        "fetched_at": None,
        "files": {k: str(v) for k, v in paths.items() if k != "units"},
    }


def summary(target: Path | None = None) -> dict[str, Any]:
    """What `GET /settings` shows for `palette`: path, name, colours, source."""
    target = Path(target) if target is not None else config.PALETTE
    record = load(target)
    return {"path": str(target),
            "name": record["name"] if record else None,
            "colours": list(record["colours"]) if record else [],
            "source": record.get("source") if record else None}


# ---------------------------------------------------------------------------
# Unit sub-palettes
# ---------------------------------------------------------------------------
def unit_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        raise PaletteError(f"{slug!r} is not a unit slug")
    return slug


def unit_palette_path(slug: str, target: Path | None = None) -> Path:
    return palette_paths(target)["units"] / f"{unit_slug(slug)}.json"


def read_unit_palette(slug: str, target: Path | None = None) -> dict[str, Any] | None:
    """The unit's record, or None when the unit has no sub-palette (the whole
    project palette then applies)."""
    path = unit_palette_path(slug, target)
    if not path.exists():
        return None
    record = read_json(path)
    if not isinstance(record, dict) or record.get("schema") != UNIT_PALETTE_SCHEMA:
        raise PaletteError(f"{path} is not a unit palette (schema {UNIT_PALETTE_SCHEMA})")
    return record


def list_unit_palettes(target: Path | None = None) -> list[str]:
    units = palette_paths(target)["units"]
    if not units.is_dir():
        return []
    return sorted(p.stem for p in units.glob("*.json"))


def colours_of_sprite(path: Path) -> list[RGB]:
    """The distinct opaque colours a sprite PNG actually uses, sorted."""
    path = Path(path)
    if not path.exists():
        raise PaletteError(f"sprite not found: {path}")
    with Image.open(path) as img:
        colours = imaging.opaque_colours(img.convert("RGBA"))
    if not colours:
        raise PaletteError(f"{path} has no opaque pixels")
    return sorted(colours)


def subset_of(colours: Sequence[RGB], project: Sequence[RGB], *,
              snap: bool) -> tuple[list[RGB], list[RGB]]:
    """(the colours as members of the project palette, the strays).

    With `snap` a stray becomes its nearest member (Oklab, the same metric
    the part lock uses) - for a sprite drawn by hand; without it the strays
    are returned for the caller to refuse - for a typed list, where a colour
    outside the palette is a typo, not a request."""
    members = set(project)
    strays = [c for c in colours if c not in members]
    if strays and snap:
        from anim.partkit import nearest_colours
        import numpy as np

        snapped = nearest_colours(np.asarray(colours, dtype=np.uint8), list(project))
        return dedupe([tuple(int(v) for v in row) for row in snapped]), strays
    return dedupe([c for c in colours if c in members]), strays


def write_unit_palette(slug: str, colours: Sequence[RGB], *, source: dict[str, Any],
                       target: Path | None = None,
                       snap_strays: bool = False) -> dict[str, Any]:
    """The unit's allowed subset of the project palette, written in place.

    Every colour must be a member of the project palette: with `snap_strays`
    an outsider is moved to its nearest member and reported in the record,
    without it the write is refused and the message lists them.
    """
    slug = unit_slug(slug)
    record = load(target)
    if record is None:
        target_shown = Path(target) if target is not None else config.PALETTE
        raise PaletteError(f"no project palette at {target_shown} - install one first "
                           f"(`anim palette --lospec <slug>` or `--file <path>`)")
    project = [rgb_of(c) for c in record["colours"]]
    kept, strays = subset_of(dedupe(colours), project, snap=snap_strays)
    if strays and not snap_strays:
        shown = ", ".join(hex_of(c) for c in strays[:8])
        raise PaletteError(f"{len(strays)} colour(s) are not in the project palette "
                           f"{record['name']!r}: {shown}"
                           + (" ..." if len(strays) > 8 else "")
                           + ". A sub-palette is a subset; pick from `anim palette --list`.")
    if not kept:
        raise PaletteError("the sub-palette would be empty")
    unit_record = {
        "schema": UNIT_PALETTE_SCHEMA,
        "unit": slug,
        "count": len(kept),
        "colours": [hex_of(c) for c in kept],
        "source": dict(source),
        "snapped": [hex_of(c) for c in strays] if strays else [],
        "project": {"name": record["name"], "count": record["count"]},
        "_note": ("The colours this unit may use, a subset of the project palette. "
                  "`post` remaps the unit's frames to these and narrows the part "
                  "palette lock to them; delete this file and the whole project "
                  "palette applies again."),
    }
    write_json_in_place(unit_palette_path(slug, target), unit_record)
    return unit_record


def remove_unit_palette(slug: str, target: Path | None = None) -> bool:
    path = unit_palette_path(slug, target)
    if not path.exists():
        return False
    path.unlink()
    return True


def unit_colours(slug: str, project: Sequence[RGB],
                 target: Path | None = None) -> tuple[list[RGB], dict[str, Any] | None]:
    """What `post` snaps a unit to: (colours, the unit record or None).

    The sub-palette when the unit has one - checked against the project
    palette it is being applied with, because a sub-palette written against
    a palette that has since changed would put colours in a frame that
    verification then rejects - else the whole project palette.
    """
    record = read_unit_palette(slug, target)
    if record is None:
        return list(project), None
    members = set(project)
    colours = [rgb_of(c) for c in record["colours"]]
    strays = [c for c in colours if c not in members]
    if strays:
        shown = ", ".join(hex_of(c) for c in strays[:6])
        raise PaletteError(
            f"{unit_palette_path(slug, target)} names {len(strays)} colour(s) that are "
            f"not in the project palette any more ({shown}); the project palette has "
            f"changed since it was written. Re-run `anim palette --unit {slug} ...` "
            f"or delete the file to use the whole palette.")
    return colours, record
