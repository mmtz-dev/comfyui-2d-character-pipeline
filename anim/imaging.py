"""Deterministic image work: palettes, alpha, and the 64x64 canvas contract.

Everything here is a pure function over PIL images. Nothing here shells out and
nothing here is random, which is the whole reason the palette snap and the
recanvas live on the CPU side instead of inside a ComfyUI graph.

Two rules from the art bible are enforced rather than assumed:
  - alpha is BINARY (0 or 255). A 64px sprite with a soft edge is a broken
    sprite, and the audit's in-place swap depends on exact dimensions.
  - every opaque pixel is a palette colour. Verified after the fact, because a
    step that silently introduced a 33rd colour would otherwise only be caught
    by eye months later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from anim import config

RGB = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
def read_palette(path: Path) -> list[RGB]:
    """Colours from a GIMP .gpl or a one-row .png, in file order.

    Order is preserved (and duplicates dropped) so the derived PNG is stable:
    `magick -remap` does not care, but a palette PNG that reshuffles itself
    between runs would make every frame's bytes change for no reason.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"palette not found: {path}\n"
            f"The art bible's winner lives at assets_src/palette.gpl and the "
            f"owner has not picked yet. Point ANIM_PALETTE (or --palette) at a "
            f"candidate meanwhile, e.g. "
            f"/assets_src/palette_candidates/grim32/grim32.gpl")
    if path.suffix.lower() == ".gpl":
        colours = _read_gpl(path)
    else:
        colours = _read_palette_png(path)
    if not colours:
        raise ValueError(f"palette {path} has no colours")
    seen: dict[RGB, None] = {}
    for colour in colours:
        seen.setdefault(colour, None)
    return list(seen)


def _read_gpl(path: Path) -> list[RGB]:
    colours: list[RGB] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith(("gimp palette", "name:", "columns:")):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[i]) for i in range(3))
        except ValueError:
            continue
        colours.append((r, g, b))
    return colours


def _read_palette_png(path: Path) -> list[RGB]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return [tuple(int(v) for v in px) for px in np.asarray(rgb).reshape(-1, 3)]


def palette_png(colours: list[RGB], dest: Path) -> Path:
    """Write the one-row PNG `magick -dither None -remap` consumes."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(colours, dtype=np.uint8).reshape(1, len(colours), 3)
    Image.fromarray(arr, mode="RGB").save(dest)
    return dest


def chroma_key_for(colours: list[RGB]) -> RGB:
    """A colour that is NOT in the palette, for flattening alpha before -remap.

    Pillow falls back to octree quantisation on RGBA and will not honour a fixed
    palette, so the snap has to happen on flat RGB. Flattening onto a palette
    colour would fuse the silhouette edge into the sprite; flattening onto a
    colour the palette cannot represent guarantees every background pixel is
    remapped somewhere harmless, and the binary mask is re-applied afterwards so
    it never survives into the output.

    Magenta first because it is the traditional key and is almost never in a
    lit-dungeon palette; if a palette does contain it, walk the 6x6x6 cube for
    the first free colour. Deterministic either way.
    """
    members = set(colours)
    for candidate in ((255, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)):
        if candidate not in members:
            return candidate
    for r in range(0, 256, 51):
        for g in range(0, 256, 51):
            for b in range(0, 256, 51):
                if (r, g, b) not in members:
                    return (r, g, b)
    raise ValueError("palette saturates the colour cube; no chroma key available")


# ---------------------------------------------------------------------------
# Alpha
# ---------------------------------------------------------------------------
def binarise_alpha(img: Image.Image, threshold: int = 128) -> Image.Image:
    """Force alpha to {0, 255}. Fully transparent pixels are zeroed to black.

    Zeroing the RGB under transparent pixels matters: `verify_frame` only checks
    opaque pixels against the palette, and leaving chroma-key green under the
    alpha would make the PNG's byte content depend on the key colour.
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).copy()
    solid = arr[:, :, 3] >= threshold
    arr[:, :, 3] = np.where(solid, 255, 0).astype(np.uint8)
    arr[~solid] = (0, 0, 0, 0)
    return Image.fromarray(arr, mode="RGBA")


def alpha_is_binary(img: Image.Image) -> bool:
    alpha = np.asarray(img.convert("RGBA"))[:, :, 3]
    return bool(np.isin(alpha, np.array(config.ALPHA_VALUES, dtype=alpha.dtype)).all())


def opaque_colours(img: Image.Image) -> set[RGB]:
    arr = np.asarray(img.convert("RGBA"))
    opaque = arr[arr[:, :, 3] > 0]
    return {tuple(int(v) for v in px[:3]) for px in opaque}


# ---------------------------------------------------------------------------
# Canvas / pivot - the paste rule from assets_src/extract_barrow_assets.py
# ---------------------------------------------------------------------------
def opaque_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the opaque pixels, or None when nothing is opaque.

    The ALPHA channel's bbox, not the image's: an RGBA image whose transparent
    pixels still carry colour has a full-image bbox and would never trim.
    """
    return img.convert("RGBA").getchannel("A").getbbox()


def trim(img: Image.Image) -> Image.Image:
    """Crop to the opaque bounding box. An all-transparent image is returned as is."""
    rgba = img.convert("RGBA")
    box = opaque_box(rgba)
    return rgba if box is None else rgba.crop(box)


def scale_to_height(img: Image.Image, height: int) -> Image.Image:
    """Nearest-neighbour scale to a target height, aspect preserved.

    NEAREST everywhere, always: any smooth filter reintroduces the soft edges
    and the off-palette blend colours the rest of this module exists to prevent.
    """
    if img.height == height or img.height == 0:
        return img
    width = max(1, round(img.width * height / img.height))
    return img.resize((width, height), Image.NEAREST)


def upscale_nearest(img: Image.Image, factor: int) -> Image.Image:
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def recanvas(img: Image.Image, canvas: tuple[int, int] = config.CANVAS,
             pivot: str = "bottom-center") -> Image.Image:
    """Place the sprite on the contract canvas at the contract pivot.

    Bottom-center means the feet sit on the LAST pixel row, not near it: the
    art bible calls out several placeholders that float and read as hovering
    once elevation lifts the token. A sprite wider or taller than the canvas is
    cropped from the centre/top rather than resized, so a caller that forgot to
    scale gets a visibly clipped sprite instead of a silently squashed one.
    """
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    width, height = canvas
    if pivot == "bottom-center":
        left = (width - img.width) // 2
        top = height - img.height
    elif pivot == "center":
        left = (width - img.width) // 2
        top = (height - img.height) // 2
    else:
        raise ValueError(f"unsupported pivot: {pivot}")
    out.paste(img, (left, top))
    return out


def fit_into_cell(img: Image.Image, cell: tuple[int, int],
                  background: RGB | None = None) -> Image.Image:
    """Bottom-center an image inside a cell, scaling DOWN only if it overflows.

    Used by `strip` for the reference sprite: the cell is the model's frame of
    reference, and a reference whose feet sit where the skeleton's feet sit is
    a far stronger identity hint than one floating in the middle.
    """
    scaled = img
    if img.width > cell[0] or img.height > cell[1]:
        ratio = min(cell[0] / img.width, cell[1] / img.height)
        scaled = img.resize((max(1, int(img.width * ratio)),
                             max(1, int(img.height * ratio))), Image.NEAREST)
    mode = "RGBA" if background is None else "RGB"
    fill = (0, 0, 0, 0) if background is None else background
    out = Image.new(mode, cell, fill)
    out.paste(scaled if mode == "RGBA" else scaled.convert("RGB"),
              ((cell[0] - scaled.width) // 2, cell[1] - scaled.height),
              scaled if scaled.mode == "RGBA" else None)
    return out


def largest_integer_scale(img: Image.Image, cell: tuple[int, int]) -> int:
    """Biggest whole-number nearest upscale of `img` that still fits `cell`."""
    if img.width == 0 or img.height == 0:
        return 1
    return max(1, min(cell[0] // img.width, cell[1] // img.height))


# ---------------------------------------------------------------------------
# Verification - run on every published frame, never skipped
# ---------------------------------------------------------------------------
def verify_frame(img: Image.Image, palette: list[RGB],
                 canvas: tuple[int, int] = config.CANVAS) -> list[str]:
    """Return the reasons this frame is not publishable. Empty list == good."""
    problems: list[str] = []
    if img.size != tuple(canvas):
        problems.append(f"dimensions are {img.size[0]}x{img.size[1]}, "
                        f"contract.canvas is {canvas[0]}x{canvas[1]}")
    if not alpha_is_binary(img):
        problems.append("alpha is not binary (must be 0 or 255 only)")
    members = set(palette)
    stray = sorted(opaque_colours(img) - members)
    if stray:
        shown = ", ".join("#%02x%02x%02x" % c for c in stray[:6])
        problems.append(f"{len(stray)} colour(s) outside the palette: {shown}"
                        + (" ..." if len(stray) > 6 else ""))
    return problems
