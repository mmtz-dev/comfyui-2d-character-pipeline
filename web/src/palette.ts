// The project palette out of `GET /settings`. The palette feature (#282)
// owns the settings fields; this reads whatever it exposes and says plainly
// when there is nothing to read. Import/fetch of palettes is not built here.

export interface ProjectPalette {
  name: string;
  colours: string[];
}

export type PaletteAnswer = { palette: ProjectPalette } | { palette: null; reason: string };

/** `#abc`, `#aabbcc`, `aabbcc`, `[170,187,204]` -> `#aabbcc`; anything else -> null. */
export function normaliseColour(value: unknown): string | null {
  if (Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((v) => Number.isInteger(v))) {
    const [r, g, b] = value as number[];
    if ([r, g, b].every((v) => v! >= 0 && v! <= 255)) {
      return "#" + [r, g, b].map((v) => v!.toString(16).padStart(2, "0")).join("");
    }
    return null;
  }
  if (typeof value !== "string") return null;
  const hex = value.trim().replace(/^#/, "").toLowerCase();
  if (/^[0-9a-f]{6}$/.test(hex)) return "#" + hex;
  if (/^[0-9a-f]{3}$/.test(hex)) return "#" + hex.split("").map((c) => c + c).join("");
  return null;
}

function coloursOf(value: unknown): string[] | null {
  if (!Array.isArray(value) || !value.length) return null;
  const colours = value.map(normaliseColour).filter((c): c is string => c !== null);
  return colours.length ? Array.from(new Set(colours)) : null;
}

/**
 * Where the colours might be, in the order tried:
 *   settings.palette = {path, name, colours: [...], source}   the palette feature's shape
 *   settings.palette = {colors: [...]}                        the same, US spelling
 *   settings.palette_colors / palette_colours = [...]         a list beside a path
 *   settings.palette = "/assets_src/x.gpl"                    the pre-#282 path string:
 *                                                             no colours, so no palette
 *   settings.palette = null                                   nothing chosen
 */
export function projectPalette(settings: Record<string, unknown> | null | undefined): PaletteAnswer {
  if (!settings) return { palette: null, reason: "no project settings" };
  const raw = settings.palette;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const record = raw as Record<string, unknown>;
    const colours = coloursOf(record.colours ?? record.colors);
    if (colours) {
      const name = typeof record.name === "string" && record.name.trim() ? record.name.trim() : "project";
      return { palette: { name, colours } };
    }
  }
  const beside = coloursOf(settings.palette_colours ?? settings.palette_colors);
  if (beside) {
    const name = typeof raw === "string" ? baseName(raw) : "project";
    return { palette: { name, colours: beside } };
  }
  if (typeof raw === "string" && raw.trim()) {
    return { palette: null, reason: `no project palette (${raw.trim()} is set, but /settings carries no colours)` };
  }
  return { palette: null, reason: "no project palette" };
}

function baseName(path: string): string {
  const last = path.trim().split(/[\\/]/).pop() ?? path;
  return last.replace(/\.(gpl|png|hex|txt)$/i, "") || "project";
}
