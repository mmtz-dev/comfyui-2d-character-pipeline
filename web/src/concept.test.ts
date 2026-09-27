import { describe, expect, it } from "vitest";
import type { Attempt, ConceptState, ConceptView, ModelsListing, RegistryRow, RunRecord } from "./api";
import {
  advancedConfirm,
  advancedSweep,
  attemptCaption,
  busyText,
  choicesFor,
  choosable,
  hiddenCount,
  familyOf,
  generateBody,
  initialLoras,
  initialModel,
  isActive,
  lockedReason,
  notInstalled,
  offerableLoras,
  reduceConcept,
  refOf,
  rowLabel,
  sameAsDefault,
  seedFor,
  unitSlug,
  mirrorOffer,
  mirroredNote,
  viewGroups,
  visibleAttempts,
  viewLabel,
  type GenerateForm,
} from "./concept";

const attempt = (index: number, extra: Partial<Attempt> = {}): Attempt => ({
  index, file: `${String(index).padStart(2, "0")}.png`, source: "txt2img", prompt: "p", seed: 100 + index,
  model: "z.safetensors", loras: [], parent: null, seconds: 8.1, url: `/jobs/knight_concept/frames/candidates/${index}.png`,
  hidden: false, file_deleted: null, rendered: true, ...extra,
});

// `mirror_of` is what the API derives from the yaws (config.CONCEPT_MIRROR_PAIRS):
// the three pairs, and null for the views on the symmetry plane.
const MIRROR: Record<string, string> = {
  right: "left", left: "right", back_right: "back_left", back_left: "back_right",
  front_right: "front_left", front_left: "front_right",
};

const view = (name: string, group: "main" | "advanced", extra: Partial<ConceptView> = {}): ConceptView => ({
  view: name, yaw: 90, elevation: name === "top" ? "top" : "side", group, prompt: `see ${name}`, default_prompt: `see ${name}`,
  candidates: [], pick: null, hidden_count: 0, pick_stale: false, state: "pending", accepted_file: null,
  mirror_of: MIRROR[name] ?? null, mirrored: false, mirrored_from: null, ...extra,
});

const state = (): ConceptState => ({
  unit: "knight", job: "knight_concept", exists: true, cell: [768, 768], locked: null,
  gates: { base: "accepted" }, defaults: {}, last_seed: 42, parts_phrase: "helm",
  base: { description: "a knight", prompt: "p", sketch: null, candidates: [attempt(0), attempt(1)], pick: 1, hidden_count: 0, pick_stale: false, accepted: true, front_url: "/f.png" },
  views: {
    back: view("back", "main", { candidates: [attempt(0, { source: "views", file: "back_00.png" })] }),
    left: view("left", "main"),
    right: view("right", "advanced"),
    top: view("top", "main"),
    back_right: view("back_right", "advanced"),
  },
  views_stale: false, estimate_seconds: 11, page: "work/_views/knight/index.html",
});

const form = (extra: Partial<GenerateForm> = {}): GenerateForm => ({
  count: 1, lockSeed: false, lastSeed: 42, model: null, loras: null, ...extra,
});

const row = (file: string, role: RegistryRow["role"], extra: Partial<RegistryRow> = {}): RegistryRow => ({
  file, family: "zimage", role, label: file, trigger: null, default_strength: role === "lora" ? 0.8 : null,
  locked: true, repo: "r", license: "apache-2.0", installed: null, ...extra,
});

const listing = (): ModelsListing => ({
  registry: "gpu/models.registry.tsv", probed: false,
  families: {
    zimage: { checkpoint: [row("z.safetensors", "checkpoint")], lora: [row("pixel.safetensors", "lora")], controlnet: [], te: [], vae: [] },
    "qwen-edit": { checkpoint: [row("q.safetensors", "checkpoint", { family: "qwen-edit" })], lora: [], controlnet: [], te: [], vae: [] },
    wan: { checkpoint: [], lora: [], controlnet: [], te: [], vae: [] },
  },
  ops: { txt2img: "zimage", img2img: "zimage", views: "qwen-edit", edit: "qwen-edit" },
  workflow_defaults: { zimage: "z.safetensors", "qwen-edit": "q.safetensors", wan: "w.safetensors" },
  default_lora: { zimage: [{ file: "pixel.safetensors", strength: 0.8 }], "qwen-edit": [], wan: [] },
});

describe("viewGroups", () => {
  it("splits the API's views into the main buttons and the advanced fold, in order", () => {
    const groups = viewGroups(state());
    expect(groups.main.map((v) => v.view)).toEqual(["back", "left", "top"]);
    expect(groups.advanced.map((v) => v.view)).toEqual(["right", "back_right"]);
    expect(viewLabel("left")).toBe("Side");
    expect(viewLabel("top")).toBe("Top-down");
    expect(viewLabel("mystery")).toBe("mystery");
  });
});

describe("seedFor", () => {
  it("resends the last seed only when locked and there is one", () => {
    expect(seedFor(true, 42)).toBe(42);
    expect(seedFor(false, 42)).toBeUndefined();
    expect(seedFor(true, null)).toBeUndefined();
  });
});

describe("generateBody", () => {
  it("txt2img carries the prompt, count and the seed when locked", () => {
    expect(generateBody({ op: "txt2img", prompt: " a knight " }, form({ count: 3, lockSeed: true })))
      .toEqual({ op: "txt2img", count: 3, seed: 42, prompt: "a knight" });
    expect(() => generateBody({ op: "txt2img", prompt: "  " }, form())).toThrow(/prompt/);
    expect(generateBody({ op: "txt2img", prompt: "x" }, form({ count: 9 })).count).toBe(4);
    expect(generateBody({ op: "txt2img", prompt: "x" }, form({ count: 0 })).count).toBe(1);
  });
  it("views takes the list and a prompt for exactly one view", () => {
    expect(generateBody({ op: "views", views: ["back", "left", "top"] }, form()))
      .toEqual({ op: "views", count: 1, views: ["back", "left", "top"] });
    expect(generateBody({ op: "views", views: ["left"], viewPrompt: "from the left" }, form()).view_prompt).toBe("from the left");
    expect(() => generateBody({ op: "views", views: ["left", "back"], viewPrompt: "x" }, form())).toThrow(/one view/);
    expect(() => generateBody({ op: "views", views: [] }, form())).toThrow(/at least one/);
  });
  it("img2img rounds the strength and keeps an optional prompt", () => {
    expect(generateBody({ op: "img2img", ref: "base:01", strength: 0.5555 }, form()))
      .toEqual({ op: "img2img", count: 1, ref: "base:01", strength: 0.56 });
    expect(generateBody({ op: "img2img", ref: "left:00", prompt: "redder" }, form()).prompt).toBe("redder");
    expect(generateBody({ op: "img2img", ref: "base:01" }, form()).strength).toBe(0.55);
    expect(() => generateBody({ op: "img2img", ref: "base:01", strength: 1.5 }, form())).toThrow(/between/);
    expect(() => generateBody({ op: "img2img" }, form())).toThrow(/attempt/);
  });
  it("edit needs an instruction", () => {
    expect(generateBody({ op: "edit", ref: "left:00", instruction: " make the helmet red " }, form()))
      .toEqual({ op: "edit", count: 1, ref: "left:00", instruction: "make the helmet red" });
    expect(() => generateBody({ op: "edit", ref: "left:00" }, form())).toThrow(/instruction/);
  });
  it("carries the model and LoRAs; an emptied list means no LoRA at all; force rides along", () => {
    const body = generateBody({ op: "txt2img", prompt: "x", force: true },
      form({ model: "z.safetensors", loras: [{ file: "a", strength: 0.3 }, { file: "b", strength: 0 }] }));
    expect(body.model).toBe("z.safetensors");
    expect(body.loras).toEqual([{ file: "a", strength: 0.3 }]);
    expect(body.force).toBe(true);
    const none = generateBody({ op: "txt2img", prompt: "x" }, form({ loras: [{ file: "a", strength: 0 }] }));
    expect(none.loras).toEqual([]);
    expect(none.lora_strength).toBe(0);
    expect(generateBody({ op: "txt2img", prompt: "x" }, form({ loras: [] })).loras).toEqual([]);
    expect(generateBody({ op: "txt2img", prompt: "x" }, form()).loras).toBeUndefined();
  });
});

describe("reduceConcept", () => {
  it("a base pick accepts the gate and clears the views", () => {
    const next = reduceConcept(state(), { type: "pick", index: 0 });
    expect(next.base.pick).toBe(0);
    expect(next.gates).toEqual({ base: "accepted" });
    expect(next.views.back!.candidates).toEqual([]);
    expect(next.locked).toBeNull();
    expect(state().views.back!.candidates).toHaveLength(1);
  });
  it("a view pick marks that view accepted", () => {
    const next = reduceConcept(state(), { type: "pickView", view: "back", index: 0 });
    expect(next.views.back!.pick).toBe(0);
    expect(next.views.back!.state).toBe("accepted");
    expect(next.views.left!.state).toBe("pending");
    expect(reduceConcept(state(), { type: "pickView", view: "nope", index: 0 })).toEqual(state());
  });
  it("a view prompt edit is remembered", () => {
    expect(reduceConcept(state(), { type: "viewPrompt", view: "left", prompt: "x" }).views.left!.prompt).toBe("x");
  });
});

describe("captions and busy text", () => {
  it("say where an attempt came from", () => {
    expect(attemptCaption(attempt(0))).toBe("txt2img · seed 100 · 8.1 s");
    expect(attemptCaption(attempt(3, { source: "img2img", parent: { target: "base", index: 1 }, strength: 0.5 })))
      .toBe("img2img of base:01 · × 0.5 · seed 103 · 8.1 s");
    expect(attemptCaption(attempt(4, { source: "upload", seed: null, seconds: null }))).toBe("upload");
    expect(attemptCaption(attempt(5, { rendered: false, seconds: null }))).toBe("txt2img · seed 105 · waiting for the GPU");
    expect(refOf("left", attempt(3))).toBe("left:03");
  });
  it("describe the run holding the GPU", () => {
    const run = { id: "r1", stage: "concept", status: "running", percent: 40 } as RunRecord;
    expect(busyText(run)).toBe("GPU: concept r1 40% - the generate buttons wait for it");
    expect(busyText(null)).toBe("");
    expect(isActive(run)).toBe(true);
    expect(isActive({ ...run, status: "cancelled" })).toBe(false);
  });
});

describe("the model picker", () => {
  it("offers a family's checkpoints and LoRAs and flags what is missing", () => {
    const l = listing();
    expect(familyOf("views", l)).toBe("qwen-edit");
    expect(familyOf("img2img", null)).toBe("zimage");
    expect(choicesFor(l, "zimage").loras.map((r) => r.file)).toEqual(["pixel.safetensors"]);
    expect(choicesFor(null, "zimage")).toEqual({ checkpoints: [], loras: [] });
    expect(rowLabel(row("a", "lora", { label: "A", installed: false }))).toBe("A (not installed)");
    expect(rowLabel(row("a", "lora", { label: "A", locked: false, installed: true }))).toBe("A (unlocked)");
    expect(rowLabel(row("a", "lora", { label: "A" }))).toBe("A");
  });
  it("starts from the unit's default, else the workflow's", () => {
    const s = state();
    expect(initialLoras(s, listing(), "zimage")).toEqual([{ file: "pixel.safetensors", strength: 0.8 }]);
    expect(initialLoras(s, listing(), "qwen-edit")).toEqual([]);
    s.defaults = { zimage: { model: "z.safetensors", loras: [] } };
    expect(initialLoras(s, listing(), "zimage")).toEqual([]);
    expect(initialModel(s, "zimage")).toBe("z.safetensors");
    expect(initialModel(s, "qwen-edit")).toBeNull();
    expect(sameAsDefault(s, "zimage", "z.safetensors", [])).toBe(true);
    expect(sameAsDefault(s, "zimage", null, [])).toBe(false);
    expect(sameAsDefault(s, "zimage", "z.safetensors", [{ file: "a", strength: 1 }])).toBe(false);
    expect(sameAsDefault(state(), "zimage", null, [])).toBe(false);
  });
});

describe("what the GPU service does not have", () => {
  const withQwenLora = (): ModelsListing => {
    const l = listing();
    l.families["qwen-edit"].lora = [row("qwen-pixel.safetensors", "lora",
      { family: "qwen-edit", label: "Qwen pixel", installed: false })];
    l.families.zimage.lora = [row("pixel.safetensors", "lora", { installed: true })];
    return l;
  };
  it("is not offerable and is named before the press", () => {
    const l = withQwenLora();
    expect(offerableLoras(l, "qwen-edit")).toEqual([]);
    expect(offerableLoras(l, "zimage").map((r) => r.file)).toEqual(["pixel.safetensors"]);
    expect(choosable(l.families["qwen-edit"].lora[0]!)).toBe(false);
    expect(choosable(l.families.zimage.lora[0]!)).toBe(true);
    expect(notInstalled(l, "qwen-edit", null, [{ file: "qwen-pixel.safetensors", strength: 0.8 }]))
      .toEqual(["Qwen pixel"]);
    expect(notInstalled(l, "zimage", null, [{ file: "pixel.safetensors", strength: 0.8 }])).toEqual([]);
  });
  it("an unprobed listing stays choosable - the stage checks again before it posts", () => {
    const l = listing();
    expect(l.families["qwen-edit"].lora).toEqual([]);
    expect(choosable(row("a", "lora", { installed: null }))).toBe(true);
    expect(notInstalled(l, "zimage", "z.safetensors", [{ file: "pixel.safetensors", strength: 0.8 }]))
      .toEqual([]);
  });
});

describe("unitSlug", () => {
  it("makes the pipeline's slug out of what the owner typed", () => {
    expect(unitSlug("Cloaked Assassin")).toBe("cloaked_assassin");
    expect(unitSlug("  Barrow Warden  ")).toBe("barrow_warden");
    expect(unitSlug("knight")).toBe("knight");
    expect(unitSlug("hound-2")).toBe("hound-2");
    expect(unitSlug("Fée du bois")).toBe("fee_du_bois");
    expect(unitSlug("knight's squire!")).toBe("knight_s_squire");
    expect(unitSlug("__knight__")).toBe("knight");
    expect(unitSlug("a  b")).toBe("a_b");
  });
  it("is empty when there is nothing to make one of", () => {
    expect(unitSlug("")).toBe("");
    expect(unitSlug("   ")).toBe("");
    expect(unitSlug("!!!")).toBe("");
  });
});

describe("hiding attempts", () => {
  const list = () => [attempt(0), attempt(1, { hidden: true }),
                      attempt(2, { hidden: true, file_deleted: "2026-09-22T10:00:00+00:00", url: null })];
  it("a strip shows what was kept, or everything behind the toggle", () => {
    expect(visibleAttempts(list(), false).map((a) => a.index)).toEqual([0]);
    expect(visibleAttempts(list(), true).map((a) => a.index)).toEqual([0, 1, 2]);
    expect(hiddenCount(list())).toBe(2);
    expect(hiddenCount([attempt(0)])).toBe(0);
  });
  it("the caption says which it is", () => {
    expect(attemptCaption(list()[1]!)).toBe("txt2img · seed 101 · 8.1 s · hidden");
    expect(attemptCaption(list()[2]!)).toBe("txt2img · seed 102 · 8.1 s · file deleted");
    expect(attemptCaption(list()[0]!)).toBe("txt2img · seed 100 · 8.1 s");
  });
  it("the accepted attempt may not be hidden, and says why", () => {
    const s = state();
    expect(lockedReason(s, "base", attempt(1))).toMatch(/accepted base/);
    expect(lockedReason(s, "base", attempt(0))).toBeNull();
    const withPick = { ...s, views: { ...s.views, back: { ...s.views.back!, pick: 0 } } };
    expect(lockedReason(withPick, "back", attempt(0))).toMatch(/the matrix reads it/);
    expect(lockedReason(withPick, "back", attempt(1))).toBeNull();
    expect(lockedReason(s, "nope", attempt(0))).toBeNull();
  });
});

describe("the mirrored view feature", () => {
  const accepted = (name: string, group: "main" | "advanced") =>
    view(name, group, { state: "accepted" as const, accepted_file: `view_${name}.png`, pick: 0 });
  const withLeft = (): ConceptState => {
    const s = state();
    return { ...s, views: { ...s.views, left: accepted("left", "main") } };
  };

  it("offers the button only for a paired view whose partner is accepted", () => {
    const offer = mirrorOffer(withLeft(), withLeft().views.right!);
    expect(offer).toMatchObject({ view: "right", partner: "left", can: true, reason: null });
    expect(offer!.label).toBe("mirror from Side");
    expect(offer!.title).toContain("--mirror-view right");
    expect(offer!.title).toMatch(/no GPU/);
    expect(offer!.title).toMatch(/sword changes hands/);
    // Both ways round: the accepted one can be re-derived from the other only
    // once that other is accepted too.
    expect(mirrorOffer(withLeft(), withLeft().views.left!)!.can).toBe(false);
  });

  it("has no button at all for a view on the symmetry plane", () => {
    const s = withLeft();
    expect(mirrorOffer(s, s.views.back!)).toBeNull();
    expect(mirrorOffer(s, s.views.top!)).toBeNull();
    expect(s.views.back!.mirror_of).toBeNull();
  });

  it("disables it with the reason when the partner is not accepted", () => {
    const s = state();
    // Drawn but never picked onto the base - `hooded_archer`'s gap.
    const drawn: ConceptState = { ...s, views: { ...s.views, left: view("left", "main", { candidates: [attempt(0)] }) } };
    const offer = mirrorOffer(drawn, drawn.views.right!)!;
    expect(offer.can).toBe(false);
    expect(offer.reason).toMatch(/not on the base yet/);
    expect(offer.title).toBe(offer.reason);
    // On the base, but rejected or still pending: a mirror inherits the verdict.
    const pending: ConceptState = { ...s, views: { ...s.views, left: view("left", "main", { accepted_file: "view_left.png", state: "pending" }) } };
    expect(mirrorOffer(pending, pending.views.right!)!.reason).toMatch(/is pending, not accepted/);
    const rejected: ConceptState = { ...s, views: { ...s.views, left: view("left", "main", { accepted_file: "view_left.png", state: "rejected" }) } };
    expect(mirrorOffer(rejected, rejected.views.right!)!.reason).toMatch(/is rejected, not accepted/);
    // A partner that is ITSELF this view flipped would only round-trip.
    const round: ConceptState = { ...s, views: { ...s.views,
      left: { ...accepted("left", "main"), mirrored: true, mirrored_from: "right" } } };
    expect(mirrorOffer(round, round.views.right!)!.reason).toMatch(/itself this one flipped/);
    // A partner the state does not carry at all (an older API, a trimmed set).
    const none: ConceptState = { ...s, views: { back: s.views.back!, right: s.views.right! } };
    expect(mirrorOffer(none, none.views.right!)!.reason).toMatch(/no left view to flip/);
  });

  it("marks a view that is a flip, naming where its pixels came from", () => {
    const s = withLeft();
    expect(mirroredNote(s.views.right!)).toBeNull();
    const flipped = { ...s.views.right!, mirrored: true, mirrored_from: "left",
                      state: "accepted" as const, accepted_file: "view_right.png" };
    expect(mirroredNote(flipped)).toBe("mirrored from Side");
    expect(mirroredNote({ ...flipped, mirrored_from: null })).toBe("mirrored");
  });

  it("folds the press in before the POST answers: accepted with no pick of its own", () => {
    const s = withLeft();
    const next = reduceConcept(s, { type: "mirrorView", view: "right" });
    expect(next.views.right).toMatchObject({ state: "accepted", pick: null, mirrored: true, mirrored_from: "left" });
    expect(s.views.right!.mirrored).toBe(false);
    // Accepting one of the view's OWN attempts undoes the mirror.
    const back = reduceConcept(next, { type: "pickView", view: "right", index: 1 });
    expect(back.views.right).toMatchObject({ state: "accepted", pick: 1, mirrored: false, mirrored_from: null });
    // A view with no partner, and a name that is not a view, change nothing.
    expect(reduceConcept(s, { type: "mirrorView", view: "back" })).toBe(s);
    expect(reduceConcept(s, { type: "mirrorView", view: "sideways" })).toBe(s);
  });
});

describe("advancedSweep", () => {
  // The fold as the API really sends it: all five advanced views.
  const withFold = (extra: Record<string, Partial<ConceptView>> = {}): ConceptState => {
    const s = state();
    const names = ["right", "back_right", "back_left", "front_left", "front_right"];
    return {
      ...s,
      views: {
        ...s.views,
        ...Object.fromEntries(names.map((n) => [n, view(n, "advanced", extra[n] ?? {})])),
      },
    };
  };
  const run = (extra: Partial<RunRecord> = {}) =>
    ({ id: "r1", stage: "concept", status: "running", percent: 40, ...extra }) as RunRecord;

  it("sweeps every advanced view a unit has not accepted, in compass order", () => {
    const sweep = advancedSweep(withFold(), null);
    expect(sweep.views).toEqual(["right", "back_right", "back_left", "front_left", "front_right"]);
    expect(sweep.label).toBe("All advanced (5)");
    expect(sweep.blocked).toBeNull();
  });

  it("leaves an accepted view alone, so a second press tops up the rest", () => {
    const sweep = advancedSweep(withFold({ right: { pick: 0 }, front_left: { pick: 2 } }), null);
    expect(sweep.views).toEqual(["back_right", "back_left", "front_right"]);
    expect(sweep.label).toBe("All advanced (3)");
    expect(sweep.blocked).toBeNull();
  });

  it("a mirrored view counts as unfinished until one of its own is accepted", () => {
    // The mirrored view feature accepts a view with no pick of its own; the
    // sweep would draw it for real, which is the point of pressing it.
    const mirrored = { mirrored: true, mirrored_from: "left", state: "accepted" as const };
    expect(advancedSweep(withFold({ right: mirrored }), null).views).toContain("right");
  });

  it("the base gate wins over everything else", () => {
    const locked = { ...withFold(), locked: "the base is not accepted yet" };
    expect(advancedSweep(locked, null).blocked).toBe("the base is not accepted yet");
    // ... even while a run holds the card.
    expect(advancedSweep(locked, run()).blocked).toBe("the base is not accepted yet");
  });

  it("a live run blocks it, a finished one does not", () => {
    expect(advancedSweep(withFold(), run()).blocked).toBe("the GPU is busy with concept r1; one at a time");
    expect(advancedSweep(withFold(), run({ status: "done" })).blocked).toBeNull();
  });

  it("says so when there is nothing left to render", () => {
    const every = Object.fromEntries(
      ["right", "back_right", "back_left", "front_left", "front_right"].map((n) => [n, { pick: 0 }]));
    const sweep = advancedSweep(withFold(every), null);
    expect(sweep.views).toEqual([]);
    expect(sweep.label).toBe("All advanced (0)");
    expect(sweep.blocked).toMatch(/every advanced view is accepted/);
  });

  it("the confirm names the views, the render total, the warm estimate and the cold penalty", () => {
    const s = withFold();
    const text = advancedConfirm(s, advancedSweep(s, null).views, 2);
    expect(text).toContain("Generate 5 advanced views of knight");
    expect(text).toContain("right, back_right, back_left, front_left, front_right");
    expect(text).toContain("2 attempts each - 10 renders at about 11 s warm");
    expect(text).toContain("roughly 2 min");      // 10 x 11 s
    expect(text).toContain("Add ~3 min");          // the cold Qwen-Image-Edit load
    expect(text).toContain("Cancel keeps the attempts that finished");
  });

  it("the confirm quotes the unit's own measured seconds, and counts one view singly", () => {
    const s = { ...withFold(), estimate_seconds: 30 };
    const text = advancedConfirm(s, ["right"], 1);
    expect(text).toContain("Generate 1 advanced view of knight");
    expect(text).toContain("1 attempt each - 1 render at about 30 s warm");
    expect(text).toContain("roughly 1 min");
  });

  it("clamps a silly count the same way every other press does", () => {
    const s = withFold();
    expect(advancedConfirm(s, ["right"], 99)).toContain(`${4} attempts each - 4 renders`);
  });
});
