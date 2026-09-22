import { describe, expect, it } from "vitest";
import type { Attempt, ConceptState, ConceptView, ModelsListing, RegistryRow, RunRecord } from "./api";
import {
  attemptCaption,
  busyText,
  choicesFor,
  familyOf,
  generateBody,
  initialLoras,
  initialModel,
  isActive,
  reduceConcept,
  refOf,
  rowLabel,
  sameAsDefault,
  seedFor,
  viewGroups,
  viewLabel,
  type GenerateForm,
} from "./concept";

const attempt = (index: number, extra: Partial<Attempt> = {}): Attempt => ({
  index, file: `${String(index).padStart(2, "0")}.png`, source: "txt2img", prompt: "p", seed: 100 + index,
  model: "z.safetensors", loras: [], parent: null, seconds: 8.1, url: `/jobs/knight_concept/frames/candidates/${index}.png`,
  rendered: true, ...extra,
});

const view = (name: string, group: "main" | "advanced", extra: Partial<ConceptView> = {}): ConceptView => ({
  view: name, yaw: 90, elevation: name === "top" ? "top" : "side", group, prompt: `see ${name}`, default_prompt: `see ${name}`,
  candidates: [], pick: null, state: "pending", accepted_file: null, ...extra,
});

const state = (): ConceptState => ({
  unit: "knight", job: "knight_concept", exists: true, cell: [768, 768], locked: null,
  gates: { base: "accepted", turnaround: "pending" }, defaults: {}, last_seed: 42, parts_phrase: "helm",
  base: { description: "a knight", prompt: "p", sketch: null, candidates: [attempt(0), attempt(1)], pick: 1, accepted: true, front_url: "/f.png" },
  views: {
    back: view("back", "main", { candidates: [attempt(0, { source: "views", file: "back_00.png" })] }),
    left: view("left", "main"),
    right: view("right", "advanced"),
    top: view("top", "main"),
    back_right: view("back_right", "advanced"),
  },
  views_stale: false, page: "work/_views/knight/index.html",
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
    expect(next.gates).toEqual({ base: "accepted", turnaround: "pending" });
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
