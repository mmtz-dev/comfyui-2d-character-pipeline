// The Concept tab as DOM (the concept generation feature). Top to bottom: the
// unit, the settings row (checkpoint + LoRAs per family, how many per press,
// the seed lock, save as the unit's default), the base (prompt, Generate,
// Upload, every attempt with the accepted one outlined), the views (All /
// Front / Back / Side / Top-down, the advanced fold, a prompt and Generate per
// view, every attempt), and under whichever attempt was clicked an action
// bar: Accept, Regenerate, img2img with its strength, Edit with an
// instruction. Every write goes to the API and the state is fetched again;
// concept.ts folds the click in first so the outline moves at once. Plain
// DOM like the board, for the same reason. While a run holds the GPU every
// generate button waits and one Cancel button does not.

import {
  ApiError,
  cancelRun,
  generateConcept,
  getConcept,
  getModels,
  listCuration,
  listJobs,
  pickConcept,
  pickConceptView,
  putConceptDefaults,
  uploadConcept,
  type Attempt,
  type ConceptState,
  type ConceptView as ViewRow,
  type LoraChoice,
  type ModelFamily,
  type ModelsListing,
  type RegistryRow,
  type RunRecord,
} from "./api";
import {
  MAIN_VIEWS,
  STRENGTH_DEFAULT,
  STRENGTH_MAX,
  STRENGTH_MIN,
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
  viewGroups,
  viewLabel,
  type GenerateAsk,
  type GenerateForm,
} from "./concept";

type Attrs = Record<string, string | boolean | ((event: Event) => void)>;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (typeof value === "function") node.addEventListener(key.replace(/^on/, ""), value);
    else if (typeof value === "boolean") node.toggleAttribute(key, value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function errorText(error: unknown): string {
  if (error instanceof ApiError) return `${error.status}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}

const FAMILIES: { family: ModelFamily; label: string; title: string }[] = [
  { family: "zimage", label: "base", title: "Z-Image-Turbo: the base (txt2img) and classic img2img" },
  { family: "qwen-edit", label: "views / edit", title: "Qwen-Image-Edit: the views of the base and instruction edits" },
];

/** What the panel lends the tab: following a run, the health line. */
export interface ConceptHost {
  follow(run: RunRecord): void;
  refreshHealth(): void;
}

interface Selected {
  target: string;
  attempt: Attempt;
}

/** One family's checkpoint + LoRA rows in the settings block. */
class FamilyPicker {
  readonly root: HTMLElement;
  private readonly modelSelect = el("select", { class: "studio-select small" });
  private readonly loraList = el("div", { class: "concept-loras" });
  private readonly addButton = el("button", { type: "button", class: "studio-button", title: "stack another LoRA", onclick: () => this.add() }, "+ LoRA");
  private readonly saveButton = el("button", { type: "button", class: "studio-button", title: "PUT /concept/{unit}/defaults: remember this for the unit", onclick: () => void this.onSave(this.family) }, "save as unit default");
  private loras: LoraChoice[] = [];
  private touched = false;
  private choices: { checkpoints: RegistryRow[]; loras: RegistryRow[] } = { checkpoints: [], loras: [] };

  constructor(readonly family: ModelFamily, label: string, title: string,
              private readonly onSave: (family: ModelFamily) => Promise<void>,
              private readonly onChange: () => void) {
    this.modelSelect.title = title;
    this.modelSelect.addEventListener("change", () => { this.touched = true; this.onChange(); });
    this.root = el("div", { class: "concept-family", title },
      el("span", { class: "concept-family-label" }, label),
      this.modelSelect, this.loraList, this.addButton, this.saveButton);
  }

  /** Repopulate from the registry and the unit's defaults (a new unit, or a save). */
  reset(listing: ModelsListing | null, state: ConceptState | null): void {
    this.choices = choicesFor(listing, this.family);
    const model = initialModel(state, this.family);
    this.modelSelect.replaceChildren(
      el("option", { value: "" }, `checkpoint: the workflow's own${listing ? ` (${listing.workflow_defaults[this.family]})` : ""}`),
      ...this.choices.checkpoints.map((row) => el("option", { value: row.file, title: row.file }, rowLabel(row))));
    this.modelSelect.value = model && this.choices.checkpoints.some((r) => r.file === model) ? model : "";
    this.loras = initialLoras(state, listing, this.family);
    this.touched = false;
    this.draw(state);
  }

  private draw(state: ConceptState | null): void {
    this.loraList.replaceChildren(...this.loras.map((lora, i) => this.loraRow(lora, i)));
    this.addButton.disabled = !this.choices.loras.length;
    this.saveButton.disabled = !state || sameAsDefault(state, this.family, this.model(), this.loras);
  }

  private loraRow(lora: LoraChoice, i: number): HTMLElement {
    const select = el("select", { class: "studio-select small", title: "the LoRA file (gpu/models.registry.tsv)" },
      ...(this.choices.loras.some((r) => r.file === lora.file) ? [] : [el("option", { value: lora.file }, `${lora.file} (not in the registry)`)]),
      ...this.choices.loras.map((row) => el("option", { value: row.file, title: row.trigger ? `trigger: ${row.trigger}` : row.file }, rowLabel(row))));
    select.value = lora.file;
    select.addEventListener("change", () => {
      const row = this.choices.loras.find((r) => r.file === select.value);
      this.loras[i] = { file: select.value, strength: row?.default_strength ?? lora.strength };
      this.touched = true;
      this.draw(null);
      this.onChange();
    });
    const strength = el("input", { class: "studio-input concept-strength", type: "number", min: "0", max: "2", step: "0.05", title: "strength_model (0 drops it)" });
    strength.value = String(lora.strength);
    strength.addEventListener("input", () => {
      this.loras[i] = { ...this.loras[i]!, strength: Number(strength.value) || 0 };
      this.touched = true;
      this.onChange();
    });
    return el("span", { class: "concept-lora" }, select, strength,
      el("button", { type: "button", class: "studio-button", title: "drop this LoRA", onclick: () => { this.loras.splice(i, 1); this.touched = true; this.draw(null); this.onChange(); } }, "×"));
  }

  private add(): void {
    const unused = this.choices.loras.find((row) => !this.loras.some((l) => l.file === row.file)) ?? this.choices.loras[0];
    if (!unused) return;
    this.loras.push({ file: unused.file, strength: unused.default_strength ?? 1 });
    this.touched = true;
    this.draw(null);
    this.onChange();
  }

  model(): string | null {
    return this.modelSelect.value || null;
  }

  /** null = untouched, the unit's default applies; a list = exactly these. */
  form(): { model: string | null; loras: LoraChoice[] | null } {
    return { model: this.model(), loras: this.touched ? this.loras.map((l) => ({ ...l })) : null };
  }

  chosen(): LoraChoice[] {
    return this.loras.map((l) => ({ ...l }));
  }

  refreshSave(state: ConceptState | null): void {
    this.saveButton.disabled = !state || sameAsDefault(state, this.family, this.model(), this.loras);
  }
}

export class ConceptTab {
  private state: ConceptState | null = null;
  private listing: ModelsListing | null = null;
  private units: string[] = [];
  private selected: Selected | null = null;
  private busy: RunRecord | null = null;
  private viewPrompts = new Map<string, HTMLTextAreaElement>();

  readonly root = el("div", { class: "concept" });
  private readonly unitSelect = el("select", { class: "studio-select", title: "the unit (GET /curation, plus every *_concept job)" });
  private readonly unitInput = el("input", { class: "studio-input small", type: "text", placeholder: "or a new unit slug", title: "a unit that has nothing yet: type its slug and press Enter" });
  private readonly pickers = FAMILIES.map((f) => new FamilyPicker(f.family, f.label, f.title, (family) => this.saveDefaults(family), () => this.refreshSaveButtons()));
  private readonly countSelect = el("select", { class: "studio-select small", title: "attempts per press (each is one /prompt on the card)" },
    ...[1, 2, 3, 4].map((n) => el("option", { value: String(n) }, `× ${n}`)));
  private readonly lockSeed = el("input", { type: "checkbox", title: "lock the seed: resend the last one, so a prompt tweak changes only what you changed" });
  private readonly seedLine = el("span", { class: "concept-seed" });
  private readonly busyLine = el("span", { class: "studio-status warn" });
  private readonly cancelButton = el("button", { type: "button", class: "studio-button", hidden: true, title: "POST /runs/{id}/cancel: stop the stage and ComfyUI's current prompt", onclick: () => void this.cancel() }, "Cancel");
  private readonly statusLine = el("div", { class: "studio-status" });
  // the base
  private readonly prompt = el("textarea", { class: "studio-input concept-prompt", rows: "2", placeholder: "who the character is - \"a plate-armoured knight with a kite shield\"" });
  private readonly generateBase = this.gpuButton("Generate", "anim concept --prompt: the base, txt2img on Z-Image-Turbo", () => void this.generate({ op: "txt2img", prompt: this.prompt.value }));
  private readonly uploadInput = el("input", { type: "file", accept: "image/png", hidden: true, onchange: () => void this.upload() });
  private readonly uploadButton = el("button", { type: "button", class: "studio-button", title: "your own PNG as a base candidate (no GPU)", onclick: () => this.uploadInput.click() }, "Upload PNG");
  private readonly baseStrip = el("div", { class: "concept-strip" });
  private readonly baseActions = el("div", { class: "concept-actions" });
  // the views
  private readonly lockLine = el("div", { class: "studio-status warn" });
  private readonly viewButtons = el("div", { class: "studio-row concept-view-buttons" });
  private readonly viewsBox = el("div", { class: "concept-views" });
  private readonly advancedBox = el("details", { class: "concept-advanced" }, el("summary", {}, "advanced views (right + the four obliques)"));

  constructor(private readonly host: ConceptHost) {
    this.unitSelect.addEventListener("change", () => { this.unitInput.value = ""; void this.load(); });
    this.unitInput.addEventListener("keydown", (event) => {
      if ((event as KeyboardEvent).key === "Enter") { event.preventDefault(); void this.loadTyped(); }
    });
    this.lockSeed.addEventListener("change", () => this.drawSeed());
    this.root.replaceChildren(
      el("div", { class: "studio-row" },
        this.unitSelect,
        el("button", { type: "button", class: "studio-button", title: "list the units again", onclick: () => void this.refreshUnits() }, "↻"),
        this.unitInput),
      el("div", { class: "concept-settings" },
        ...this.pickers.map((p) => p.root),
        el("div", { class: "studio-row" },
          this.countSelect,
          el("label", { class: "concept-lock" }, this.lockSeed, " lock seed"),
          this.seedLine)),
      el("div", { class: "studio-row concept-busy" }, this.busyLine, this.cancelButton),
      this.statusLine,
      el("h3", { class: "concept-h" }, "Base"),
      el("p", { class: "studio-help" }, "The front-facing reference every later stage trusts. Generate from the prompt or upload your own; click an attempt to accept it (that is the base gate) or rework it."),
      this.prompt,
      el("div", { class: "studio-row" }, this.generateBase, this.uploadButton, this.uploadInput),
      this.baseStrip,
      this.baseActions,
      el("h3", { class: "concept-h" }, "Views"),
      el("p", { class: "studio-help" }, "The accepted base seen from the other sides, drawn from it through Qwen-Image-Edit. Front is the base itself; Side is the left profile (the right is mirrored in the sheet). Accept one per view and the turnaround hands each direction its nearest."),
      this.lockLine,
      this.viewButtons,
      this.viewsBox,
      this.advancedBox,
    );
  }

  private gpuButton(label: string, title: string, run: () => void): HTMLButtonElement {
    return el("button", { type: "button", class: "studio-button gpu", title: `${title}\nStarts a GPU job on ComfyUI - one at a time.`, disabled: true, onclick: run }, label);
  }

  // --- loading ---------------------------------------------------------------
  async start(): Promise<void> {
    try {
      this.listing = await getModels();
    } catch (error) {
      this.say(`could not read /models - ${errorText(error)}`, "warn");
    }
    await this.refreshUnits();
  }

  async refreshUnits(): Promise<void> {
    const before = this.unitSelect.value;
    try {
      const [listing, jobs] = await Promise.all([listCuration(), listJobs()]);
      const names = new Set(listing.units.map((u) => u.unit));
      for (const job of jobs) if (job.name.endsWith("_concept") && job.slug) names.add(job.slug);
      this.units = [...names].sort();
      this.unitSelect.replaceChildren(
        el("option", { value: "" }, this.units.length ? "pick a unit" : "no units yet - type a slug"),
        ...this.units.map((unit) => el("option", { value: unit }, unit)));
      if (this.units.includes(before)) this.unitSelect.value = before;
      else if (this.units.length === 1) this.unitSelect.value = this.units[0]!;
    } catch (error) {
      this.say(`could not list units - ${errorText(error)}`, "bad");
      return;
    }
    await this.load();
  }

  private async loadTyped(): Promise<void> {
    const slug = this.unitInput.value.trim().toLowerCase();
    if (!/^[a-z0-9][a-z0-9_-]*$/.test(slug)) {
      this.say("a unit slug is letters, digits, - and _", "bad");
      return;
    }
    if (!this.units.includes(slug)) {
      this.units.push(slug);
      this.units.sort();
      this.unitSelect.append(el("option", { value: slug }, slug));
    }
    this.unitSelect.value = slug;
    await this.load();
  }

  private unit(): string {
    return this.unitSelect.value;
  }

  private async load(): Promise<void> {
    const unit = this.unit();
    this.selected = null;
    if (!unit) {
      this.state = null;
      this.draw();
      return;
    }
    try {
      const state = await getConcept(unit);
      const same = this.state?.unit === unit;
      this.state = state;
      if (!same) {
        for (const picker of this.pickers) picker.reset(this.listing, state);
        this.prompt.value = state.base.description ?? "";
        this.viewPrompts.clear();
      } else {
        this.refreshSaveButtons();
      }
    } catch (error) {
      this.state = null;
      this.say(`could not read /concept/${unit} - ${errorText(error)}`, "bad");
    }
    this.draw();
  }

  /** The panel tells the tab a run ended, so the attempts reload. */
  async runFinished(): Promise<void> {
    this.setBusy(null);
    if (this.state) await this.load();
  }

  /** The run holding the GPU (or null): every generate button waits for it. */
  setBusy(run: RunRecord | null): void {
    this.busy = isActive(run) ? run : null;
    this.busyLine.textContent = busyText(this.busy);
    this.cancelButton.hidden = !this.busy;
    this.cancelButton.disabled = false;
    this.enableButtons();
  }

  /** Every GPU button in the tab: off without a unit, while a run holds the
   * card, or when the base gate locks that one (`.locked`). */
  private enableButtons(): void {
    const off = !this.state || !!this.busy;
    for (const button of this.root.querySelectorAll<HTMLButtonElement>("button.gpu")) {
      button.disabled = off || button.classList.contains("locked");
    }
    this.uploadButton.disabled = !this.state;
  }

  private refreshSaveButtons(): void {
    for (const picker of this.pickers) picker.refreshSave(this.state);
  }

  private say(text: string, kind: "" | "ok" | "warn" | "bad" = ""): void {
    this.statusLine.textContent = text;
    this.statusLine.className = `studio-status ${kind}`.trim();
  }

  // --- the form ----------------------------------------------------------------
  private form(op: GenerateAsk["op"]): GenerateForm {
    const family = familyOf(op, this.listing);
    const picker = this.pickers.find((p) => p.family === family)!;
    return {
      count: Number(this.countSelect.value) || 1,
      lockSeed: this.lockSeed.checked,
      lastSeed: this.state?.last_seed ?? null,
      ...picker.form(),
    };
  }

  private drawSeed(): void {
    const last = this.state?.last_seed;
    this.seedLine.textContent = last === null || last === undefined ? "seed: random" : this.lockSeed.checked ? `seed ${last} (locked)` : `last seed ${last}`;
  }

  // --- writes --------------------------------------------------------------------
  private async generate(ask: GenerateAsk): Promise<void> {
    const state = this.state;
    if (!state) return;
    let body;
    try {
      body = generateBody(ask, this.form(ask.op));
    } catch (error) {
      this.say(errorText(error), "bad");
      return;
    }
    const what = ask.op === "views" ? `${ask.views!.map(viewLabel).join(", ")} view${ask.views!.length === 1 ? "" : "s"}` : ask.op === "txt2img" ? "the base" : `${ask.op} of ${ask.ref}`;
    this.say(`starting ${what} ...`);
    try {
      const result = await generateConcept(state.unit, body);
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          this.say(`GPU busy with ${refused.run.id} (${refused.run.stage}); one at a time`, "warn");
          this.host.follow(refused.run);
          this.setBusy(refused.run);
        } else if (refused.status === 423) {
          this.say(`gate: ${refused.error}`, "warn");
        } else {
          this.say(`${refused.status}: ${refused.error}`, "bad");
        }
        return;
      }
      this.say(`${what}: run ${result.run.id} (× ${body.count}${body.seed !== undefined ? `, seed ${body.seed}` : ""})`, "ok");
      this.host.follow(result.run);
      this.setBusy(result.run);
    } catch (error) {
      this.say(`could not start - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }

  private async cancel(): Promise<void> {
    const run = this.busy;
    if (!run) return;
    this.cancelButton.disabled = true;
    this.say(`cancelling ${run.id} ...`);
    try {
      const result = await cancelRun(run.id);
      if ("refused" in result) {
        this.say(`${result.refused.status}: ${result.refused.error}`, "warn");
        this.setBusy(null);
        return;
      }
      this.say(`${run.id} cancelled`, "ok");
    } catch (error) {
      this.say(`could not cancel - ${errorText(error)}`, "bad");
      this.cancelButton.disabled = false;
    } finally {
      this.host.refreshHealth();
    }
  }

  private async upload(): Promise<void> {
    const state = this.state;
    const file = this.uploadInput.files?.[0];
    this.uploadInput.value = "";
    if (!state || !file) return;
    this.say(`uploading ${file.name} ...`);
    try {
      this.state = await uploadConcept(state.unit, file, file.name);
      this.say(`${file.name} is candidate ${String(this.state.uploaded?.index ?? "?").padStart(2, "0")}`, "ok");
      this.draw();
    } catch (error) {
      this.say(`upload refused - ${errorText(error)}`, "bad");
    }
  }

  private async accept(selected: Selected): Promise<void> {
    const state = this.state;
    if (!state) return;
    const { target, attempt } = selected;
    if (target === "base") {
      if (state.base.pick !== null && state.base.pick !== attempt.index
          && !window.confirm(`accept ${refOf(target, attempt)} as the base?\n\nThe views drawn from the current base are cleared and the turnaround goes back to pending.`)) return;
      this.apply(reduceConcept(state, { type: "pick", index: attempt.index }));
      try {
        this.state = await pickConcept(state.unit, attempt.index);
        this.say(`base:${String(attempt.index).padStart(2, "0")} accepted - the base gate is open; the views can be drawn`, "ok");
      } catch (error) {
        this.say(`could not accept - ${errorText(error)}`, "bad");
      }
    } else {
      this.apply(reduceConcept(state, { type: "pickView", view: target, index: attempt.index }));
      try {
        this.state = await pickConceptView(state.unit, target, attempt.index);
        this.say(`${refOf(target, attempt)} accepted for the turnaround`, "ok");
      } catch (error) {
        this.say(`could not accept - ${errorText(error)}`, "bad");
      }
    }
    this.selected = null;
    this.draw();
  }

  private async saveDefaults(family: ModelFamily): Promise<void> {
    const state = this.state;
    const picker = this.pickers.find((p) => p.family === family);
    if (!state || !picker) return;
    try {
      this.state = await putConceptDefaults(state.unit, family, picker.model(), picker.chosen());
      this.say(`${state.unit}'s ${family} default saved`, "ok");
      this.refreshSaveButtons();
    } catch (error) {
      this.say(`could not save the default - ${errorText(error)}`, "bad");
    }
  }

  private apply(state: ConceptState): void {
    this.state = state;
    this.draw();
  }

  // --- drawing ---------------------------------------------------------------------
  private draw(): void {
    const state = this.state;
    this.drawSeed();
    this.enableButtons();
    if (!state) {
      this.baseStrip.replaceChildren();
      this.baseActions.replaceChildren();
      this.viewButtons.replaceChildren();
      this.viewsBox.replaceChildren();
      this.lockLine.textContent = "";
      return;
    }
    this.drawBase(state);
    this.drawViews(state);
    this.enableButtons();
  }

  private tile(target: string, attempt: Attempt, picked: boolean): HTMLElement {
    const selected = this.selected?.target === target && this.selected.attempt.index === attempt.index;
    const image = attempt.url && attempt.rendered
      ? el("img", { src: `${attempt.url}?t=${attempt.sha256 ?? attempt.seconds ?? ""}`, alt: refOf(target, attempt), loading: "lazy" })
      : el("div", { class: "concept-tile-miss" }, attempt.rendered ? "no file" : "…");
    return el("figure", {
      class: `concept-tile${picked ? " picked" : ""}${selected ? " selected" : ""}`,
      title: `${refOf(target, attempt)}\n${attempt.prompt ?? ""}`,
      onclick: () => { this.selected = selected ? null : { target, attempt }; this.draw(); },
    }, image, el("figcaption", {}, `${String(attempt.index).padStart(2, "0")}${picked ? " ✓" : ""} · ${attemptCaption(attempt)}`));
  }

  private drawBase(state: ConceptState): void {
    const tiles = state.base.candidates.map((attempt) => this.tile("base", attempt, state.base.pick === attempt.index));
    this.baseStrip.replaceChildren(...(tiles.length ? tiles : [el("div", { class: "studio-help" }, "no attempts yet - write a prompt and Generate, or Upload")]));
    this.baseActions.replaceChildren(...(this.selected?.target === "base" ? [this.actionBar(state, this.selected)] : []));
  }

  private drawViews(state: ConceptState): void {
    const locked = state.locked;
    this.lockLine.textContent = locked ? `locked: ${locked}` : state.views_stale ? "the views were drawn from an earlier base - draw them again" : "";
    const groups = viewGroups(state);
    const all = this.gpuButton("All", "back, side and top-down in one run", () => void this.generate({ op: "views", views: [...MAIN_VIEWS] }));
    const front = el("button", { type: "button", class: "studio-button", title: "the front is the accepted base itself - nothing to render", disabled: true }, "Front ✓");
    const buttons = groups.main.map((view) => this.gpuButton(viewLabel(view.view), `anim concept --views ${view.view}`, () => void this.generate({ op: "views", views: [view.view], viewPrompt: this.viewPromptOf(view) })));
    this.viewButtons.replaceChildren(all, front, ...buttons);
    for (const button of [all, ...buttons]) button.classList.toggle("locked", !!locked);
    const frontTile = state.base.front_url
      ? el("figure", { class: "concept-tile picked front", title: "the accepted base - the front view" },
          el("img", { src: `${state.base.front_url}?t=${state.base.pick ?? ""}`, alt: "front" }), el("figcaption", {}, "front ✓ · the base"))
      : el("figure", { class: "concept-tile front" }, el("div", { class: "concept-tile-miss" }, "no base"), el("figcaption", {}, "front · accept a base first"));
    this.viewsBox.replaceChildren(
      el("div", { class: "concept-view-row" }, el("div", { class: "concept-view-head" }, el("b", {}, "Front")), el("div", { class: "concept-strip" }, frontTile)),
      ...groups.main.map((view) => this.viewRow(state, view, !!locked)));
    const summary = this.advancedBox.firstElementChild!;
    this.advancedBox.replaceChildren(summary, ...groups.advanced.map((view) => this.viewRow(state, view, !!locked)));
  }

  private viewPromptOf(view: ViewRow): string {
    return this.viewPrompts.get(view.view)?.value ?? view.prompt;
  }

  private viewRow(state: ConceptState, view: ViewRow, locked: boolean): HTMLElement {
    let prompt = this.viewPrompts.get(view.view);
    if (!prompt) {
      prompt = el("textarea", { class: "studio-input concept-prompt small", rows: "2", title: "this view's sentence for Qwen-Image-Edit; the base's parts phrase is already in it" });
      prompt.value = view.prompt;
      this.viewPrompts.set(view.view, prompt);
    }
    const reset = el("button", { type: "button", class: "studio-button", title: "back to the config sentence", onclick: () => { prompt!.value = view.default_prompt; } }, "reset");
    const generate = this.gpuButton("Generate", `anim concept --views ${view.view} --view-prompt ...`, () => void this.generate({ op: "views", views: [view.view], viewPrompt: prompt!.value }));
    generate.classList.toggle("locked", locked);
    const tiles = view.candidates.map((attempt) => this.tile(view.view, attempt, view.pick === attempt.index));
    const status = view.state === "accepted" ? "accepted" : view.state === "rejected" ? "rejected" : view.candidates.length ? "pick one" : "";
    return el("div", { class: `concept-view-row ${view.state}` },
      el("div", { class: "concept-view-head" },
        el("b", {}, viewLabel(view.view)), el("span", { class: "studio-help" }, ` yaw ${view.yaw}${view.elevation === "top" ? ", from above" : ""} · ${status}`)),
      el("div", { class: "studio-row" }, prompt, reset, generate),
      el("div", { class: "concept-strip" }, ...(tiles.length ? tiles : [el("span", { class: "studio-help" }, "no attempts")])),
      ...(this.selected?.target === view.view ? [this.actionBar(state, this.selected)] : []));
  }

  /** Accept · Regenerate · img2img · Edit for the clicked attempt. */
  private actionBar(state: ConceptState, selected: Selected): HTMLElement {
    const { target, attempt } = selected;
    const ref = refOf(target, attempt);
    const isBase = target === "base";
    const picked = isBase ? state.base.pick === attempt.index : state.views[target]?.pick === attempt.index;
    const accept = el("button", { type: "button", class: "studio-button primary", disabled: picked || !attempt.rendered,
      title: isBase ? "anim concept --pick NN: this is the base gate" : `anim concept --pick-view ${target}=NN`,
      onclick: () => void this.accept(selected) }, picked ? "accepted ✓" : "Accept");
    const regenerate = this.gpuButton("Regenerate", isBase ? "txt2img again with the prompt above (new seed unless locked)" : "this view again with its prompt (new seed unless locked)",
      () => void this.generate(isBase
        ? { op: "txt2img", prompt: this.prompt.value }
        : { op: "views", views: [target], viewPrompt: this.viewPromptOf(state.views[target]!) }));
    const strength = el("input", { type: "range", min: String(STRENGTH_MIN), max: String(STRENGTH_MAX), step: "0.05", class: "concept-range", title: "how much is redrawn (KSampler denoise)" });
    strength.value = String(attempt.source === "img2img" && attempt.strength ? attempt.strength : STRENGTH_DEFAULT);
    const strengthLabel = el("span", { class: "concept-seed" }, `× ${strength.value}`);
    strength.addEventListener("input", () => { strengthLabel.textContent = `× ${strength.value}`; });
    const img2imgPrompt = el("input", { class: "studio-input", type: "text", placeholder: "prompt (blank: the parent's)", title: "a prompt for the redraw; blank keeps the parent's" });
    const img2img = this.gpuButton("img2img", `anim concept --img2img ${ref} --strength: classic redraw on Z-Image, keeps the composition`,
      () => void this.generate({ op: "img2img", ref, strength: Number(strength.value), prompt: img2imgPrompt.value }));
    const instruction = el("input", { class: "studio-input", type: "text", placeholder: "instruction - \"make the helmet red\"", title: "what should change; Qwen-Image-Edit keeps the rest" });
    const edit = this.gpuButton("Edit", `anim concept --edit ${ref} --instruction ...`,
      () => void this.generate({ op: "edit", ref, instruction: instruction.value }));
    const locked = !isBase && !!state.locked;
    for (const button of [regenerate, img2img, edit]) button.classList.toggle("locked", locked);
    this.enableButtons();
    const meta = [attemptCaption(attempt), attempt.model ? `model ${attempt.model}` : null,
      attempt.loras?.length ? `LoRA ${attempt.loras.map((l) => `${l.file} × ${l.strength}`).join(", ")}` : null,
      attempt.instruction ? `"${attempt.instruction}"` : null].filter(Boolean).join(" · ");
    return el("div", { class: "concept-actions-bar" },
      el("div", { class: "concept-actions-head" }, el("b", {}, ref), " ", el("span", { class: "studio-help" }, meta)),
      attempt.prompt ? el("div", { class: "concept-actions-prompt", title: "the prompt this attempt was made from" }, attempt.prompt) : null,
      el("div", { class: "studio-row" }, accept, regenerate),
      el("div", { class: "studio-row" }, img2img, strength, strengthLabel, img2imgPrompt),
      el("div", { class: "studio-row" }, edit, instruction));
  }
}
