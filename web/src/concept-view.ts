// The Concept and Views tabs as DOM (the concept generation feature, split in
// two by the pipeline tabs feature). One class, one state, two roots:
// `root` is the Concept tab - the base (prompt, Generate, Upload, every
// attempt with the accepted one outlined) - and `viewsRoot` the Views tab
// (All / Front / Back / Side / Top-down, the advanced fold, a prompt and
// Generate per view, every attempt, and "measure part palettes"). The
// settings row (checkpoint + LoRAs per family, how many per press, the seed
// lock) and the busy / status lines are ONE block that `show` moves into
// whichever of the two is on screen, so a choice made on one tab holds on the
// other. Under whichever attempt was clicked sits an action bar: Accept,
// Regenerate, img2img with its strength, Edit with an instruction. On the
// Views tab the click also opens the attempt in Piskel beside the panel.
// Every write goes to the API and the state is fetched again; concept.ts
// folds the click in first so the outline moves at once. While a run holds
// the GPU every generate button waits and one Cancel button does not.

import {
  ApiError,
  cancelRun,
  deleteAttemptFile,
  generateConcept,
  getConcept,
  getModels,
  hideAttempt,
  mirrorConceptView,
  pickConcept,
  pickConceptView,
  putConceptDefaults,
  startRun,
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
  advancedConfirm,
  advancedSweep,
  STRENGTH_MAX,
  STRENGTH_MIN,
  attemptCaption,
  busyText,
  choicesFor,
  choosable,
  familyOf,
  generateBody,
  hiddenCount,
  initialLoras,
  initialModel,
  isActive,
  lockedReason,
  mirrorOffer,
  mirroredNote,
  notInstalled,
  offerableLoras,
  reduceConcept,
  refOf,
  rowLabel,
  sameAsDefault,
  viewGroups,
  visibleAttempts,
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

/** What the panel lends the tabs: following a run, the health line, opening
 * one attempt in the pixel editor (the retouch feature), and hearing that the
 * state changed (the ✓ on the Concept and Views tabs). */
export interface ConceptHost {
  follow(run: RunRecord): void;
  refreshHealth(): void;
  editPixels(unit: string, target: string, attempt: Attempt): void;
  changed(state: ConceptState | null): void;
}

interface Selected {
  target: string;
  attempt: Attempt;
}

/** Whatever carries the unit's model defaults: the Concept tab's state (read
 * from the concept job). */
type WithDefaults = Pick<ConceptState, "defaults">;

/** One family's checkpoint + LoRA rows in the settings block. */
export class FamilyPicker {
  readonly root: HTMLElement;
  private readonly modelSelect = el("select", { class: "studio-select small" });
  private readonly loraList = el("div", { class: "concept-loras" });
  private readonly addButton = el("button", { type: "button", class: "studio-button", title: "stack another LoRA", onclick: () => this.add() }, "+ LoRA");
  private readonly saveButton = el("button", { type: "button", class: "studio-button", title: "PUT /concept/{unit}/defaults: remember this for the unit", onclick: () => void this.onSave(this.family) }, "save as unit default");
  private loras: LoraChoice[] = [];
  private touched = false;
  private choices: { checkpoints: RegistryRow[]; loras: RegistryRow[] } = { checkpoints: [], loras: [] };
  private offerable: RegistryRow[] = [];

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
  reset(listing: ModelsListing | null, state: WithDefaults | null): void {
    this.choices = choicesFor(listing, this.family);
    this.offerable = offerableLoras(listing, this.family);
    const model = initialModel(state, this.family);
    this.modelSelect.replaceChildren(
      el("option", { value: "" }, `checkpoint: the workflow's own${listing ? ` (${listing.workflow_defaults[this.family]})` : ""}`),
      ...this.choices.checkpoints.map((row) =>
        el("option", { value: row.file, title: row.file, disabled: !choosable(row) }, rowLabel(row))));
    this.modelSelect.value = model && this.choices.checkpoints.some((r) => r.file === model) ? model : "";
    this.loras = initialLoras(state, listing, this.family);
    this.touched = false;
    this.draw(state);
  }

  private draw(state: WithDefaults | null): void {
    this.loraList.replaceChildren(...this.loras.map((lora, i) => this.loraRow(lora, i)));
    this.addButton.disabled = !this.offerable.length;
    this.addButton.title = this.offerable.length ? "stack another LoRA"
      : `no ${this.family} LoRA is installed on the GPU service (gpu/models.registry.tsv lists them)`;
    this.saveButton.disabled = !state || sameAsDefault(state, this.family, this.model(), this.loras);
  }

  private loraRow(lora: LoraChoice, i: number): HTMLElement {
    const select = el("select", { class: "studio-select small", title: "the LoRA file (gpu/models.registry.tsv)" },
      ...(this.choices.loras.some((r) => r.file === lora.file) ? [] : [el("option", { value: lora.file }, `${lora.file} (not in the registry)`)]),
      ...this.choices.loras.map((row) =>
        el("option", { value: row.file, disabled: !choosable(row),
          title: row.installed === false ? `${row.file} is not on the GPU service - gpu/download_models.sh`
            : row.trigger ? `trigger: ${row.trigger}` : row.file }, rowLabel(row))));
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
    const offerable = this.offerable;
    const unused = offerable.find((row) => !this.loras.some((l) => l.file === row.file)) ?? offerable[0];
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

  refreshSave(state: WithDefaults | null): void {
    this.saveButton.disabled = !state || sameAsDefault(state, this.family, this.model(), this.loras);
  }
}

export class ConceptTab {
  private state: ConceptState | null = null;
  private listing: ModelsListing | null = null;
  private unitName = "";
  private selected: Selected | null = null;
  private busy: RunRecord | null = null;
  private viewPrompts = new Map<string, HTMLTextAreaElement>();

  /** The Concept tab. */
  readonly root = el("div", { class: "concept" });
  /** The Views tab. */
  readonly viewsRoot = el("div", { class: "concept concept-views-tab" });
  /** Settings, busy and status: moved into whichever tab is showing. */
  private readonly head = el("div", { class: "concept-head" });
  private readonly pickers = FAMILIES.map((f) => new FamilyPicker(f.family, f.label, f.title, (family) => this.saveDefaults(family), () => this.refreshSaveButtons()));
  private readonly countSelect = el("select", { class: "studio-select small", title: "attempts per press (each is one /prompt on the card)" },
    ...[1, 2, 3, 4].map((n) => el("option", { value: String(n) }, `× ${n}`)));
  private readonly lockSeed = el("input", { type: "checkbox", title: "lock the seed: resend the last one, so a prompt tweak changes only what you changed" });
  private readonly showHidden = el("input", { type: "checkbox", title: "show the attempts you have hidden, greyed, with Restore" });
  private readonly inPlace = el("input", { type: "checkbox", title: "the retouch feature: img2img and Edit write over the attempt you clicked instead of adding a new one" });
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
  private readonly partsLine = el("div", { class: "studio-row concept-parts" });

  constructor(private readonly host: ConceptHost) {
    this.lockSeed.addEventListener("change", () => this.drawSeed());
    this.showHidden.addEventListener("change", () => { this.selected = null; this.draw(); });
    this.head.replaceChildren(
      el("details", { class: "concept-settings-fold" },
        el("summary", {}, "model settings: checkpoints, LoRAs, attempts per press, seed"),
        el("div", { class: "concept-settings" },
          ...this.pickers.map((p) => p.root),
          el("div", { class: "studio-row" },
            this.countSelect,
            el("label", { class: "concept-lock" }, this.lockSeed, " lock seed"),
            el("label", { class: "concept-lock" }, this.showHidden, " show hidden"),
            el("label", { class: "concept-lock" }, this.inPlace, " edit in place"),
            this.seedLine))),
      el("div", { class: "studio-row concept-busy" }, this.busyLine, this.cancelButton),
      this.statusLine);
    this.root.replaceChildren(
      this.head,
      el("h3", { class: "concept-h" }, "Base"),
      el("p", { class: "studio-help" }, "The front-facing reference every later stage trusts. Generate from the prompt or upload your own; click an attempt to accept it (that is the base gate) or rework it. Edit pixels opens it in Piskel beside the panel."),
      this.prompt,
      el("div", { class: "studio-row" }, this.generateBase, this.uploadButton, this.uploadInput),
      this.baseStrip,
      this.baseActions,
    );
    this.viewsRoot.replaceChildren(
      el("p", { class: "studio-help" }, "The accepted base seen from the other sides, drawn from it through Qwen-Image-Edit. Front is the base itself; Side is the left profile. Accept one per view: the Animate tab draws each direction from its own. Click an attempt to open it in Piskel."),
      this.lockLine,
      this.viewButtons,
      this.viewsBox,
      this.advancedBox,
      this.partsLine,
    );
  }

  /** Put the shared settings / busy / status block on the tab being shown. */
  show(which: "concept" | "views"): void {
    const root = which === "concept" ? this.root : this.viewsRoot;
    if (this.head.parentElement === root) return;
    root.prepend(this.head);
    // Open on Concept, where the settings are used most; folded on Views.
    (this.head.firstElementChild as HTMLDetailsElement).open = which === "concept";
  }

  private gpuButton(label: string, title: string, run: () => void): HTMLButtonElement {
    return el("button", { type: "button", class: "studio-button gpu", title: `${title}\nStarts a GPU job on ComfyUI - one at a time.`, disabled: true, onclick: run }, label);
  }

  // --- loading ---------------------------------------------------------------
  async start(): Promise<void> {
    try {
      // Probed: a file the service does not have must not be offered - the
      // Qwen pixel LoRA is registered and deliberately not installed.
      this.listing = await getModels(true);
    } catch (error) {
      this.say(`could not read /models - ${errorText(error)}`, "warn");
    }
  }

  /** The unit bar picked a unit ("" for none). */
  async setUnit(unit: string): Promise<void> {
    this.unitName = unit;
    await this.load();
  }

  private unit(): string {
    return this.unitName;
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
    for (const root of [this.root, this.viewsRoot]) {
      for (const button of root.querySelectorAll<HTMLButtonElement>("button.gpu")) {
        button.disabled = off || button.classList.contains("locked");
      }
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
  /** The advanced views sweep feature: confirm what it costs, then hand the
   * whole list to the one `op: "views"` run, which renders them in order. */
  private async sweepAdvanced(): Promise<void> {
    const state = this.state;
    if (!state) return;
    const { views, blocked } = advancedSweep(state, this.busy);
    if (blocked) {
      this.say(blocked, "warn");
      return;
    }
    if (!window.confirm(advancedConfirm(state, views, this.form("views").count))) return;
    await this.generate({ op: "views", views });
  }

  private async generate(ask: GenerateAsk): Promise<void> {
    const state = this.state;
    if (!state) return;
    let body;
    const form = this.form(ask.op);
    try {
      body = generateBody(ask, form);
      if (this.inPlace.checked && (ask.op === "img2img" || ask.op === "edit")) {
        body.in_place = true;
        body.count = 1;
      }
    } catch (error) {
      this.say(errorText(error), "bad");
      return;
    }
    // The GPU service validates every model filename before it executes, so a
    // registered-but-missing file is a 400 on the whole graph. Say so here.
    const family = familyOf(ask.op, this.listing);
    const missing = notInstalled(this.listing, family, form.model, form.loras ?? []);
    if (missing.length) {
      this.say(`${missing.join(" and ")} ${missing.length === 1 ? "is" : "are"} not installed on the `
        + "GPU service - pick another, or install it with gpu/download_models.sh", "bad");
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
          && !window.confirm(`accept ${refOf(target, attempt)} as the base?\n\nThe views drawn from the current base are cleared.`)) return;
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
        this.say(`${refOf(target, attempt)} accepted`, "ok");
      } catch (error) {
        this.say(`could not accept - ${errorText(error)}`, "bad");
      }
    }
    this.selected = null;
    this.draw();
  }

  /** The mirrored view feature: derive this view from its accepted partner.
   * No GPU, so it does not go through the run manager and does not wait for
   * the busy line - it is a copy with the pixels reversed. */
  private async mirror(view: ViewRow): Promise<void> {
    const state = this.state;
    if (!state) return;
    const offer = mirrorOffer(state, view);
    if (!offer?.can) return;
    this.apply(reduceConcept(state, { type: "mirrorView", view: view.view }));
    try {
      this.state = await mirrorConceptView(state.unit, view.view);
      const warning = this.state.warning;
      this.say(`${view.view} is now the ${offer.partner} view flipped - no GPU spent`
               + (warning ? `. ${warning}` : ""), warning ? "warn" : "ok");
    } catch (error) {
      this.say(`could not mirror - ${errorText(error)}`, "bad");
    }
    this.selected = null;
    this.draw();
  }

  private async setHidden(selected: Selected, hidden: boolean): Promise<void> {
    const state = this.state;
    if (!state) return;
    const { target, attempt } = selected;
    const ref = refOf(target, attempt);
    try {
      this.state = await hideAttempt(state.unit, target, attempt.index, hidden);
      this.say(hidden
        ? `${ref} hidden - the file and the record stay; "show hidden" brings it back`
        : `${ref} is back in the strip`, "ok");
    } catch (error) {
      this.say(`could not ${hidden ? "hide" : "restore"} ${ref} - ${errorText(error)}`, "bad");
    }
    this.selected = null;
    this.draw();
  }

  private async deleteFile(selected: Selected): Promise<void> {
    const state = this.state;
    if (!state) return;
    const { target, attempt } = selected;
    const ref = refOf(target, attempt);
    if (!window.confirm(`delete ${ref} (${attempt.file})?\n\nThe PNG is removed from disk. `
      + "Its row stays - the prompt, seed and what it was made from - so the number is never "
      + "reused. This cannot be undone.")) return;
    try {
      this.state = await deleteAttemptFile(state.unit, target, attempt.index);
      this.say(`${ref}: ${attempt.file} deleted; its record stays`, "ok");
    } catch (error) {
      this.say(`could not delete ${ref} - ${errorText(error)}`, "bad");
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
    this.host.changed(state);
    if (!state) {
      this.baseStrip.replaceChildren(el("span", { class: "studio-help" }, this.unit() ? "" : "pick a unit above, or type a new one"));
      this.baseActions.replaceChildren();
      this.viewButtons.replaceChildren();
      this.viewsBox.replaceChildren();
      this.partsLine.replaceChildren();
      this.lockLine.textContent = this.unit() ? "" : "pick a unit above";
      return;
    }
    this.drawBase(state);
    this.drawViews(state);
    this.drawParts(state);
    this.enableButtons();
  }

  /** The part palettes every direction snaps to: `post --parts` over the
   * accepted base, once (CPU, no GPU). Offered until it has run. */
  private drawParts(state: ConceptState): void {
    if (state.base.pick === null) {
      this.partsLine.replaceChildren();
      return;
    }
    const job = state.base_parts_job;
    const measure = el("button", { type: "button", class: "studio-button",
      title: `anim post ${job} --parts: SAM over the accepted base, once, to record what colour each part is. CPU, no GPU.`,
      onclick: () => void this.measureParts(state.unit, job) },
      state.base_parts ? "measure part palettes again" : "measure part palettes");
    this.partsLine.replaceChildren(measure, el("span", { class: state.base_parts ? "studio-help" : "studio-status warn" },
      state.base_parts ? " part palettes measured - every direction snaps its colours to them"
        : " no part palettes yet - every direction locks its colours to them"));
  }

  private async measureParts(unit: string, job: string): Promise<void> {
    if (!window.confirm(`Measure ${unit}'s part palettes from its accepted base?\n\n`
      + `This runs anim post ${job} --parts: about 45 s on the CPU, no GPU.`)) return;
    try {
      const result = await startRun("post", { job, parts: true });
      if ("refused" in result) {
        this.say(`${result.refused.status}: ${result.refused.error}`, result.refused.status === 409 ? "warn" : "bad");
        return;
      }
      this.say(`measuring part palettes: run ${result.run.id}`, "ok");
      this.host.follow(result.run);
    } catch (error) {
      this.say(`could not start - ${errorText(error)}`, "bad");
    }
  }

  private tile(target: string, attempt: Attempt, picked: boolean): HTMLElement {
    const selected = this.selected?.target === target && this.selected.attempt.index === attempt.index;
    const image = attempt.url && attempt.rendered
      ? el("img", { src: `${attempt.url}?t=${attempt.sha256 ?? attempt.seconds ?? ""}`, alt: refOf(target, attempt), loading: "lazy" })
      : el("div", { class: "concept-tile-miss" },
          attempt.file_deleted ? "file deleted" : attempt.rendered ? "no file" : "…");
    return el("figure", {
      class: `concept-tile${picked ? " picked" : ""}${selected ? " selected" : ""}${attempt.hidden ? " hidden-attempt" : ""}`,
      title: `${refOf(target, attempt)}\n${attempt.prompt ?? ""}`,
      onclick: () => {
        // On the Views tab, selecting is editing: the attempt opens in Piskel
        // beside the panel. The base opens only from its Edit pixels button,
        // so a click on Concept still just selects an attempt to compare.
        const editable = target !== "base" && !!attempt.url && attempt.rendered && !attempt.file_deleted;
        if (editable && this.state) this.host.editPixels(this.state.unit, target, attempt);
        this.selected = selected && !editable ? null : { target, attempt };
        this.draw();
      },
    }, image, el("figcaption", {}, `${String(attempt.index).padStart(2, "0")}${picked ? " ✓" : ""} · ${attemptCaption(attempt)}`));
  }

  private drawBase(state: ConceptState): void {
    const shown = visibleAttempts(state.base.candidates, this.showHidden.checked);
    const tiles = shown.map((attempt) => this.tile("base", attempt, state.base.pick === attempt.index));
    const hidden = hiddenCount(state.base.candidates);
    this.baseStrip.replaceChildren(
      ...(tiles.length ? tiles : [el("div", { class: "studio-help" },
        hidden ? `every attempt is hidden (${hidden}) - "show hidden" brings them back`
          : "no attempts yet - write a prompt and Generate, or Upload")]),
      ...(hidden && !this.showHidden.checked
        ? [el("div", { class: "studio-help concept-hidden-note" }, `${hidden} hidden`)] : []));
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
      ? el("figure", { class: `concept-tile picked front${state.base.pick_stale ? " stale" : ""}`,
          title: state.base.pick_stale
            ? "the accepted base was edited after it was accepted - Accept it again to push the edit through"
            : "the accepted base - the front view" },
          el("img", { src: `${state.base.front_url}?t=${state.base.pick ?? ""}`, alt: "front" }),
          el("figcaption", {}, state.base.pick_stale ? "front · edited since accepted" : "front ✓ · the base"))
      : el("figure", { class: "concept-tile front" }, el("div", { class: "concept-tile-miss" }, "no base"), el("figcaption", {}, "front · accept a base first"));
    this.viewsBox.replaceChildren(
      el("div", { class: "concept-view-row" }, el("div", { class: "concept-view-head" }, el("b", {}, "Front")), el("div", { class: "concept-strip" }, frontTile)),
      ...groups.main.map((view) => this.viewRow(state, view, !!locked)));
    const summary = this.advancedBox.firstElementChild!;
    // The advanced views sweep feature: one press for the whole fold, sitting
    // above the rows it fills. It is a `.gpu` button like every other, so the
    // busy handling and the gate lock come for free; `blocked` adds its own
    // reason on top (nothing left to render) in the same `.locked` idiom.
    const sweep = advancedSweep(state, this.busy);
    const sweepButton = this.gpuButton(sweep.label,
      `anim concept --views advanced${sweep.views.length ? ` (${sweep.views.join(", ")})` : ""}`,
      () => void this.sweepAdvanced());
    sweepButton.classList.toggle("locked", !!sweep.blocked);
    if (sweep.blocked) sweepButton.title = sweep.blocked;
    this.advancedBox.replaceChildren(summary,
      el("div", { class: "concept-advanced-sweep" }, sweepButton),
      ...groups.advanced.map((view) => this.viewRow(state, view, !!locked)));
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
    // The mirrored view feature: the three paired views get a second way to be
    // filled that costs nothing. Disabled with the reason when the partner is
    // not accepted - the same `.locked` idiom the GPU buttons use for the gate.
    const offer = mirrorOffer(state, view);
    const mirror = offer
      ? el("button", { type: "button", class: `studio-button${offer.can ? "" : " locked"}`,
          title: offer.title, disabled: !offer.can,
          onclick: () => void this.mirror(view) }, offer.label)
      : null;
    const note = mirroredNote(view);
    const shown = visibleAttempts(view.candidates, this.showHidden.checked);
    const tiles = shown.map((attempt) => this.tile(view.view, attempt, view.pick === attempt.index));
    const hidden = view.hidden_count;
    const status = view.state === "accepted" ? "accepted" : view.state === "rejected" ? "rejected" : view.candidates.length ? "pick one" : "";
    return el("div", { class: `concept-view-row ${view.state}${view.mirrored ? " mirrored" : ""}` },
      el("div", { class: "concept-view-head" },
        el("b", {}, viewLabel(view.view)),
        el("span", { class: "studio-help" },
          ` yaw ${view.yaw}${view.elevation === "top" ? ", from above" : ""} · ${status}`
          + (hidden && !this.showHidden.checked ? ` · ${hidden} hidden` : "")),
        ...(note ? [el("span", { class: "concept-mirror-mark",
          title: "the accepted picture here is that view reversed, not a generated one - the same pixels, the other hand" }, `⇄ ${note}`)] : [])),
      el("div", { class: "studio-row" }, prompt, reset, generate, ...(mirror ? [mirror] : [])),
      el("div", { class: "concept-strip" }, ...(tiles.length ? tiles : [el("span", { class: "studio-help" }, "no attempts")])),
      ...(this.selected?.target === view.view ? [this.actionBar(state, this.selected)] : []));
  }

  /** Accept · Regenerate · img2img · Edit for the clicked attempt. */
  private actionBar(state: ConceptState, selected: Selected): HTMLElement {
    const { target, attempt } = selected;
    const ref = refOf(target, attempt);
    const isBase = target === "base";
    const picked = isBase ? state.base.pick === attempt.index : state.views[target]?.pick === attempt.index;
    const accept = el("button", { type: "button", class: "studio-button primary",
      disabled: picked || !attempt.rendered || !!attempt.file_deleted,
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
    // Hide / Restore, and - only once hidden - Delete file. The accepted
    // attempt is refused here and in the API: every later stage reads it.
    const editPixels = el("button", {
      type: "button", class: "studio-button",
      disabled: !attempt.rendered || !!attempt.file_deleted,
      title: "open this attempt in the pixel editor; Save writes it back over itself, and "
        + "rubbing an area out lets Inpaint redraw just that",
      onclick: () => this.host.editPixels(state.unit, target, attempt),
    }, "Edit pixels");
    const why = lockedReason(state, target, attempt);
    const hide = el("button", {
      type: "button", class: "studio-button", disabled: !!why,
      title: why ?? (attempt.hidden ? "show this attempt again"
        : "take it out of the strip; the file and the record stay"),
      onclick: () => void this.setHidden(selected, !attempt.hidden),
    }, attempt.hidden ? "Restore" : "Hide");
    const deleteFile = attempt.hidden && !attempt.file_deleted
      ? el("button", {
          type: "button", class: "studio-button", disabled: !!why,
          title: why ?? `delete ${attempt.file} from disk; the row stays as its record`,
          onclick: () => void this.deleteFile(selected),
        }, "Delete file")
      : null;
    const stale = (isBase ? state.base.pick_stale : state.views[target]?.pick_stale) && picked
      ? el("div", { class: "studio-status warn" },
          "edited since accepted - Accept again to push it through to the base render")
      : null;
    const meta = [attemptCaption(attempt), attempt.model ? `model ${attempt.model}` : null,
      attempt.loras?.length ? `LoRA ${attempt.loras.map((l) => `${l.file} × ${l.strength}`).join(", ")}` : null,
      attempt.instruction ? `"${attempt.instruction}"` : null].filter(Boolean).join(" · ");
    return el("div", { class: "concept-actions-bar" },
      el("div", { class: "concept-actions-head" }, el("b", {}, ref), " ", el("span", { class: "studio-help" }, meta)),
      stale,
      attempt.prompt ? el("div", { class: "concept-actions-prompt", title: "the prompt this attempt was made from" }, attempt.prompt) : null,
      el("div", { class: "studio-row" }, accept, regenerate, editPixels, hide, deleteFile),
      el("div", { class: "studio-row" }, img2img, strength, strengthLabel, img2imgPrompt),
      el("div", { class: "studio-row" }, edit, instruction));
  }
}
