// The studio panel: the pipeline beside the Piskel canvas (the pipeline tabs
// feature). Top to bottom, outside the scrolling body: the unit bar (one
// unit for every tab), the three tabs in pipeline order - Concept → Views →
// Animate, each with a ✓ once its step is done - and the edit bar (what is
// open in Piskel, Save / Inpaint / Done, the project palette as swatches).
// Then the tab's body: the base (concept-view.ts), its views (the same class's
// second root) or every animation (animate-view.ts). At the bottom, the run
// footer: one line for the run being followed, with Cancel, that opens into
// its log and the recent runs.
//
// Concept is wide - a strip of attempts needs the room - until a base attempt
// is opened for editing; Views and Animate always sit at the draggable split
// width with Piskel beside them, because clicking a view or a frame there
// opens it in Piskel. Plain DOM, no framework: the page is one screen and
// Piskel itself is jQuery-era JavaScript, so a second rendering model would be
// a second thing to learn.

import {
  ApiError,
  cancelRun,
  getAnimate,
  getConcept,
  getHealth,
  getRun,
  getSettings,
  inpaintAnimate,
  inpaintAttempt,
  listRuns,
  listUnits,
  putAnimateImage,
  putConceptImage,
  type Attempt,
  type ConceptState,
  type RunRecord,
} from "./api";
import { ConceptTab } from "./concept-view";
import { AnimateTab, type AnimateCell } from "./animate-view";
import { unitSlug } from "./concept";
import { projectPalette } from "./palette";
import { clampEditWidth, discardQuestion, mustConfirmLeave, readEditWidth } from "./editing";
import { TABS, TAB_LABELS, layoutFor, readTab, readUnit, stepsDone, type Tab } from "./tabs";
import {
  exportFrames,
  frameCount,
  frameSize,
  isDirty,
  markSaved,
  mountOverCanvas,
  onPiskelEvent,
  openAnimation,
  selectColour,
  setFrameTags,
  setProjectPalette,
} from "./piskel-bridge";
import { describeCells, describeRun, followRun, runTone, succeeded, viewOf, type RunView } from "./sse";

const DEFAULT_FPS = 12;
const TAB_KEY = "anim-studio.panel-tab";
const UNIT_KEY = "anim-studio.unit";
const EDIT_WIDTH_KEY = "anim-studio.edit-width";
const TAB_HELP: Record<Tab, string> = {
  concept: "the base character: generate or upload, accept one",
  views: "the base from the other sides: one accepted view per side",
  animate: "every animation: keyframes first, then the in-betweens",
};

/** One attempt open in the editor (the retouch feature): Save writes it back
 * over itself, and Inpaint sends the canvas for a masked redraw. */
interface OpenedAttempt {
  unit: string;
  target: string;
  index: number;
  file: string;
  /** The attempt's positional index in its frame set, from its URL. */
  position: number;
  width: number;
  height: number;
  /** Set when it is a frame attempt: `target` is then the frame's id, and
   * Save / Inpaint go back to the Animate routes. */
  animate?: AnimateCell;
}

/** The edit bar: what it is about. Set from the moment an open starts until
 * Done; `opened` is set only once the load worked, so a failed open leaves
 * the bar with its error and Done, and nothing to save to. */
interface Editing {
  label: string;
}

// --- tiny DOM helpers ------------------------------------------------------
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

function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private mode; the choice just does not survive a reload
  }
}

function status(node: HTMLElement, text: string, kind: "" | "ok" | "warn" | "bad" = ""): void {
  node.textContent = text;
  node.className = `studio-status ${kind}`.trim();
}

function errorText(error: unknown): string {
  if (error instanceof ApiError) return `${error.status}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}

// --- the panel ---------------------------------------------------------------
export class StudioPanel {
  private opened: OpenedAttempt | null = null;
  private editing: Editing | null = null;
  private saving = false;
  private stopFollowing: (() => void) | null = null;
  private followed: RunView | null = null;
  private tab: Tab = "concept";
  private unit = "";
  private units: string[] = [];
  private gpuRunSeen: string | null = null;

  // The unit bar
  private readonly unitSelect = el("select", { class: "studio-select", title: "the unit every tab works on (GET /units)" });
  private readonly unitInput = el("input", { class: "studio-input small", type: "text", placeholder: "or a new unit - type a name, Enter", title: "a unit that has nothing yet: type a name and press Enter (\"Cloaked Assassin\" becomes cloaked_assassin)" });
  private readonly unitStatus = el("div", { class: "studio-status" });
  // The tabs
  private readonly tabButtons = new Map<Tab, HTMLButtonElement>();
  // The edit bar
  private readonly editLabel = el("b");
  private readonly editDirty = el("span", { class: "studio-dirty", hidden: true }, "● unsaved");
  private readonly saveButton = el("button", { type: "button", class: "studio-button primary",
    title: "write the canvas back over the attempt it came from, same size, alpha rounded to 0/255",
    onclick: () => void this.save() }, "Save → the attempt");
  private readonly doneButton = el("button", { type: "button", class: "studio-button",
    title: "stop editing: the image stays on the canvas, but nothing is saved from it any more",
    onclick: () => this.finishEditing() }, "Done");
  private readonly saveStatus = el("div", { class: "studio-status" });
  // The retouch feature: the masked redraw of the open attempt.
  private readonly inpaintRow = el("div", { class: "studio-inpaint", hidden: true });
  private readonly inpaintPrompt = el("input", { class: "studio-input", type: "text", placeholder: "what goes in the erased area", title: "the prompt for the masked redraw; blank keeps the attempt's own" });
  private readonly inpaintStrength = el("input", { type: "range", min: "0.3", max: "0.9", step: "0.05", value: "0.9", class: "concept-range", title: "how hard the erased region is redrawn (KSampler denoise)" });
  private readonly inpaintButton = el("button", { type: "button", class: "studio-button gpu", title: "redraw what you rubbed out; everything else is pasted back untouched\nStarts a GPU job on ComfyUI - one at a time.", onclick: () => void this.inpaint() }, "Inpaint");
  private readonly swatches = el("div", { class: "studio-swatches", title: "the project palette: click for the pen colour, right-click for the second colour" });
  private readonly editBar = el("div", { class: "studio-editbar", hidden: true });
  // The folded panel's copy: the same actions, no inputs.
  private readonly floatLabel = el("b");
  private readonly floatDirty = el("span", { class: "studio-dirty", hidden: true }, "● unsaved");
  private readonly floatSave = el("button", { type: "button", class: "studio-button small primary", onclick: () => void this.save() }, "Save");
  private readonly floatInpaint = el("button", { type: "button", class: "studio-button small gpu", onclick: () => void this.inpaint() }, "Inpaint");
  private readonly floatDone = el("button", { type: "button", class: "studio-button small", onclick: () => this.finishEditing() }, "Done");
  private readonly floatBar = el("div", { class: "studio-float", hidden: true },
    this.floatLabel, this.floatDirty, this.floatSave, this.floatInpaint, this.floatDone);
  // The draggable right edge that sets the split width.
  private readonly widthHandle = el("div", { class: "studio-width-handle", hidden: true, title: "drag to set the panel's width beside Piskel (remembered)" });
  // The run footer
  private readonly runDot = el("span", { class: "studio-run-dot" });
  private readonly runLine = el("span", { class: "studio-run-line" }, "no run followed");
  private readonly runBar = el("div", { class: "studio-bar-fill" });
  private readonly runCancel = el("button", { type: "button", class: "studio-button small", hidden: true,
    title: "POST /runs/{id}/cancel: stop the stage and ComfyUI's current prompt",
    onclick: (event: Event) => { event.stopPropagation(); void this.cancelFollowed(); } }, "Cancel");
  private readonly runToggle = el("span", { class: "studio-help studio-run-toggle" }, "▲ log");
  private readonly runLog = el("pre", { class: "studio-log" });
  private readonly runsList = el("ul", { class: "studio-runs" });
  private readonly footer = el("footer", { class: "studio-footer" });
  // The tabs' bodies
  private readonly concept = new ConceptTab({
    follow: (run) => this.follow(run),
    refreshHealth: () => void this.refreshHealth(),
    editPixels: (unit, target, attempt) => void this.editAttempt(unit, target, attempt),
    changed: (state) => this.drawSteps(state),
  });
  private readonly animate = new AnimateTab({
    follow: (run) => this.follow(run),
    refreshHealth: () => void this.refreshHealth(),
    editPixels: (cell, frame, attempt, mode) =>
      void this.editAttempt(cell.unit, frame, attempt, cell, mode),
  });
  private readonly bodies: Record<Tab, HTMLElement> = {
    concept: this.concept.root,
    views: this.concept.viewsRoot,
    animate: this.animate.root,
  };
  private readonly health: HTMLElement;

  constructor(private readonly root: HTMLElement, health: HTMLElement) {
    this.health = health;
    this.build();
  }

  private build(): void {
    // The unit bar
    this.unitSelect.addEventListener("change", () => { this.unitInput.value = ""; void this.setUnit(this.unitSelect.value); });
    this.unitInput.addEventListener("keydown", (event) => {
      if ((event as KeyboardEvent).key === "Enter") { event.preventDefault(); void this.typeUnit(); }
    });
    const unitBar = el("div", { class: "studio-unitbar" },
      el("div", { class: "studio-row" },
        el("span", { class: "studio-unitbar-label" }, "Unit"),
        this.unitSelect,
        el("button", { type: "button", class: "studio-button", title: "list the units again", onclick: () => void this.refreshUnits() }, "↻"),
        this.unitInput),
      this.unitStatus);

    // The tabs, as the pipeline's steps
    const tabs = el("nav", { class: "studio-tabs" });
    TABS.forEach((tab, i) => {
      if (i) tabs.append(el("span", { class: "studio-tab-arrow" }, "→"));
      const button = el("button", { type: "button", class: "studio-tab", title: TAB_HELP[tab], onclick: () => this.showTab(tab) },
        el("span", { class: "studio-tab-n" }, String(i + 1)), TAB_LABELS[tab]);
      this.tabButtons.set(tab, button);
      tabs.append(button);
    });

    this.buildEditBar();
    this.root.before(unitBar, tabs, this.editBar);
    this.root.replaceChildren(...TABS.map((tab) =>
      el("section", { class: "studio-section", "data-tab": tab }, this.bodies[tab])));

    // The run footer
    this.footer.replaceChildren(
      el("div", { class: "studio-footer-line", title: "the run being followed - click for its log and the recent runs",
        onclick: () => this.footer.classList.toggle("open") },
        this.runDot, this.runLine, el("div", { class: "studio-bar" }, this.runBar), this.runCancel, this.runToggle),
      el("div", { class: "studio-footer-body" },
        this.runLog,
        el("div", { class: "studio-row" },
          el("button", { type: "button", class: "studio-button small", title: "GET /runs", onclick: () => void this.refreshRuns() }, "recent runs ↻")),
        this.runsList));
    this.root.after(this.footer);
  }

  // --- lifecycle -------------------------------------------------------------
  async start(): Promise<void> {
    this.showTab(readTab(readStored(TAB_KEY)));
    await Promise.all([this.refreshHealth(), this.refreshRuns(), this.refreshPalette(),
      this.concept.start(), this.animate.start()]);
    await this.refreshUnits(readUnit(readStored(UNIT_KEY)));
  }

  showTab(tab: Tab): void {
    this.tab = tab;
    for (const [name, button] of this.tabButtons) button.classList.toggle("active", name === tab);
    for (const node of this.root.querySelectorAll<HTMLElement>("section[data-tab]")) node.hidden = node.dataset.tab !== tab;
    if (tab !== "animate") this.concept.show(tab);
    this.applyLayout();
    store(TAB_KEY, tab);
  }

  /** Wide on Concept with nothing open, split beside Piskel otherwise. */
  private applyLayout(): void {
    const split = layoutFor(this.tab, this.editing !== null) === "split";
    document.body.classList.toggle("studio-split", split);
    document.body.classList.toggle("studio-wide", !split);
    this.widthHandle.hidden = !split;
  }

  /** The ✓ on Concept and Views, from the unit's concept state. */
  private drawSteps(state: ConceptState | null): void {
    const done = stepsDone(state);
    for (const [tab, button] of this.tabButtons) {
      button.classList.toggle("done", done[tab]);
      button.querySelector(".studio-tab-n")!.textContent = done[tab] ? "✓" : String(TABS.indexOf(tab) + 1);
    }
  }

  // --- the unit bar ----------------------------------------------------------
  private async refreshUnits(want?: string): Promise<void> {
    const before = want ?? this.unit;
    try {
      this.units = await listUnits();
    } catch (error) {
      status(this.unitStatus, `could not list units - ${errorText(error)}`, "bad");
      return;
    }
    status(this.unitStatus, "");
    const pick = before && !this.units.includes(before) ? [...this.units, before].sort() : this.units;
    this.unitSelect.replaceChildren(
      el("option", { value: "" }, pick.length ? "pick a unit" : "no units yet - type a name"),
      ...pick.map((unit) => el("option", { value: unit }, unit)));
    const chosen = pick.includes(before) ? before : pick.length === 1 ? pick[0]! : "";
    this.unitSelect.value = chosen;
    await this.setUnit(chosen);
  }

  private async typeUnit(): Promise<void> {
    const typed = this.unitInput.value.trim();
    const slug = unitSlug(typed);
    if (!slug) {
      status(this.unitStatus, `"${typed}" has nothing a unit slug can be made of - letters or digits, `
        + "e.g. \"Cloaked Assassin\" -> cloaked_assassin", "bad");
      return;
    }
    this.unitInput.value = "";
    status(this.unitStatus, slug !== typed ? `unit ${slug} (from "${typed}") - start on the Concept tab` : "");
    if (![...this.unitSelect.options].some((o) => o.value === slug)) {
      this.unitSelect.append(el("option", { value: slug }, slug));
    }
    this.unitSelect.value = slug;
    await this.setUnit(slug);
    this.showTab("concept");
  }

  /** Every tab follows the one unit. */
  private async setUnit(unit: string): Promise<void> {
    this.unit = unit;
    if (unit) store(UNIT_KEY, unit);
    await Promise.all([this.concept.setUnit(unit), this.animate.setUnit(unit)]);
  }

  // --- the edit bar ------------------------------------------------------------
  private buildEditBar(): void {
    this.inpaintRow.replaceChildren(
      el("div", { class: "studio-row" }, this.inpaintButton, this.inpaintStrength),
      el("div", { class: "studio-row" }, this.inpaintPrompt));
    this.editBar.replaceChildren(
      el("div", { class: "studio-editbar-what" }, "editing ", this.editLabel, this.editDirty),
      el("div", { class: "studio-row" }, this.saveButton, this.doneButton, this.swatches),
      this.inpaintRow,
      this.saveStatus,
    );
    document.body.append(this.widthHandle);
    mountOverCanvas(this.floatBar);
    this.applyEditWidth(readEditWidth(readStored(EDIT_WIDTH_KEY)));
    this.widthHandle.addEventListener("pointerdown", (event) => this.dragWidth(event));
    // Piskel says after every stroke, undo and load whether the canvas
    // still matches what was loaded or saved: the "● unsaved" marker.
    onPiskelEvent("PISKEL_SAVED_STATUS_UPDATE", () => this.syncEditBar());
    this.syncEditBar();
  }

  /** The split width, as the custom property the CSS clamps again against
   * the window (a remembered width from a wider screen still fits). */
  private applyEditWidth(width: number): void {
    document.body.style.setProperty("--studio-edit-w", `${width}px`);
    this.widthHandle.dataset.width = `${width} px`;
  }

  private dragWidth(event: PointerEvent): void {
    event.preventDefault();
    const handle = this.widthHandle;
    handle.setPointerCapture(event.pointerId);
    document.body.classList.add("studio-dragging");
    let width = readEditWidth(readStored(EDIT_WIDTH_KEY));
    const move = (moved: PointerEvent) => {
      width = clampEditWidth(moved.clientX, window.innerWidth);
      this.applyEditWidth(width);
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      document.body.classList.remove("studio-dragging");
      store(EDIT_WIDTH_KEY, String(width));
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  }

  /** Everything the bar shows follows `editing` / `opened` / `saving`. */
  private syncEditBar(): void {
    const editing = this.editing;
    const opened = this.opened;
    document.body.classList.toggle("studio-editing", editing !== null);
    this.editBar.hidden = editing === null;
    this.floatBar.hidden = editing === null;
    this.applyLayout();
    if (!editing) return;
    const dirty = opened !== null && isDirty();
    this.editLabel.textContent = editing.label;
    this.floatLabel.textContent = editing.label;
    this.editDirty.hidden = !dirty;
    this.floatDirty.hidden = !dirty;
    for (const button of [this.saveButton, this.floatSave]) {
      button.hidden = !opened;
      button.disabled = this.saving;
    }
    this.inpaintRow.hidden = !opened;
    this.floatInpaint.hidden = !opened;
    this.floatInpaint.disabled = this.inpaintButton.disabled;
  }

  /** The unsaved guard: true when it is fine to drop what is open. */
  private mayLeave(): boolean {
    if (!mustConfirmLeave(this.editing, this.opened !== null && isDirty())) return true;
    return window.confirm(discardQuestion(this.editing!.label));
  }

  /** Done: the edit bar goes, the image stays on the canvas, and Save /
   * Inpaint go with the bar so nothing can overwrite the attempt from it.
   * On Concept the panel widens back. */
  private finishEditing(): void {
    if (!this.editing || !this.mayLeave()) return;
    this.editing = null;
    this.opened = null;
    status(this.saveStatus, "");
    this.syncEditBar();
  }

  /**
   * The retouch feature: open ONE attempt in Piskel, and remember where it
   * came from so Save writes it back over itself and Inpaint can send the
   * canvas for a masked redraw. `reload` is the attempt coming back after an
   * inpaint: no unsaved question, the edit carries on. Opening the same
   * attempt again (a second click) leaves the canvas alone.
   */
  async editAttempt(unit: string, target: string, attempt: Attempt,
                    animate?: AnimateCell, mode: "pixels" | "inpaint" = "pixels", reload = false): Promise<void> {
    const open = this.opened;
    if (!reload && open && open.unit === unit && open.target === target && open.index === attempt.index
        && open.animate?.action === animate?.action && open.animate?.direction === animate?.direction) return;
    if (!reload && !this.mayLeave()) return;
    const label = `${unit} ${animate ? `${animate.action} ${animate.direction} ` : ""}${target}:${String(attempt.index).padStart(2, "0")}`;
    this.editing = { label };
    if (!reload) this.opened = null;
    this.syncEditBar();
    if (!attempt.url) {
      status(this.saveStatus, `${target}:${attempt.index} has no image to edit`, "bad");
      return;
    }
    if (!reload) status(this.saveStatus, `opening ${attempt.file} ...`);
    try {
      // The frame routes are positional, so the URL the state gave is the
      // index to write back to - not the attempt's own number.
      const position = Number(/\/(\d+)\.png$/.exec(attempt.url)?.[1] ?? NaN);
      if (!Number.isInteger(position)) throw new Error(`cannot tell which frame ${attempt.url} is`);
      const loaded = await openAnimation(label, [`${attempt.url}?t=${Date.now()}`], DEFAULT_FPS);
      setFrameTags([null]);
      this.opened = { unit, target, index: attempt.index, file: attempt.file, position,
        width: loaded.width, height: loaded.height, animate };
      if (reload) {
        status(this.saveStatus, `the inpaint came back: ${attempt.file} is on the canvas as it was saved`, "ok");
      } else {
        status(this.saveStatus,
          `${attempt.file}, ${loaded.width}×${loaded.height} - `
          + (mode === "inpaint" ? "rub out the area to redraw with the eraser, then press Inpaint"
            : "paint it, then Save; rub an area out and Inpaint redraws it"), "ok");
      }
    } catch (error) {
      status(this.saveStatus, errorText(error), "bad");
    } finally {
      this.syncEditBar();
    }
  }

  /** The one frame in the editor, alpha binarised - the canvas as it stands. */
  private async canvas(): Promise<{ png: Blob; rounded: number } | null> {
    const opened = this.opened;
    if (!opened) return null;
    const count = frameCount();
    const [width, height] = frameSize();
    if (count !== 1) {
      status(this.saveStatus, `the editor has ${count} frames; an attempt is one image`, "bad");
      return null;
    }
    if (width !== opened.width || height !== opened.height) {
      status(this.saveStatus, `the canvas is ${width}×${height}, the attempt is ${opened.width}×${opened.height}; resize back first`, "bad");
      return null;
    }
    const frames = await exportFrames();
    const frame = frames[0]!;
    return { png: frame.png, rounded: frame.alpha.rounded };
  }

  /** Save the canvas over the attempt it came from. */
  private async save(): Promise<void> {
    const attempt = this.opened;
    if (!attempt) return;
    this.saving = true;
    this.syncEditBar();
    status(this.saveStatus, `saving ${attempt.file} ...`);
    try {
      const canvas = await this.canvas();
      if (!canvas) return;
      if (attempt.animate) {
        await putAnimateImage(attempt.unit, attempt.animate.action, attempt.animate.direction,
          attempt.target, attempt.index, canvas.png);
      } else {
        await putConceptImage(attempt.unit, attempt.target, attempt.index, canvas.png);
      }
      markSaved();
      status(this.saveStatus,
        `${attempt.file} saved; erased pixels became white. Its record keeps the seed it was `
        + "generated from and now says it was edited by hand.", "ok");
      // Editing carries on: the tab re-reads so the thumbnail shows the paint.
      await (attempt.animate ? this.animate.runFinished() : this.concept.runFinished());
    } catch (error) {
      status(this.saveStatus, `save stopped - ${errorText(error)}`, "bad");
    } finally {
      this.saving = false;
      this.syncEditBar();
    }
  }

  /** Send the canvas for a masked redraw: what was rubbed out is redrawn. */
  private async inpaint(): Promise<void> {
    const attempt = this.opened;
    if (!attempt) return;
    if (!window.confirm("Inpaint: this starts a GPU job on ComfyUI (one at a time on the card).\n\n"
      + "The area you rubbed out is redrawn; everything else is pasted back untouched.\n\nContinue?")) return;
    this.inpaintButton.disabled = true;
    this.syncEditBar();
    status(this.saveStatus, "sending the canvas ...");
    try {
      const canvas = await this.canvas();
      if (!canvas) return;
      const options = {
        prompt: this.inpaintPrompt.value.trim() || undefined,
        strength: Number(this.inpaintStrength.value),
      };
      const cell = attempt.animate;
      const result = cell
        ? await inpaintAnimate(attempt.unit, cell.action, cell.direction, attempt.target, attempt.index, canvas.png, options)
        : await inpaintAttempt(attempt.unit, attempt.target, attempt.index, canvas.png, options);
      if ("refused" in result) {
        const { refused } = result;
        status(this.saveStatus, refused.status === 409
          ? `GPU busy with ${refused.run?.id ?? "another run"}; one at a time`
          : `${refused.status}: ${refused.error}`, refused.status === 409 ? "warn" : "bad");
        if (refused.run) this.follow(refused.run);
        return;
      }
      status(this.saveStatus, `inpainting ${attempt.file}: run ${result.run.id}`, "ok");
      this.follow(result.run);
    } catch (error) {
      status(this.saveStatus, `inpaint stopped - ${errorText(error)}`, "bad");
    } finally {
      this.inpaintButton.disabled = false;
      this.syncEditBar();
    }
  }

  // --- health --------------------------------------------------------------------
  private async refreshHealth(): Promise<void> {
    try {
      const health = await getHealth();
      this.health.textContent = health.gpu_run ? `GPU: ${health.gpu_run}` : `API v${health.version}`;
      this.health.className = `studio-health ${health.gpu_run ? "warn" : "ok"}`;
      this.health.title = `${health.work_dir}\nComfyUI at ${health.comfyui_url}`;
      // A GPU run started anywhere (the CLI, another tab) holds every tab's
      // GPU buttons; one that ended frees them.
      if (health.gpu_run !== this.gpuRunSeen) {
        this.gpuRunSeen = health.gpu_run;
        let run: RunRecord | null = null;
        if (health.gpu_run) {
          try {
            run = await getRun(health.gpu_run);
          } catch {
            run = null;
          }
        }
        this.concept.setBusy(run);
        this.animate.setBusy(run);
      }
    } catch (error) {
      this.health.textContent = "API unreachable";
      this.health.className = "studio-health bad";
      this.health.title = errorText(error);
    }
  }

  // --- the run footer ------------------------------------------------------------
  private follow(run: RunRecord): void {
    this.stopFollowing?.();
    const initial = viewOf(run);
    this.renderRun(initial);
    this.stopFollowing = followRun(initial, (view) => {
      this.renderRun(view);
      if (view.status !== "queued" && view.status !== "running") {
        void this.refreshRuns();
        void this.refreshHealth();
        // Every tab re-reads: a run started anywhere changes what they show
        // (a new attempt, a measured palette, a sweep's frames).
        void this.concept.runFinished();
        void this.animate.runFinished();
        if (run.gpu && this.opened && succeeded(view)) this.reopenAfterInpaint(this.opened);
      }
    });
  }

  /** An inpaint rewrote the attempt that is open: show what came back. */
  private reopenAfterInpaint(attempt: OpenedAttempt): void {
    const cell = attempt.animate;
    if (cell) {
      void getAnimate(attempt.unit, cell.action, cell.direction).then((state) => {
        const row = state.frames.find((f) => f.id === attempt.target)?.candidates
          .find((a) => a.index === attempt.index);
        if (row) void this.editAttempt(attempt.unit, attempt.target, row, cell, "pixels", true);
      }).catch(() => undefined);
    } else {
      void getConcept(attempt.unit).then((state) => {
        const rows = attempt.target === "base"
          ? state.base.candidates : state.views[attempt.target]?.candidates ?? [];
        const row = rows.find((a) => a.index === attempt.index);
        if (row) void this.editAttempt(attempt.unit, attempt.target, row, undefined, "pixels", true);
      }).catch(() => undefined);
    }
  }

  private renderRun(view: RunView): void {
    this.followed = view;
    const active = view.status === "queued" || view.status === "running";
    this.runDot.className = `studio-run-dot ${view.status}`;
    this.runLine.textContent = `${view.id}${view.gpu ? " (GPU)" : ""} · ${describeRun(view)}`;
    this.runLine.className = `studio-run-line ${runTone(view.status)}`.trim();
    this.runBar.style.width = `${view.percent ?? 0}%`;
    this.runBar.className = `studio-bar-fill ${view.status}`;
    this.runCancel.hidden = !active;
    this.runCancel.disabled = false;
    this.runLog.textContent = view.log.join("\n");
    this.runLog.scrollTop = this.runLog.scrollHeight;
  }

  private async cancelFollowed(): Promise<void> {
    const view = this.followed;
    if (!view) return;
    this.runCancel.disabled = true;
    try {
      const result = await cancelRun(view.id);
      if ("refused" in result) this.runLine.textContent = `${result.refused.status}: ${result.refused.error}`;
    } catch (error) {
      this.runLine.textContent = `could not cancel - ${errorText(error)}`;
      this.runCancel.disabled = false;
    } finally {
      void this.refreshHealth();
    }
  }

  private async refreshRuns(): Promise<void> {
    try {
      const runs = (await listRuns()).slice(0, 8);
      this.runsList.replaceChildren(
        ...runs.map((run) =>
          el("li", { class: `studio-run ${run.status}` },
            el("a", {
            href: "#",
            // The tally is the hover, not the line: the list is one row per
            // run and must stay scannable, but "0 of 8 cells done" is the
            // first thing wanted about a row that says `failed`.
            title: [run.argv.join(" "), describeCells(run.cells ?? null)].filter(Boolean).join("\n"),
            onclick: (event: Event) => { event.preventDefault(); void this.followById(run.id); },
          }, `${run.status}  ${run.stage}${run.gpu ? " (GPU)" : ""}  ${run.started.replace("T", " ").slice(5, 16)}`))),
      );
      if (!runs.length) this.runsList.replaceChildren(el("li", { class: "studio-run" }, "no runs yet"));
      // Nothing followed yet: pick up a run that is still going.
      const live = runs.find((run) => run.status === "running" || run.status === "queued");
      if (!this.followed && live) this.follow(live);
    } catch (error) {
      this.runsList.replaceChildren(el("li", { class: "studio-run failed" }, errorText(error)));
    }
  }

  private async followById(id: string): Promise<void> {
    try {
      this.follow(await getRun(id));
    } catch (error) {
      this.runLine.textContent = errorText(error);
    }
  }

  // --- the palette strip -----------------------------------------------------------
  private async refreshPalette(): Promise<void> {
    try {
      const answer = projectPalette(await getSettings());
      if (!answer.palette) {
        this.swatches.replaceChildren(el("span", { class: "studio-help", title: answer.reason }, "no palette"));
        return;
      }
      const { name, colours } = answer.palette;
      setProjectPalette(name, colours);
      this.swatches.title = `${name}: ${colours.length} colours - click for the pen colour, right-click for the second colour`;
      this.swatches.replaceChildren(
        ...colours.map((colour) => el("span", {
          class: "studio-swatch", title: colour, style: `background:${colour}`,
          onclick: () => selectColour(colour),
          oncontextmenu: (event: Event) => { event.preventDefault(); selectColour(colour, true); },
        })));
    } catch (error) {
      this.swatches.replaceChildren(el("span", { class: "studio-help", title: errorText(error) }, "no palette"));
    }
  }
}
