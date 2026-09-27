// The Animate tab as DOM (the animate tab feature; the grid and the sweep are
// the pipeline tabs feature's): 1 every animation of the unit as an actions x
// directions grid - click a cell to open it, shift-click to tick it, and the
// sweep box draws the ticked cells' keyframes (then, once approved, their
// in-betweens) as one run; 2 the frame row with its toolbar - clicking a
// frame also opens its picture in Piskel beside the panel; 3 the selected
// frame (big picture, its skeleton, every attempt, prompt, draw from, the
// pose slider, and Generate / Approve / Reject / Pixel edit / Inpaint);
// 4 the player (full size and game size), and the output line. Plain DOM
// like the other tabs; the logic is animate.ts.

import {
  ApiError,
  addAnimateFrame,
  approveReviewed,
  cancelRun,
  generateAnimate,
  getAnimate,
  getAnimateGrid,
  getModels,
  judgeAnimateFrame,
  mirrorAnimate,
  outputAnimate,
  putAnimateFrame,
  removeAnimateFrame,
  sweepAnimate,
  type AnimateFrame,
  type AnimateGrid,
  type AnimateRequest,
  type AnimateState,
  type Attempt,
  type Batch,
  type DrawFrom,
  type ModelsListing,
  type RunRecord,
  type SweepBatch,
} from "./api";
import { attemptCaption, busyText, isActive } from "./concept";
import {
  CELL_TEXT,
  STATUS_TEXT,
  approvedLine,
  attemptOrigin,
  batchBody,
  batchTargets,
  cellKey,
  cellMark,
  cellTitle,
  clampFrame,
  currentAttempt,
  drawFromBlocked,
  drawFromText,
  frameBody,
  frameImage,
  generateAllButton,
  mirrorOffer,
  nextPlayIndex,
  playInterval,
  playList,
  reduceAnimate,
  skeletonUrl,
  sliderText,
  startFrame,
  sweepConfirm,
  sweepPlan,
  toggleTick,
  viewLabel,
  type PlayFrame,
} from "./animate";

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

export interface AnimateCell {
  unit: string;
  action: string;
  direction: string;
}

/** What the panel lends the tab: following a run, the health line, and
 * opening one attempt in the pixel editor - to paint it, or to rub an area
 * out for Inpaint. */
export interface AnimateHost {
  follow(run: RunRecord): void;
  refreshHealth(): void;
  editPixels(cell: AnimateCell, frame: string, attempt: Attempt, mode: "pixels" | "inpaint"): void;
}

/** The selected frame's form, kept across redraws so a typed prompt or a
 * moved slider survives a state reload; dropped when the frame changes. */
interface FrameForm {
  id: string;
  prompt: HTMLTextAreaElement;
  drawFrom: HTMLSelectElement;
  skeleton: HTMLInputElement;
  slider: HTMLInputElement;
}

export class AnimateTab {
  private state: AnimateState | null = null;
  private grid: AnimateGrid | null = null;
  private unitName = "";
  private ticked: string[] = [];
  private listing: ModelsListing | null = null;
  private busy: RunRecord | null = null;
  private selected: string | null = null;
  private form: FrameForm | null = null;
  private playTimer: number | null = null;
  private playIndex = -1;
  private playing: PlayFrame[] = [];
  private action = "attack";
  private direction = "s270";

  readonly root = el("div", { class: "animate" });
  private readonly statusLine = el("div", { class: "studio-status" });
  private readonly busyLine = el("span", { class: "studio-status warn" });
  private readonly cancelButton = el("button", { type: "button", class: "studio-button", hidden: true, title: "POST /runs/{id}/cancel", onclick: () => void this.cancel() }, "Cancel");
  private readonly gridBox = el("div", { class: "animate-grid" });
  private readonly sweepBox = el("div", { class: "animate-sweep" });
  private readonly cellTitleLine = el("div", { class: "animate-cell-title" });
  private readonly toolbar = el("div", { class: "animate-toolbar" });
  private readonly strip = el("div", { class: "animate-strip" });
  private readonly detail = el("div", { class: "animate-detail" });
  private readonly outputLine = el("div", { class: "animate-output" });
  private readonly modelSelect = el("select", { class: "studio-select small", title: "the Qwen-Image-Edit checkpoint (installed ones from gpu/models.registry.tsv)" });
  private readonly lockSeed = el("input", { type: "checkbox", title: "lock the seed: resend the last one, so a prompt or pose change is the only change" });
  // the player
  private readonly playFull = el("img", { class: "animate-play-full", alt: "full size" });
  private readonly playGame = el("img", { class: "animate-play-game", alt: "game size" });
  private readonly playButton = el("button", { type: "button", class: "studio-button primary", onclick: () => this.togglePlay() }, "▶ Play");
  private readonly fps = el("input", { type: "range", min: "4", max: "24", step: "1", value: "12", class: "concept-range", title: "playback speed; the timing template is 12 fps, one tick per frame" });
  private readonly fpsText = el("span", { class: "concept-seed" }, "12 fps");
  private readonly approvedOnly = el("input", { type: "checkbox", title: "play only the approved frames" });
  private readonly loop = el("input", { type: "checkbox", title: "loop, or play once and stop" });
  private readonly playNow = el("div", { class: "studio-help" });
  private readonly gameCaption = el("figcaption", { class: "studio-help" }, "in game");

  constructor(private readonly host: AnimateHost) {
    this.approvedOnly.checked = true;
    this.loop.checked = true;
    this.fps.addEventListener("input", () => {
      this.fpsText.textContent = `${this.fps.value} fps`;
      if (this.playTimer !== null) this.startPlay();
    });
    this.approvedOnly.addEventListener("change", () => { if (this.playTimer !== null) this.startPlay(); else this.showStill(); });
    this.root.replaceChildren(
      el("div", { class: "studio-row" },
        this.modelSelect,
        el("label", { class: "concept-lock" }, this.lockSeed, " lock seed"),
        el("button", { type: "button", class: "studio-button", title: "read the grid and the animation again", onclick: () => void this.reload() }, "↻")),
      el("div", { class: "studio-row concept-busy" }, this.busyLine, this.cancelButton),
      this.statusLine,
      this.stepHead(1, "Pick an animation", "click a cell to open it · shift-click to tick it for a sweep"),
      el("div", { class: "animate-panel" },
        this.gridBox,
        el("div", { class: "animate-legend" },
          ...(["none", "keys", "keys_ok", "done"] as const).map((s) =>
            el("span", { class: `animate-chip c-${s}` }, CELL_TEXT[s])),
          el("span", { class: "animate-chip c-blocked" }, "blocked (no view)")),
        this.sweepBox),
      this.cellTitleLine,
      this.stepHead(2, "Frames: generate, check, fix", "clicking a frame opens it in Piskel"),
      el("div", { class: "animate-panel" },
        this.toolbar,
        this.strip,
        el("div", { class: "animate-legend" },
          ...(["empty", "generating", "review", "approved", "rejected"] as const).map((s) =>
            el("span", { class: `animate-chip s-${s}` }, STATUS_TEXT[s])),
          el("span", { class: "animate-key-note" }, "gold label = keyframe"))),
      this.stepHead(3, "The selected frame"),
      el("div", { class: "animate-panel" }, this.detail),
      this.stepHead(4, "Play it"),
      el("div", { class: "animate-panel animate-player" },
        el("figure", {}, el("div", { class: "animate-stage" }, this.playFull), el("figcaption", { class: "studio-help" }, "full size")),
        el("figure", {}, el("div", { class: "animate-stage small" }, this.playGame), this.gameCaption),
        el("div", { class: "animate-play-controls" },
          this.playButton,
          el("label", { class: "concept-lock" }, "speed ", this.fps, this.fpsText),
          el("label", { class: "concept-lock" }, this.approvedOnly, " approved frames only"),
          el("label", { class: "concept-lock" }, this.loop, " loop"),
          this.playNow)),
      this.outputLine,
    );
  }

  private stepHead(n: number, title: string, note?: string): HTMLElement {
    return el("div", { class: "animate-step" }, el("b", {}, String(n)), el("span", {}, title),
      note ? el("span", { class: "studio-help" }, ` ${note}`) : null);
  }

  private gpuButton(label: string, title: string, run: () => void, primary = false): HTMLButtonElement {
    return el("button", { type: "button", class: `studio-button gpu${primary ? " primary" : ""}`, title: `${title}\nStarts a GPU job on ComfyUI - one at a time.`, onclick: run }, label);
  }

  // --- loading ---------------------------------------------------------------
  async start(): Promise<void> {
    try {
      this.listing = await getModels(true);
    } catch (error) {
      this.say(`could not read /models - ${errorText(error)}`, "warn");
    }
    this.fillModels();
  }

  private fillModels(): void {
    const rows = (this.listing?.families["qwen-edit"]?.checkpoint ?? []).filter((r) => r.installed !== false);
    const fallback = this.listing?.workflow_defaults["qwen-edit"];
    this.modelSelect.replaceChildren(
      el("option", { value: "" }, `model: ${fallback ? `the workflow's own (${fallback})` : "the workflow's own"}`),
      ...rows.filter((r) => r.file !== fallback).map((r) => el("option", { value: r.file, title: r.file }, r.label || r.file)));
  }

  /** The unit bar picked a unit ("" for none). */
  async setUnit(unit: string): Promise<void> {
    const changed = unit !== this.unitName;
    this.unitName = unit;
    if (changed) this.ticked = [];
    await this.reload(changed);
  }

  private async reload(changed = false): Promise<void> {
    await Promise.all([this.loadGrid(), this.load(changed)]);
  }

  private cell(): AnimateCell {
    return { unit: this.unitName, action: this.action, direction: this.direction };
  }

  private async loadGrid(): Promise<void> {
    const unit = this.unitName;
    if (!unit) {
      this.grid = null;
      this.drawGrid();
      return;
    }
    try {
      this.grid = await getAnimateGrid(unit);
    } catch (error) {
      this.grid = null;
      this.say(`could not read /animate/${unit} - ${errorText(error)}`, "bad");
    }
    this.drawGrid();
  }

  private async load(changed: boolean): Promise<void> {
    const { unit, action, direction } = this.cell();
    if (changed) { this.selected = null; this.form = null; }
    if (!unit) {
      this.state = null;
      this.draw();
      return;
    }
    try {
      this.state = await getAnimate(unit, action, direction);
    } catch (error) {
      this.state = null;
      this.say(`could not read /animate/${unit}/${action}/${direction} - ${errorText(error)}`, "bad");
    }
    this.draw();
  }

  /** The panel tells the tab a run ended, so the frames reload. */
  async runFinished(): Promise<void> {
    this.setBusy(null);
    if (this.unitName) await this.reload();
  }

  setBusy(run: RunRecord | null): void {
    this.busy = isActive(run) ? run : null;
    this.busyLine.textContent = busyText(this.busy);
    this.cancelButton.hidden = !this.busy;
    this.cancelButton.disabled = false;
    this.enableButtons();
  }

  private enableButtons(): void {
    const off = !this.state || !!this.busy;
    for (const button of this.root.querySelectorAll<HTMLButtonElement>("button.gpu")) {
      button.disabled = off || button.classList.contains("locked");
    }
  }

  private say(text: string, kind: "" | "ok" | "warn" | "bad" = ""): void {
    this.statusLine.textContent = text;
    this.statusLine.className = `studio-status ${kind}`.trim();
  }

  private apply(state: AnimateState): void {
    this.state = state;
    this.draw();
  }

  // --- writes ----------------------------------------------------------------
  private async launch(body: AnimateRequest, what: string, frames: string[]): Promise<void> {
    const state = this.state;
    if (!state) return;
    this.say(`starting ${what} ...`);
    try {
      const result = await generateAnimate(state.unit, state.action, state.direction, body);
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          this.say(`GPU busy with ${refused.run.id} (${refused.run.stage}); one at a time`, "warn");
          this.host.follow(refused.run);
          this.setBusy(refused.run);
        } else {
          this.say(`${refused.status === 423 ? "gate" : refused.status}: ${refused.error}`, refused.status === 423 ? "warn" : "bad");
        }
        return;
      }
      this.say(`${what}: run ${result.run.id}`, "ok");
      this.apply(reduceAnimate(state, { type: "generating", frames }));
      this.host.follow(result.run);
      this.setBusy(result.run);
    } catch (error) {
      this.say(`could not start - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }

  private batch(batch: Batch): void {
    const state = this.state;
    if (!state) return;
    const targets = batchTargets(state.frames, batch);
    if (!targets.length) {
      this.say("nothing to draw for that press", "warn");
      return;
    }
    const ids = targets.map((i) => state.frames[i]!.id);
    void this.launch(batchBody(batch, this.modelSelect.value || null),
      `${targets.length} frame(s): ${targets.map((i) => state.frames[i]!.label).join(", ")}`, ids);
  }

  private generateFrame(frame: AnimateFrame): void {
    const state = this.state;
    const form = this.form;
    if (!state) return;
    const mine = form && form.id === frame.id ? form : null;
    const drawFrom = (mine?.drawFrom.value as DrawFrom | undefined) ?? frame.draw_from;
    const skeleton = mine ? mine.skeleton.checked && !!state.rig : !!state.rig;
    const blocked = drawFromBlocked(state, frame, drawFrom, skeleton);
    if (blocked) {
      this.say(blocked, "warn");
      return;
    }
    let body: AnimateRequest;
    try {
      body = frameBody({
        frame, prompt: mine?.prompt.value ?? frame.prompt, drawFrom, skeleton,
        skeletonFrame: mine ? clampFrame(state, Number(mine.slider.value)) : null,
        model: this.modelSelect.value || null, lockSeed: this.lockSeed.checked, lastSeed: state.last_seed,
      });
    } catch (error) {
      this.say(errorText(error), "bad");
      return;
    }
    void this.launch(body, `${frame.label} (frame ${frame.position})`, [frame.id]);
  }

  private async cancel(): Promise<void> {
    const run = this.busy;
    if (!run) return;
    this.cancelButton.disabled = true;
    try {
      const result = await cancelRun(run.id);
      this.say("refused" in result ? `${result.refused.status}: ${result.refused.error}` : `${run.id} cancelled`,
        "refused" in result ? "warn" : "ok");
      if ("refused" in result) this.setBusy(null);
    } catch (error) {
      this.say(`could not cancel - ${errorText(error)}`, "bad");
      this.cancelButton.disabled = false;
    } finally {
      this.host.refreshHealth();
    }
  }

  private async call(what: string, op: (s: AnimateState) => Promise<AnimateState>): Promise<void> {
    const state = this.state;
    if (!state) return;
    try {
      const next = await op(state);
      this.apply(next);
      void this.loadGrid();
      const extra = next.output.error ? ` · output: ${next.output.error}` : next.written ? ` · ${next.written.frames} frames written to work/${next.written.job}` : "";
      this.say(`${what}${extra}`, next.output.error ? "warn" : "ok");
    } catch (error) {
      this.say(`${what} - ${errorText(error)}`, "bad");
      await this.load(false);
    }
  }

  private judge(frame: AnimateFrame, verdict: "approve" | "reject", index?: number): void {
    const state = this.state;
    if (!state) return;
    const shown = index ?? frame.current;
    if (shown === null || shown === undefined) {
      this.say(`${frame.label} has no attempt to ${verdict} yet`, "warn");
      return;
    }
    this.apply(reduceAnimate(state, { type: "judge", frame: frame.id, verdict: verdict === "approve" ? "approved" : "rejected", index: shown }));
    void this.call(`${frame.label} ${verdict === "approve" ? "approved" : "rejected"}`,
      (s) => judgeAnimateFrame(s.unit, s.action, s.direction, frame.id, verdict, index));
  }

  private selectAttempt(frame: AnimateFrame, attempt: Attempt): void {
    const state = this.state;
    if (!state) return;
    this.apply(reduceAnimate(state, { type: "select", frame: frame.id, index: attempt.index }));
    void this.call(`${frame.label} shows attempt ${String(attempt.index).padStart(2, "0")}`,
      (s) => putAnimateFrame(s.unit, s.action, s.direction, frame.id, { current: attempt.index }));
  }

  // --- drawing ---------------------------------------------------------------
  private draw(): void {
    const state = this.state;
    this.cellTitleLine.textContent = state ? `${state.action} · ${state.direction} (${viewLabel(state.view)})` : "";
    if (!state) {
      this.toolbar.replaceChildren();
      this.strip.replaceChildren(el("span", { class: "studio-help" }, this.unitName ? "" : "pick a unit above"));
      this.detail.replaceChildren();
      this.outputLine.replaceChildren();
      this.stopPlay();
      this.enableButtons();
      return;
    }
    if (!this.selected || !state.frames.some((f) => f.id === this.selected)) {
      this.selected = state.frames.find((f) => f.key && f.status !== "approved")?.id ?? state.frames[0]?.id ?? null;
    }
    this.drawToolbar(state);
    this.strip.replaceChildren(...state.frames.map((frame) => this.frameCard(state, frame)));
    this.drawDetail(state);
    this.drawOutput(state);
    this.gameCaption.textContent = `${state.height} px, in game`;
    if (this.playTimer === null) this.showStill();
    this.enableButtons();
  }

  /** The grid: rows the studio's actions, columns the eight side facings. */
  private drawGrid(): void {
    const grid = this.grid;
    if (!grid) {
      this.gridBox.replaceChildren(el("span", { class: "studio-help" }, this.unitName ? "" : "pick a unit above"));
      this.drawSweep();
      return;
    }
    const head = el("tr", {}, el("th", {}),
      ...grid.directions.map((d) => el("th", { title: d.blocked ? `${d.direction}: ${d.blocked}` : `${d.direction}: drawn from the ${viewLabel(d.view)} view` },
        d.url ? el("img", { src: d.url, alt: d.view, loading: "lazy" }) : null,
        el("span", {}, viewLabel(d.view)))));
    const rows = grid.actions.map((action) => el("tr", {}, el("th", { class: "animate-grid-action" }, action),
      ...grid.directions.map(({ direction, view }) => {
        const cell = grid.cells[action]?.[direction];
        if (!cell) return el("td", {});
        const key = cellKey(action, direction);
        const open = action === this.action && direction === this.direction;
        return el("td", {
          class: `c-${cell.blocked && cell.state === "none" ? "blocked" : cell.state}${open ? " open" : ""}${this.ticked.includes(key) ? " ticked" : ""}`,
          title: cellTitle(action, direction, view, cell),
          onclick: (event: Event) => this.clickCell(action, direction, (event as MouseEvent).shiftKey),
        }, cellMark(cell));
      })));
    this.gridBox.replaceChildren(el("table", { class: "animate-grid-table" }, el("thead", {}, head), el("tbody", {}, ...rows)));
    this.drawSweep();
  }

  private clickCell(action: string, direction: string, tick: boolean): void {
    if (tick) {
      const next = toggleTick(this.ticked, cellKey(action, direction), this.grid?.max_cells ?? 0);
      if (next.refused) this.say(next.refused, "warn");
      this.ticked = next.ticked;
      this.drawGrid();
      return;
    }
    this.action = action;
    this.direction = direction;
    this.drawGrid();
    void this.load(true);
  }

  /** The sweep box: what the ticked cells would draw, as one run. */
  private drawSweep(): void {
    const grid = this.grid;
    if (!grid || !this.ticked.length) {
      this.sweepBox.replaceChildren(el("span", { class: "studio-help" },
        "Shift-click cells to tick them, then draw all their keyframes as one run. In-betweens follow once a cell's keys are approved."));
      return;
    }
    const keys = sweepPlan(grid, this.ticked, "keys");
    const between = sweepPlan(grid, this.ticked, "inbetweens");
    const keysButton = this.gpuButton(`Generate keyframes (${keys.cells.length})`,
      `anim animate --sweep ... --batch keys: every keyframe not approved yet, ~${keys.renders} render(s)`,
      () => void this.sweep("keys"), true);
    keysButton.classList.toggle("locked", !keys.cells.length || !keys.renders || !!grid.locked);
    const betweenButton = this.gpuButton(`Generate in-betweens (${between.cells.length})`,
      between.blocked.length ? `waiting on the keyframe gate: ${between.blocked.map((b) => `${b.key} - ${b.why}`).join("; ")}`
        : `anim animate --sweep ... --batch inbetweens: ~${between.renders} render(s)`,
      () => void this.sweep("inbetweens"));
    betweenButton.classList.toggle("locked", !between.cells.length || !between.renders || !!grid.locked);
    this.sweepBox.replaceChildren(
      el("div", { class: "studio-row" },
        el("b", {}, "Sweep"), ` ${this.ticked.length} ticked `,
        el("span", { class: "studio-help" }, `· keys ≈ ${keys.renders} render(s) · in-betweens ≈ ${between.renders}`)),
      el("div", { class: "studio-row" }, keysButton, betweenButton,
        el("button", { type: "button", class: "studio-button", onclick: () => { this.ticked = []; this.drawGrid(); } }, "clear ticks")),
      ...(keys.blocked.length ? [el("div", { class: "studio-status warn" },
        `skipped: ${keys.blocked.map((b) => `${b.key} - ${b.why}`).join("; ")}`)] : []));
    this.enableButtons();
  }

  private async sweep(batch: SweepBatch): Promise<void> {
    const grid = this.grid;
    if (!grid) return;
    const plan = sweepPlan(grid, this.ticked, batch);
    if (!plan.cells.length) {
      this.say("nothing in the ticked cells can be drawn by that press", "warn");
      return;
    }
    if (!window.confirm(sweepConfirm(grid.unit, plan, batch))) return;
    this.say(`starting the sweep ...`);
    try {
      const result = await sweepAnimate(grid.unit, plan.cells, batch, this.modelSelect.value || null);
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          this.say(`GPU busy with ${refused.run.id} (${refused.run.stage}); one at a time`, "warn");
          this.host.follow(refused.run);
          this.setBusy(refused.run);
        } else {
          this.say(`${refused.status === 423 ? "gate" : refused.status}: ${refused.error}`, refused.status === 423 ? "warn" : "bad");
        }
        return;
      }
      this.say(`sweep: ${plan.cells.length} animation(s), run ${result.run.id}`, "ok");
      this.host.follow(result.run);
      this.setBusy(result.run);
    } catch (error) {
      this.say(`could not start - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }

  private drawToolbar(state: AnimateState): void {
    const all = generateAllButton(state.frames);
    const generateAll = this.gpuButton(all.label, all.title, () => { if (all.batch) this.batch(all.batch); }, true);
    generateAll.classList.toggle("locked", !all.batch || !!state.locked);
    const keysOnly = this.gpuButton("Generate keyframes only", "a new attempt on every keyframe that is not approved", () => this.batch("keys"));
    keysOnly.classList.toggle("locked", !batchTargets(state.frames, "keys").length || !!state.locked);
    const rejected = this.gpuButton("Regenerate rejected", "a new attempt on every rejected frame", () => this.batch("rejected"));
    rejected.classList.toggle("locked", !state.counts.rejected || !!state.locked);
    const approveAll = el("button", { type: "button", class: "studio-button", disabled: !state.counts.review,
      title: "approve every frame waiting for review", onclick: () => void this.call("every reviewed frame approved",
        (s) => approveReviewed(s.unit, s.action, s.direction)) }, "Approve all reviewed");
    const offer = mirrorOffer(state);
    const mirror = offer ? el("button", { type: "button", class: "studio-button", disabled: offer.disabled || !!this.busy, title: offer.title,
      onclick: () => {
        if (!window.confirm(`${offer.label}?\n\n${state.mirror.warning ?? ""}\n\nThe flipped frames come back TO REVIEW; nothing is approved for you.`)) return;
        void this.call("mirrored", (s) => mirrorAnimate(s.unit, s.action, s.direction)).then(() => {
          if (this.state?.warning) this.say(`mirrored ${this.state.mirrored ?? ""} frame(s) - ${this.state.warning}`, "warn");
        });
      } }, offer.label) : null;
    const notes = [state.locked ? `locked: ${state.locked}` : null, state.reference_error, state.rig_error].filter(Boolean) as string[];
    this.toolbar.replaceChildren(
      el("div", { class: "studio-row" }, generateAll, keysOnly, rejected, approveAll, mirror,
        el("span", { class: "animate-count" }, approvedLine(state.frames))),
      ...notes.map((n) => el("div", { class: "studio-status warn" }, n)),
      ...(state.mirror.last ? [el("div", { class: "studio-help" }, `mirrored from ${state.mirror.last.from} (${state.mirror.last.frames} frames) - ${state.mirror.last.warning}`)] : []));
  }

  private frameCard(state: AnimateState, frame: AnimateFrame): HTMLElement {
    const image = frame.status === "generating"
      ? el("div", { class: "animate-spin" })
      : frameImage(frame) ? el("img", { src: frameImage(frame)!, alt: frame.label, loading: "lazy" })
        : el("span", { class: "studio-help" }, "empty");
    const regenerate = this.gpuButton("↻", `generate ${frame.label}`, () => this.generateFrame(frame));
    regenerate.classList.toggle("locked", !!state.locked);
    const approve = el("button", { type: "button", class: "studio-button", title: "approve the attempt it shows", disabled: frame.current === null || frame.status === "approved",
      onclick: (e) => { e.stopPropagation(); this.judge(frame, "approve"); } }, "✓");
    const reject = el("button", { type: "button", class: "studio-button", title: "reject the attempt it shows", disabled: frame.current === null || frame.status === "rejected",
      onclick: (e) => { e.stopPropagation(); this.judge(frame, "reject"); } }, "✗");
    regenerate.addEventListener("click", (e) => e.stopPropagation());
    return el("div", {
      class: `animate-frame${frame.key ? " key" : ""}${frame.id === this.selected ? " selected" : ""}`,
      title: `${frame.label} · frame ${frame.position} (${frame.id})`,
      onclick: () => {
        const changed = this.selected !== frame.id;
        this.selected = frame.id;
        if (changed) this.form = null;
        this.draw();
        // Selecting is editing: the picture opens in Piskel beside the panel.
        const attempt = currentAttempt(frame);
        if (attempt?.url && !attempt.file_deleted) this.host.editPixels(this.cell(), frame.id, attempt, "pixels");
      },
    },
      el("div", { class: "animate-frame-label" }, el("span", {}, frame.label), el("span", {}, String(frame.position))),
      el("div", { class: "animate-frame-pic" }, image),
      el("span", { class: `animate-chip s-${frame.status}` }, STATUS_TEXT[frame.status]),
      el("div", { class: "animate-frame-buttons" }, regenerate, approve, reject));
  }

  private formFor(state: AnimateState, frame: AnimateFrame): FrameForm {
    if (this.form && this.form.id === frame.id) return this.form;
    const prompt = el("textarea", { class: "studio-input concept-prompt", rows: "3", title: "the pose, in words - the rest of the prompt (which image is which, keep the character) is added for you" });
    prompt.value = frame.prompt;
    const drawFrom = el("select", { class: "studio-select small" },
      el("option", { value: "view" }, `the ${viewLabel(state.view)} view`),
      el("option", { value: "keys" }, "the approved keyframes either side"),
      el("option", { value: "both" }, "both"));
    drawFrom.value = frame.draw_from;
    const skeleton = el("input", { type: "checkbox", title: "pose it after the rig clip's skeleton at the slider's instant" });
    skeleton.checked = !!state.rig && frame.draw_from !== "both";
    const slider = el("input", { type: "range", min: "0", max: String(Math.max(0, (state.rig?.frames ?? 1) - 1)), step: "1", class: "animate-scrub" });
    slider.value = String(startFrame(frame));
    this.form = { id: frame.id, prompt, drawFrom, skeleton, slider };
    return this.form;
  }

  private drawDetail(state: AnimateState): void {
    const frame = state.frames.find((f) => f.id === this.selected);
    if (!frame) {
      this.detail.replaceChildren(el("span", { class: "studio-help" }, "click a frame"));
      return;
    }
    const form = this.formFor(state, frame);
    const skelImg = el("img", { alt: "the pose for this frame" });
    const sliderLabel = el("span", { class: "concept-seed" });
    const drawNote = el("div", { class: "studio-help" });
    const refresh = () => {
      const value = clampFrame(state, Number(form.slider.value));
      sliderLabel.textContent = sliderText(state, frame, value);
      form.slider.disabled = !form.skeleton.checked || !state.rig;
      if (state.rig && form.skeleton.checked) skelImg.src = skeletonUrl(state, value);
      skelImg.hidden = !state.rig || !form.skeleton.checked;
      const how = form.drawFrom.value as DrawFrom;
      const blocked = drawFromBlocked(state, frame, how, form.skeleton.checked && !!state.rig);
      drawNote.textContent = blocked ?? drawFromText(state, frame, how, form.skeleton.checked && !!state.rig);
      drawNote.className = blocked ? "studio-status warn" : "studio-help";
    };
    form.slider.oninput = refresh;
    form.skeleton.onchange = refresh;
    form.drawFrom.onchange = refresh;
    form.skeleton.disabled = !state.rig;
    refresh();

    const attempt = currentAttempt(frame);
    const big = frameImage(frame);
    const attemptsRow = el("div", { class: "animate-attempts" },
      ...(frame.candidates.length ? frame.candidates.filter((a) => !a.hidden).map((a) => {
        const shown = a.index === frame.current;
        const judged = frame.verdict?.index === a.index ? ` ${frame.verdict.state === "approved" ? "✓" : "✗"}` : "";
        return el("figure", {
          class: `animate-attempt${shown ? " current" : ""}`,
          title: `${frame.id}:${String(a.index).padStart(2, "0")} · ${[attemptOrigin(a), attemptCaption(a)].filter(Boolean).join(" · ")}\nclick to make it the frame's picture`,
          onclick: () => { if (!shown && a.url && !a.file_deleted) this.selectAttempt(frame, a); },
        }, a.url && !a.file_deleted && (a.rendered || a.source === "mirror")
          ? el("img", { src: `${a.url}?t=${(a.sha256 ?? "").slice(0, 12)}`, alt: String(a.index), loading: "lazy" })
          : el("span", { class: "studio-help" }, a.file_deleted ? "deleted" : "…"),
        el("figcaption", {}, `${String(a.index).padStart(2, "0")}${judged}`));
      }) : [el("span", { class: "studio-help" }, "no attempts yet")]));

    // the row's layout: keyframe mark, add, remove
    const phase = el("select", { class: "studio-select small", title: "the keyframe's phase label" },
      ...state.phases.map((p) => el("option", { value: p }, p)));
    phase.value = frame.phase ?? "action";
    const keyToggle = el("input", { type: "checkbox", title: "a keyframe: drawn from the view by default, and the in-betweens either side are drawn from it" });
    keyToggle.checked = frame.key;
    const setKey = () => void this.call(keyToggle.checked ? `frame ${frame.position} is a ${phase.value} keyframe` : `frame ${frame.position} is an in-between`,
      (s) => putAnimateFrame(s.unit, s.action, s.direction, frame.id, { key: keyToggle.checked, phase: keyToggle.checked ? phase.value : null }));
    keyToggle.onchange = setKey;
    phase.onchange = () => { if (keyToggle.checked) setKey(); };
    const full = state.frames.length >= state.max_frames;
    const addBefore = el("button", { type: "button", class: "studio-button", disabled: full, title: "insert an in-between before this frame",
      onclick: () => void this.call("frame added", (s) => addAnimateFrame(s.unit, s.action, s.direction, frame.position - 1)) }, "+ before");
    const addAfter = el("button", { type: "button", class: "studio-button", disabled: full, title: "insert an in-between after this frame",
      onclick: () => void this.call("frame added", (s) => addAnimateFrame(s.unit, s.action, s.direction, frame.position)) }, "+ after");
    const remove = el("button", { type: "button", class: "studio-button", disabled: state.frames.length <= 1,
      title: "take this frame out of the row; its attempts stay in the record",
      onclick: () => {
        if (!window.confirm(`Remove ${frame.label} (frame ${frame.position}) from the row?\n\nIts ${frame.candidates.length} attempt(s) stay on disk and in the record.`)) return;
        this.selected = null;
        void this.call("frame removed", (s) => removeAnimateFrame(s.unit, s.action, s.direction, frame.id));
      } }, "Remove frame");

    const generate = this.gpuButton("Generate this frame", `anim animate --frame ${frame.id} --draw-from ... [--skeleton-frame N]`, () => this.generateFrame(frame), true);
    generate.classList.toggle("locked", !!state.locked);
    const approve = el("button", { type: "button", class: "studio-button", disabled: frame.current === null || frame.status === "approved",
      onclick: () => this.judge(frame, "approve") }, "✓ Approve");
    const reject = el("button", { type: "button", class: "studio-button", disabled: frame.current === null || frame.status === "rejected",
      onclick: () => this.judge(frame, "reject") }, "✗ Reject");
    const editable = !!attempt && !!attempt.url && !attempt.file_deleted;
    const pixel = el("button", { type: "button", class: "studio-button", disabled: !editable,
      title: "open this attempt in the pixel editor (Piskel); Save writes it back over itself - an approved frame goes back to review",
      onclick: () => { if (attempt) this.host.editPixels(this.cell(), frame.id, attempt, "pixels"); } }, "🖉 Pixel edit");
    const inpaint = el("button", { type: "button", class: "studio-button", disabled: !editable,
      title: "open this attempt in the editor; rub out the area to redraw, then press Inpaint there (Z-Image, masked)",
      onclick: () => { if (attempt) this.host.editPixels(this.cell(), frame.id, attempt, "inpaint"); } }, "🖌 Inpaint");
    const reset = el("button", { type: "button", class: "studio-button", title: "back to the pre-filled sentence",
      onclick: () => { form.prompt.value = frame.default_prompt; } }, "reset");
    const resetPose = el("button", { type: "button", class: "studio-button", title: "back to the frame's instant in the timing template",
      onclick: () => { form.slider.value = String(frame.default_skeleton_frame ?? 0); refresh(); } }, "template");
    const verdict = frame.verdict ? ` · ${frame.verdict.state} ${String(frame.verdict.index).padStart(2, "0")}` : "";

    this.detail.replaceChildren(
      el("div", { class: "animate-detail-left" },
        el("div", { class: "animate-big" },
          el("figure", {}, el("div", { class: "animate-box" }, big ? el("img", { src: big, alt: frame.label })
            : el("span", { class: "studio-help" }, frame.status === "generating" ? "generating…" : "no attempt yet")),
          el("figcaption", { class: "studio-help" }, attempt ? `${frame.id}:${String(attempt.index).padStart(2, "0")} · ${attemptOrigin(attempt) || attemptCaption(attempt)}` : "full size")),
          el("figure", {}, el("div", { class: "animate-box skeleton" }, skelImg),
            el("figcaption", { class: "studio-help" }, "pose for this frame (the mannequin at this instant)"))),
        el("div", { class: "studio-help" }, "Attempts for this frame (all kept, click to swap in)"),
        attemptsRow),
      el("div", { class: "animate-detail-right" },
        el("div", { class: "animate-detail-title" }, `${frame.label} · frame ${frame.position} of ${state.frames.length}${frame.key ? " · keyframe" : ""} · ${viewLabel(state.view)} · ${state.action}`,
          el("span", { class: `animate-chip s-${frame.status}` }, `${STATUS_TEXT[frame.status]}${verdict}`)),
        el("div", { class: "studio-row" },
          el("label", { class: "concept-lock" }, keyToggle, " keyframe"), phase, addBefore, addAfter, remove),
        el("div", { class: "studio-help" }, "Prompt"),
        el("div", { class: "studio-row" }, form.prompt, reset),
        el("div", { class: "studio-help" }, "Draw from"),
        el("div", { class: "studio-row" }, form.drawFrom, el("label", { class: "concept-lock" }, form.skeleton, " skeleton")),
        drawNote,
        el("div", { class: "studio-help" }, "Pose moment in the mannequin's clip: ", sliderLabel),
        el("div", { class: "studio-row" }, form.slider, resetPose),
        el("div", { class: "studio-row" }, generate, approve, reject, pixel, inpaint)));
  }

  private drawOutput(state: AnimateState): void {
    const out = state.output;
    let text: string;
    let kind = "studio-help";
    if (out.error) { text = `The finished frames were not written: ${out.error}`; kind = "studio-status warn"; }
    else if (out.written && out.stale) { text = `work/${out.job} is older than the approvals - write it again.`; kind = "studio-status warn"; }
    else if (out.written) text = `Every frame approved - written to work/${out.job} (${out.at ?? ""}); \`anim export\` packs it.`;
    else if (out.ready) text = "Every frame approved - write them to the game layout.";
    else text = `When every frame is approved they are written to work/${out.job}, the job \`anim export\` reads.`;
    const replace = !!out.error && out.error.includes("another route");
    const write = el("button", { type: "button", class: "studio-button", disabled: !out.ready,
      title: replace ? "moves the video route's job aside to work/_replaced/ first" : "the approved frames through the pipeline's post chain into the cell's job",
      onclick: () => {
        if (replace && !window.confirm(`work/${out.job} holds this cell from the video route. Move it to work/_replaced/ and write the Animate frames?`)) return;
        void this.call("finished frames written", (s) => outputAnimate(s.unit, s.action, s.direction, replace));
      } }, replace ? "Replace and write" : out.written ? "Write again" : "Write to the game layout");
    this.outputLine.replaceChildren(el("span", { class: kind }, text), " ", write);
  }

  // --- the player -----------------------------------------------------------
  private togglePlay(): void {
    if (this.playTimer !== null) this.stopPlay();
    else this.startPlay();
  }

  private startPlay(): void {
    this.stopPlay();
    const state = this.state;
    if (!state) return;
    this.playing = playList(state.frames, this.approvedOnly.checked);
    if (!this.playing.length) {
      this.playNow.textContent = this.approvedOnly.checked ? "nothing approved yet" : "no frame has a picture yet";
      return;
    }
    this.playButton.textContent = "■ Stop";
    this.playIndex = -1;
    const step = () => {
      const next = nextPlayIndex(this.playIndex, this.playing.length, this.loop.checked);
      if (next === null) {
        this.stopPlay();
        return;
      }
      this.playIndex = next;
      this.show(this.playing[next]!);
    };
    step();
    this.playTimer = window.setInterval(step, playInterval(Number(this.fps.value)));
  }

  private stopPlay(): void {
    if (this.playTimer !== null) window.clearInterval(this.playTimer);
    this.playTimer = null;
    this.playButton.textContent = "▶ Play";
  }

  private showStill(): void {
    const state = this.state;
    if (!state) return;
    const list = playList(state.frames, this.approvedOnly.checked);
    const selected = list.find((f) => f.id === this.selected) ?? list[0];
    if (selected) this.show(selected);
    else {
      this.playFull.removeAttribute("src");
      this.playGame.removeAttribute("src");
      this.playNow.textContent = this.approvedOnly.checked ? "nothing approved yet" : "no frame has a picture yet";
    }
  }

  private show(frame: PlayFrame): void {
    if (this.playFull.getAttribute("src") !== frame.full) this.playFull.src = frame.full;
    if (frame.game) {
      if (this.playGame.getAttribute("src") !== frame.game) this.playGame.src = frame.game;
    } else this.playGame.removeAttribute("src");
    this.playNow.textContent = `${frame.label} (frame ${frame.position})`;
  }
}
