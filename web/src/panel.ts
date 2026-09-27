// The studio panel: the pipeline beside the Piskel canvas. Five tabs. The
// Pipeline tab is four sections - Job (open a job's frames), Save (write them
// back), Pipeline (start stages, follow the run), Palette (the project's
// colours in Piskel's list); the Board tab is the curation board
// (board-view.ts), the Concept tab the concept generation feature
// (concept-view.ts), the Animate tab the animate tab feature
// (animate-view.ts) and the Matrix tab the action matrix feature
// (matrix-view.ts) - all four widen the panel because a grid of images needs
// the room. The Run section shows on all five. Plain DOM, no framework:
// the page is one screen and Piskel itself is jQuery-era JavaScript, so a
// second rendering model would be a second thing to learn.

import {
  ApiError,
  getConcept,
  getHealth,
  getAnimate,
  getJob,
  getPicks,
  getRun,
  getSettings,
  listFrames,
  listJobs,
  listRuns,
  inpaintAnimate,
  inpaintAttempt,
  putAnimateImage,
  putConceptImage,
  putFrame,
  startRun,
  type Attempt,
  type FrameEntry,
  type JobRow,
  type RunRecord,
} from "./api";
import { BoardView } from "./board-view";
import { ConceptTab } from "./concept-view";
import { AnimateTab, type AnimateCell } from "./animate-view";
import { MatrixTab } from "./matrix-view";
import { chooseFrameSet, phaseLabels, type OpenableSet } from "./frames";
import { projectPalette } from "./palette";
import {
  exportFrames,
  frameCount,
  frameSize,
  openAnimation,
  setFrameTags,
  setProjectPalette,
} from "./piskel-bridge";
import { describeCells, describeRun, followRun, runTone, succeeded, viewOf, type RunView } from "./sse";

const DEFAULT_FPS = 12;
const TAB_KEY = "anim-studio.panel-tab";
type Tab = "pipeline" | "board" | "concept" | "animate" | "matrix";
/** Every tab, so a remembered one that no longer exists falls back rather
 * than hiding every section. */
const TABS: Tab[] = ["pipeline", "board", "concept", "animate", "matrix"];

/** One concept attempt open in the editor (the retouch feature): Save writes
 * it back over itself, and Inpaint sends the canvas for a masked redraw. */
interface OpenedAttempt {
  unit: string;
  target: string;
  index: number;
  file: string;
  /** Set when it is a frame attempt (the animate tab feature): `target` is
   * then the frame's id, and Save / Inpaint go back to that tab's routes. */
  animate?: AnimateCell;
}

/** What is open in the editor, if anything. */
interface Opened {
  job: JobRow | { name: string };
  set: OpenableSet | "candidates" | "views" | "animate";
  entries: FrameEntry[];
  fps: number;
  width: number;
  height: number;
  /** Set when what is open is one concept attempt, not a job's frame set. */
  attempt?: OpenedAttempt;
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

function section(title: string, tab: Tab | "both", ...children: Node[]): HTMLElement {
  return el("section", { class: "studio-section", "data-tab": tab }, el("h2", {}, title), ...children);
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
  private opened: Opened | null = null;
  private jobs: JobRow[] = [];
  private stopFollowing: (() => void) | null = null;

  // Job
  private readonly jobSelect = el("select", { class: "studio-select", title: "the job (work/<job>)" });
  private readonly jobStatus = el("div", { class: "studio-status" });
  private readonly openButton = el("button", { type: "button", class: "studio-button primary", onclick: () => void this.open() }, "Open frames");
  // Save
  private readonly saveButton = el("button", { type: "button", class: "studio-button primary", disabled: true, onclick: () => void this.save() }, "Save frames");
  private readonly saveStatus = el("div", { class: "studio-status" });
  // The retouch feature: shown only while one concept attempt is open.
  private readonly inpaintRow = el("div", { class: "studio-row", hidden: true });
  private readonly inpaintPrompt = el("input", { class: "studio-input", type: "text", placeholder: "what goes in the erased area", title: "the prompt for the masked redraw; blank keeps the attempt's own" });
  private readonly inpaintStrength = el("input", { type: "range", min: "0.3", max: "0.9", step: "0.05", value: "0.9", class: "concept-range", title: "how hard the erased region is redrawn (KSampler denoise)" });
  private readonly inpaintButton = el("button", { type: "button", class: "studio-button gpu", title: "redraw what you rubbed out; everything else is pasted back untouched\nStarts a GPU job on ComfyUI - one at a time.", onclick: () => void this.inpaint() }, "Inpaint");
  // Pipeline
  private readonly postSource = el("select", { class: "studio-select small", title: "post --input: which frames post reads" },
    el("option", { value: "" }, "input: auto"),
    el("option", { value: "render" }, "input: render"),
    el("option", { value: "clip" }, "input: clip"));
  private readonly pipelineStatus = el("div", { class: "studio-status" });
  private readonly stageButtons: HTMLButtonElement[] = [];
  // Run
  private readonly runTitle = el("div", { class: "studio-run-title" }, "no run followed");
  private readonly runBar = el("div", { class: "studio-bar-fill" });
  private readonly runLine = el("div", { class: "studio-status" });
  private readonly runLog = el("pre", { class: "studio-log" });
  private readonly runsList = el("ul", { class: "studio-runs" });
  // Palette
  private readonly paletteStatus = el("div", { class: "studio-status" });
  private readonly swatches = el("div", { class: "studio-swatches" });
  // Board (the curation board feature)
  private readonly board = new BoardView({
    follow: (run) => this.follow(run),
    openJob: (name) => void this.openByName(name),
    refreshHealth: () => void this.refreshHealth(),
  });
  // Concept (the concept generation feature)
  private readonly concept = new ConceptTab({
    follow: (run) => this.follow(run),
    refreshHealth: () => void this.refreshHealth(),
    editPixels: (unit, target, attempt) => void this.editAttempt(unit, target, attempt),
  });
  // Animate (the animate tab feature): every frame of one animation by hand,
  // so it borrows the same three things - and the editor for Pixel edit and
  // Inpaint on a frame's attempt.
  private readonly animate = new AnimateTab({
    follow: (run) => this.follow(run),
    refreshHealth: () => void this.refreshHealth(),
    editPixels: (cell, frame, attempt, mode) =>
      void this.editAttempt(cell.unit, frame, attempt, cell, mode),
  });
  // Matrix (the action matrix feature): the tab LAUNCHES cells, the Board
  // reviews them - so it borrows the panel's run following and its editor,
  // and sends the owner to the Board for the gates a blocked cell names.
  private readonly matrix = new MatrixTab({
    follow: (run) => this.follow(run),
    openJob: (name) => void this.openByName(name),
    refreshHealth: () => void this.refreshHealth(),
    showBoard: () => this.showTab("board"),
  });
  private readonly tabButtons = new Map<Tab, HTMLButtonElement>();
  private tab: Tab = "pipeline";
  private gpuRunSeen: string | null = null;
  // Health
  private readonly health: HTMLElement;

  constructor(private readonly root: HTMLElement, health: HTMLElement) {
    this.health = health;
    this.build();
  }

  private build(): void {
    this.jobSelect.addEventListener("change", () => this.describeSelectedJob());
    this.inpaintRow.replaceChildren(this.inpaintButton, this.inpaintStrength, this.inpaintPrompt);

    const cpu = (label: string, title: string, run: () => void) =>
      this.stageButton(label, title, run, false);
    const gpu = (label: string, title: string, run: () => void) =>
      this.stageButton(label, title, run, true);

    const tabButton = (tab: Tab, label: string, title: string) => {
      const button = el("button", { type: "button", class: "studio-tab", title, onclick: () => this.showTab(tab) }, label);
      this.tabButtons.set(tab, button);
      return button;
    };

    this.root.replaceChildren(
      el("div", { class: "studio-tabs" },
        tabButton("pipeline", "Pipeline", "open a job, save frames, start stages"),
        tabButton("board", "Board", "the curation board: accept / reject / regenerate per cell and per frame"),
        tabButton("concept", "Concept", "the concept generation feature: the base, its views, regenerate / img2img / edit, the model and LoRA picker"),
        tabButton("animate", "Animate", "the animate tab feature: draw every frame of one animation by hand, direction by direction, approve each, play it, ship it"),
        tabButton("matrix", "Matrix", "the action matrix: pick actions × directions and generate")),
      section(
        "Matrix",
        "matrix",
        el("p", { class: "studio-help" },
          "Rows are the studio's actions, columns the eight facings - the whole grid, whether or not it has run. "
          + "Tick what to render and Generate: one run walks the cells, checkpointing after each. A cell that cannot "
          + "render says why; accept what it names on the Board and it un-greys."),
        this.matrix.root,
      ),
      section(
        "Concept",
        "concept",
        this.concept.root,
      ),
      section(
        "Animate",
        "animate",
        this.animate.root,
      ),
      section(
        "Board",
        "board",
        el("p", { class: "studio-help" },
          "Rows are the studio's actions, columns the unit's directions. Click a frame to accept, reject, clear; a rejected frame leaves the job's picks. Accepting a cell locks it."),
        this.board.root,
      ),
      section(
        "Job",
        "pipeline",
        el("div", { class: "studio-row" },
          this.jobSelect,
          el("button", { type: "button", class: "studio-button", title: "GET /jobs again", onclick: () => void this.refreshJobs() }, "↻")),
        el("div", { class: "studio-row" }, this.openButton),
        this.jobStatus,
      ),
      section(
        "Save",
        "pipeline",
        el("p", { class: "studio-help" },
          "Writes every frame back to the set it came from, same size, alpha rounded to 0/255. Adding or removing frames is not saved back yet."),
        el("div", { class: "studio-row" }, this.saveButton),
        this.inpaintRow,
        this.saveStatus,
      ),
      section(
        "Pipeline",
        "pipeline",
        el("div", { class: "studio-row" },
          cpu("pick --phases", "anim pick <job> --phases: the key poses, tagged start / wind-up / action / wind-down / end",
            () => this.run("pick", { job: this.jobName(), phases: true })),
          cpu("post", "anim post <job>: strip or clip -> 64x64 frames",
            () => this.run("post", this.postArgs())),
          this.postSource),
        el("div", { class: "studio-row" },
          gpu("clip", "anim clip <job>: every rig frame -> one generated clip (Wan 2.2)",
            () => this.run("clip", { job: this.jobName() })),
          gpu("bakeoff", "anim bakeoff --unit <slug> --action attack: two routes x two rigs",
            () => this.run("bakeoff", { unit: this.slug(), action: "attack" }))),
        el("p", { class: "studio-help" }, "The base character and its views are on the Concept tab."),
        this.pipelineStatus,
      ),
      section(
        "Run",
        "both",
        this.runTitle,
        el("div", { class: "studio-bar" }, this.runBar),
        this.runLine,
        this.runLog,
        el("div", { class: "studio-row" },
          el("button", { type: "button", class: "studio-button", title: "GET /runs", onclick: () => void this.refreshRuns() }, "recent runs ↻")),
        this.runsList,
      ),
      section(
        "Palette",
        "pipeline",
        el("div", { class: "studio-row" },
          el("button", { type: "button", class: "studio-button", title: "GET /settings", onclick: () => void this.refreshPalette() }, "project palette ↻")),
        this.paletteStatus,
        this.swatches,
      ),
    );
  }

  private stageButton(label: string, title: string, run: () => void, gpu: boolean): HTMLButtonElement {
    const button = el("button", {
      type: "button",
      class: `studio-button ${gpu ? "gpu" : ""}`.trim(),
      title: gpu ? `${title}\nStarts a GPU job on ComfyUI - one at a time.` : title,
      disabled: true,
      onclick: () => {
        if (gpu && !window.confirm(`${label}: this starts a GPU job on ComfyUI (one at a time on the card).\n\nContinue?`)) return;
        run();
      },
    }, label);
    this.stageButtons.push(button);
    return button;
  }

  // --- lifecycle -------------------------------------------------------------
  async start(): Promise<void> {
    let remembered: string | null = null;
    try {
      remembered = localStorage.getItem(TAB_KEY);
    } catch {
      remembered = null;
    }
    this.showTab(TABS.includes(remembered as Tab) ? (remembered as Tab) : "pipeline");
    await Promise.all([this.refreshHealth(), this.refreshJobs(), this.refreshRuns(), this.refreshPalette(),
      this.board.start(), this.concept.start(), this.animate.start(), this.matrix.start()]);
  }

  /** The Pipeline tab, or one of the three grid tabs; the grids widen the panel. */
  showTab(tab: Tab): void {
    this.tab = tab;
    document.body.classList.toggle("studio-board", tab !== "pipeline");
    for (const [name, button] of this.tabButtons) button.classList.toggle("active", name === tab);
    for (const node of this.root.querySelectorAll<HTMLElement>("section[data-tab]")) {
      const owner = node.dataset.tab;
      node.hidden = owner !== "both" && owner !== tab;
    }
    try {
      localStorage.setItem(TAB_KEY, tab);
    } catch {
      // private mode; the choice just does not survive a reload
    }
  }

  /** Open a job by name (the board's "open in the editor" link). */
  private async openByName(name: string): Promise<void> {
    if (!this.jobs.some((job) => job.name === name)) await this.refreshJobs();
    this.jobSelect.value = name;
    this.describeSelectedJob();
    this.showTab("pipeline");
    await this.open();
  }

  private async refreshHealth(): Promise<void> {
    try {
      const health = await getHealth();
      this.health.textContent = health.gpu_run ? `GPU: ${health.gpu_run}` : `API v${health.version}`;
      this.health.className = `studio-health ${health.gpu_run ? "warn" : "ok"}`;
      this.health.title = `${health.work_dir}\nComfyUI at ${health.comfyui_url}`;
      // A GPU run started anywhere (the CLI, the board) holds the Concept
      // and Matrix tabs' buttons too; one that ended frees them.
      if (health.gpu_run !== this.gpuRunSeen) {
        this.gpuRunSeen = health.gpu_run;
        if (health.gpu_run) {
          let run: RunRecord | null = null;
          try {
            run = await getRun(health.gpu_run);
          } catch {
            run = null;
          }
          this.concept.setBusy(run);
          this.animate.setBusy(run);
          this.matrix.setBusy(run);
          this.board.setBusy(run);
        } else {
          this.concept.setBusy(null);
          this.animate.setBusy(null);
          this.matrix.setBusy(null);
          this.board.setBusy(null);
        }
      }
    } catch (error) {
      this.health.textContent = "API unreachable";
      this.health.className = "studio-health bad";
      this.health.title = errorText(error);
    }
  }

  // --- Job -------------------------------------------------------------------
  private async refreshJobs(): Promise<void> {
    const before = this.jobSelect.value;
    try {
      this.jobs = await listJobs();
    } catch (error) {
      status(this.jobStatus, `could not list jobs - ${errorText(error)}`, "bad");
      return;
    }
    this.jobSelect.replaceChildren(
      el("option", { value: "" }, this.jobs.length ? "pick a job" : "no jobs in work/"),
      ...this.jobs.map((job) =>
        el("option", { value: job.name }, `${job.name}  ·  ${job.stages.join(" › ") || "empty"}`)),
    );
    if (this.jobs.some((job) => job.name === before)) this.jobSelect.value = before;
    this.describeSelectedJob();
  }

  private selectedJob(): JobRow | null {
    return this.jobs.find((job) => job.name === this.jobSelect.value) ?? null;
  }

  private jobName(): string {
    return this.selectedJob()?.name ?? "";
  }

  private slug(): string {
    return this.selectedJob()?.slug ?? "";
  }

  private describeSelectedJob(): void {
    const job = this.selectedJob();
    for (const button of this.stageButtons) button.disabled = !job;
    this.openButton.disabled = !job;
    if (!job) {
      status(this.jobStatus, "");
      return;
    }
    const parts = [job.slug && job.action ? `${job.slug} / ${job.action}` : job.slug || job.action, job.direction,
      job.frames ? `${job.frames} source frames` : null, `updated ${job.updated.replace("T", " ").slice(0, 16)}`];
    status(this.jobStatus, parts.filter(Boolean).join("  ·  "));
    // post --input defaults to clip when that is what the job has.
    if (!this.postSource.value) this.postSource.value = job.stages.includes("clip") && !job.stages.includes("render") ? "clip" : "";
  }

  private async open(): Promise<void> {
    const job = this.selectedJob();
    if (!job) return;
    this.openButton.disabled = true;
    status(this.jobStatus, `opening ${job.name} ...`);
    try {
      const [detail, listing, picks] = await Promise.all([getJob(job.name), listFrames(job.name), getPicks(job.name)]);
      const chosen = chooseFrameSet(listing.stages);
      if (!chosen) {
        throw new Error(`${job.name} has no post, clip or frames PNGs yet - run a stage first`);
      }
      const fps = typeof detail.manifest.fps === "number" && detail.manifest.fps > 0 ? detail.manifest.fps : DEFAULT_FPS;
      const urls = chosen.entries.map((entry) => `${entry.url}?t=${Date.now()}`);
      const loaded = await openAnimation(`${job.name} (${chosen.set})`, urls, fps);
      this.opened = { job, set: chosen.set, entries: chosen.entries, fps, width: loaded.width, height: loaded.height };
      const tags = phaseLabels(chosen.set, chosen.entries.length, picks);
      setFrameTags(tags);
      const tagged = tags.filter(Boolean).length;
      status(this.jobStatus,
        `${job.name}: ${chosen.set}, ${loaded.frames} frames, ${loaded.width}×${loaded.height}, ${fps} fps`
        + (tagged ? `, ${tagged} phase tags` : picks ? ", no phase tags (run pick --phases)" : ""), "ok");
      this.saveButton.disabled = false;
      this.saveButton.textContent = `Save frames → ${chosen.set}`;
      this.inpaintRow.hidden = true;
      status(this.saveStatus, "");
    } catch (error) {
      status(this.jobStatus, errorText(error), "bad");
    } finally {
      this.openButton.disabled = !this.selectedJob();
    }
  }

  /**
   * The retouch feature: open ONE concept attempt in Piskel (not a job's whole
   * set), show the editor, and remember where it came from so Save writes it
   * back over itself and Inpaint can send the canvas for a masked redraw.
   */
  async editAttempt(unit: string, target: string, attempt: Attempt,
                    animate?: AnimateCell, mode: "pixels" | "inpaint" = "pixels"): Promise<void> {
    if (!attempt.url) {
      status(this.saveStatus, `${target}:${attempt.index} has no image to edit`, "bad");
      return;
    }
    const job = animate ? `${unit}_animate_${animate.action}_${animate.direction}` : `${unit}_concept`;
    const set = animate ? "animate" : target === "base" ? "candidates" : "views";
    status(this.jobStatus, `opening ${attempt.file} ...`);
    this.showTab("pipeline");
    try {
      // The frame routes are positional, so the URL the state gave is the
      // index to write back to - not the attempt's own number.
      const index = Number(/\/(\d+)\.png$/.exec(attempt.url)?.[1] ?? NaN);
      if (!Number.isInteger(index)) throw new Error(`cannot tell which frame ${attempt.url} is`);
      const loaded = await openAnimation(`${unit} ${target}:${attempt.index}`,
        [`${attempt.url}?t=${Date.now()}`], DEFAULT_FPS);
      setFrameTags([null]);
      this.opened = {
        job: { name: job }, set, fps: DEFAULT_FPS, width: loaded.width, height: loaded.height,
        entries: [{ index, file: attempt.file, url: attempt.url }],
        attempt: { unit, target, index: attempt.index, file: attempt.file, animate },
      };
      status(this.jobStatus,
        `${unit} ${target}:${String(attempt.index).padStart(2, "0")} (${attempt.file}), `
        + `${loaded.width}×${loaded.height} - `
        + (mode === "inpaint" ? "rub out the area to redraw with the eraser, then press Inpaint below"
          : "paint it, then Save; rub an area out and Inpaint redraws it"), "ok");
      this.saveButton.disabled = false;
      this.saveButton.textContent = "Save → the attempt";
      // The retouch feature's masked redraw: the Concept tab's and, since the
      // animate tab feature, a frame's attempt on the Animate tab.
      this.inpaintRow.hidden = false;
      status(this.saveStatus, "");
    } catch (error) {
      status(this.jobStatus, errorText(error), "bad");
    }
  }

  /** One frame out of the editor, alpha binarised - the canvas as it stands. */
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
  private async saveAttempt(attempt: OpenedAttempt): Promise<void> {
    this.saveButton.disabled = true;
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
      status(this.saveStatus,
        `${attempt.file} saved; erased pixels became white. Its record keeps the seed it was `
        + "generated from and now says it was edited by hand.", "ok");
      this.showTab(attempt.animate ? "animate" : "concept");
      await (attempt.animate ? this.animate.runFinished() : this.concept.runFinished());
    } catch (error) {
      status(this.saveStatus, `save stopped - ${errorText(error)}`, "bad");
    } finally {
      this.saveButton.disabled = false;
    }
  }

  /** Send the canvas for a masked redraw: what was rubbed out is redrawn. */
  private async inpaint(): Promise<void> {
    const attempt = this.opened?.attempt;
    if (!attempt) return;
    if (!window.confirm("Inpaint: this starts a GPU job on ComfyUI (one at a time on the card).\n\n"
      + "The area you rubbed out is redrawn; everything else is pasted back untouched.\n\nContinue?")) return;
    this.inpaintButton.disabled = true;
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
    }
  }

  // --- Save --------------------------------------------------------------------
  private async save(): Promise<void> {
    const opened = this.opened;
    if (!opened) return;
    if (opened.attempt) {
      await this.saveAttempt(opened.attempt);
      return;
    }
    const count = frameCount();
    if (count !== opened.entries.length) {
      status(this.saveStatus,
        `the editor has ${count} frames but ${opened.job.name}/${opened.set} has ${opened.entries.length}; `
        + "adding or removing frames is not saved back yet - match the count, or run a stage", "bad");
      return;
    }
    const [width, height] = frameSize();
    if (width !== opened.width || height !== opened.height) {
      status(this.saveStatus, `the frames are ${width}×${height} now, the set is ${opened.width}×${opened.height}; resize back before saving`, "bad");
      return;
    }
    this.saveButton.disabled = true;
    status(this.saveStatus, `saving ${count} frames ...`);
    try {
      const frames = await exportFrames();
      let rounded = 0;
      for (const frame of frames) {
        const entry = opened.entries[frame.index]!;
        const saved = await putFrame(opened.job.name, opened.set, entry.index, frame.png);
        rounded += frame.alpha.rounded + saved.alpha_rounded;
        status(this.saveStatus, `saved ${frame.index + 1}/${count} ...`);
      }
      status(this.saveStatus,
        `saved ${count} frames to ${opened.job.name}/${opened.set}` + (rounded ? ` (${rounded} soft-alpha pixels rounded)` : ""), "ok");
    } catch (error) {
      status(this.saveStatus, `save stopped - ${errorText(error)}`, "bad");
    } finally {
      this.saveButton.disabled = false;
    }
  }

  // --- Pipeline ----------------------------------------------------------------
  private postArgs(): Record<string, unknown> {
    const args: Record<string, unknown> = { job: this.jobName() };
    if (this.postSource.value) args.source = this.postSource.value;
    return args;
  }

  private async run(stage: string, args: Record<string, unknown>): Promise<void> {
    if (!args.job && !args.unit) {
      status(this.pipelineStatus, "pick a job first", "bad");
      return;
    }
    status(this.pipelineStatus, `starting ${stage} ...`);
    try {
      const result = await startRun(stage, args);
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          status(this.pipelineStatus, `GPU busy with ${refused.run.id} (${refused.run.stage}); one at a time`, "warn");
          this.follow(refused.run);
        } else if (refused.status === 409) {
          status(this.pipelineStatus, `busy: ${refused.error}`, "warn");
        } else {
          status(this.pipelineStatus, `${refused.status}: ${refused.error}`, "bad");
        }
        return;
      }
      status(this.pipelineStatus, `${stage} started: ${result.run.id}`, "ok");
      this.follow(result.run);
    } catch (error) {
      status(this.pipelineStatus, errorText(error), "bad");
    } finally {
      void this.refreshHealth();
    }
  }

  // --- Run ---------------------------------------------------------------------
  private follow(run: RunRecord): void {
    this.stopFollowing?.();
    const initial = viewOf(run);
    this.renderRun(initial);
    this.stopFollowing = followRun(initial, (view) => {
      this.renderRun(view);
      if (view.status !== "queued" && view.status !== "running") {
        void this.refreshJobs();
        void this.refreshRuns();
        void this.refreshHealth();
        // The view goes with it: a tab that started a run is where the
        // owner is looking when it ends, and "0 of 8 cells done" belongs
        // there and not only in the panel's run line.
        if (this.tab === "board") void this.board.runFinished(view);
        if (run.gpu) void this.concept.runFinished();
        if (run.gpu) void this.animate.runFinished();
        // A matrix run writes its state file after every cell, so the grid is
        // worth re-reading whenever any GPU run ends - a clip or a post from
        // elsewhere changes what a cell shows too.
        if (run.gpu) void this.matrix.runFinished(view);
        if (run.gpu && this.opened?.attempt && succeeded(view)) {
          // An inpaint rewrote the attempt that is open: show what came back.
          const attempt = this.opened.attempt;
          const cell = attempt.animate;
          if (cell) {
            void getAnimate(attempt.unit, cell.action, cell.direction).then((state) => {
              const row = state.frames.find((f) => f.id === attempt.target)?.candidates
                .find((a) => a.index === attempt.index);
              if (row) void this.editAttempt(attempt.unit, attempt.target, row, cell);
            }).catch(() => undefined);
          } else void getConcept(attempt.unit).then((state) => {
            const rows = attempt.target === "base"
              ? state.base.candidates : state.views[attempt.target]?.candidates ?? [];
            const row = rows.find((a) => a.index === attempt.index);
            if (row) void this.editAttempt(attempt.unit, attempt.target, row);
          }).catch(() => undefined);
        }
      }
    });
  }

  private renderRun(view: RunView): void {
    this.runTitle.textContent = `${view.id}${view.gpu ? "  (GPU)" : ""}`;
    this.runBar.style.width = `${view.percent ?? 0}%`;
    this.runBar.className = `studio-bar-fill ${view.status}`;
    status(this.runLine, describeRun(view), runTone(view.status));
    this.runLog.textContent = view.log.join("\n");
    this.runLog.scrollTop = this.runLog.scrollHeight;
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
    } catch (error) {
      this.runsList.replaceChildren(el("li", { class: "studio-run failed" }, errorText(error)));
    }
  }

  private async followById(id: string): Promise<void> {
    try {
      this.follow(await getRun(id));
    } catch (error) {
      status(this.runLine, errorText(error), "bad");
    }
  }

  // --- Palette -----------------------------------------------------------------
  private async refreshPalette(): Promise<void> {
    try {
      const answer = projectPalette(await getSettings());
      if (!answer.palette) {
        status(this.paletteStatus, answer.reason, "warn");
        this.swatches.replaceChildren();
        return;
      }
      const { name, colours } = answer.palette;
      setProjectPalette(name, colours);
      status(this.paletteStatus, `${name}: ${colours.length} colours, selected in Piskel's palette list`, "ok");
      this.swatches.replaceChildren(
        ...colours.map((colour) => el("span", { class: "studio-swatch", title: colour, style: `background:${colour}` })));
    } catch (error) {
      status(this.paletteStatus, `could not read /settings - ${errorText(error)}`, "bad");
    }
  }
}
