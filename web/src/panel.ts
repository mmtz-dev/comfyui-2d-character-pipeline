// The studio panel: the pipeline beside the Piskel canvas. Two tabs. The
// Pipeline tab is four sections - Job (open a job's frames), Save (write them
// back), Pipeline (start stages, follow the run), Palette (the project's
// colours in Piskel's list); the Board tab is the curation board
// (board-view.ts), which widens the panel because a grid of frame strips
// needs the room. The Run section shows on both. Plain DOM, no framework:
// the page is one screen and Piskel itself is jQuery-era JavaScript, so a
// second rendering model would be a second thing to learn.

import {
  ApiError,
  getHealth,
  getJob,
  getPicks,
  getRun,
  getSettings,
  listFrames,
  listJobs,
  listRuns,
  putFrame,
  startRun,
  type FrameEntry,
  type JobRow,
  type RunRecord,
} from "./api";
import { BoardView } from "./board-view";
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
import { describeRun, followRun, viewOf, type RunView } from "./sse";

const DEFAULT_FPS = 12;
const TAB_KEY = "anim-studio.panel-tab";
type Tab = "pipeline" | "board";

/** What is open in the editor, if anything. */
interface Opened {
  job: JobRow;
  set: OpenableSet;
  entries: FrameEntry[];
  fps: number;
  width: number;
  height: number;
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
  // Pipeline
  private readonly postSource = el("select", { class: "studio-select small", title: "post --input: which frames post reads" },
    el("option", { value: "" }, "input: auto"),
    el("option", { value: "render" }, "input: render"),
    el("option", { value: "clip" }, "input: clip"));
  private readonly conceptPrompt = el("input", { class: "studio-input", type: "text", placeholder: "concept prompt (who the character is)" });
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
  private readonly tabButtons = new Map<Tab, HTMLButtonElement>();
  private tab: Tab = "pipeline";
  // Health
  private readonly health: HTMLElement;

  constructor(private readonly root: HTMLElement, health: HTMLElement) {
    this.health = health;
    this.build();
  }

  private build(): void {
    this.jobSelect.addEventListener("change", () => this.describeSelectedJob());

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
        tabButton("board", "Board", "the curation board: accept / reject / regenerate per cell and per frame")),
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
        el("div", { class: "studio-row" },
          this.conceptPrompt,
          gpu("concept", "anim concept --unit <slug> --prompt ...: candidate base characters",
            () => this.run("concept", { unit: this.slug(), prompt: this.conceptPrompt.value.trim() }))),
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
    this.showTab(remembered === "board" ? "board" : "pipeline");
    await Promise.all([this.refreshHealth(), this.refreshJobs(), this.refreshRuns(), this.refreshPalette(), this.board.start()]);
  }

  /** The Pipeline tab or the Board tab; the board widens the panel. */
  showTab(tab: Tab): void {
    this.tab = tab;
    document.body.classList.toggle("studio-board", tab === "board");
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
      status(this.saveStatus, "");
    } catch (error) {
      status(this.jobStatus, errorText(error), "bad");
    } finally {
      this.openButton.disabled = !this.selectedJob();
    }
  }

  // --- Save --------------------------------------------------------------------
  private async save(): Promise<void> {
    const opened = this.opened;
    if (!opened) return;
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
    if (stage === "concept" && !args.prompt) {
      status(this.pipelineStatus, "concept needs a prompt", "bad");
      this.conceptPrompt.focus();
      return;
    }
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
        if (this.tab === "board") void this.board.runFinished();
      }
    });
  }

  private renderRun(view: RunView): void {
    this.runTitle.textContent = `${view.id}${view.gpu ? "  (GPU)" : ""}`;
    this.runBar.style.width = `${view.percent ?? 0}%`;
    this.runBar.className = `studio-bar-fill ${view.status}`;
    const kind = view.status === "done" ? "ok" : view.status === "failed" || view.status === "lost" ? "bad" : "";
    status(this.runLine, describeRun(view), kind);
    this.runLog.textContent = view.log.join("\n");
    this.runLog.scrollTop = this.runLog.scrollHeight;
  }

  private async refreshRuns(): Promise<void> {
    try {
      const runs = (await listRuns()).slice(0, 8);
      this.runsList.replaceChildren(
        ...runs.map((run) =>
          el("li", { class: `studio-run ${run.status}` },
            el("a", { href: "#", title: run.argv.join(" "), onclick: (event: Event) => { event.preventDefault(); void this.followById(run.id); } },
              `${run.status}  ${run.stage}${run.gpu ? " (GPU)" : ""}  ${run.started.replace("T", " ").slice(5, 16)}`))),
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
