// The Matrix tab as DOM (the action matrix feature). Top to bottom: the unit,
// the prerequisite banner, the actions, the directions with the "all 8" toggle,
// what the selection costs, Generate, and the whole five-by-eight grid. Plain
// DOM like the board and the Concept tab, for the same reason; every decision
// worth testing lives in matrix.ts and this file only builds nodes and wires
// handlers.
//
// The tab LAUNCHES and the Board REVIEWS (constraint 7). A press is
// `POST /jobs {"stage": "matrix"}` - the same route `anim matrix` takes and
// every other stage's button takes, so the 409 for a busy card, the 423 for a
// pending gate and the run's log all come for free. There is no
// `POST /matrix/.../generate` and there is nothing to accept or reject here.
//
// A blocked cell is never a silent grey square: it carries the stage's own
// sentence as text AND as its title, and it wears the `.locked` class the
// Concept tab already styles, so the two tabs refuse in one visual language.

import {
  ApiError,
  getMatrix,
  listCuration,
  startRun,
  type MatrixCell,
  type MatrixState,
  type RunRecord,
} from "./api";
import { isActive } from "./concept";
import { describeRun, runTone, succeeded, type RunView } from "./sse";
import {
  ACTIONS,
  SIDE_DIRECTIONS,
  actionReason,
  argsFor,
  blockedText,
  cellClass,
  closureText,
  costText,
  defaultSelection,
  directionReason,
  frameAt,
  generateBlocked,
  gridOf,
  phaseChips,
  runnableKeys,
  selectionText,
  setAllDirs,
  toggleAction,
  toggleDirection,
  type Selection,
} from "./matrix";

const SCALE = 2;

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

/** What the panel lends the tab: following the run it starts, opening a
 * finished cell in the editor, the health line, and the Board - which is
 * where a pending gate is accepted, so the banner can send the owner there. */
export interface MatrixHost {
  follow(run: RunRecord): void;
  openJob(name: string): void;
  refreshHealth(): void;
  showBoard(): void;
}

export class MatrixTab {
  private state: MatrixState | null = null;
  private units: string[] = [];
  private selection: Selection = defaultSelection(null);
  private busy: RunRecord | null = null;
  /** Stops whatever cell is playing inline; one player at a time. */
  private stopPlaying: (() => void) | null = null;

  readonly root = el("div", { class: "matrix" });
  private readonly unitSelect = el("select", { class: "studio-select", title: "the unit (GET /curation)" });
  private readonly banner = el("div", { class: "studio-status" });
  private readonly boardButton = el("button", {
    type: "button", class: "studio-button",
    title: "the Board is where the base and its views are accepted - the things a cell waits for",
    onclick: () => this.host.showBoard(),
  }, "open the Board");
  private readonly actionsBox = el("div", { class: "studio-row matrix-pick" });
  private readonly directionsBox = el("div", { class: "studio-row matrix-pick" });
  private readonly allDirs = el("input", {
    type: "checkbox",
    title: "every one of the side elevation's eight facings (--all-dirs); off is s0 alone",
  });
  private readonly costLine = el("div", { class: "studio-help matrix-cost" });
  private readonly generateButton = el("button", {
    type: "button", class: "studio-button gpu",
    title: "anim matrix <unit>: rig → clip → pick --phases → post for every selected cell"
      + "\nStarts a GPU job on ComfyUI - one at a time.",
    disabled: true,
    onclick: () => void this.generate(),
  }, "Generate");
  private readonly busyLine = el("span", { class: "studio-status warn" });
  private readonly statusLine = el("div", { class: "studio-status" });
  private readonly grid = el("div", { class: "board-grid" });
  private readonly footLine = el("div", { class: "studio-help" });

  constructor(private readonly host: MatrixHost) {
    this.unitSelect.addEventListener("change", () => { this.say(""); void this.load(); });
    this.allDirs.addEventListener("change", () => this.select(setAllDirs(this.selection, this.allDirs.checked)));
    this.root.replaceChildren(
      el("div", { class: "studio-row" },
        this.unitSelect,
        el("button", { type: "button", class: "studio-button", title: "GET /curation again", onclick: () => void this.refreshUnits() }, "↻"),
        el("button", { type: "button", class: "studio-button", title: "GET /matrix/{unit} again", onclick: () => void this.load() }, "reload the matrix")),
      el("div", { class: "studio-row" }, this.banner, this.boardButton),
      el("div", { class: "matrix-picks" },
        el("span", { class: "matrix-pick-label", title: "the rows: config.STUDIO_ACTIONS" }, "actions"),
        this.actionsBox,
        el("span", { class: "matrix-pick-label", title: "the columns: the side elevation's eight facings" }, "directions"),
        this.directionsBox),
      this.costLine,
      el("div", { class: "studio-row" }, this.generateButton, this.busyLine),
      this.statusLine,
      this.grid,
      this.footLine,
    );
  }

  // --- loading ---------------------------------------------------------------
  async start(): Promise<void> {
    await this.refreshUnits();
  }

  async refreshUnits(): Promise<void> {
    const before = this.unitSelect.value;
    try {
      const listing = await listCuration();
      this.units = listing.units.map((u) => u.unit);
      this.unitSelect.replaceChildren(
        el("option", { value: "" }, this.units.length ? "pick a unit" : "no units in work/"),
        ...listing.units.map((u) => el("option", { value: u.unit }, u.unit)),
      );
      if (this.units.includes(before)) this.unitSelect.value = before;
      else if (this.units.length === 1) this.unitSelect.value = this.units[0]!;
    } catch (error) {
      this.say(`could not list units - ${errorText(error)}`, "bad");
      return;
    }
    await this.load();
  }

  private unit(): string {
    return this.unitSelect.value;
  }

  private async load(): Promise<void> {
    const unit = this.unit();
    if (!unit) {
      this.state = null;
      this.draw();
      return;
    }
    try {
      const state = await getMatrix(unit);
      // A different unit is a different grid, so the selection goes back to
      // the stage's own default rather than carrying a tick that made sense
      // for the last unit's accepted views.
      const same = this.state?.unit === state.unit;
      this.state = state;
      if (!same) this.selection = defaultSelection(state);
    } catch (error) {
      this.state = null;
      this.say(`could not read /matrix/${unit} - ${errorText(error)}`, "bad");
    }
    this.draw();
  }

  /** The panel tells the tab a run ended, so the cells reload - and, with the
   * finished view, what came of it. A matrix is minutes of card per cell, so
   * "0 of 8 cells done, 8 failed" is the first thing wanted when the grid
   * comes back unchanged. */
  async runFinished(view?: RunView): Promise<void> {
    this.setBusy(null);
    if (view && !succeeded(view)) this.say(describeRun(view), runTone(view.status));
    if (this.state) await this.load();
  }

  /** The run holding the GPU (or null). A matrix is minutes of card per cell,
   * and the `/health` poll sees a run started anywhere - the CLI included -
   * so Generate waits for whatever holds the card, not only for this tab. */
  setBusy(run: RunRecord | null): void {
    this.busy = isActive(run) ? run : null;
    this.busyLine.textContent = this.busy
      ? `GPU: ${this.busy.stage} ${this.busy.id}${this.busy.percent !== null ? ` ${this.busy.percent}%` : ""} - Generate waits for it`
      : "";
    this.enableButtons();
  }

  private select(selection: Selection): void {
    this.selection = selection;
    this.draw();
  }

  private say(text: string, kind: "" | "ok" | "warn" | "bad" = ""): void {
    this.statusLine.textContent = text;
    this.statusLine.className = `studio-status ${kind}`.trim();
  }

  /** Every GPU button of the tab: off while a run holds the card, without a
   * unit, or when the selection is one the stage would refuse - and the
   * reason is the button's title, so the refusal is readable before the press. */
  private enableButtons(): void {
    const why = generateBlocked(this.selection, this.state, this.busy);
    const cells = runnableKeys(this.selection, this.state).length;
    this.generateButton.disabled = why !== null;
    this.generateButton.classList.toggle("locked", why !== null);
    this.generateButton.title = why
      ?? `anim matrix ${this.unit()}: ${cells} cell${cells === 1 ? "" : "s"}, `
        + "rig → clip → pick --phases → post each\nStarts a GPU job on ComfyUI - one at a time.";
    this.generateButton.textContent = cells ? `Generate ${cells} cell${cells === 1 ? "" : "s"}` : "Generate";
    for (const button of this.root.querySelectorAll<HTMLButtonElement>("button.gpu.matrix-again")) {
      button.disabled = !!this.busy;
    }
  }

  // --- the press ---------------------------------------------------------------
  private async generate(selection: Selection = this.selection): Promise<void> {
    const state = this.state;
    if (!state) return;
    const keys = runnableKeys(selection, state);
    if (!keys.length) return;
    const per = state.estimate_seconds || 0;
    if (!window.confirm(
      `Generate ${keys.length} cell${keys.length === 1 ? "" : "s"} of ${state.unit}:\n\n`
      + `${keys.join(", ")}\n\n`
      + `Each is a whole clip - rig, Wan, pick --phases, post - about ${Math.round(per)} s warm, `
      + `so roughly ${Math.max(1, Math.round((keys.length * per) / 60))} min on the card.\n`
      + "This starts a GPU job on ComfyUI (one at a time on the card); the run checkpoints "
      + "after every cell, so Cancel keeps what finished.\n\nContinue?")) return;
    this.say(`starting the matrix: ${keys.length} cell(s) ...`);
    try {
      const result = await startRun("matrix", argsFor(state.unit, selection));
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
      this.say(`matrix started: run ${result.run.id} - ${keys.length} cell(s), `
        + "checkpointed after each one", "ok");
      this.host.follow(result.run);
      this.setBusy(result.run);
    } catch (error) {
      this.say(`could not start - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }

  /** One cell again - what a failed cell offers. The same stage and the same
   * route, asked for exactly that action and that direction. */
  private async generateOne(cell: MatrixCell): Promise<void> {
    await this.generate({ actions: [cell.action], directions: [cell.direction], allDirs: false });
  }

  // --- drawing -----------------------------------------------------------------
  private draw(): void {
    this.stopPlaying?.();
    const state = this.state;
    this.banner.textContent = blockedText(state);
    this.banner.className = `studio-status ${!state || !state.selectable ? "warn" : "ok"}`;
    this.boardButton.hidden = !state || state.selectable === Object.keys(state.cells).length;
    this.drawActions(state);
    this.drawDirections(state);
    this.costLine.textContent = state
      ? `${selectionText(this.selection, state)} · ${costText(this.selection, state)}`
      : "";
    this.enableButtons();
    this.drawGrid(state);
    this.footLine.textContent = state
      ? `${state.rig.name} rig · ${state.max_cells ? `at most ${state.max_cells} cells a run (the #277 hold)` : "no cell cap"}`
        + ` · ${state.path}${state.updated ? ` · updated ${state.updated.replace("T", " ").slice(0, 16)}` : ""}`
      : "";
  }

  private drawActions(state: MatrixState | null): void {
    this.actionsBox.replaceChildren(...ACTIONS.map(({ action, verb }) => {
      const why = actionReason(state, action);
      const box = el("input", { type: "checkbox" });
      box.checked = this.selection.actions.includes(action);
      box.disabled = !state || why !== null;
      box.addEventListener("change", () => this.select(toggleAction(this.selection, action)));
      return el("label", {
        class: `matrix-pick-item${why ? " locked" : ""}`,
        title: why ?? `pipeline verb: ${verb} - the job is ${state?.unit ?? "<unit>"}_${verb}[_<dir>]`,
      }, box, ` ${action}`);
    }));
  }

  private drawDirections(state: MatrixState | null): void {
    const directions = state?.directions?.length ? state.directions : SIDE_DIRECTIONS;
    this.allDirs.checked = this.selection.allDirs;
    this.allDirs.disabled = !state;
    this.directionsBox.replaceChildren(
      ...directions.map((direction) => {
        const why = directionReason(state, direction);
        const box = el("input", { type: "checkbox" });
        box.checked = this.selection.directions.includes(direction);
        box.disabled = !state || why !== null;
        box.addEventListener("change", () => this.select(toggleDirection(this.selection, direction)));
        return el("label", {
          class: `matrix-pick-item${why ? " locked" : ""}`,
          title: why ?? (direction === "s0" ? "the side view game-02 ships" : direction),
        }, box, ` ${direction}`);
      }),
      el("label", { class: "matrix-pick-item" }, this.allDirs, " all 8 directions"),
    );
  }

  private drawGrid(state: MatrixState | null): void {
    if (!state) {
      this.grid.replaceChildren(el("p", { class: "studio-help" },
        "Pick a unit: the matrix is every studio action of it in every facing, and this tab is "
        + "where a selection of those cells is generated."));
      return;
    }
    const grid = gridOf(state, this.selection);
    const table = el("table", { class: "board-table" });
    table.append(el("thead", {}, el("tr", {},
      el("th", {}, "action"),
      ...grid.directions.map((dir) => el("th", { title: dir === "s0" ? "the side view game-02 ships" : dir }, dir)))));
    const body = el("tbody");
    for (const row of grid.rows) {
      body.append(el("tr", {},
        el("th", { title: `pipeline verb: ${row.verb}` }, row.action),
        ...row.cells.map((entry) => el("td", {}, this.cellNode(state, entry.cell, entry.key, entry.selected)))));
    }
    table.append(body);
    this.grid.replaceChildren(table);
  }

  private cellNode(state: MatrixState, cell: MatrixCell | null, key: string, selected: boolean): HTMLElement {
    const classes = `board-cell matrix-cell ${cellClass(cell, selected)}`;
    if (!cell) {
      return el("div", { class: classes, title: `${key} is not in GET /matrix/${state.unit}` }, "—");
    }
    const head = el("div", { class: "board-cell-head" },
      cell.status === "done" || cell.status === "kept"
        ? el("a", { href: "#", class: "board-job", title: `open ${cell.job} in the editor`,
            onclick: (event: Event) => { event.preventDefault(); this.host.openJob(cell.job); } }, cell.job)
        : el("span", { class: "board-job" }, cell.job),
      el("span", { class: `board-status ${cell.status.replace(" ", "-")}` }, cell.status));
    // A blocked cell says the stage's own sentence, as text and as the title:
    // a grey square nobody can explain is the thing this feature refuses to
    // ship (constraint 6). `.locked` is the Concept tab's own greying.
    if (!cell.runnable) {
      return el("div", { class: classes, title: cell.reason ?? "blocked" },
        head,
        el("div", { class: "matrix-reason" }, cell.reason ?? "blocked"),
        cell.wanted.length ? el("div", { class: "studio-help" }, `wants: ${cell.wanted.join(" → ")}`) : null);
    }
    if (cell.status === "failed") {
      return el("div", { class: classes, title: cell.error ?? "failed" },
        head,
        el("div", { class: "matrix-reason" }, cell.error ?? "failed"),
        el("div", { class: "studio-row board-actions" }, this.againButton(cell)));
    }
    if (cell.status === "not run") {
      return el("div", { class: classes, title: `no ${cell.job} yet - tick ${cell.action} and ${cell.direction}, then Generate` },
        head,
        el("div", { class: "matrix-reason ready" }, "—"),
        el("div", { class: "studio-help" }, `${cell.clip ?? "no clip"} · reference ${cell.view ?? "the base"}`));
    }
    return this.doneNode(state, cell, classes, head);
  }

  /** A finished cell: the first frame at x2, what it cost, the phase tags it
   * came back with, the loop's closure, and the two things worth doing with
   * it - watch it here, or open it in the editor. */
  private doneNode(state: MatrixState, cell: MatrixCell, classes: string, head: HTMLElement): HTMLElement {
    const stamp = state.updated ?? "";
    const film = el("div", { class: "matrix-film" },
      ...cell.strip.map((url, i) => {
        const img = el("img", {
          src: `${url}?t=${stamp}`, alt: `${cell.key} frame ${i}`,
          width: String(64 * SCALE), height: String(64 * SCALE), loading: "lazy",
        });
        img.hidden = i !== 0;
        return img;
      }));
    const play = el("button", {
      type: "button", class: "studio-button",
      title: `play the ${cell.frames} finished frames at ${cell.fps} fps${cell.loop ? ", looping" : " once"}`,
      disabled: cell.strip.length < 2,
      onclick: () => this.play(cell, film, play),
    }, "Play");
    const chips = phaseChips(cell);
    const meta = [
      `${cell.frames} frame${cell.frames === 1 ? "" : "s"}`,
      cell.gpu_seconds !== null ? `gpu ${cell.gpu_seconds} s` : null,
      cell.seconds !== null ? `${cell.seconds} s total` : null,
      cell.seed !== null ? `seed ${cell.seed}` : null,
      cell.curation ? `board: ${cell.curation}` : null,
    ].filter(Boolean).join(" · ");
    const closure = closureText(cell);
    return el("div", { class: classes, title: `${cell.job} · ${cell.clip ?? "no clip"} · reference ${cell.view ?? "the base"}` },
      head,
      film,
      chips.length
        ? el("div", { class: "matrix-phases" }, ...chips.map((chip) =>
            el("span", { class: "matrix-phase", title: `frames ${chip.frames.join(", ")}` }, chip.tag)))
        : el("div", { class: "studio-help" }, "no phase tags (pick --phases did not tag it)"),
      el("div", { class: "studio-help board-meta" }, meta),
      // Loop closure is INFORMATION: idle, walk and run are cycles and Wan
      // does not know it, so a cycle that does not close is worth saying and
      // not worth calling a failure.
      closure ? el("div", { class: `studio-help matrix-closure${closure.startsWith("loop does not") ? " open" : ""}` }, closure) : null,
      cell.status === "kept" ? el("div", { class: "studio-help" }, "picks kept - your hand-edited picks.json was not re-picked") : null,
      el("div", { class: "studio-row board-actions" },
        play,
        el("button", { type: "button", class: "studio-button",
          title: `open ${cell.job} in Piskel with its phase tags on the timeline`,
          onclick: () => this.host.openJob(cell.job) }, "Open in Piskel"),
        this.againButton(cell)),
    );
  }

  private againButton(cell: MatrixCell): HTMLButtonElement {
    return el("button", {
      type: "button", class: "studio-button gpu matrix-again",
      title: `anim matrix ${cell.key.split("/")[0]} --dirs ${cell.direction}: this one cell again`
        + "\nStarts a GPU job on ComfyUI - one at a time.",
      disabled: !!this.busy,
      onclick: () => void this.generateOne(cell),
    }, cell.status === "failed" ? "Generate again" : "Regenerate");
  }

  /** The inline player: the frames are already in the DOM, so playing is
   * showing one of them at the cell's own fps. `frameAt` does the arithmetic
   * and a non-looping action (attack, jump) stops on its last frame rather
   * than snapping back to the first. */
  private play(cell: MatrixCell, film: HTMLElement, button: HTMLButtonElement): void {
    const frames = [...film.querySelectorAll("img")];
    const stopping = button.textContent === "Stop";
    // One player at a time: whatever was running stops (and puts its own
    // button back), and a second press on the same cell is the stop.
    this.stopPlaying?.();
    if (stopping || frames.length < 2) return;
    const started = performance.now();
    let handle = 0;
    const stop = () => {
      cancelAnimationFrame(handle);
      frames.forEach((img, i) => { img.hidden = i !== 0; });
      button.textContent = "Play";
      this.stopPlaying = null;
    };
    const tick = (now: number) => {
      const n = frameAt(now - started, cell.fps, frames.length);
      frames.forEach((img, i) => { img.hidden = i !== n; });
      if (!cell.loop && n === frames.length - 1) {
        this.stopPlaying = null;
        button.textContent = "Play";
        return;
      }
      handle = requestAnimationFrame(tick);
    };
    this.stopPlaying = stop;
    button.textContent = "Stop";
    handle = requestAnimationFrame(tick);
  }
}
