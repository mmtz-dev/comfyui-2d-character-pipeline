// The curation board as DOM: the Board tab of the studio panel. Unit picker,
// mode selector, the gates, "waiting on you", then the grid - rows the five
// studio actions, columns the unit's directions, each cell its frame strip at
// x2 nearest-neighbour with a mark per frame and accept / reject / regenerate
// for the cell. Every write goes to the API and the whole board is fetched
// again afterwards; board.ts folds the click in first so the page answers at
// once. Plain DOM like panel.ts, for the same reason. Under the gate sit the
// views of the base, accepted one by one - the rest of the human gate.

import {
  ApiError,
  getBoard,
  getSettings,
  listCuration,
  putCell,
  putGate,
  putSettings,
  putView,
  regenerateCell,
  startRun,
  type BoardCell,
  type BoardState,
  type CurationMode,
  type RunRecord,
  type StripFrame,
} from "./api";
import type { ViewTile } from "./board";
import { MODES, MODE_HELP, cycleMark, gateButtons, gatesText, gridOf, partsArgs, partsOffer, reduceBoard, tally, viewTiles, viewsText, waitingText } from "./board";
import { isActive } from "./concept";
import { describeRun, runTone, succeeded, type RunView } from "./sse";

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

/** What the panel lends the board: following a run, opening a job in Piskel. */
export interface BoardHost {
  follow(run: RunRecord): void;
  openJob(name: string): void;
  refreshHealth(): void;
}

export class BoardView {
  private board: BoardState | null = null;
  private units: string[] = [];

  readonly root = el("div", { class: "board" });
  private readonly unitSelect = el("select", { class: "studio-select", title: "the unit (GET /curation)" });
  private readonly modeSelect = el("select", { class: "studio-select small", title: "curation mode - a project setting (PUT /settings)" },
    ...MODES.map((mode) => el("option", { value: mode, title: MODE_HELP[mode] }, mode)));
  private readonly gatesLine = el("div", { class: "board-gates" });
  private readonly viewsBox = el("div", { class: "board-views" });
  /** The GPU run the panel last saw, so every `button.gpu` here waits for it
   * the way the Concept and Matrix tabs already do. */
  private busy: RunRecord | null = null;
  private readonly waitingLine = el("div", { class: "studio-status" });
  private readonly statusLine = el("div", { class: "studio-status" });
  private readonly grid = el("div", { class: "board-grid" });
  private readonly tallyLine = el("div", { class: "studio-help" });

  constructor(private readonly host: BoardHost) {
    this.unitSelect.addEventListener("change", () => { this.say(""); void this.load(); });
    this.modeSelect.addEventListener("change", () => void this.setMode(this.modeSelect.value as CurationMode));
    this.root.replaceChildren(
      el("div", { class: "studio-row" },
        this.unitSelect,
        el("button", { type: "button", class: "studio-button", title: "GET /curation again", onclick: () => void this.refreshUnits() }, "↻"),
        this.modeSelect),
      this.gatesLine,
      this.viewsBox,
      this.waitingLine,
      this.statusLine,
      this.grid,
      this.tallyLine,
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
      this.modeSelect.value = listing.mode;
      this.unitSelect.replaceChildren(
        el("option", { value: "" }, this.units.length ? "pick a unit" : "no units in work/"),
        ...listing.units.map((u) => el("option", { value: u.unit }, `${u.unit}  ·  ${u.cells} cell${u.cells === 1 ? "" : "s"}${u.record ? "" : "  (legacy, ungated)"}`)),
      );
      if (this.units.includes(before)) this.unitSelect.value = before;
      else if (this.units.length === 1) this.unitSelect.value = this.units[0]!;
    } catch (error) {
      this.say(`could not list units - ${errorText(error)}`, "bad");
      return;
    }
    await this.load();
  }

  /** Show the given unit (the panel calls this when a job is opened). */
  async show(unit: string): Promise<void> {
    if (!this.units.includes(unit)) await this.refreshUnits();
    if (this.units.includes(unit) && this.unitSelect.value !== unit) {
      this.unitSelect.value = unit;
      await this.load();
    }
  }

  private async load(): Promise<void> {
    const unit = this.unitSelect.value;
    if (!unit) {
      this.board = null;
      this.render();
      return;
    }
    try {
      this.apply(await getBoard(unit));
    } catch (error) {
      this.say(`could not read the board - ${errorText(error)}`, "bad");
    }
  }

  private apply(board: BoardState): void {
    this.board = board;
    this.modeSelect.value = board.mode;
    this.render();
  }

  private say(text: string, kind: "" | "ok" | "warn" | "bad" = ""): void {
    this.statusLine.textContent = text;
    this.statusLine.className = `studio-status ${kind}`.trim();
  }

  // --- writes ------------------------------------------------------------------
  private async setMode(mode: CurationMode): Promise<void> {
    try {
      const settings = await getSettings();
      await putSettings({ ...settings, curation: mode });
      if (this.board) this.apply(reduceBoard(this.board, { type: "mode", mode }));
      this.say(`curation mode: ${mode}`, "ok");
      await this.load();
    } catch (error) {
      this.say(`could not set the mode - ${errorText(error)}`, "bad");
    }
  }

  private async toggleGate(gate: string): Promise<void> {
    const board = this.board;
    if (!board) return;
    const state = board.gates[gate] === "accepted" ? "pending" : "accepted";
    this.apply(reduceBoard(board, { type: "gates", gates: { [gate]: state } }));
    try {
      this.apply(await putGate(board.unit, gate, state));
      this.say(`${gate} gate: ${state}`, "ok");
    } catch (error) {
      this.say(`could not set the ${gate} gate - ${errorText(error)}`, "bad");
      await this.load();
    }
  }

  /** The views step under the base gate: click cycles pending -> accepted -> rejected. */
  private async setView(tile: ViewTile): Promise<void> {
    const board = this.board;
    if (!board) return;
    this.apply(reduceBoard(board, { type: "view", view: tile.view, state: tile.next }));
    try {
      this.apply(await putView(board.unit, tile.view, tile.next));
      this.say(`view ${tile.view}: ${tile.next}${tile.next === "rejected" ? " - the matrix refuses the facings that want it" : ""}`, "ok");
    } catch (error) {
      this.say(`could not set the ${tile.view} view - ${errorText(error)}`, "bad");
      await this.load();
    }
  }

  private async mark(cell: BoardCell, frame: StripFrame): Promise<void> {
    const board = this.board;
    if (!board || cell.status === "accepted") return;
    const mark = cycleMark(frame.mark);
    this.apply(reduceBoard(board, { type: "mark", key: cell.key, index: frame.index, mark }));
    try {
      const saved = await putCell(board.unit, cell.action, cell.direction, { frames: { [String(frame.index)]: mark } });
      this.apply(reduceBoard(this.board ?? board, { type: "cell", cell: saved }));
      this.say(mark === "reject"
        ? `${cell.key}: frame ${frame.index} rejected - dropped from ${cell.job}'s picks (re-run strip / post)`
        : mark === "accept" ? `${cell.key}: frame ${frame.index} accepted` : `${cell.key}: frame ${frame.index} cleared`, "ok");
    } catch (error) {
      this.say(`could not mark frame ${frame.index} - ${errorText(error)}`, "bad");
      await this.load();
    }
  }

  private async setStatus(cell: BoardCell, status: BoardCell["status"]): Promise<void> {
    const board = this.board;
    if (!board) return;
    this.apply(reduceBoard(board, { type: "cell", cell: { ...cell, status } }));
    try {
      const saved = await putCell(board.unit, cell.action, cell.direction, { status });
      this.apply(reduceBoard(this.board ?? board, { type: "cell", cell: saved }));
      this.say(`${cell.key}: ${status}`, status === "rejected" ? "warn" : "ok");
    } catch (error) {
      this.say(`could not set ${cell.key} - ${errorText(error)}`, "bad");
      await this.load();
    }
  }

  private async regenerate(cell: BoardCell): Promise<void> {
    const board = this.board;
    if (!board || !cell.job) return;
    const stage = cell.stage ?? "render";
    if (!window.confirm(`regenerate ${cell.key} (${cell.job}) with a new seed: this starts a GPU job (anim ${stage}) on ComfyUI, one at a time on the card.\n\nContinue?`)) return;
    this.say(`regenerating ${cell.key} ...`);
    try {
      const result = await regenerateCell(board.unit, cell.action, cell.direction);
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          this.say(`GPU busy with ${refused.run.id} (${refused.run.stage}); one at a time`, "warn");
          this.host.follow(refused.run);
        } else if (refused.status === 423) {
          this.say(`gate: ${refused.error}`, "warn");
        } else {
          this.say(`${refused.status}: ${refused.error}`, "bad");
        }
        return;
      }
      this.say(`${cell.key}: ${stage} started with seed ${result.cell.seed} (was ${result.cell.regenerated_from ?? "derived"}) - run ${result.run.id}`, "ok");
      this.host.follow(result.run);
      await this.load();
    } catch (error) {
      this.say(`could not regenerate - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }


  /** The part palettes every matrix cell locks to: `post --parts` over the accepted
   * base. A CPU stage, so it queues in a CPU slot and never waits for the card -
   * which is why this button is NOT `.gpu` and stays pressable while a GPU run
   * is live. */
  private async measureParts(): Promise<void> {
    const board = this.board;
    if (!board) return;
    const offer = partsOffer(board);
    if (!offer) return;
    if (!window.confirm(offer.confirm)) return;
    this.say(`measuring ${offer.job}'s part palettes ...`);
    try {
      const result = await startRun("post", partsArgs(offer));
      if ("refused" in result) {
        const { refused } = result;
        if (refused.status === 409 && refused.run) {
          this.say(`busy with ${refused.run.id} (${refused.run.stage})`, "warn");
          this.host.follow(refused.run);
        } else if (refused.status === 423) {
          this.say(`gate: ${refused.error}`, "warn");
        } else {
          this.say(`${refused.status}: ${refused.error}`, "bad");
        }
        return;
      }
      this.say(`measuring part palettes: run ${result.run.id} - CPU, ~45 s; `
        + "the matrix cells unlock when it lands", "ok");
      this.host.follow(result.run);
    } catch (error) {
      this.say(`could not measure the part palettes - ${errorText(error)}`, "bad");
    } finally {
      this.host.refreshHealth();
    }
  }

  /** The parts button, when the base has no measured palettes. */
  private partsButton(board: BoardState): HTMLElement | null {
    const offer = partsOffer(board);
    if (!offer) return null;
    return el("button", {
      type: "button",
      class: "studio-button",
      title: offer.title,
      onclick: () => void this.measureParts(),
    }, offer.label);
  }

  /** The panel tells the board a run ended, so the strips reload - and, when
   * it has the finished view, what came of it: the run\'s own tally, so a
   * batch whose every cell failed does not read as a success. */
  async runFinished(view?: RunView): Promise<void> {
    if (view && !succeeded(view)) this.say(describeRun(view), runTone(view.status));
    if (this.board) await this.load();
  }

  /** The panel's /health poll, pushed in. Without it the
   * Board had no busy state at all, so its `regenerate` button stayed pressable
   * while another stage held the card and only found out through the 409. The
   * tab's own presses are not the problem; a run started from the CLI, the
   * Concept tab or the Matrix tab is. */
  setBusy(run: RunRecord | null): void {
    this.busy = isActive(run) ? run : null;
    this.enableGpuButtons();
  }

  private enableGpuButtons(): void {
    const held = this.busy;
    for (const button of this.root.querySelectorAll<HTMLButtonElement>("button.gpu")) {
      // A button already disabled for its own reason (no job, a pending gate)
      // stays disabled; this only adds the card being busy on top of that.
      if (button.dataset.lockedReason) continue;
      button.disabled = held !== null;
      if (held) button.title = `GPU busy with ${held.stage} ${held.id} - one at a time on the card`;
      else if (button.dataset.freeTitle) button.title = button.dataset.freeTitle;
    }
  }

  // --- drawing -----------------------------------------------------------------
  private render(): void {
    const board = this.board;
    if (!board) {
      this.gatesLine.replaceChildren();
      this.viewsBox.replaceChildren();
      this.waitingLine.textContent = "";
      this.grid.replaceChildren(el("p", { class: "studio-help" }, "Pick a unit: every job of it that is a studio action (attack, idle, walk, run, jump) is a cell."));
      this.tallyLine.textContent = "";
      return;
    }
    // board.ts decides which gate is acceptable and what it says; this only
    // builds the node (panel.ts:1 - the pure / view split).
    this.gatesLine.replaceChildren(
      el("span", { class: "board-gates-text", title: MODE_HELP[board.mode] },
        gatesText(board.gates, board.mode, board.gate_blocked ?? {})),
      ...gateButtons(board).map((gate) => el("button", {
        type: "button",
        class: `studio-button board-gate ${gate.state}${gate.blocked ? " blocked" : ""}`,
        title: gate.title,
        disabled: gate.disabled,
        onclick: () => void this.toggleGate(gate.gate),
      }, gate.label)),
    );
    if (!board.exists) {
      this.gatesLine.append(el("span", { class: "board-legacy", title: `${board.path} is written on the first edit` }, "legacy - no record, nothing gated"));
    }
    const measure = this.partsButton(board);
    if (measure) this.gatesLine.append(measure);
    this.renderViews(board);
    const waiting = waitingText(board.waiting);
    this.waitingLine.textContent = waiting;
    this.waitingLine.className = `studio-status ${board.waiting.length ? "warn" : "ok"}`;
    this.waitingLine.title = board.waiting.map((w) => `${w.key}: ${w.why}`).join("\n");

    const grid = gridOf(board);
    const table = el("table", { class: "board-table" });
    table.append(el("thead", {}, el("tr", {},
      el("th", {}, "action"),
      ...grid.directions.map((dir) => el("th", { title: dir === "s0" ? "the side view game-02 ships" : dir }, dir)))));
    const body = el("tbody");
    for (const row of grid.rows) {
      body.append(el("tr", {},
        el("th", { title: `pipeline verb: ${row.verb}` }, row.action),
        ...row.cells.map((cell, i) => el("td", {}, cell ? this.cellNode(cell) : this.emptyCell(row.verb, grid.directions[i]!)))));
    }
    table.append(body);
    this.grid.replaceChildren(table);
    const t = tally(board);
    this.tallyLine.textContent = `${t.cells} cells: ${t.accepted} accepted, ${t.rejected} rejected, ${t.pending} pending · ${t.marked} frame marks · ${board.path}`;
  }

  /** The views of the base under the base gate (the multi-view reference
   * feature): one tile per view with accept / reject per view. */
  private renderViews(board: BoardState): void {
    const text = viewsText(board);
    const head = el("div", { class: "board-views-head" },
      el("span", { class: "board-views-title" }, "views of the base"),
      el("span", { class: "studio-help", title: "anim concept --unit <unit> --views [all] then --pick-view back=NN ...; the matrix hands each direction its nearest accepted view" }, text));
    // All eight slots, always: a missing one is a tile that says how to make
    // it, not an absence. One flat sentence could never name the gap, and the
    // gap is the thing the owner has to act on (s0 wants `right`).
    const ring = el("div", { class: "board-ring" });
    for (const tile of viewTiles(board)) {
      const wants = tile.wantedBy.length ? ` · ${tile.wantedBy.join(", ")} wants it` : "";
      if (tile.missing) {
        // Two different gaps: an attempt nobody picked, and a view nobody
        // drew. The tile says which, because the answers differ.
        const what = tile.attempts ? `${tile.attempts} attempt${tile.attempts === 1 ? "" : "s"}` : "not rendered";
        ring.append(el("figure", { class: `board-tile board-view missing${tile.attempts ? " unpicked" : ""}`, title: `${tile.view}: not on the base - ${tile.how}${wants}` },
          el("div", { class: "board-tile-miss" }, what),
          el("figcaption", {}, tile.view),
          el("span", { class: "board-view-how" }, tile.attempts ? "pick on Concept" : "Concept tab"),
        ));
        continue;
      }
      const label = `${tile.view}: ${tile.state}${tile.provisional ? " (provisional)" : ""}`;
      // The mirrored view feature: a flip is marked, not silently shown as a
      // rendered view - it is the same pixels as its partner with the handed
      // gear on the other side, which is the one thing about it worth knowing.
      const mirror = tile.mirrored
        ? ` · the ${tile.mirroredFrom ?? "partner"} view flipped, no GPU spent on it`
        : "";
      ring.append(el("figure", { class: `board-tile board-view ${tile.state}${tile.provisional ? " provisional" : ""}${tile.mirrored ? " mirrored" : ""}`, title: `${label} · yaw ${tile.yaw}${wants}${mirror} · click: ${tile.next}` },
        tile.url
          ? el("img", { src: `${tile.url}?t=${board.views[tile.view]?.picked ?? ""}`, alt: `${tile.view} view`, width: String(64 * SCALE), height: String(64 * SCALE), loading: "lazy" })
          : el("div", { class: "board-tile-miss" }, "no file"),
        el("figcaption", {}, tile.view),
        ...(tile.mirrored ? [el("span", { class: "board-view-mirror" }, `⇄ ${tile.mirroredFrom ?? "mirrored"}`)] : []),
        el("button", {
          type: "button",
          class: `studio-button board-gate ${tile.state}`,
          title: `set the ${tile.view} view ${tile.next}`,
          disabled: board.mode === "end",
          onclick: () => void this.setView(tile),
        }, tile.state === "accepted" ? "accepted" : tile.state === "rejected" ? "rejected" : "accept"),
      ));
    }
    this.viewsBox.replaceChildren(head, ring);
  }

  private emptyCell(verb: string, direction: string): HTMLElement {
    const name = direction === "s0" ? `${this.board?.unit}_${verb}` : `${this.board?.unit}_${verb}_${direction}`;
    return el("div", { class: "board-cell empty", title: `no job ${name} yet - anim rig ... --action ${verb}${direction === "s0" ? "" : " --yaw/--elevation"}` }, "—");
  }

  private cellNode(cell: BoardCell): HTMLElement {
    const locked = cell.status === "accepted";
    const waiting = this.board?.waiting.some((w) => w.kind === "cell" && w.key === cell.key) ?? false;
    const strip = el("div", { class: "board-strip" });
    if (!cell.strip.length) {
      strip.append(el("span", { class: "studio-help" }, cell.stale ? "post/ is stale against the picks - re-run post" : "no frames yet"));
    }
    for (const frame of cell.strip) {
      const mark = frame.mark ?? "none";
      strip.append(el("button", {
        type: "button",
        class: `board-frame mark-${mark}`,
        title: `frame ${frame.index}${frame.mark ? `: ${frame.mark}` : ""}${locked ? " (cell accepted - locked)" : " - click: accept, reject, clear"}`,
        disabled: locked,
        onclick: () => void this.mark(cell, frame),
      },
        el("img", { src: `${frame.url}?t=${cell.decided_at ?? ""}`, alt: `frame ${frame.index}`, height: String(64 * SCALE), loading: "lazy" }),
        el("span", { class: "board-frame-n" }, String(frame.index)),
        frame.mark ? el("span", { class: "board-frame-mark" }, frame.mark === "accept" ? "✓" : "✗") : null,
      ));
    }
    const rejected = Object.values(cell.frames).filter((m) => m === "reject").length;
    const meta = [
      cell.set ? `${cell.set}: ${cell.strip.length} frame${cell.strip.length === 1 ? "" : "s"}` : "no frames",
      rejected ? `${rejected} rejected` : null,
      cell.seed !== null ? `seed ${cell.seed}` : cell.current_seed !== null ? `seed ${cell.current_seed}` : null,
      cell.rerender_requested ? "rerender requested" : null,
    ].filter(Boolean).join(" · ");
    const button = (label: string, title: string, cls: string, run: () => void, disabled = false) =>
      el("button", { type: "button", class: `studio-button ${cls}`.trim(), title, disabled, onclick: run }, label);
    return el("div", { class: `board-cell status-${cell.status}${waiting ? " waiting" : ""}` },
      el("div", { class: "board-cell-head" },
        el("a", { href: "#", class: "board-job", title: `open ${cell.job} in the editor`, onclick: (event: Event) => { event.preventDefault(); if (cell.job) this.host.openJob(cell.job); } }, cell.job ?? cell.key),
        el("span", { class: `board-status ${cell.status}` }, cell.status)),
      strip,
      el("div", { class: "studio-help board-meta" }, meta),
      el("div", { class: "studio-row board-actions" },
        locked
          ? button("reopen", "set the cell pending again so frames can be marked", "", () => void this.setStatus(cell, "pending"))
          : button("accept", "accept the cell: locks its frames", "ok", () => void this.setStatus(cell, "accepted"), !cell.set),
        button("reject", "reject the cell: it waits for a regenerate", "bad", () => void this.setStatus(cell, "rejected"), locked || cell.status === "rejected"),
        button("regenerate", `anim ${cell.stage ?? "render"} ${cell.job} --seed <new>: the cell again with a new seed\nStarts a GPU job on ComfyUI - one at a time.`, "gpu", () => void this.regenerate(cell), !cell.job)),
    );
  }
}
