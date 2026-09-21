"""Stage: `anim curate` - the curation board from the shell

The headless side of the curation board feature (anim/curation.py has the
rules; this is only a way to type them). The API's board and this command
write the same `work/<unit>_curation.json` through the same functions.

    anim curate knight                              # print the board: gates, cells, what is waiting
    anim curate knight --accept-cell attack/s0      # a cell key is <studio action>/<direction>
    anim curate knight --reject-cell walk/s0
    anim curate knight --pending-cell attack/s0     # reopen (an accepted cell is locked)
    anim curate knight --reject-frame attack/s0/3   # drops frame 3 from the job's picks.json
    anim curate knight --accept-frame attack/s0/3
    anim curate knight --clear-frame attack/s0/3    # a cleared reject comes back into the picks
    anim curate knight --accept-gate base           # what `concept --pick` does for you
    anim curate knight --accept-gate turnaround
    anim curate knight --regenerate attack/s0       # records a new seed, prints the GPU line to run
    anim curate --mode every                        # the project setting (gates | every | end)

Several flags on one line are applied in the order above; `--regenerate` does
NOT start the GPU - it records the seed and prints `anim clip … --seed N`
(the board's regenerate button starts the run through the API instead, so
the one-at-a-time rule and the confirm dialog apply).
"""

from __future__ import annotations

import argparse
from typing import Any

from anim import config, curation
from anim.api import settings

NAME = "curate"
HELP = "accept / reject / regenerate a unit's cells and gates (the curation board)"


def _frame_ref(value: str) -> tuple[str, str, int]:
    """`attack/s0/3` -> (action, direction, 3)."""
    head, sep, tail = value.rpartition("/")
    if not sep or not tail.isdigit():
        raise SystemExit(f"anim curate: a frame is <action>/<direction>/<n>, not {value!r}")
    action, direction = curation.parse_key(head)
    return action, direction, int(tail)


def print_board(board: dict[str, Any]) -> None:
    print(f"curate: {board['unit']}  mode {board['mode']}  "
          f"gates base={board['gates']['base']} turnaround={board['gates']['turnaround']}"
          f"{'' if board['exists'] else '  (no record yet: legacy, nothing gated)'}")
    if not board["cells"]:
        print("  no cells - no job of this unit is a studio action")
    for key, cell in board["cells"].items():
        marks = cell.get("frames") or {}
        rejected = sum(1 for m in marks.values() if m == "reject")
        seed = cell.get("seed")
        print(f"  {key:<14} {cell['status']:<9} {cell.get('job') or '-':<28} "
              f"{cell.get('set') or 'no frames':<9} {len(cell.get('strip') or [])} frame(s)"
              f"{'' if not marks else f', {len(marks)} marked, {rejected} rejected'}"
              f"{'' if seed is None else f', seed {seed}'}")
    if board["waiting"]:
        print("  waiting on you:")
        for item in board["waiting"]:
            print(f"    {item['key']:<12} {item['why']}")
    else:
        print("  nothing waiting")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("unit", nargs="?", default=None, help="unit slug, e.g. knight")
    parser.add_argument("--accept-cell", action="append", default=[], metavar="KEY")
    parser.add_argument("--reject-cell", action="append", default=[], metavar="KEY")
    parser.add_argument("--pending-cell", action="append", default=[], metavar="KEY",
                        help="reopen a cell (an accepted cell is locked)")
    parser.add_argument("--accept-frame", action="append", default=[], metavar="KEY/N")
    parser.add_argument("--reject-frame", action="append", default=[], metavar="KEY/N",
                        help="also drops frame N from the job's picks.json")
    parser.add_argument("--clear-frame", action="append", default=[], metavar="KEY/N")
    parser.add_argument("--accept-gate", action="append", default=[],
                        choices=list(curation.GATES), metavar="GATE")
    parser.add_argument("--pending-gate", action="append", default=[],
                        choices=list(curation.GATES), metavar="GATE")
    parser.add_argument("--regenerate", action="append", default=[], metavar="KEY",
                        help="record a new seed for the cell and print the GPU "
                             "command; nothing is started")
    parser.add_argument("--seed", type=int, default=None,
                        help="the seed --regenerate records (default: a fresh one)")
    parser.add_argument("--mode", choices=list(curation.MODES), default=None,
                        help="set the project's curation mode (work/project.json)")


def run_stage(args: argparse.Namespace) -> int:
    work_dir = config.WORK_DIR
    if args.mode:
        current = settings.stored(work_dir)
        current["curation"] = args.mode
        settings.save(work_dir, current)
        print(f"curate: curation mode -> {args.mode} ({settings.settings_path(work_dir)})")
    edits = any([args.accept_cell, args.reject_cell, args.pending_cell, args.accept_frame,
                 args.reject_frame, args.clear_frame, args.accept_gate, args.pending_gate,
                 args.regenerate])
    if not args.unit:
        if edits:
            raise SystemExit("anim curate: a unit is required for cell, frame and gate edits")
        units = curation.list_units(work_dir)
        if not units:
            print("curate: no units with studio-action jobs under work/")
        for row in units:
            print(f"  {row['unit']:<16} {row['cells']} cell job(s)"
                  f"{'  curation.json' if row['record'] else ''}")
        return 0
    unit = args.unit
    try:
        for gate in args.pending_gate:
            curation.set_gate(unit, gate, "pending", work_dir)
            print(f"curate: gate {gate} -> pending")
        for gate in args.accept_gate:
            curation.set_gate(unit, gate, "accepted", work_dir)
            print(f"curate: gate {gate} -> accepted")
        for key in args.pending_cell:
            cell = curation.set_cell_status(unit, *curation.parse_key(key), "pending", work_dir)
            print(f"curate: {key} -> pending ({cell['job']})")
        for ref in args.clear_frame:
            action, direction, n = _frame_ref(ref)
            curation.set_frames(unit, action, direction, {n: None}, work_dir)
            print(f"curate: {ref} cleared")
        for ref in args.accept_frame:
            action, direction, n = _frame_ref(ref)
            curation.set_frames(unit, action, direction, {n: "accept"}, work_dir)
            print(f"curate: {ref} accepted")
        for ref in args.reject_frame:
            action, direction, n = _frame_ref(ref)
            cell = curation.set_frames(unit, action, direction, {n: "reject"}, work_dir)
            picks = cell.get("picks")
            print(f"curate: {ref} rejected"
                  + (f"; picks now {picks}" if picks is not None else ""))
        for key in args.reject_cell:
            cell = curation.set_cell_status(unit, *curation.parse_key(key), "rejected", work_dir)
            print(f"curate: {key} -> rejected ({cell['job']})")
        for key in args.accept_cell:
            cell = curation.set_cell_status(unit, *curation.parse_key(key), "accepted", work_dir)
            print(f"curate: {key} -> accepted ({cell['job']})")
        for key in args.regenerate:
            action, direction = curation.parse_key(key)
            cell = curation.regenerate(unit, action, direction, _print_only, work_dir,
                                       seed=args.seed)
            print(f"curate: {key} seed {cell['seed']} (was {cell['regenerated_from']}); run:")
            print(f"  anim {cell['stage']} {cell['job']} --seed {cell['seed']}")
    except curation.CurationError as exc:
        raise SystemExit(f"anim curate: {exc}") from None
    print_board(curation.board(unit, work_dir))
    return 0


def _print_only(stage: str, *, args: dict[str, Any]) -> dict[str, Any]:
    """`regenerate`'s launcher when there is nobody to start the GPU."""
    return {"stage": stage, "args": args, "started": False}


run = run_stage
