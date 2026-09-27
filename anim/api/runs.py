"""Running a stage as a background job - `POST /jobs`, `/runs`, and the GPU rule.

A run is the CLI. `POST /jobs {"stage": "pick", "args": {"job": "x", "n": 3}}`
becomes `python -m anim pick x -n 3` in a subprocess with `ANIM_WORK_DIR`
pointed at the API's work/, its stdout captured line by line. That is the
whole trick: the API never re-implements a stage's argument handling or
its steps, it builds an argv from the stage's own argparse definition and
runs the same entry point the owner types. A stage that gains a flag gains it
here with no change.

Why a subprocess and not a thread calling `run_stage`: the stages print their
progress, and two threads printing into one process share one stdout; a stage
that fails hard (a segfault in onnxruntime, a torch OOM) would take the API
with it; and the GPU stages hold a `requests` session open for minutes, which a
thread cannot be told to abandon but a process can be killed.

THE GPU RULE. `base`, `render`, `clip`, `concept`, `animate`, `bakeoff` and
`run` (which renders when it can) reach ComfyUI, and the card is
24 GB: a second model load on top of a running one is an OOM twenty minutes in. So GPU stages start only
from an explicit POST - nothing here polls or prefetches - and only ONE may be
active at a time; a second is refused with 409 and the id of the one running.
CPU stages run concurrently up to `cpu_slots`, then 409 as well.

Percent: a stage reports it by printing `[n/m] ...` (what `anim run` prints
per step) or a line `progress: 42%`; the manager parses those and nothing
else. Everything is also written under work/_runs/<id>.{json,log} so `GET
/runs` survives a restart - a run the API no longer owns is reported `lost`.

OUTCOME, NOT EXIT CODE. A batch stage (`bakeoff`, an `animate` batch or sweep)
catches each cell's exception, records it and goes on to the next, because one
bad cell must not cost the other seven. It used to exit 0 for doing that, and a
run whose every cell failed read `done` at 100% here - see `anim.outcome` for
the 2026-09-24 bug. Such a stage now prints a last line `outcome: done=0
failed=8 blocked=0 of=8`; the manager keeps it on the record as `cells`, and
the run's `status` and final `percent` come from THAT rather than from the
process result. Hence the status `partial`: the run finished, and not every
cell it planned did. A stage that prints no tally is unchanged - a clean exit
is `done`, a non-zero one `failed`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from anim import cli, config, outcome
from anim.stages import STAGE_ORDER
from anim.util import read_json, write_json_in_place

# A GPU stage gets three things at once: the one-GPU-at-a-time 409, ComfyUI's
# /interrupt on Cancel (a Cancel that only killed the process would leave the
# card finishing a render nobody wants), and the GPU badge in the panel.
GPU_STAGES = frozenset({"base", "render", "clip", "concept", "animate", "bakeoff", "run"})
# `review` is a page, not a step - there is nothing to run.
RUNNABLE = [name for name in STAGE_ORDER if name != "review"]

RUNS_DIRNAME = "_runs"
ACTIVE = ("queued", "running")
# A run ends in exactly one of these, and `done` is the only full success.
# `partial` is the batch stages' own: the process finished and some of the cells
# it planned did not (`anim.outcome`, `_status_for`). `web/src/sse.ts` keeps the
# same five in `FINISHED` and refuses a `done` event carrying anything else.
FINISHED = ("done", "partial", "failed", "lost", "cancelled")
# How long a cancelled stage gets to exit on SIGTERM before SIGKILL.
CANCEL_GRACE_S = 5.0

_STEP = re.compile(r"^\[(\d+)/(\d+)\]")
_PROGRESS = re.compile(r"^progress:\s*(\d{1,3})\s*%", re.IGNORECASE)


class RunRefused(Exception):
    """Refused before a process was started. `status` is the HTTP code."""

    def __init__(self, status: int, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.extra = extra


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# What a finished process means
# ---------------------------------------------------------------------------
def _cells(value: Any) -> dict[str, int] | None:
    """A `cells` tally off disk or off a stage's line, or None. Anything that
    is not a map of ints is dropped rather than trusted - the record is a file
    on disk that a person may have opened."""
    if not isinstance(value, dict):
        return None
    counts = {str(k): int(v) for k, v in value.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return counts or None


def _status_for(cancelled: bool, returncode: int,
                cells: dict[str, int] | None) -> str:
    """The run's end status.

    A cancel is a cancel whatever the tally says. Otherwise the TALLY WINS over
    the exit code when there is one, because it is the finer signal and the
    reason this field stopped lying: a batch stage exited 0 with eight
    failed cells (DEVLOG 2026-09-24), and `partial` / `failed` there is the truth the panel needs.
    The exit code alone answers for every stage that counts no cells.
    """
    if cancelled:
        return "cancelled"
    if cells is not None:
        planned = cells.get("of", 0)
        done = cells.get("done", 0)
        if planned <= 0:
            # A tally of nothing planned: `--dirs` matched no cell. There was
            # no work to fail at, so the process result stands.
            return "done" if returncode == 0 else "failed"
        if done >= planned:
            return "done"
        # Not every cell finished. `failed` when none did - whether they raised
        # or were blocked by the prerequisite gate, nothing came of the run.
        return "partial" if done else "failed"
    return "done" if returncode == 0 else "failed"


def _final_percent(status: str, cells: dict[str, int] | None,
                   percent: int | None) -> int | None:
    """Where the bar stops.

    A full success is 100 - that is the old unconditional rule, kept, and it is
    the only case that earns it. With a tally the bar is REBASED onto the cells
    that actually finished, so an all-failed run ends at 0 and a four-of-eight
    run at 50. That is the one place the bar is not monotone: a batch advances
    `[n/m]` per cell FINISHED, failures included, so it can climb to 100 during
    a run that produced nothing, and the last frame corrects it. Rejected:
    leaving the bar at 100 and letting the status colour carry the bad news -
    the bar is what the eye reads first, and a full green bar over the word
    "failed" is the same lie in a smaller font.

    With no tally the bar is left exactly where the stage's own lines put it,
    which is why a crashed `pick` still shows how far it got.
    """
    if status == "done":
        return 100
    if cells:
        planned, done = cells.get("of", 0), cells.get("done", 0)
        if planned > 0:
            return max(0, min(100, round(100 * done / planned)))
    return percent


# ---------------------------------------------------------------------------
# argv from the stage's own parser
# ---------------------------------------------------------------------------
def stage_parser(stage: str) -> argparse.ArgumentParser:
    parser = cli.build_parser()
    sub = next(a for a in parser._actions
               if isinstance(a, argparse._SubParsersAction))
    try:
        return sub.choices[stage]
    except KeyError:
        raise RunRefused(400, f"unknown stage {stage!r}; one of {RUNNABLE}") from None


def _flag(action: argparse.Action) -> str:
    longs = [s for s in action.option_strings if s.startswith("--")]
    return longs[0] if longs else action.option_strings[0]


def argv_for(stage: str, args: dict[str, Any]) -> list[str]:
    """`{"job": "x", "n": 3, "force": true, "audit": false}` -> the CLI's argv.

    Keys are the parser's `dest` names (what `--help` shows, dashes as
    underscores). Positionals go in the parser's order; a true store_true adds
    its flag, a false store_false adds its (`audit: false` -> `--no-audit`); a
    list repeats the flag or fills an nargs. Unknown keys are refused, not
    dropped - a silently ignored `seed` is a render nobody asked for.
    """
    parser = stage_parser(stage)
    actions = [a for a in parser._actions
               if not isinstance(a, argparse._HelpAction)]
    known = {a.dest for a in actions}
    unknown = sorted(set(args) - known)
    if unknown:
        raise RunRefused(400, f"{stage}: unknown argument(s) {unknown}; "
                              f"known: {sorted(known)}")
    positionals: list[str] = []
    options: list[str] = []
    for action in actions:
        if action.dest not in args:
            continue
        value = args[action.dest]
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not action.option_strings:
            positionals.extend(str(v) for v in values)
            continue
        flag = _flag(action)
        if isinstance(action, argparse._StoreTrueAction):
            if value is True:
                options.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if value is False:
                options.append(flag)
        elif isinstance(action, argparse._CountAction):
            options.extend([flag] * int(value))
        elif isinstance(action, argparse._AppendAction):
            for v in values:
                options.extend([flag, str(v)])
        elif action.nargs in ("+", "*") or isinstance(action.nargs, int):
            options.append(flag)
            options.extend(str(v) for v in values)
        else:
            if value is None:
                continue
            if isinstance(value, bool):
                raise RunRefused(400, f"{stage}: {action.dest} takes a value, "
                                      f"not a boolean")
            options.extend([flag, str(value)])
    return options + positionals


def check_argv(stage: str, argv: list[str]) -> None:
    """Parse with the real parser, so a bad request is a 400 with argparse's
    own words rather than a run that dies on its first line."""
    parser = cli.build_parser()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            parser.parse_args([stage, *argv])
        except SystemExit:
            message = err.getvalue().strip().splitlines()
            raise RunRefused(400, message[-1] if message else "bad arguments",
                             argv=argv) from None


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
@dataclass
class Run:
    id: str
    stage: str
    argv: list[str]
    gpu: bool
    status: str = "queued"
    percent: int | None = None
    started: str = field(default_factory=_now)
    finished: str | None = None
    returncode: int | None = None
    pid: int | None = None
    log: list[str] = field(default_factory=list)
    # The tally a batch stage printed, `{done, failed, blocked, of}`, or None
    # for a stage that has no cells to count. It is what makes "8 of 8 cells
    # failed" sayable where the process result alone only said "exited 0".
    cells: dict[str, int] | None = None
    # The concept generation feature: a run can be cancelled from the studio.
    # The handle is the process itself; `cancelled` is what the exit reports.
    cancelled: bool = False
    proc: Any = field(default=None, repr=False, compare=False)
    # The manager's condition, shared so `events()` can wait on one lock.
    cond: threading.Condition = field(default_factory=threading.Condition,
                                      repr=False, compare=False)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def record(self, *, with_log: bool = False, tail: int | None = None
               ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id, "stage": self.stage, "argv": list(self.argv),
            "gpu": self.gpu, "status": self.status, "percent": self.percent,
            "started": self.started, "finished": self.finished,
            "returncode": self.returncode, "pid": self.pid,
            "cancelled": self.cancelled,
            "cells": dict(self.cells) if self.cells else None,
            "log_lines": len(self.log), "url": f"/runs/{self.id}",
            "events": f"/runs/{self.id}/events",
        }
        if with_log:
            row["log"] = list(self.log[-tail:] if tail else self.log)
        return row


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------
class RunManager:
    """Starts stages as subprocesses and remembers them. One per app."""

    def __init__(self, work_dir: Path, *, cpu_slots: int | None = None,
                 comfyui_url: str | None = None,
                 python: list[str] | None = None) -> None:
        self.work_dir = Path(work_dir)
        self.runs_dir = self.work_dir / RUNS_DIRNAME
        self.cpu_slots = config.API_CPU_SLOTS if cpu_slots is None else cpu_slots
        self.comfyui_url = comfyui_url
        # How a stage is launched; overridable so a test can run something
        # that is not the CLI (a shell line that prints and sleeps).
        self.python = python or [sys.executable, "-m", "anim"]
        self.cond = threading.Condition()
        self.runs: dict[str, Run] = {}
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        """Runs from before this process: their files are the record, and any
        that was still active belonged to a process that no longer exists.

        Every field is read with a default, so a record written before it
        existed still loads: a run from before the outcome tally has no `cells`
        and is served with `cells: null`. Migrating in memory rather than
        rewriting the files keeps `_runs/` append-only - the alternative,
        back-filling a tally the old run never reported, would be inventing it.
        """
        if not self.runs_dir.is_dir():
            return
        for path in sorted(self.runs_dir.glob("*.json")):
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            # A status this build does not know - a record from a newer API, or
            # a hand-edited file - is `lost`, which is what "the API cannot
            # account for this run" already means. Guessing would be worse.
            status = row.get("status", "lost")
            run = Run(id=row["id"], stage=row["stage"], argv=list(row["argv"]),
                      gpu=bool(row.get("gpu")),
                      status=status if status in ACTIVE + FINISHED else "lost",
                      percent=row.get("percent"), started=row.get("started", ""),
                      finished=row.get("finished"),
                      returncode=row.get("returncode"), pid=row.get("pid"),
                      cells=_cells(row.get("cells")),
                      cancelled=bool(row.get("cancelled")), cond=self.cond)
            log_path = self.runs_dir / f"{run.id}.log"
            if log_path.exists():
                run.log = log_path.read_text(encoding="utf-8",
                                             errors="replace").splitlines()
            if run.active:
                run.status = "lost"
                run.finished = run.finished or _now()
                self._save(run)
            self.runs[run.id] = run

    def _save(self, run: Run) -> None:
        write_json_in_place(self.runs_dir / f"{run.id}.json", run.record())

    # --- listing -----------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        with self.cond:
            rows = [run.record() for run in self.runs.values()]
        rows.sort(key=lambda row: row["started"], reverse=True)
        return rows

    def get(self, run_id: str) -> Run | None:
        with self.cond:
            return self.runs.get(run_id)

    def active_gpu(self) -> Run | None:
        with self.cond:
            return next((r for r in self.runs.values() if r.gpu and r.active), None)

    def active_cpu(self) -> list[Run]:
        with self.cond:
            return [r for r in self.runs.values() if not r.gpu and r.active]

    # --- starting ----------------------------------------------------------
    def start(self, stage: str, *, args: dict[str, Any] | None = None,
              argv: list[str] | None = None) -> Run:
        """Refuses (RunRefused) or returns the queued run; never blocks."""
        if stage not in RUNNABLE:
            raise RunRefused(400, f"unknown stage {stage!r}; one of {RUNNABLE}")
        if argv is None:
            argv = argv_for(stage, dict(args or {}))
        elif args:
            raise RunRefused(400, "give `args` or `argv`, not both")
        argv = [str(a) for a in argv]
        check_argv(stage, argv)
        gpu = stage in GPU_STAGES
        with self.cond:
            if gpu:
                busy = next((r for r in self.runs.values() if r.gpu and r.active), None)
                if busy is not None:
                    raise RunRefused(
                        409, f"a GPU stage is already running ({busy.stage}, "
                             f"run {busy.id}); one at a time on a 24 GB card",
                        run=busy.record())
            else:
                active = [r for r in self.runs.values() if not r.gpu and r.active]
                if len(active) >= self.cpu_slots:
                    raise RunRefused(
                        409, f"{len(active)} CPU stage(s) already running "
                             f"(limit {self.cpu_slots}); try again when one finishes",
                        runs=[r.record() for r in active])
            run = Run(id=self._new_id(stage), stage=stage, argv=argv, gpu=gpu,
                      cond=self.cond)
            self.runs[run.id] = run
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._save(run)
        threading.Thread(target=self._drive, args=(run,), daemon=True,
                         name=f"anim-run-{run.id}").start()
        return run

    def _new_id(self, stage: str) -> str:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        while True:
            candidate = f"{stamp}-{stage}-{secrets.token_hex(2)}"
            if candidate not in self.runs:
                return candidate

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["ANIM_WORK_DIR"] = str(self.work_dir)
        env["PYTHONUNBUFFERED"] = "1"
        if self.comfyui_url:
            env["COMFYUI_URL"] = self.comfyui_url
        return env

    def _drive(self, run: Run) -> None:
        """The run's own thread: spawn, stream the log, record the exit."""
        command = [*self.python, run.stage, *run.argv]
        log_path = self.runs_dir / f"{run.id}.log"
        try:
            # Its own session, so a cancel's killpg takes the stage and
            # whatever it spawned (ffmpeg, the audit) and nothing else.
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=self._env(),
                encoding="utf-8", errors="replace", start_new_session=True)
        except OSError as exc:
            with self.cond:
                run.status = "failed"
                run.finished = _now()
                run.log.append(f"could not start {command[0]}: {exc}")
                self.cond.notify_all()
            self._save(run)
            return
        with self.cond:
            run.status = "running"
            run.pid = proc.pid
            run.proc = proc
            if run.cancelled:
                # Cancelled between queued and running: stop it now.
                self._kill(proc)
            self.cond.notify_all()
        self._save(run)
        assert proc.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                log.write(line + "\n")
                log.flush()
                percent = self._percent(line)
                cells = self._outcome(line)
                with self.cond:
                    run.log.append(line)
                    if percent is not None:
                        run.percent = percent
                    if cells is not None:
                        run.cells = cells
                    self.cond.notify_all()
        returncode = proc.wait()
        with self.cond:
            run.returncode = returncode
            run.status = _status_for(run.cancelled, returncode, run.cells)
            run.percent = _final_percent(run.status, run.cells, run.percent)
            run.finished = _now()
            run.proc = None
            self.cond.notify_all()
        self._save(run)

    # --- cancelling --------------------------------------------------------
    @staticmethod
    def _kill(proc: Any, signal_number: int = signal.SIGTERM) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal_number)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def cancel(self, run_id: str) -> Run:
        """Stop a queued or running stage: SIGTERM to its process group,
        SIGKILL after `CANCEL_GRACE_S`, and - for a GPU stage - ComfyUI's own
        `/interrupt`, so the card drops the prompt the dead process was
        waiting on instead of finishing an image nobody wants. The run's
        thread records the exit as `cancelled`. 404 unknown, 409 not active."""
        with self.cond:
            run = self.runs.get(run_id)
            if run is None:
                raise RunRefused(404, f"no run {run_id!r}")
            if not run.active:
                raise RunRefused(409, f"run {run_id} is {run.status}; nothing to cancel",
                                 run=run.record())
            run.cancelled = True
            run.log.append("cancelled from the API")
            proc = run.proc
            self.cond.notify_all()
        if proc is not None:
            self._kill(proc)
            try:
                proc.wait(timeout=CANCEL_GRACE_S)
            except subprocess.TimeoutExpired:
                self._kill(proc, signal.SIGKILL)
        if run.gpu and self.comfyui_url:
            from anim import comfy
            comfy.ComfyClient(self.comfyui_url, timeout=3.0).interrupt()
        return run

    @staticmethod
    def _outcome(line: str) -> dict[str, int] | None:
        """A batch stage's `outcome: done=.. failed=.. blocked=.. of=..`."""
        return outcome.parse(line)

    @staticmethod
    def _percent(line: str) -> int | None:
        step = _STEP.match(line)
        if step:
            n, m = int(step.group(1)), int(step.group(2))
            return max(0, min(100, round(100 * n / m))) if m else None
        progress = _PROGRESS.match(line.strip())
        if progress:
            return max(0, min(100, int(progress.group(1))))
        return None

    # --- SSE ---------------------------------------------------------------
    def events(self, run: Run, *, keepalive: float = 15.0) -> Iterator[str]:
        """The run's log as server-sent events, then `done`, then the end.

        Yields already-written text/event-stream frames: every log line as a
        `log` event, each new percent as `progress`, and one `done` (or
        `partial` / `failed` / `lost`) when the run is over. A connection that
        opens after the run finished still gets the whole log, then the end - so
        a page reloaded mid-run misses nothing.

        The final event carries `cells` as well as the status, so the panel can
        say "0 of 8 cells done, 8 failed" without a second fetch of the record.
        """
        cursor = 0
        last_percent: int | None = None
        while True:
            with self.cond:
                while cursor >= len(run.log) and run.active \
                        and run.percent == last_percent:
                    if not self.cond.wait(timeout=keepalive):
                        break
                lines = run.log[cursor:]
                cursor = len(run.log)
                percent = run.percent
                active = run.active
                status = run.status
                returncode = run.returncode
                cells = dict(run.cells) if run.cells else None
            if not lines and percent == last_percent and active:
                yield ": keepalive\n\n"
                continue
            for line in lines:
                yield _event("log", {"line": line})
            if percent != last_percent:
                last_percent = percent
                yield _event("progress", {"percent": percent})
            if not active:
                yield _event("done", {"status": status, "returncode": returncode,
                                      "percent": percent, "cells": cells})
                return


def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"
