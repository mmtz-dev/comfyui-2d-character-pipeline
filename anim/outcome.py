"""The tally a batch stage ends with, and how it travels to the studio.

THE BUG THIS EXISTS FOR (2026-09-24, DEVLOG). A batch stage ran on
hooded_archer, all eight directions raised `ValueError: no part table for unit ...`, every one
was caught and recorded - which is correct, one bad cell must not kill the
batch - and the process exited 0. `anim.api.runs._drive` reads a clean exit as
success, so the run record said `status: done, percent: 100` for a run that
produced nothing. "The process exited cleanly" and "the work succeeded" are two
different facts and the pipeline was only carrying the first one.

So a batch stage now ends by REPORTING ITS TALLY, on stdout, in the same
register as the `[n/m]` progress lines and `progress: 42%`:

    outcome: done=0 failed=8 blocked=0 of=8

`anim.api.runs` parses that line and nothing else, keeps it on the run record
as `cells`, and derives the run's status and final percent from it. A stage
that prints no such line behaves exactly as before - `pick`, `post`, `render`
and the rest are single-shot, they raise or they succeed, and there is nothing
to tally.

WHY A PRINTED LINE AND NOT A RETURN VALUE. The API runs a stage as a
subprocess on purpose (see `anim.api.runs`), so stdout is the only channel that
exists between a stage and the manager. Rejected: writing the tally into
`work/_runs/<id>.json` from the stage (the stage does not know the run id, and
the API owns that file), and having the manager re-read the stage's own state
file (it would have to know each stage's file name, schema and which key
counts - the coupling the subprocess design exists to avoid).

EXIT CODES. The tally also has to reach a shell, a script and CI, which see
nothing but `$?`:

    0  every planned cell is done - a full success
    3  PARTIAL: some cells are done and some are not
    4  NOTHING: zero of N cells are done

The middle value is deliberate. Folding "partial" into 0 is the bug above
wearing a smaller hat, and folding it into "failed" would make a nineteen-cell
success with one flaky cell indistinguishable from a total loss - the owner
re-runs those two differently. `blocked` counts as not-done: a cell the
prerequisite gate refused did not happen, whatever the reason, and a run of
eight blocked cells must not read 0/green.

A MODE THAT RUNS NOTHING REPORTS NOTHING. `--dry-run` and `--page-only` are
asked to run no cells, so they succeed by running none; they print no
`outcome:` line and exit 0. Reporting `done=0 of=8` there would fail a dry run
for doing its job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# Full success; some cells done, some not; none done.
EXIT_OK = 0
EXIT_PARTIAL = 3
EXIT_NOTHING = 4

# What `anim.api.runs` matches. Anchored at column 0 like `_STEP`, so a stage
# quoting the word "outcome" mid-sentence cannot be mistaken for a tally.
LINE = re.compile(r"^outcome:\s*(?P<body>(?:\w+=\d+\s*)+)$")

# The counters a tally may carry. `of` is separate from done+failed+blocked
# because a cancelled run leaves cells that never reached any of the three:
# eight planned, three done, five never started is `done=3 of=8`, and the
# missing five are exactly what makes it partial rather than done.
KEYS = ("done", "failed", "blocked", "of")


@dataclass(frozen=True)
class Outcome:
    """What a batch stage accomplished, in cells."""

    done: int = 0
    failed: int = 0
    blocked: int = 0
    of: int | None = None

    @property
    def planned(self) -> int:
        """How many cells this run was asked for."""
        return self.of if self.of is not None else self.done + self.failed + self.blocked

    @property
    def counts(self) -> dict[str, int]:
        return {"done": self.done, "failed": self.failed,
                "blocked": self.blocked, "of": self.planned}

    @property
    def line(self) -> str:
        """The one line stdout carries to the run manager."""
        return ("outcome: " +
                " ".join(f"{k}={v}" for k, v in self.counts.items()))

    @property
    def exit_code(self) -> int:
        """0 / 3 / 4, per the module docstring."""
        planned = self.planned
        if planned <= 0:
            # Nothing was asked for - `--dirs` matched an empty selection, say.
            # There is no work to have failed at.
            return EXIT_OK
        if self.done >= planned:
            return EXIT_OK
        return EXIT_PARTIAL if self.done else EXIT_NOTHING

    @property
    def phrase(self) -> str:
        """What a human reads: `0 of 8 cells done, 8 failed`."""
        parts = [f"{self.done} of {self.planned} done"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.blocked:
            parts.append(f"{self.blocked} blocked")
        missing = self.planned - self.done - self.failed - self.blocked
        if missing > 0:
            parts.append(f"{missing} not run")
        return ", ".join(parts)

    def report(self, log: Callable[[str], None] = print) -> int:
        """Print the tally and answer with the stage's exit code. The last
        thing a batch stage's `run_stage` does, so the line is the last thing
        the manager parses before the process exits."""
        log(self.line)
        return self.exit_code


def count(records: dict[str, Any], keys: Sequence[str]) -> Outcome:
    """The tally of the cell records `keys` names, in the shared per-cell
    status vocabulary (`done` / `kept` / `failed` / `blocked` / `not run`).

    `kept` counts as done - it means the chain ran and the owner's hand-edited
    picks were left alone, which is a finished cell, and `matrix._state`'s
    `summary` has always counted it that way.
    """
    keys = list(keys)
    rows = [(records.get(k) or {}) for k in keys]
    statuses = [r.get("status") or "not run" for r in rows]
    return Outcome(done=sum(1 for s in statuses if s in ("done", "kept")),
                   failed=sum(1 for s in statuses if s == "failed"),
                   blocked=sum(1 for s in statuses if s == "blocked"),
                   of=len(keys))


def parse(line: str) -> dict[str, int] | None:
    """`outcome: done=0 failed=8 blocked=0 of=8` -> the counts, else None.

    Unknown keys are dropped rather than refused: a stage that one day reports
    `skipped=2` must not make the whole tally unreadable to an older API.
    """
    match = LINE.match(line)
    if not match:
        return None
    counts: dict[str, int] = {}
    for pair in match.group("body").split():
        key, _, value = pair.partition("=")
        if key in KEYS:
            counts[key] = int(value)
    if not counts:
        return None
    counts.setdefault("of", sum(counts.get(k, 0)
                                for k in ("done", "failed", "blocked")))
    return counts
