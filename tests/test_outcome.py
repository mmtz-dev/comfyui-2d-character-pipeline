"""`anim.outcome` - the tally a batch stage ends with.

What is pinned here:
  1. The tally counts the shared per-cell vocabulary (`done` / `kept` /
     `failed` / `blocked` / `not run`) over the cells THIS run planned.
  2. The exit code: 0 only for a full success, EXIT_PARTIAL when some cells
     are done, EXIT_NOTHING when none are - so the 2026-09-24 bug (eight
     failed directions, exit 0) cannot come back at the stage's edge.
  3. The line a stage prints and the manager parses round-trips, and nothing
     else in a log is mistaken for it.
"""

from __future__ import annotations

from anim import outcome


# ---------------------------------------------------------------------------
# 1. Counting
# ---------------------------------------------------------------------------
def test_the_tally_counts_the_cells_this_run_planned():
    records = {
        "attack/s0": {"status": "done"},
        "attack/s45": {"status": "kept"},      # hand-edited picks: still done
        "walk/s0": {"status": "failed"},
        "walk/s45": {"status": "blocked"},
        "idle/s0": {"status": "not run"},
        "jump/s0": {"status": "done"},         # an earlier run's cell
    }
    planned = ["attack/s0", "attack/s45", "walk/s0", "walk/s45", "idle/s0"]
    tally = outcome.count(records, planned)
    assert tally.counts == {"done": 2, "failed": 1, "blocked": 1, "of": 5}, \
        "`jump/s0` is in the file but not in this run's plan"
    assert tally.phrase == "2 of 5 done, 1 failed, 1 blocked, 1 not run"
    # A cell the plan names and the file has never heard of is `not run`.
    assert outcome.count({}, ["a", "b"]).counts == \
        {"done": 0, "failed": 0, "blocked": 0, "of": 2}


# ---------------------------------------------------------------------------
# 2. The exit code
# ---------------------------------------------------------------------------
def test_only_a_full_success_exits_zero():
    full = outcome.Outcome(done=8, of=8)
    none = outcome.Outcome(done=0, failed=8, of=8)
    some = outcome.Outcome(done=5, failed=3, of=8)
    assert full.exit_code == outcome.EXIT_OK == 0
    assert none.exit_code == outcome.EXIT_NOTHING
    assert some.exit_code == outcome.EXIT_PARTIAL
    assert none.exit_code != some.exit_code, \
        "all-failed and partial are different answers to `did it work?`"
    # Blocked counts as not-done: the run asked for eight cells and got none.
    assert outcome.Outcome(blocked=8, of=8).exit_code == outcome.EXIT_NOTHING
    # A cancel leaves cells that reached no status at all.
    assert outcome.Outcome(done=3, of=8).exit_code == outcome.EXIT_PARTIAL
    # Nothing planned is nothing to fail at (an empty `--dirs` selection).
    assert outcome.Outcome().exit_code == outcome.EXIT_OK


def test_report_prints_the_line_and_answers_with_the_code():
    lines: list[str] = []
    code = outcome.Outcome(done=0, failed=8, of=8).report(lines.append)
    assert lines == ["outcome: done=0 failed=8 blocked=0 of=8"]
    assert code == outcome.EXIT_NOTHING


# ---------------------------------------------------------------------------
# 3. The line the run manager reads
# ---------------------------------------------------------------------------
def test_the_line_round_trips_and_nothing_else_is_taken_for_one():
    tally = outcome.Outcome(done=5, failed=2, blocked=1, of=8)
    assert outcome.parse(tally.line) == tally.counts
    # `of` may be left out; it is then the three counters added up.
    assert outcome.parse("outcome: done=1 failed=1") == \
        {"done": 1, "failed": 1, "of": 2}
    # An unknown counter is dropped, not refused - an older API must still
    # read a tally a newer stage has added a column to.
    assert outcome.parse("outcome: done=2 skipped=3 of=5")["done"] == 2
    for line in ["  outcome: done=1 of=1",            # not at column 0
                 "the outcome: done=1 of=1",
                 "outcome: done=1 of=1 and then some",
                 "outcome: nope=1",
                 "outcome:",
                 "[1/8] attack/s0: done"]:
        assert outcome.parse(line) is None, line
