"""One module per pipeline stage.

Each exposes NAME, HELP, add_arguments(parser) and run(args) -> int, so
anim/cli.py can wire them up without knowing anything about them.
Stage order: extract -> pose -> pick -> review -> strip -> render -> post -> publish.
`rig` replaces extract + pose when the motion comes from a rig clip instead of a
filmed one; `clip` (the clip render feature) sits between rig and pick and
replaces strip + render when the frames come from a generated clip; `base` is
the standing-sprite variant; `run` chains the whole thing in one command.
"""

STAGE_ORDER = [
    "extract",
    "pose",
    "rig",
    "clip",
    "pick",
    "review",
    "strip",
    "render",
    "post",
    "publish",
    "base",
    "run",
]
