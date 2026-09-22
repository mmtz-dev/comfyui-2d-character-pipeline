"""One module per pipeline stage.

Each exposes NAME, HELP, add_arguments(parser) and run(args) -> int, so
anim/cli.py can wire them up without knowing anything about them.
Stage order: extract -> pose -> pick -> review -> strip -> render -> post -> publish.
`rig` replaces extract + pose when the motion comes from a rig clip instead of a
filmed one; `clip` (the clip render feature) sits between rig and pick and
replaces strip + render when the frames come from a generated clip; `base` is
the standing-sprite variant; `concept` (the character concept feature) comes
before `base` in the studio's chain - text or sketch to candidate base
characters, one of which is picked as the unit's reference; `bakeoff` (the
frame route bake-off feature) runs one action through both in-between routes
on both rigs and writes the page the owner rules on; `turnaround` (the turnaround
feature) renders one idle still per direction from the accepted base and
writes the identity gate page; `palette` (the palette
feature) installs the project palette from Lospec or a file and the per-unit
sub-palettes `post` snaps to; `export` (the spritesheet export feature) packs
a unit's accepted frames into sheets + atlas.json + a Godot SpriteFrames, with
`publish` as its game-02 adapter; `curate` (the curation board feature) is the
board from the shell - accept / reject / regenerate a unit's cells and gates;
`run` chains the whole thing in one command.
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
    "export",
    "base",
    "concept",
    "bakeoff",
    "turnaround",
    "palette",
    "curate",
    "run",
]
