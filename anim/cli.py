"""`anim` - the CPU side of the pose-to-pixel animation pipeline feature.

    anim <stage> [options]

Stages run in order; each reads what the previous one wrote under work/<job>/:

    extract -> pose -> pick -> review -> strip -> render -> post -> publish
    rig     ---------^   (a rig clip instead of a filmed one: replaces extract + pose)
    rig -> clip -> pick -> post --input clip -> publish   (the clip render feature)

`base` generates a unit's standing sprite; `run` chains the whole thing.

Every stage is a module in anim/stages/ exposing NAME, HELP, add_arguments and
run. Adding a stage means adding a module and listing it in
anim/stages/__init__.py - no changes here.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from anim import __version__, config
from anim.stages import STAGE_ORDER

EPILOG = """\
stage order:
  extract -> pose -> pick -> review -> strip -> render -> post -> publish
  rig replaces extract + pose when the motion is a rig clip (KayKit Knight)
  clip renders every rig frame as one Wan 2.2 Fun-Control video; then
    pick -> post --input clip -> publish (the clip render feature)
  base is the standing-sprite variant of the same chain
  run chains extract..strip, then render..publish if the GPU is up

the usual one-liners:
  anim run capture/knight_strike.mp4 --unit knight --action strike
  anim rig rigs/kaykit-adventurers/Knight.glb --clip Hit_A --unit knight --action hit

where things live (override with the env vars in anim/config.py):
  capture/   filmed clips, dropped in by the owner
  rigs/      CC0 rigged characters, each with a rig.json provenance sidecar
  work/      per-job scratch: frames, keypoints, picks, strips, intermediates
  assets     game-02's published PNGs and sidecars

the GPU:
  only `render`, `clip` and `base` touch it, over HTTP at $COMFYUI_URL
  (default http://host.docker.internal:8188 - ComfyUI runs on the OTHER Docker
  engine, so localhost would not reach it from inside this container)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anim",
        description=(
            "Film a motion in profile, get palette-locked 64x64 pixel frames "
            "that match the unit's own sprite and play in game-02."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"anim {__version__}")
    parser.add_argument(
        "--comfyui-url",
        default=config.COMFYUI_URL,
        help=f"GPU image service base URL (default: {config.COMFYUI_URL})",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="repeat for more output"
    )

    sub = parser.add_subparsers(dest="stage", metavar="<stage>", required=True)
    for name in STAGE_ORDER:
        mod = importlib.import_module(f"anim.stages.{name}")
        stage_parser = sub.add_parser(
            mod.NAME,
            help=mod.HELP,
            description=(mod.__doc__ or "").strip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        mod.add_arguments(stage_parser)
        stage_parser.set_defaults(_run=mod.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
