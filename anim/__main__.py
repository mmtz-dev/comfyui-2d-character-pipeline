"""Allow `python -m anim` as well as the `anim` console script."""

from anim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
