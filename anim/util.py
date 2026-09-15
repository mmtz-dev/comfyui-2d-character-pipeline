"""Small shared helpers. No stage-specific knowledge lives here.

The only genuinely load-bearing one is `write_bytes_in_place`. Read its
docstring before replacing any write in this package with a rename.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex digest of a file, read in chunks so a 200 MB clip does not sit in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """The exact bytes a workflow hash is taken over.

    Sorted keys and no incidental whitespace, so the same graph always hashes
    the same however it was assembled. This is what goes in
    `generation.workflow_sha256`, and it must stay stable forever - a frame's
    sidecar is a promise that the frame can be re-made.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_bytes_in_place(path: Path, data: bytes) -> Path:
    """Overwrite a file's CONTENT, never its directory entry.

    Docker Desktop's file share (gRPC-FUSE / virtiofs) caches by inode. A
    write-temp-then-rename - the usual atomic-save trick, and what most image
    libraries do internally - leaves a NEW inode behind the same name, and a
    container that already has the old one open keeps serving the old bytes.
    That is the "Mac edits a PNG, the game still shows the old sprite" quirk in
    the Mac edit share feature, and it is why `publish` writes through this.

    Opening with "wb" truncates in place and keeps the inode, which is the
    `cat > file` behaviour the plan asks for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
    return path


def write_text_in_place(path: Path, text: str) -> Path:
    return write_bytes_in_place(path, text.encode("utf-8"))


def write_json_in_place(path: Path, obj: Any) -> Path:
    """Pretty JSON with a trailing newline, written in place.

    Indented rather than canonical because every one of these files is read by a
    human at some point (job.json, picks.json, the sidecar) and diffed in git.
    The canonical form is only used for hashing.
    """
    return write_text_in_place(path, json.dumps(obj, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cmd: list[str], *, check: bool = True,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run an external tool (ffmpeg, magick, pixelfixer) and capture both streams.

    Output is captured rather than inherited so a failure can be reported with
    the tool's own words instead of "returned 1".
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc


def have(tool: str) -> bool:
    """Is this external tool on PATH?"""
    return shutil.which(tool) is not None


def tool_version(tool: str, *args: str) -> str:
    """One-line version string for the post-chain report, or '(missing)'."""
    if not have(tool):
        return "(missing)"
    try:
        proc = run([tool, *args], check=False, timeout=30)
    except Exception as exc:  # pragma: no cover - defensive
        return f"(unreadable: {exc})"
    line = (proc.stdout or proc.stderr).strip().splitlines()
    return line[0] if line else "(no output)"
