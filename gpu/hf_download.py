#!/usr/bin/env python3
"""Download exactly the files named in gpu/models.lock into the ComfyUI basedir.

Runs INSIDE a throwaway python:3.12-slim container on Docker Desktop
(the Fedora host has Python 3.14 and no pip). Never run this on the host.

Mounts expected inside the container:
    /models   -> "/mnt/private/AI Data/comfyui-basedir/models"   (rw)
    /spec     -> tools/anim-pipeline/gpu                          (ro)

Idempotent: a file whose size already matches the lock is skipped. Pass
--verify to also sha256 every present file (slow: reads ~35 GiB).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

LOCK = Path("/spec/models.lock")
DEST_ROOT = Path("/models")
# Keep the HF blob cache on the big disk, not the container's overlay fs.
CACHE = Path(os.environ.get("HF_HOME", "/models/.hf-cache"))


def rows():
    for line in LOCK.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 8:
            sys.exit(f"malformed models.lock row ({len(parts)} cols): {line!r}")
        repo, rev, rfile, subdir, name, size, sha, lic = parts
        # Guard the bug class that bit us once: a lock row whose hash column is
        # empty would make every verification "fail" and silently discard a
        # perfectly good 12 GiB download. Refuse to run on such a lock.
        if len(sha) != 64:
            sys.exit(f"models.lock row for {name!r} has no usable sha256 ({sha!r}); "
                     "regenerate it with gpu/relock_models.py")
        yield repo, rev, rfile, subdir, name, int(size), sha, lic


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="sha256 every file that is already present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = list(rows())
    total = sum(r[5] for r in plan)
    print(f"[plan] {len(plan)} files, {total / 2**30:.2f} GiB", flush=True)
    free = shutil.disk_usage(DEST_ROOT).free
    print(f"[disk] {free / 2**30:.1f} GiB free at {DEST_ROOT}", flush=True)
    if free < total * 1.15 and not args.dry_run:
        return print(f"[FATAL] need ~{total * 1.15 / 2**30:.0f} GiB (blob cache + copy)") or 1

    failures: list[str] = []
    for i, (repo, rev, rfile, subdir, name, size, sha, lic) in enumerate(plan, 1):
        dest = DEST_ROOT / subdir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tag = f"[{i}/{len(plan)}] {subdir}/{name}"

        if dest.exists() and dest.stat().st_size == size:
            if args.verify:
                got = sha256(dest)
                ok = got == sha
                print(f"{tag} present, sha256 {'OK' if ok else f'MISMATCH got={got} want={sha}'}", flush=True)
                if not ok:
                    failures.append(f"{name}: sha256 mismatch")
            else:
                print(f"{tag} present ({size / 2**30:.2f} GiB), skipping", flush=True)
            continue

        print(f"{tag} <- {repo}@{rev[:8]}:{rfile}  ({size / 2**30:.2f} GiB, {lic})", flush=True)
        if args.dry_run:
            continue
        try:
            blob = hf_hub_download(
                repo_id=repo, filename=rfile, revision=rev,
                cache_dir=str(CACHE), local_dir=None,
            )
        except Exception as exc:  # noqa: BLE001 - log and keep going
            print(f"{tag} DOWNLOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        # Copy (not symlink) into the ComfyUI tree: ComfyUI and the container
        # that serves it must see a real file at a stable path.
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(blob, tmp)
        got_size = tmp.stat().st_size
        if got_size != size:
            tmp.unlink(missing_ok=True)
            print(f"{tag} SIZE MISMATCH got={got_size} want={size}", flush=True)
            failures.append(f"{name}: size {got_size} != {size}")
            continue
        got_sha = sha256(tmp)
        if got_sha != sha:
            tmp.unlink(missing_ok=True)
            print(f"{tag} SHA256 MISMATCH got={got_sha} want={sha}", flush=True)
            failures.append(f"{name}: sha256 mismatch")
            continue
        tmp.replace(dest)
        print(f"{tag} OK sha256={got_sha}", flush=True)

    if failures:
        print("\n[FAILURES]", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    print("\n[done] all files present and verified against models.lock", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
