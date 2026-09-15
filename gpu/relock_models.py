#!/usr/bin/env python3
"""Regenerate gpu/models.lock from the Hugging Face API.

Run this (not a text editor) whenever the model list changes. It is the only
writable source for models.lock. Standard library only, so it runs with the
host's Python 3.14 - it makes HTTP calls and writes one file, nothing else.

    python3 gpu/relock_models.py

Each file's size and sha256 are read from TWO independent endpoints and must
agree, and every repo must be ungated, or nothing is written. That guard exists
because the two endpoints spell the hash differently:
  /api/models/<repo>?blobs=true            -> siblings[].lfs.sha256
  /api/models/<repo>/paths-info/<revision> -> [].lfs.oid
Reading only one spelling silently produced an all-empty hash column once.

To change WHICH files are downloaded, edit WANT below and re-run.
"""
from __future__ import annotations

import datetime
import json
import sys
import urllib.request

# (repo_id, path inside the repo, dest subdir under models/, installed filename)
WANT = [
    # --- stage 1, pose pass: Z-Image-Turbo + Fun ControlNet Union 2.1 ---
    ("Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
     "diffusion_models", "z_image_turbo_int8_convrot.safetensors"),
    ("Comfy-Org/z_image_turbo",
     "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
     "text_encoders", "qwen_3_4b_fp8_mixed.safetensors"),
    ("Comfy-Org/z_image_turbo",
     "split_files/vae/ae.safetensors",
     "vae", "z_image_ae.safetensors"),  # renamed: see models.lock header
    ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
     "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
     "model_patches",
     "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"),

    # --- stage 2, identity pass: Qwen-Image-Edit-2509 INT4 (Nunchaku) ---
    # INT4, not NVFP4: the A5000 is Ampere (sm_86) and has no FP8/FP4 tensor
    # cores. r128 over r32 for quality; lightning-251115 is the newest 4-step.
    ("nunchaku-ai/nunchaku-qwen-image-edit-2509",
     "lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
     "diffusion_models",
     "svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/vae/qwen_image_vae.safetensors",
     "vae", "qwen_image_vae.safetensors"),

    # --- optional style LoRA for the identity pass (trigger: "Pixel Art") ---
    # WARNING: this repo's advertised sha256 does not match the bytes HF serves
    # (see the KNOWN DISCREPANCY block in models.lock). The lock row it produces
    # will fail verification and the file will not be installed. That is the
    # correct, safe outcome - the LoRA is optional. Do not "fix" it by weakening
    # the hash check.
    ("prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA",
     "Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors",
     "loras", "Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors"),
]

# Hard licence gate: this project ships commercially, so nothing else is allowed.
ALLOWED_LICENCES = {"apache-2.0", "mit", "cc0-1.0"}


def get(url: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    meta = {}
    for repo in dict.fromkeys(r[0] for r in WANT):
        d = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
        meta[repo] = {
            "sha": d["sha"],
            "license": (d.get("cardData") or {}).get("license"),
            "gated": d.get("gated"),
            "files": {s["rfilename"]: s for s in d.get("siblings", [])},
        }

    rows, total, problems = [], 0, []
    for repo, rfile, subdir, name in WANT:
        m = meta[repo]
        if m["gated"]:
            problems.append(f"{repo}: GATED - needs an accepted licence + HF_TOKEN")
        if (m["license"] or "").lower() not in ALLOWED_LICENCES:
            problems.append(f"{repo}: licence {m['license']!r} is not commercially usable")
        if rfile not in m["files"]:
            problems.append(f"{repo}: {rfile} no longer exists in the repo")
            continue

        s = m["files"][rfile]
        size = s["size"]
        lfs = s.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid") or ""

        info = get(f"https://huggingface.co/api/models/{repo}/paths-info/{m['sha']}",
                   {"paths": [rfile]})[0]
        info_sha = (info.get("lfs") or {}).get("oid") or ""

        if len(sha) != 64:
            problems.append(f"{name}: bad sha256 from ?blobs=true: {sha!r}")
        if sha != info_sha:
            problems.append(f"{name}: hash disagreement blobs={sha!r} paths-info={info_sha!r}")
        if info["size"] != size:
            problems.append(f"{name}: size disagreement {size} vs {info['size']}")

        total += size
        rows.append((repo, m["sha"], rfile, subdir, name, size, sha, m["license"]))

    if problems:
        print("REFUSING TO WRITE models.lock:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    header = f"""# =====================================================================
# models.lock - the GPU image service feature
# =====================================================================
# RESOLVED LOCKFILE, not hand-written. Regenerate with gpu/relock_models.py.
# Resolved {datetime.date.today().isoformat()} from the Hugging Face API; every sha256 and size was
# read from BOTH /api/models/<repo>?blobs=true and
# /api/models/<repo>/paths-info/<revision> and the two reads agreed.
# Every repo below was confirmed ungated and Apache-2.0 via the same API.
#
# Consumed by gpu/download_models.sh -> gpu/hf_download.py, which refuses to
# install a file whose size or sha256 does not match a row here.
# Verify a file by hand with:  sha256sum <file>
#
# Total: {total} bytes = {total / 2**30:.2f} GiB ({total / 1e9:.2f} GB)
#
# TAB-separated columns:
#   repo_id  repo_revision  repo_file  dest_subdir  dest_filename  size_bytes  sha256  license
#
# dest_subdir is relative to the basedir's models/ dir, i.e. the container's
# /basedir/models/<dest_subdir>/<dest_filename>.
#
# RENAME: z_image_turbo's vae ships as `ae.safetensors` and is installed as
# `z_image_ae.safetensors`, because extra_model_paths.yaml also exposes the
# pre-existing Flux `ae.safetensors` and ComfyUI merges vae folders by filename.
# The workflows reference the renamed file.
# ======================================================================"""

    with open("models.lock", "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    for r in rows:
        print(f"  {r[3]}/{r[4]}  {r[5] / 2**30:.2f} GiB  sha256={r[6][:16]}...")
    print(f"\nwrote models.lock: {len(rows)} files, {total / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
