"""HTTP client for the GPU image service feature (ComfyUI).

Same drive protocol as upstream's drivers (`POST /prompt`, poll `/history/<id>`,
read the image specs out of the history entry) with three changes the fork
needs:

  1. The base URL is a config value, never a constant. The CPU pipeline is on
     Docker Desktop and ComfyUI is on the native daemon - two engines - so
     `localhost` is wrong from in here and `host.docker.internal` is right.
  2. Inputs go up through `POST /upload/image` rather than being written into a
     shared input folder. The shared folder still works and is used as a
     fallback, but the upload endpoint is the only path that works when the
     basedir is not mounted, and it is the only path a test can fake.
  3. Nodes are patched by `_meta.title`, never by node id. Re-exporting a
     workflow renumbers nodes, and silently swapping the reference sprite for
     the skeleton would produce a posed skeleton instead of a posed character.

Nothing in here reaches for a GPU on its own. `reachable()` is how every caller
finds out the service is down BEFORE doing work that would be thrown away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from anim import config
from anim.util import canonical_json, sha256_bytes

PATCH_PREFIX = "PATCH"


class ComfyError(RuntimeError):
    """The service answered, and the answer was no."""


class ComfyUnreachable(ComfyError):
    """The service did not answer at all - almost always "the daemon is off"."""


@dataclass
class OutputImage:
    filename: str
    subfolder: str
    type: str


class ComfyClient:
    def __init__(self, base_url: str | None = None, *, timeout: float = 30.0,
                 session: requests.Session | None = None) -> None:
        self.base_url = (base_url or config.COMFYUI_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # --- plumbing ----------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        try:
            return self.session.get(self._url(path), timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ComfyUnreachable(f"GET {self._url(path)}: {exc}") from exc

    # --- the four endpoints we use ----------------------------------------
    def system_stats(self) -> dict[str, Any]:
        response = self._get("/system_stats")
        if response.status_code != 200:
            raise ComfyError(f"/system_stats returned {response.status_code}")
        return response.json()

    def reachable(self) -> bool:
        """True when the service answers /system_stats. Never raises."""
        try:
            self.system_stats()
            return True
        except (ComfyError, ValueError):
            return False

    def object_info(self, class_type: str) -> dict[str, Any]:
        """`GET /object_info/<class>` -> that node's schema. The concept
        generation feature reads the loaders' combo lists out of it (which
        checkpoint / LoRA files ComfyUI can actually see) for `GET /models`."""
        response = self._get(f"/object_info/{class_type}")
        if response.status_code != 200:
            raise ComfyError(f"/object_info/{class_type} returned {response.status_code}")
        payload = response.json()
        return payload.get(class_type) or {}

    def combo_values(self, class_type: str, input_name: str) -> list[str]:
        """The file list a loader's combo input offers, or [] when the node
        class is not installed or the input is not a combo."""
        try:
            info = self.object_info(class_type)
        except ComfyError:
            return []
        required = ((info.get("input") or {}).get("required") or {})
        spec = required.get(input_name)
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
        return []

    def interrupt(self) -> bool:
        """`POST /interrupt`: stop the prompt ComfyUI is executing now. Used by
        a cancelled run so the card does not finish an image nobody wants.
        Never raises - a cancel must not fail because the service is off."""
        try:
            response = self.session.post(self._url("/interrupt"), timeout=self.timeout)
        except requests.RequestException:
            return False
        return response.status_code == 200

    def upload_image(self, path: Path, *, subfolder: str = "",
                     overwrite: bool = True, name: str | None = None) -> str:
        """Upload one file and return the name a LoadImage node should use.

        ComfyUI answers with the name it actually stored under (it de-duplicates
        by appending a counter unless overwrite is set), and that returned name -
        not the local filename - is what gets patched into the workflow.

        `name` overrides the stored filename: the clip render feature uploads
        `clip/control.mp4` from every job, and `LoadVideo` lists only the input
        ROOT, so the name has to carry the job to keep two jobs apart. The
        endpoint is `/upload/image` for videos too - it writes whatever bytes
        it is given (server.py `image_upload`); the content type is not checked.
        """
        path = Path(path)
        name = name or path.name
        mime = "video/mp4" if name.lower().endswith(".mp4") else "image/png"
        files = {"image": (name, path.read_bytes(), mime)}
        data = {"overwrite": "true" if overwrite else "false", "type": "input"}
        if subfolder:
            data["subfolder"] = subfolder
        try:
            response = self.session.post(self._url("/upload/image"), files=files,
                                         data=data, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ComfyUnreachable(f"POST /upload/image: {exc}") from exc
        if response.status_code not in (200, 201):
            raise ComfyError(f"/upload/image returned {response.status_code}: "
                             f"{response.text[:400]}")
        payload = response.json()
        name = payload.get("name") or path.name
        sub = payload.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    def post_prompt(self, workflow: dict[str, Any],
                    client_id: str | None = None) -> str:
        body: dict[str, Any] = {"prompt": workflow}
        if client_id:
            body["client_id"] = client_id
        try:
            response = self.session.post(self._url("/prompt"), json=body,
                                         timeout=self.timeout)
        except requests.RequestException as exc:
            raise ComfyUnreachable(f"POST /prompt: {exc}") from exc
        if response.status_code != 200:
            raise ComfyError(f"ComfyUI rejected the workflow "
                             f"({response.status_code}): {response.text[:800]}")
        payload = response.json()
        if "prompt_id" not in payload:
            raise ComfyError(f"no prompt_id in /prompt response: {payload}")
        return payload["prompt_id"]

    def history(self, prompt_id: str) -> dict[str, Any]:
        response = self._get(f"/history/{prompt_id}")
        if response.status_code != 200:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    def view(self, image: OutputImage) -> bytes:
        response = self._get("/view", params={"filename": image.filename,
                                              "subfolder": image.subfolder,
                                              "type": image.type})
        if response.status_code != 200:
            raise ComfyError(f"/view {image.filename} returned "
                             f"{response.status_code}")
        return response.content

    # --- the drive loop ----------------------------------------------------
    def wait(self, prompt_id: str, *, timeout_s: float = 900.0,
             poll_s: float = 2.0, sleep=None, now=None) -> dict[str, Any]:
        """Block until the job finishes; raise on an errored or timed-out job.

        `sleep` and `now` are injectable so a test can drive the loop without
        actually waiting - the loop is the part worth testing, not time.
        """
        import time as _time

        sleep = sleep or _time.sleep
        now = now or _time.monotonic
        started = now()
        while True:
            entry = self.history(prompt_id).get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(f"job {prompt_id} errored: "
                                     f"{status.get('messages')}")
                if status.get("completed") is False and not entry.get("outputs"):
                    pass  # queued but not finished; keep polling
                else:
                    return entry
            if now() - started > timeout_s:
                raise ComfyError(f"job {prompt_id} did not finish within "
                                 f"{timeout_s:.0f}s")
            sleep(poll_s)

    @staticmethod
    def outputs(entry: dict[str, Any]) -> list[OutputImage]:
        images: list[OutputImage] = []
        for node_outputs in (entry.get("outputs") or {}).values():
            for spec in node_outputs.get("images") or []:
                if spec.get("filename"):
                    images.append(OutputImage(spec["filename"],
                                              spec.get("subfolder", ""),
                                              spec.get("type", "output")))
        return images


# ---------------------------------------------------------------------------
# Workflow templating
# ---------------------------------------------------------------------------
def load_workflow(name: str, workflow_dir: Path | None = None) -> dict[str, Any]:
    base = Path(workflow_dir) if workflow_dir is not None else config.WORKFLOW_DIR
    path = base / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"no workflow at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def patch_points(workflow: dict[str, Any]) -> dict[str, str]:
    """`_meta.title` -> node id, for every node marked PATCH.

    Titles are the contract between gpu/workflows/*.json and this package, which
    is why NOTES.md lists them per workflow.
    """
    found: dict[str, str] = {}
    for node_id, node in workflow.items():
        title = (node.get("_meta") or {}).get("title", "")
        if title.startswith(PATCH_PREFIX):
            found[title] = node_id
    return found


def find_node(workflow: dict[str, Any], needle: str) -> str:
    """Node id whose PATCH title contains `needle` (case-insensitive)."""
    needle = needle.lower()
    matches = [node_id for title, node_id in patch_points(workflow).items()
               if needle in title.lower()]
    if not matches:
        raise KeyError(
            f"no node titled 'PATCH ...{needle}...'; the workflow's patch "
            f"points are {sorted(patch_points(workflow))}")
    if len(matches) > 1:
        raise KeyError(f"{needle!r} matches {len(matches)} patch points; "
                       f"be more specific")
    return matches[0]


def patch(workflow: dict[str, Any], needle: str, **inputs: Any) -> dict[str, Any]:
    """Set inputs on the node whose PATCH title matches. Mutates and returns it."""
    node_id = find_node(workflow, needle)
    node_inputs = workflow[node_id].setdefault("inputs", {})
    for key, value in inputs.items():
        if key not in node_inputs:
            raise KeyError(
                f"node {node_id} ({workflow[node_id].get('class_type')}) has no "
                f"input {key!r}; it has {sorted(node_inputs)}")
        node_inputs[key] = value
    return workflow


def bypass(workflow: dict[str, Any], needle: str, passthrough: str) -> dict[str, Any]:
    """Remove the node whose PATCH title matches and wire every consumer of its
    output straight to whatever fed its `passthrough` input. Mutates and returns.

    Why this exists: ComfyUI validates every combo input against the loader's
    file list BEFORE execution, so a `LoraLoaderModelOnly` naming a file that
    is not on disk is rejected with `value_not_in_list` even at strength 0.0 -
    the zero-strength early return in `load_lora` never gets to run. NOTES.md
    had that wrong; found on the first identity pass (knight_strike,
    2026-09-15). Bypassing the node is exactly what the editor's Ctrl+B does.
    """
    node_id = find_node(workflow, needle)
    source = workflow[node_id].get("inputs", {}).get(passthrough)
    if not isinstance(source, list):
        raise KeyError(f"node {node_id} input {passthrough!r} is not a link; "
                       f"nothing to pass through")
    for other_id, other in workflow.items():
        if other_id == node_id:
            continue
        for key, value in list((other.get("inputs") or {}).items()):
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                other["inputs"][key] = list(source)
    del workflow[node_id]
    return workflow


def patch_checkpoint(workflow: dict[str, Any], file: str) -> dict[str, Any]:
    """The concept generation feature: swap the model the graph loads. The
    loader is titled `PATCH checkpoint (...)` - "checkpoint", not "model",
    because `find_node` matches substrings and the LoRA title already says
    `strength_model`. Whichever of `unet_name` / `model_name` / `ckpt_name`
    the loader has is the input that takes the file."""
    node_id = find_node(workflow, "checkpoint")
    inputs = workflow[node_id].setdefault("inputs", {})
    key = next((k for k in ("unet_name", "model_name", "ckpt_name") if k in inputs), None)
    if key is None:
        raise KeyError(f"node {node_id} ({workflow[node_id].get('class_type')}) has no "
                       f"model-file input; it has {sorted(inputs)}")
    inputs[key] = file
    return workflow


def chain_loras(workflow: dict[str, Any], needle: str,
                loras: list[tuple[str, float]] | list[dict[str, Any]]) -> list[str]:
    """Set the LoRA node titled `needle` to the first LoRA and clone it once
    per extra one, chained model -> model, so N LoRAs stack the way N
    `LoraLoaderModelOnly` nodes in a row do in the editor. No LoRAs: the
    node is bypassed (`bypass`), never left at strength 0 - ComfyUI validates
    the filename even then. Returns the ids of the nodes that carry a LoRA,
    in chain order. Mutates the workflow.
    """
    rows = [(r["file"], float(r["strength"])) if isinstance(r, dict) else (r[0], float(r[1]))
            for r in loras]
    rows = [(f, s) for f, s in rows if s != 0]
    if not rows:
        bypass(workflow, needle, "model")
        return []
    anchor = find_node(workflow, needle)
    patch(workflow, needle, lora_name=rows[0][0], strength_model=rows[0][1])
    if len(rows) == 1:
        return [anchor]
    numeric = [int(k) for k in workflow if str(k).isdigit()]
    next_id = (max(numeric) + 1) if numeric else 1000
    consumers = [(other_id, key) for other_id, other in workflow.items()
                 if other_id != anchor
                 for key, value in (other.get("inputs") or {}).items()
                 if isinstance(value, list) and len(value) == 2 and value[0] == anchor]
    ids = [anchor]
    previous = anchor
    for offset, (file, strength) in enumerate(rows[1:], 1):
        node_id = str(next_id)
        next_id += 1
        clone = json.loads(json.dumps(workflow[anchor]))
        clone["inputs"]["model"] = [previous, 0]
        clone["inputs"]["lora_name"] = file
        clone["inputs"]["strength_model"] = strength
        title = (clone.get("_meta") or {}).get("title", "LoRA")
        clone.setdefault("_meta", {})["title"] = f"{title} +{offset}".replace(PATCH_PREFIX, "chained", 1)
        workflow[node_id] = clone
        ids.append(node_id)
        previous = node_id
    for other_id, key in consumers:
        workflow[other_id]["inputs"][key] = [previous, 0]
    return ids


def unlink(workflow: dict[str, Any], needle: str) -> dict[str, Any]:
    """Remove the node whose PATCH title matches and drop every input that
    pointed at it, leaving those inputs unset (they must be optional on the
    consumer - `TextEncodeQwenImageEditPlus.image2/3` are). Mutates and returns.
    """
    node_id = find_node(workflow, needle)
    for other_id, other in workflow.items():
        if other_id == node_id:
            continue
        for key, value in list((other.get("inputs") or {}).items()):
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                del other["inputs"][key]
    del workflow[node_id]
    return workflow


def prune_orphans(workflow: dict[str, Any], *,
                  outputs: tuple[str, ...] = ("SaveImage", "SaveVideo")
                  ) -> list[str]:
    """Drop every node whose output nothing consumes, except the output nodes.
    Repeats until stable, so a loader feeding only a bypassed node goes too.

    After `bypass()` removes the ControlNet apply node, its ModelPatchLoader and
    the LoadImage that fed it are still in the graph. ComfyUI only executes
    what an output node reaches, so they would not RUN - but a `LoadImage`
    naming a file that was never uploaded is still validated, and the
    provenance hash and `models_lock.for_workflow` would still list a control
    model the image never saw. Returns the removed node ids.
    """
    removed: list[str] = []
    while True:
        consumed: set[str] = set()
        for node in workflow.values():
            for value in (node.get("inputs") or {}).values():
                if isinstance(value, list) and len(value) == 2 and \
                        isinstance(value[0], str):
                    consumed.add(value[0])
        orphans = [node_id for node_id, node in workflow.items()
                   if node_id not in consumed
                   and node.get("class_type") not in outputs]
        if not orphans:
            return removed
        for node_id in orphans:
            del workflow[node_id]
            removed.append(node_id)


def workflow_sha256(workflow: dict[str, Any]) -> str:
    """Hash of the graph AS SENT - the provenance record, not the file on disk."""
    return sha256_bytes(canonical_json(workflow))
