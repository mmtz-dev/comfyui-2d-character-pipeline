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

    def upload_image(self, path: Path, *, subfolder: str = "",
                     overwrite: bool = True) -> str:
        """Upload one PNG and return the name a LoadImage node should use.

        ComfyUI answers with the name it actually stored under (it de-duplicates
        by appending a counter unless overwrite is set), and that returned name -
        not the local filename - is what gets patched into the workflow.
        """
        path = Path(path)
        files = {"image": (path.name, path.read_bytes(), "image/png")}
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


def workflow_sha256(workflow: dict[str, Any]) -> str:
    """Hash of the graph AS SENT - the provenance record, not the file on disk."""
    return sha256_bytes(canonical_json(workflow))
