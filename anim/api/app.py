"""The FastAPI app - every route of the pipeline API feature.

    uvicorn --factory anim.api.app:create_app --host 0.0.0.0 --port 8720

Routes (all JSON unless said otherwise):

    GET  /jobs                             every job under work/: name, stages done, direction, action, updated
    GET  /jobs/{job}                       job.json + stage status + frame-set sizes
    GET  /jobs/{job}/frames                every PNG set (frames, pose, depth, clip, control, strip, render, post, candidates) with URLs
    GET  /jobs/{job}/frames/{stage}/{n}.png  the PNG itself
    PUT  /jobs/{job}/frames/{stage}/{n}.png  overwrite it (image/png body, same size, alpha rounded)
    GET  /jobs/{job}/picks                 the review state (ReviewApp.state)
    PUT  /jobs/{job}/picks  {"picks": []}  rewrite picks.json exactly as `anim review` did
    POST /jobs/{job}/regenerate            flag rerender_requested in job.json (never the GPU)
    POST /jobs  {"stage", "args" | "argv"} run a stage in the background -> 202 + the run
                                           (423 when the unit's curation gate blocks it)
    GET  /runs                             every run this API knows, newest first
    GET  /runs/{run}                       status + log
    GET  /runs/{run}/events                text/event-stream: log, progress, done
    POST /runs/{run}/cancel                stop a queued or running stage -> 202 + the run (status `cancelled`)
    GET  /settings   PUT /settings         work/project.json (palette shown expanded)
    GET  /palette                          the project palette record (palette.json)
    POST /palette  {"lospec"} | {"file"} | {"unit", "colours" | "from_sprite"}
                                           run `anim palette` in the background -> 202 + the run
    GET  /palette/units                    the units that have a sub-palette
    GET  /palette/units/{unit}             one unit's sub-palette (404: the whole palette applies)
    GET  /curation                         the units with cells or a curation record, and the mode
    GET  /curation/{unit}                  the board: gates, cells with their strips, what is waiting
    PUT  /curation/{unit}/cells/{action}/{dir}  {"status"} | {"frames": {n: accept|reject|null}}
    POST /curation/{unit}/cells/{action}/{dir}/regenerate   new seed + the cell's GPU stage -> 202
    PUT  /curation/{unit}/gates/{gate}  {"state"}   base | turnaround -> pending | accepted
    PUT  /curation/{unit}/views/{view}  {"state"}   a view of the base -> pending | accepted | rejected
    GET  /export/{unit}                    the last `anim export` of a unit: files, animations, warnings
    GET  /turnaround/{unit}                the turnaround: per direction status, seconds, the stills' URLs, the identity table
    GET  /concept/{unit}                   the concept generation feature: the base attempts, every view's attempts, picks, defaults
    POST /concept/{unit}/generate          {"op": txt2img|views|img2img|edit, ...} -> `anim concept` in the background -> 202 + the run
                                           (423 while the base gate is pending for a view op)
    PUT  /concept/{unit}/upload            the owner's PNG (image/png body) as a base candidate
    POST /concept/{unit}/pick  {"index"}   accept that base candidate (= the base gate)
    POST /concept/{unit}/views/{view}/pick {"index"}  accept that view attempt for the turnaround
    PUT  /concept/{unit}/defaults          {"family", "model", "loras"}: the unit's default checkpoint + LoRAs
    GET  /models                           the model registry by family and role (?probe=1 asks ComfyUI what is installed)
    GET  /health                           {"ok": true, "work_dir": ..., "comfyui": reachable?}
    GET  /                                 web/dist/ when built, else the jobs page
    GET  /jobs/{job}/review                the click-to-pick page

The app is a layer over the stage functions and `ReviewApp`, never a copy of
them: a job is read by anim.job, picks are written by anim.stages.review, and
a run is the CLI in a subprocess (anim.api.runs). `create_app` takes every
path so the tests run it on a temp work/ with no environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, \
    RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anim import __version__, comfy, config, curation, models_lock, palettekit
from anim.api import concept as concept_api
from anim.api import jobs as jobs_api
from anim.api import pages, settings
from anim.api import runs as runs_api
from anim.api.runs import RunManager, RunRefused
from anim.job import Job
from anim.stages import concept as concept_stage
from anim.stages import export as export_stage
from anim.stages import turnaround as turnaround_stage
from anim.stages.review import ReviewApp


class RunRequest(BaseModel):
    """`POST /jobs`. `args` are the stage's `--help` names; `argv` is the raw
    CLI tail for a caller that already speaks it. One or the other."""

    stage: str
    args: dict[str, Any] | None = None
    argv: list[str] | None = None


class PicksRequest(BaseModel):
    picks: list[int] = Field(default_factory=list)


class CellRequest(BaseModel):
    """`PUT /curation/{unit}/cells/{action}/{dir}` (the curation board
    feature): a `status` for the cell, and/or `frames` marks merged over the
    ones there are (`null` clears a mark)."""

    status: str | None = None
    frames: dict[str, str | None] | None = None


class GateRequest(BaseModel):
    state: str


class RegenerateRequest(BaseModel):
    seed: int | None = None


class ConceptRequest(BaseModel):
    """`POST /concept/{unit}/generate` (the concept generation feature). One
    `op`; the rest are that op's `anim concept` flags by their `--help`
    names, so `args()` is what the run manager turns into argv."""

    op: str
    prompt: str | None = None
    views: list[str] | str | None = None
    view_prompt: str | None = None
    ref: str | None = None
    strength: float | None = None
    instruction: str | None = None
    count: int | None = Field(default=None, ge=1, le=4)
    seed: int | None = None
    model: str | None = None
    loras: list[str | dict[str, Any]] | None = None
    lora_strength: float | None = None
    sketch: str | None = None
    fresh: bool = False
    force: bool = False

    def args(self, unit: str) -> dict[str, Any]:
        args: dict[str, Any] = {"slug": unit}
        if self.op == "txt2img":
            if not self.prompt or not self.prompt.strip():
                raise HTTPException(400, "txt2img needs a `prompt`")
            args["prompt"] = self.prompt.strip()
            if self.sketch:
                args["sketch"] = self.sketch
            if self.fresh:
                args["fresh"] = True
        elif self.op == "views":
            views = self.views
            if isinstance(views, list):
                views = ",".join(v.strip() for v in views if v.strip())
            args["views"] = views if views else ""
            if self.view_prompt and self.view_prompt.strip():
                args["view_prompt"] = self.view_prompt.strip()
        elif self.op in ("img2img", "edit"):
            if not self.ref:
                raise HTTPException(400, f"{self.op} needs a `ref` (base:NN or <view>:NN)")
            args[self.op] = self.ref
            if self.op == "img2img":
                if self.strength is not None:
                    args["strength"] = self.strength
                if self.prompt and self.prompt.strip():
                    args["prompt"] = self.prompt.strip()
            else:
                if not self.instruction or not self.instruction.strip():
                    raise HTTPException(400, "edit needs an `instruction`")
                args["instruction"] = self.instruction.strip()
        else:
            raise HTTPException(400, f"op is txt2img, views, img2img or edit, not {self.op!r}")
        if self.count is not None:
            args["count"] = self.count
        if self.seed is not None:
            args["seed"] = self.seed
        if self.model:
            args["model"] = self.model
        if self.loras is not None:
            specs = []
            for spec in self.loras:
                if isinstance(spec, dict):
                    file = str(spec.get("file") or "").strip()
                    if not file:
                        raise HTTPException(400, "a LoRA needs a `file`")
                    strength = spec.get("strength")
                    specs.append(f"{file}:{strength}" if strength is not None else file)
                else:
                    specs.append(str(spec))
            args["lora"] = specs
        if self.lora_strength is not None:
            args["lora_strength"] = self.lora_strength
        if self.force:
            args["force"] = True
        return args


class PickRequest(BaseModel):
    index: int
    parts: str | None = None
    avoid: str | None = None


class DefaultsRequest(BaseModel):
    family: str = "zimage"
    model: str | None = None
    loras: list[str | dict[str, Any]] = Field(default_factory=list)


class PaletteRequest(BaseModel):
    """`POST /palette` (the palette feature). One of `lospec` / `file` for the
    project palette, or `unit` with `colours` / `from_sprite` for a unit's
    sub-palette; `clear` with `unit` removes it. Becomes `anim palette ...`
    through the run manager like any stage."""

    lospec: str | None = None
    file: str | None = None
    name: str | None = None
    author: str | None = None
    unit: str | None = None
    colours: list[str] | str | None = None
    from_sprite: str | None = None
    clear: bool = False

    def args(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.unit:
            args["unit"] = self.unit
            if self.clear:
                args["clear"] = True
            elif self.colours:
                args["colours"] = (",".join(self.colours) if isinstance(self.colours, list)
                                   else self.colours)
            elif self.from_sprite:
                args["from_sprite"] = self.from_sprite
            else:
                raise HTTPException(400, "a unit needs `colours`, `from_sprite` or `clear`")
        elif self.lospec and self.file:
            raise HTTPException(400, "give `lospec` or `file`, not both")
        elif self.lospec:
            args["lospec"] = self.lospec
        elif self.file:
            args["file"] = self.file
            if self.name:
                args["name"] = self.name
            if self.author:
                args["author"] = self.author
        else:
            raise HTTPException(400, "give `lospec`, `file`, or `unit` with "
                                     "`colours` / `from_sprite` / `clear`")
        return args


def create_app(work_dir: Path | None = None, *, web_dist: Path | None = None,
               comfyui_url: str | None = None, cpu_slots: int | None = None,
               runs: RunManager | None = None) -> FastAPI:
    work_dir = Path(work_dir) if work_dir is not None else config.WORK_DIR
    web_dist = Path(web_dist) if web_dist is not None else config.WEB_DIST
    comfyui_url = comfyui_url or config.COMFYUI_URL
    manager = runs or RunManager(work_dir, cpu_slots=cpu_slots,
                                 comfyui_url=comfyui_url)

    app = FastAPI(title="anim pipeline API", version=__version__,
                  description=(__doc__ or "").split("Routes")[0].strip())
    app.state.work_dir = work_dir
    app.state.runs = manager

    def _job(name: str) -> Job:
        job = jobs_api.open_job(name, work_dir)
        if job is None:
            raise HTTPException(404, f"no job {name!r} in {work_dir}")
        return job

    def _review(job: Job) -> ReviewApp:
        try:
            return ReviewApp(job)
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc)) from None

    # --- health ------------------------------------------------------------
    @app.get("/health")
    def health(probe: bool = False) -> dict[str, Any]:
        """`?probe=1` also asks ComfyUI whether it is up (a 3 s timeout when
        the native daemon is off, which is the normal case - so opt-in)."""
        busy = manager.active_gpu()
        row: dict[str, Any] = {
            "ok": True, "version": __version__, "work_dir": str(work_dir),
            "comfyui_url": comfyui_url, "gpu_run": busy.id if busy else None,
            "cpu_runs": [r.id for r in manager.active_cpu()]}
        if probe:
            row["comfyui"] = comfy.ComfyClient(comfyui_url, timeout=3.0).reachable()
        return row

    # --- jobs --------------------------------------------------------------
    @app.get("/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs_api.list_jobs(work_dir)

    def _gate(stage: str, args: dict[str, Any] | None, argv: list[str] | None
              ) -> JSONResponse | None:
        """The curation board feature's 423: the unit's gate is pending. The
        run manager's own refusals (unknown stage, bad args) come first."""
        try:
            line = argv if argv is not None else runs_api.argv_for(stage, dict(args or {}))
        except RunRefused:
            return None
        reason = curation.gate_for_request(stage, [str(a) for a in line], work_dir)
        if reason is None:
            return None
        return JSONResponse(status_code=423, content={"error": reason, "gate": True})

    @app.post("/jobs", status_code=202)
    def start_run(body: RunRequest, response: Response) -> Any:
        locked = _gate(body.stage, body.args, body.argv)
        if locked is not None:
            return locked
        try:
            run = manager.start(body.stage, args=body.args, argv=body.argv)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{run.id}"
        return run.record()

    @app.get("/jobs/{name}")
    def get_job(name: str) -> dict[str, Any]:
        return jobs_api.detail(_job(name))

    @app.get("/jobs/{name}/frames")
    def list_frames(name: str) -> dict[str, Any]:
        return jobs_api.frames(_job(name))

    @app.get("/jobs/{name}/frames/{stage}/{index:int}.png")
    def get_frame(name: str, stage: str, index: int) -> FileResponse:
        path = jobs_api.frame_path(_job(name), stage, index)
        if path is None:
            raise HTTPException(404, f"no {stage} frame {index} in {name}")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "no-store"})

    @app.put("/jobs/{name}/frames/{stage}/{index:int}.png")
    async def put_frame(name: str, stage: str, index: int, request: Request
                        ) -> dict[str, Any]:
        """The web pixel editor's save: the body is the PNG, nothing else.
        409 when its size differs from the file it replaces, 415 when it is
        not a PNG, 404 when there is no such frame to replace."""
        job = _job(name)
        data = await request.body()
        try:
            return jobs_api.save_frame(job, stage, index, data)
        except jobs_api.FrameRejected as exc:
            raise HTTPException(exc.status, str(exc)) from None

    @app.get("/jobs/{name}/picks")
    def get_picks(name: str) -> dict[str, Any]:
        return _review(_job(name)).state()

    @app.put("/jobs/{name}/picks")
    def put_picks(name: str, body: PicksRequest) -> dict[str, Any]:
        return _review(_job(name)).set_picks(body.picks)

    @app.post("/jobs/{name}/regenerate")
    def regenerate(name: str) -> dict[str, Any]:
        return _review(_job(name)).regenerate()

    @app.get("/jobs/{name}/review", response_class=HTMLResponse)
    def review_page(name: str) -> str:
        return pages.review_page(_job(name).name)

    # --- runs --------------------------------------------------------------
    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return manager.list()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, tail: int | None = None) -> dict[str, Any]:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return run.record(with_log=True, tail=tail)

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return StreamingResponse(
            manager.events(run), media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    @app.post("/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str) -> Any:
        """The concept generation feature's Cancel: stop the stage's process
        (and ComfyUI's current prompt for a GPU stage); the run ends as
        `cancelled` and its GPU slot is free again."""
        try:
            run = manager.cancel(run_id)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        return run.record()

    # --- settings ----------------------------------------------------------
    @app.get("/settings")
    def get_settings() -> dict[str, Any]:
        return settings.load(work_dir)

    @app.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body is not JSON") from None
        try:
            return settings.save(work_dir, body)
        except settings.SettingsError as exc:
            raise HTTPException(400, str(exc)) from None

    # --- the palette feature ----------------------------------------------
    @app.get("/palette")
    def get_palette() -> dict[str, Any]:
        path = settings.palette_path(work_dir)
        record = palettekit.load(path)
        if record is None:
            raise HTTPException(404, f"no project palette at {path}; POST /palette "
                                     f"with a Lospec slug or a file")
        return {**record, "path": str(path), "units": palettekit.list_unit_palettes(path)}

    @app.post("/palette", status_code=202)
    def set_palette(body: PaletteRequest, response: Response) -> Any:
        args = body.args()
        args["target"] = str(settings.palette_path(work_dir))
        try:
            run = manager.start("palette", args=args)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{run.id}"
        return run.record()

    @app.get("/palette/units")
    def list_unit_palettes() -> dict[str, Any]:
        path = settings.palette_path(work_dir)
        return {"units": palettekit.list_unit_palettes(path), "path": str(path)}

    @app.get("/palette/units/{unit}")
    def get_unit_palette(unit: str) -> dict[str, Any]:
        path = settings.palette_path(work_dir)
        try:
            record = palettekit.read_unit_palette(unit, path)
        except palettekit.PaletteError as exc:
            raise HTTPException(400, str(exc)) from None
        if record is None:
            raise HTTPException(404, f"{unit!r} has no sub-palette; the whole project "
                                     f"palette applies")
        return {**record, "path": str(palettekit.unit_palette_path(unit, path))}

    # --- the curation board feature ---------------------------------------
    def _curated(call: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except curation.CurationError as exc:
            raise HTTPException(exc.status, str(exc)) from None

    @app.get("/curation")
    def list_curation() -> dict[str, Any]:
        return {"units": curation.list_units(work_dir), "mode": curation.mode_of(work_dir)}

    @app.get("/curation/{unit}")
    def get_board(unit: str) -> dict[str, Any]:
        return _curated(curation.board, unit, work_dir)

    @app.put("/curation/{unit}/cells/{action}/{direction}")
    def put_cell(unit: str, action: str, direction: str, body: CellRequest
                 ) -> dict[str, Any]:
        """Marks first, then the status: a `{"status": "accepted", "frames":
        {...}}` records the marks and then locks the cell."""
        if body.status is None and body.frames is None:
            raise HTTPException(400, "give `status` and/or `frames`")
        if body.frames is not None:
            _curated(curation.set_frames, unit, action, direction, body.frames, work_dir)
        if body.status is not None:
            _curated(curation.set_cell_status, unit, action, direction, body.status,
                     work_dir)
        key = _curated(curation.cell_key, action, direction)
        return _curated(curation.board, unit, work_dir)["cells"][key]

    @app.post("/curation/{unit}/cells/{action}/{direction}/regenerate", status_code=202)
    def regenerate_cell(unit: str, action: str, direction: str, response: Response,
                        body: RegenerateRequest | None = None) -> Any:
        """A new seed, then the cell's GPU stage through the run manager: the
        unit's gate is the 423, a busy card the 409, as for `POST /jobs`."""
        key = _curated(curation.cell_key, action, direction)
        cell = _curated(curation.board, unit, work_dir)["cells"].get(key)
        if cell is None or not cell.get("job"):
            raise HTTPException(404, f"{unit} has no job for cell {key}")
        locked = _gate(cell["stage"], {"job": cell["job"]}, None)
        if locked is not None:
            return locked
        try:
            out = _curated(curation.regenerate, unit, action, direction, manager.start,
                           work_dir, seed=body.seed if body else None)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{out['run']['id']}"
        return out

    @app.put("/curation/{unit}/gates/{gate}")
    def put_gate(unit: str, gate: str, body: GateRequest) -> dict[str, Any]:
        _curated(curation.set_gate, unit, gate, body.state, work_dir)
        return _curated(curation.board, unit, work_dir)

    @app.put("/curation/{unit}/views/{view}")
    def put_view(unit: str, view: str, body: GateRequest) -> dict[str, Any]:
        """The views step under the base gate (the multi-view reference
        feature): accept / reject / reopen one view of the base."""
        _curated(curation.set_views, unit, {view: body.state}, work_dir)
        return _curated(curation.board, unit, work_dir)

    # --- the spritesheet export feature -----------------------------------
    @app.get("/export/{unit}")
    def get_export(unit: str) -> dict[str, Any]:
        """What the last `anim export <unit>` wrote under `work/_export/<unit>/`
        - the atlas's summary and every file. An export runs like any stage:
        `POST /jobs {"stage": "export", "args": {"unit": ...}}`."""
        if not export_stage.UNIT_RE.match(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        row = export_stage.listing(unit, work_dir)
        if row is None:
            raise HTTPException(404, f"no export of {unit!r} yet; POST /jobs "
                                     f"{{\"stage\": \"export\", \"args\": "
                                     f"{{\"unit\": \"{unit}\"}}}}")
        return row

    # --- the turnaround feature -------------------------------------------
    @app.get("/turnaround/{unit}")
    def get_turnaround(unit: str) -> dict[str, Any]:
        """What `anim turnaround <unit>` recorded: the state file with, per
        direction, its status, seconds and the URLs of the still, the raw
        clip frame, the control frame, the skeleton and the depth through
        the frame-set routes above. The board reads this to show the eight
        stills as the thing the `turnaround` gate accepts. A turnaround runs
        like any stage: `POST /jobs {"stage": "turnaround", "args": {"unit": ...}}`."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        row = turnaround_stage.listing(unit, work_dir)
        if row is None:
            raise HTTPException(404, f"no turnaround of {unit!r} yet; POST /jobs "
                                     f"{{\"stage\": \"turnaround\", \"args\": "
                                     f"{{\"unit\": \"{unit}\"}}}}")
        return row

    # --- the concept generation feature -----------------------------------
    def _concept_state(unit: str) -> dict[str, Any]:
        row = concept_api.state(unit, work_dir)
        if row is None:
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        return row

    @app.get("/concept/{unit}")
    def get_concept(unit: str) -> dict[str, Any]:
        """The base attempts, every view's attempts, the picks, the unit's
        model defaults and whether the views are locked behind the base gate."""
        return _concept_state(unit)

    @app.post("/concept/{unit}/generate", status_code=202)
    def generate_concept(unit: str, body: ConceptRequest, response: Response) -> Any:
        """`anim concept ...` in the background: txt2img of the base, views of
        the accepted base, img2img or an edit of one attempt. 423 while a
        view op waits for the base gate, 409 while the GPU is busy."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        args = body.args(unit)
        locked = _gate("concept", args, None)
        if locked is not None:
            return locked
        try:
            run = manager.start("concept", args=args)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{run.id}"
        return run.record()

    @app.put("/concept/{unit}/upload")
    async def upload_concept(unit: str, request: Request) -> dict[str, Any]:
        """The owner's own image as a base candidate: the body is the PNG.
        415 when it is not one. No GPU, no run - the state comes straight back."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        data = await request.body()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(415, "the body is not a PNG")
        name = request.headers.get("x-filename") or "upload.png"
        try:
            row = concept_stage.register_upload(unit, data, name=Path(name).name, work_dir=work_dir)
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from None
        return {**_concept_state(unit), "uploaded": row}

    @app.post("/concept/{unit}/pick")
    def pick_concept(unit: str, body: PickRequest) -> dict[str, Any]:
        """Accept a base candidate: `anim concept --pick NN`, which IS the
        base gate. Optional `parts` / `avoid` phrases ride along."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        try:
            concept_stage.pick(unit, body.index, parts=body.parts, avoid=body.avoid,
                               work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return _concept_state(unit)

    @app.post("/concept/{unit}/views/{view}/pick")
    def pick_concept_view(unit: str, view: str, body: PickRequest) -> dict[str, Any]:
        """Accept one view attempt for the turnaround: `--pick-view view=NN`."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        if view not in config.CONCEPT_VIEWS:
            raise HTTPException(404, f"no view {view!r}; one of {list(config.CONCEPT_VIEWS)}")
        try:
            concept_stage.pick_view(unit, {view: body.index}, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return _concept_state(unit)

    @app.put("/concept/{unit}/defaults")
    def put_concept_defaults(unit: str, body: DefaultsRequest) -> dict[str, Any]:
        """The unit's default checkpoint + LoRAs for one family, checked
        against the registry; a later generate without its own picks uses them."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        try:
            concept_stage.set_defaults(unit, body.family, model=body.model, loras=body.loras,
                                       work_dir=work_dir)
        except (ValueError, models_lock.RegistryError) as exc:
            raise HTTPException(400, str(exc)) from None
        return _concept_state(unit)

    @app.get("/models")
    def get_models(probe: bool = False) -> dict[str, Any]:
        """gpu/models.registry.tsv by family and role; `?probe=1` also asks
        ComfyUI which files it can see (3 s timeout, opt-in like /health)."""
        return concept_api.models(probe=probe, comfyui_url=comfyui_url)

    # --- the pages ---------------------------------------------------------
    if web_dist.is_dir() and (web_dist / "index.html").exists():
        # The editor, once the web pixel editor feature builds it. Mounted
        # last so every API route above wins over a file of the same name.
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def index() -> str:
            return pages.index_page()

        @app.get("/index.html", include_in_schema=False)
        def index_html() -> RedirectResponse:
            return RedirectResponse("/")

    return app
