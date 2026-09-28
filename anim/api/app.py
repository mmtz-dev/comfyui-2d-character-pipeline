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
    GET  /runs/{run}                       status + log (+ `cells`, a batch stage's tally)
    GET  /runs/{run}/events                text/event-stream: log, progress, done
    POST /runs/{run}/cancel                stop a queued or running stage -> 202 + the run (status `cancelled`)
    GET  /settings   PUT /settings         work/project.json (palette shown expanded)
    GET  /palette                          the project palette record (palette.json)
    POST /palette  {"lospec"} | {"file"} | {"unit", "colours" | "from_sprite"}
                                           run `anim palette` in the background -> 202 + the run
    GET  /palette/units                    the units that have a sub-palette
    GET  /palette/units/{unit}             one unit's sub-palette (404: the whole palette applies)
    POST /curation/{unit}/cells/{action}/{dir}/regenerate   new seed + the cell's GPU stage -> 202
    GET  /export/{unit}                    the last `anim export` of a unit: files, animations, warnings
    GET  /concept/{unit}                   the concept generation feature: the base attempts, every view's attempts, picks, defaults
    POST /concept/{unit}/generate          {"op": txt2img|views|img2img|edit, ...} -> `anim concept` in the background -> 202 + the run
                                           (423 while the base gate is pending for a view op)
    PUT  /concept/{unit}/upload            the owner's PNG (image/png body) as a base candidate
    POST /concept/{unit}/pick  {"index"}   accept that base candidate (= the base gate)
    POST /concept/{unit}/views/{view}/pick {"index"}  accept that view attempt (the Animate tab draws its direction from it)
    PUT  /concept/{unit}/attempts/{target}/{index}/image   the retouch feature: write this PNG over that attempt (the editor's Save)
    POST /concept/{unit}/attempts/{target}/{index}/inpaint the canvas (RGBA, erased = redraw) -> `anim concept --inpaint` in the background -> 202 + the run
    PUT  /concept/{unit}/attempts/{target}/{index}  {"hidden": bool}: hide one attempt from the strips (the accepted one is refused)
    DELETE /concept/{unit}/attempts/{target}/{index}  delete a hidden attempt's PNG; the row stays as its record
    PUT  /concept/{unit}/defaults          {"family", "model", "loras"}: the unit's default checkpoint + LoRAs
    GET  /units                           every unit the studio knows (the unit bar)
    GET  /animate/{unit}                   the Animate tab's grid: every action x side direction, each cell's state
    POST /animate/{unit}/sweep             {"cells": [{"action", "direction"}], "batch": keys|inbetweens} -> ONE
                                           `anim animate --sweep` run over those animations -> 202 + the run
    GET  /animate/{unit}/{action}/{dir}   the animate tab feature: the compass, the actions, the frame row
                                           (per frame its status, attempts, neighbours, prompt), the mirror, the output
    POST /animate/{unit}/{action}/{dir}/generate   {"frame"} | {"batch": all|keys|rejected|inbetweens}, + prompt,
                                           draw_from, skeleton, skeleton_frame, count, seed, model -> `anim animate` -> 202 + the run
    POST /animate/{unit}/{action}/{dir}/frames      {"after"}: add an in-between after frame `after` (0: first)
    PUT  /animate/{unit}/{action}/{dir}/frames/{id} {"key", "phase"} and/or {"current"}: mark / un-mark, show an attempt
    DELETE /animate/{unit}/{action}/{dir}/frames/{id}   take the frame out of the row (its attempts are kept)
    POST /animate/{unit}/{action}/{dir}/frames/{id}/approve  {"index"?}   approve the current attempt (or that one)
    POST /animate/{unit}/{action}/{dir}/frames/{id}/reject   {"index"?}   reject it
    POST /animate/{unit}/{action}/{dir}/approve-reviewed      every frame to review -> approved
    POST /animate/{unit}/{action}/{dir}/mirror      flip the partner direction's frames in (no GPU; to review) + `warning`
    POST /animate/{unit}/{action}/{dir}/output      {"replace_job"?}: the approved frames -> work/<unit>_<verb>[_<dir>]/
    PUT  /animate/{unit}/{action}/{dir}/attempts/{id}/{index}  {"hidden": bool}; DELETE deletes a hidden attempt's PNG
    PUT  /animate/{unit}/{action}/{dir}/attempts/{id}/{index}/image    the editor's Save over that attempt
    POST /animate/{unit}/{action}/{dir}/attempts/{id}/{index}/inpaint  the canvas (erased = redraw) -> `anim animate --inpaint` -> 202
    GET  /animate/{unit}/{action}/{dir}/skeleton/{n}.png  scrub frame n of the rig clip at that direction (CPU, cached)
    GET  /animate/{unit}/{action}/{dir}/game/{id}.png     the frame's current attempt at game size (CPU, cached)
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
from anim.api import animate as animate_api
from anim.api import concept as concept_api
from anim.api import jobs as jobs_api
from anim.api import pages, settings
from anim.api import runs as runs_api
from anim.api.runs import RunManager, RunRefused
from anim.job import Job
from anim.stages import concept as concept_stage
from anim.stages import export as export_stage
from anim.stages import animate as animate_stage
from anim.stages.review import ReviewApp


class RunRequest(BaseModel):
    """`POST /jobs`. `args` are the stage's `--help` names; `argv` is the raw
    CLI tail for a caller that already speaks it. One or the other."""

    stage: str
    args: dict[str, Any] | None = None
    argv: list[str] | None = None


class PicksRequest(BaseModel):
    picks: list[int] = Field(default_factory=list)


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
    in_place: bool = False
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
            if self.in_place:
                # The retouch feature: one image, written over the attempt.
                args["in_place"] = True
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
        if self.count is not None and not self.in_place:
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


class AnimateRequest(BaseModel):
    """`POST /animate/{unit}/{action}/{dir}/generate` (the animate tab
    feature): one `frame` (its id, number or phase), or a `batch`. The rest
    are `anim animate`'s flags by their `--help` names."""

    frame: str | None = None
    batch: str | None = None
    prompt: str | None = None
    draw_from: str | None = None
    skeleton: bool | None = None
    skeleton_frame: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1, le=4)
    seed: int | None = None
    model: str | None = None
    loras: list[str | dict[str, Any]] | None = None
    force: bool = False

    def args(self, unit: str, action: str, direction: str) -> dict[str, Any]:
        args: dict[str, Any] = {"slug": unit, "action": action, "direction": direction}
        if self.batch is not None:
            if self.frame:
                raise HTTPException(400, "give a `frame` or a `batch`, not both")
            if self.batch not in animate_stage.BATCHES:
                raise HTTPException(400, f"batch is one of {list(animate_stage.BATCHES)}")
            args["batch"] = self.batch
        elif self.frame:
            args["frame"] = str(self.frame)
            if self.prompt and self.prompt.strip():
                args["prompt"] = self.prompt.strip()
            if self.skeleton_frame is not None:
                args["skeleton_frame"] = self.skeleton_frame
        else:
            raise HTTPException(400, "give a `frame` or a `batch`")
        if self.draw_from:
            if self.draw_from not in config.ANIMATE_DRAW_FROM:
                raise HTTPException(400, f"draw_from is one of {list(config.ANIMATE_DRAW_FROM)}")
            args["draw_from"] = self.draw_from
        if self.skeleton is not None and "skeleton_frame" not in args:
            args["skeleton"] = bool(self.skeleton)
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
        if self.force:
            args["force"] = True
        return args


class SweepRequest(BaseModel):
    """`POST /animate/{unit}/sweep` (the keyframe sweep feature): the
    animations to sweep, `{"action", "direction"}` each, and what to draw -
    `keys`, or `inbetweens` once a cell's keys are approved."""

    cells: list[dict[str, str]] = Field(min_length=1)
    batch: str = "keys"
    model: str | None = None
    force: bool = False

    def args(self, unit: str) -> dict[str, Any]:
        if self.batch not in animate_stage.SWEEP_BATCHES:
            raise HTTPException(400, f"batch is one of {list(animate_stage.SWEEP_BATCHES)}")
        specs = []
        for cell in self.cells:
            try:
                action, direction = animate_stage.parse_cell(
                    f"{cell.get('action', '')}:{cell.get('direction', '')}")
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            specs.append(f"{action}:{direction}")
        cap = config.ANIMATE_SWEEP_MAX_CELLS
        if cap and len(set(specs)) > cap:
            raise HTTPException(400, f"{len(set(specs))} cells; one sweep takes at most {cap}")
        args: dict[str, Any] = {"slug": unit, "sweep": specs, "batch": self.batch}
        if self.model:
            args["model"] = self.model
        if self.force:
            args["force"] = True
        return args


class FrameAddRequest(BaseModel):
    after: int = Field(ge=0)


class FramePutRequest(BaseModel):
    key: bool | None = None
    phase: str | None = None
    current: int | None = None


class VerdictRequest(BaseModel):
    index: int | None = None


class OutputRequest(BaseModel):
    replace_job: bool = False


class HiddenRequest(BaseModel):
    hidden: bool = True


class PickRequest(BaseModel):
    index: int
    parts: str | None = None
    avoid: str | None = None


class ViewPickRequest(BaseModel):
    """`POST /concept/{unit}/views/{view}/pick`. Either an attempt of this
    view (`index`), or the mirrored view feature's `mirror: true`, which
    derives it from its accepted partner and needs no index - the picture is
    not one of this view's attempts."""

    index: int | None = None
    mirror: bool = False


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
        """The base gate's 423: the unit's gate is pending. The
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
    def pick_concept_view(unit: str, view: str, body: ViewPickRequest) -> dict[str, Any]:
        """Accept one view of the base: `--pick-view view=NN`, or with
        `mirror: true` the mirrored view feature's `--mirror-view view` - the
        horizontal flip of its accepted partner, no GPU.

        One route rather than a sibling: both ways END in the same thing, one
        accepted view under `<unit>_base/render/view_<view>.png` with the
        curation step set, and the Concept tab already posts here. A second
        endpoint would have had to repeat the slug check, the view check and
        the refusal mapping to say "the other kind of accept".

        A mirror's `warning` (the handedness line) rides back on the state so
        the tab can show it; 400 carries the refusal - no partner, or a
        partner nobody has accepted.
        """
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        if view not in config.CONCEPT_VIEWS:
            raise HTTPException(404, f"no view {view!r}; one of {list(config.CONCEPT_VIEWS)}")
        warning: str | None = None
        try:
            if body.mirror:
                warning = concept_stage.mirror_view(unit, [view], work_dir=work_dir,
                                                    log=lambda _line: None)["warning"]
            elif body.index is None:
                raise HTTPException(400, "give an attempt `index`, or `mirror: true` to "
                                         "derive this view from its accepted partner")
            else:
                concept_stage.pick_view(unit, {view: body.index}, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return {**_concept_state(unit), "warning": warning}

    def _attempt_ref(unit: str, target: str, index: int) -> str:
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        if target != concept_stage.BASE_TARGET and target not in config.CONCEPT_VIEWS:
            raise HTTPException(404, f"no target {target!r}; 'base' or one of "
                                     f"{list(config.CONCEPT_VIEWS)}")
        return f"{target}:{index:02d}"

    @app.put("/concept/{unit}/attempts/{target}/{index:int}")
    def hide_attempt(unit: str, target: str, index: int, body: HiddenRequest
                     ) -> dict[str, Any]:
        """Hide one attempt from the strips and the views page, or show it
        again. The row and its file stay - the accepted one is refused (400),
        because every later stage reads it."""
        ref = _attempt_ref(unit, target, index)
        try:
            concept_stage.hide_attempt(unit, ref, hidden=body.hidden, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return _concept_state(unit)

    @app.delete("/concept/{unit}/attempts/{target}/{index:int}")
    def delete_attempt_file(unit: str, target: str, index: int) -> dict[str, Any]:
        """Delete a HIDDEN attempt's PNG. The row stays with its prompt, seed
        and parent, so its number is never reused and anything made from it
        still says where it came from. 400 when it is not hidden, or is the
        accepted one."""
        ref = _attempt_ref(unit, target, index)
        try:
            concept_stage.delete_attempt_file(unit, ref, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return _concept_state(unit)

    @app.put("/concept/{unit}/attempts/{target}/{index:int}/image")
    async def replace_attempt_image(unit: str, target: str, index: int, request: Request
                                    ) -> dict[str, Any]:
        """The retouch feature: the editor's Save. The body is the PNG, of the
        attempt's own size; transparency is flattened onto white. The row keeps
        its generation record and gains an `edited` block."""
        ref = _attempt_ref(unit, target, index)
        data = await request.body()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(415, "the body is not a PNG")
        try:
            row = concept_stage.replace_attempt(unit, ref, data, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        return {**_concept_state(unit), "edited": row}

    @app.post("/concept/{unit}/attempts/{target}/{index:int}/inpaint", status_code=202)
    async def inpaint_attempt(unit: str, target: str, index: int, request: Request,
                              response: Response, prompt: str | None = None,
                              strength: float | None = None, seed: int | None = None,
                              model: str | None = None) -> Any:
        """The retouch feature's masked redraw: the body is the canvas as the
        editor left it (RGBA - what you painted is the source, what you rubbed
        out is the mask). It is written under the job and redrawn by `anim
        concept --inpaint` through the run manager, so it queues behind the
        card like every other GPU op and Cancel stops it."""
        ref = _attempt_ref(unit, target, index)
        data = await request.body()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(415, "the body is not a PNG")
        try:
            mask_path = concept_stage.stash_canvas(unit, ref, data, work_dir=work_dir)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None
        args: dict[str, Any] = {"slug": unit, "inpaint": ref, "mask": str(mask_path)}
        if prompt and prompt.strip():
            args["prompt"] = prompt.strip()
        if strength is not None:
            args["strength"] = strength
        if seed is not None:
            args["seed"] = seed
        if model:
            args["model"] = model
        try:
            run = manager.start("concept", args=args)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{run.id}"
        return run.record()

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

    # --- the animate tab feature ------------------------------------------
    def _animate_cell(unit: str, action: str, direction: str) -> tuple[str, str]:
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        try:
            return (animate_stage.action_name(action), animate_stage.direction_name(direction))
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None

    def _animate_state(unit: str, action: str, direction: str) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        row = animate_api.state(unit, action, direction, work_dir)
        if row is None:
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        return row

    def _animate_call(call: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except (ValueError, FileNotFoundError, palettekit.PaletteError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400,
                                str(exc)) from None

    def _animate_run(args: dict[str, Any], response: Response) -> Any:
        locked = _gate("animate", args, None)
        if locked is not None:
            return locked
        try:
            run = manager.start("animate", args=args)
        except RunRefused as exc:
            return JSONResponse(status_code=exc.status,
                                content={"error": str(exc), **exc.extra})
        response.headers["Location"] = f"/runs/{run.id}"
        return run.record()

    @app.get("/units")
    def list_units() -> dict[str, Any]:
        """Every unit the studio knows: a slug any job carries, or a curation
        record - what the unit bar lists."""
        return {"units": animate_api.units(work_dir)}

    @app.get("/animate/{unit}")
    def get_animate_grid(unit: str) -> dict[str, Any]:
        """Every studio action x every side direction: each animation's state
        (none / keys / keys_ok / done), counts, mirrored, and why it is blocked."""
        row = animate_api.grid(unit, work_dir)
        if row is None:
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        return row

    @app.post("/animate/{unit}/sweep", status_code=202)
    def sweep_animate(unit: str, body: SweepRequest, response: Response) -> Any:
        """`anim animate --sweep ... --batch keys|inbetweens` in the background:
        ONE run over several animations. Blocked cells are skipped and reported
        in the run's log; 423 while the base gate is pending, 409 while the GPU
        is busy."""
        if not jobs_api.valid_name(unit):
            raise HTTPException(404, f"{unit!r} is not a unit slug")
        return _animate_run(body.args(unit), response)

    @app.get("/animate/{unit}/{action}/{direction}")
    def get_animate(unit: str, action: str, direction: str) -> dict[str, Any]:
        """The compass, the actions with their frame counts, the frame row
        with each frame's status, attempts and neighbouring keyframes, the
        rig clip, the mirror offer and the output."""
        return _animate_state(unit, action, direction)

    @app.post("/animate/{unit}/{action}/{direction}/generate", status_code=202)
    def generate_animate(unit: str, action: str, direction: str, body: AnimateRequest,
                         response: Response) -> Any:
        """`anim animate ...` in the background: one frame, or a batch (Generate
        all's two steps, keyframes only, the rejected, the in-betweens). 423
        while the base gate is pending, 409 while the GPU is busy. A frame
        that may not be drawn - the view is not accepted, a neighbour keyframe
        is not approved - is refused here with 400, before the run."""
        action, direction = _animate_cell(unit, action, direction)
        args = body.args(unit, action, direction)
        locked = _gate("animate", args, None)
        if locked is not None:
            return locked
        if body.frame:
            _job, record = animate_stage.open_job(unit, action, direction, work_dir,
                                                  create=False)
            position = _animate_call(animate_stage.find_frame, record, body.frame)
            frame = record["frames"][position]
            how = body.draw_from or animate_stage.default_draw_from(
                frame, animate_stage.view_keys(record))
            job = Job.open(animate_stage.job_name(unit, action, direction), work_dir)
            view = (_animate_call(animate_stage.reference, unit, direction, work_dir)
                    if how in ("view", "both", "copy") else None)
            images = _animate_call(animate_stage.resolve_draw_from, record, job, position,
                                   how, view)
            skeleton = how != "copy" and (body.skeleton_frame is not None
                                          or body.skeleton is True)
            if how != "copy" and body.skeleton is None and body.skeleton_frame is None:
                try:
                    animate_stage.rig_clip(action)
                    skeleton = True
                except (ValueError, FileNotFoundError):
                    skeleton = False
            if len(images) + int(skeleton) > config.ANIMATE_MAX_IMAGES:
                raise HTTPException(400, f"{len(images) + int(skeleton)} images; Qwen-Image-Edit "
                                         f"takes {config.ANIMATE_MAX_IMAGES} - untick the "
                                         f"skeleton, or draw from the keys or the view alone")
        return _animate_run(args, response)

    @app.post("/animate/{unit}/{action}/{direction}/frames")
    def add_animate_frame(unit: str, action: str, direction: str,
                          body: FrameAddRequest) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        row = _animate_call(animate_stage.add_frame, unit, action, direction, body.after,
                            work_dir=work_dir)
        return {**_animate_state(unit, action, direction), "added": row["id"]}

    @app.put("/animate/{unit}/{action}/{direction}/frames/{frame}")
    def put_animate_frame(unit: str, action: str, direction: str, frame: str,
                          body: FramePutRequest) -> dict[str, Any]:
        """Mark / un-mark a keyframe (`key`, `phase`), and/or make attempt
        `current` the one the frame shows."""
        action, direction = _animate_cell(unit, action, direction)
        if body.key is None and body.current is None:
            raise HTTPException(400, "give `key` (+ `phase`) and/or `current`")
        if body.key is not None:
            _animate_call(animate_stage.set_key, unit, action, direction, frame, key=body.key,
                          phase=body.phase, work_dir=work_dir)
        if body.current is not None:
            _animate_call(animate_stage.select, unit, action, direction, frame, body.current,
                          work_dir=work_dir)
        return _animate_state(unit, action, direction)

    @app.delete("/animate/{unit}/{action}/{direction}/frames/{frame}")
    def delete_animate_frame(unit: str, action: str, direction: str,
                             frame: str) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        _animate_call(animate_stage.remove_frame, unit, action, direction, frame,
                      work_dir=work_dir)
        return _animate_state(unit, action, direction)

    @app.post("/animate/{unit}/{action}/{direction}/frames/{frame}/{verdict}")
    def judge_animate_frame(unit: str, action: str, direction: str, frame: str,
                            verdict: str, body: VerdictRequest | None = None) -> dict[str, Any]:
        """Approve or reject the frame's current attempt (or `index`). The
        approval that completes the row writes the finished frames."""
        action, direction = _animate_cell(unit, action, direction)
        state = {"approve": "approved", "reject": "rejected"}.get(verdict)
        if state is None:
            raise HTTPException(404, f"a frame is approved or rejected, not {verdict!r}")
        _animate_call(animate_stage.judge, unit, action, direction, frame, state,
                      index=(body.index if body else None), work_dir=work_dir,
                      palette=settings.palette_path(work_dir))
        return _animate_state(unit, action, direction)

    @app.post("/animate/{unit}/{action}/{direction}/approve-reviewed")
    def approve_reviewed(unit: str, action: str, direction: str) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        ids = _animate_call(animate_stage.approve_reviewed, unit, action, direction,
                            work_dir=work_dir, palette=settings.palette_path(work_dir))
        return {**_animate_state(unit, action, direction), "approved": ids}

    @app.post("/animate/{unit}/{action}/{direction}/mirror")
    def mirror_animate(unit: str, action: str, direction: str) -> dict[str, Any]:
        """The partner direction's frames, flipped, as new attempts to review.
        No GPU. Always a press; the handedness warning rides back."""
        action, direction = _animate_cell(unit, action, direction)
        result = _animate_call(animate_stage.mirror, unit, action, direction,
                               work_dir=work_dir)
        return {**_animate_state(unit, action, direction), "warning": result["warning"],
                "mirrored": result["frames"]}

    @app.post("/animate/{unit}/{action}/{direction}/output")
    def output_animate(unit: str, action: str, direction: str,
                       body: OutputRequest | None = None) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        result = _animate_call(animate_stage.output, unit, action, direction,
                               replace_job=bool(body and body.replace_job), work_dir=work_dir,
                               palette=settings.palette_path(work_dir))
        return {**_animate_state(unit, action, direction), "written": result}

    @app.put("/animate/{unit}/{action}/{direction}/attempts/{frame}/{index:int}")
    def hide_animate_attempt(unit: str, action: str, direction: str, frame: str, index: int,
                             body: HiddenRequest) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        _animate_call(animate_stage.hide, unit, action, direction, f"{frame}:{index:02d}",
                      hidden=body.hidden, work_dir=work_dir)
        return _animate_state(unit, action, direction)

    @app.delete("/animate/{unit}/{action}/{direction}/attempts/{frame}/{index:int}")
    def delete_animate_file(unit: str, action: str, direction: str, frame: str,
                            index: int) -> dict[str, Any]:
        action, direction = _animate_cell(unit, action, direction)
        _animate_call(animate_stage.delete_file, unit, action, direction,
                      f"{frame}:{index:02d}", work_dir=work_dir)
        return _animate_state(unit, action, direction)

    @app.put("/animate/{unit}/{action}/{direction}/attempts/{frame}/{index:int}/image")
    async def replace_animate_image(unit: str, action: str, direction: str, frame: str,
                                    index: int, request: Request) -> dict[str, Any]:
        """The editor's Save over one attempt (the retouch feature's
        `replace`, on a frame's store). A judged attempt goes back to review."""
        action, direction = _animate_cell(unit, action, direction)
        data = await request.body()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(415, "the body is not a PNG")
        row = _animate_call(animate_stage.replace, unit, action, direction,
                            f"{frame}:{index:02d}", data, work_dir=work_dir)
        return {**_animate_state(unit, action, direction), "edited": row}

    @app.post("/animate/{unit}/{action}/{direction}/attempts/{frame}/{index:int}/inpaint",
              status_code=202)
    async def inpaint_animate(unit: str, action: str, direction: str, frame: str, index: int,
                              request: Request, response: Response, prompt: str | None = None,
                              strength: float | None = None, seed: int | None = None) -> Any:
        """The retouch feature's masked redraw on a frame's attempt: the canvas
        is stashed under the job and `anim animate --inpaint` redraws it
        through the run manager."""
        action, direction = _animate_cell(unit, action, direction)
        data = await request.body()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(415, "the body is not a PNG")
        ref = f"{frame}:{index:02d}"
        mask_path = _animate_call(animate_stage.stash_canvas, unit, action, direction, ref,
                                  data, work_dir=work_dir)
        args: dict[str, Any] = {"slug": unit, "action": action, "direction": direction,
                                "inpaint": ref, "mask": str(mask_path)}
        if prompt and prompt.strip():
            args["prompt"] = prompt.strip()
        if strength is not None:
            args["strength"] = strength
        if seed is not None:
            args["seed"] = seed
        return _animate_run(args, response)

    @app.get("/animate/{unit}/{action}/{direction}/skeleton/{frame:int}.png")
    def animate_skeleton(unit: str, action: str, direction: str, frame: int) -> FileResponse:
        """One scrub frame of the rig clip at this direction's yaw, drawn by the
        pose renderer on the CPU; the whole clip is rendered on the first ask
        and cached under the animate job."""
        action, direction = _animate_cell(unit, action, direction)
        path = _animate_call(animate_stage.skeleton_file, unit, action, direction, frame,
                             work_dir=work_dir)
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/animate/{unit}/{action}/{direction}/game/{frame}.png")
    def animate_game(unit: str, action: str, direction: str, frame: str) -> FileResponse:
        """The frame's current attempt at game size: the pipeline's own
        matte, downscale and palette snap, cached by the source's hash."""
        action, direction = _animate_cell(unit, action, direction)
        path = _animate_call(animate_stage.game_file, unit, action, direction, frame,
                             work_dir=work_dir, palette=settings.palette_path(work_dir))
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

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
