"""The FastAPI app - every route of the pipeline API feature.

    uvicorn --factory anim.api.app:create_app --host 0.0.0.0 --port 8720

Routes (all JSON unless said otherwise):

    GET  /jobs                             every job under work/: name, stages done, direction, action, updated
    GET  /jobs/{job}                       job.json + stage status + frame-set sizes
    GET  /jobs/{job}/frames                every PNG set (frames, pose, depth, clip, strip, render, post, candidates) with URLs
    GET  /jobs/{job}/frames/{stage}/{n}.png  the PNG itself
    PUT  /jobs/{job}/frames/{stage}/{n}.png  overwrite it (image/png body, same size, alpha rounded)
    GET  /jobs/{job}/picks                 the review state (ReviewApp.state)
    PUT  /jobs/{job}/picks  {"picks": []}  rewrite picks.json exactly as `anim review` did
    POST /jobs/{job}/regenerate            flag rerender_requested in job.json (never the GPU)
    POST /jobs  {"stage", "args" | "argv"} run a stage in the background -> 202 + the run
    GET  /runs                             every run this API knows, newest first
    GET  /runs/{run}                       status + log
    GET  /runs/{run}/events                text/event-stream: log, progress, done
    GET  /settings   PUT /settings         work/project.json
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

from anim import __version__, comfy, config
from anim.api import jobs as jobs_api
from anim.api import pages, settings
from anim.api.runs import RunManager, RunRefused
from anim.job import Job
from anim.stages.review import ReviewApp


class RunRequest(BaseModel):
    """`POST /jobs`. `args` are the stage's `--help` names; `argv` is the raw
    CLI tail for a caller that already speaks it. One or the other."""

    stage: str
    args: dict[str, Any] | None = None
    argv: list[str] | None = None


class PicksRequest(BaseModel):
    picks: list[int] = Field(default_factory=list)


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

    @app.post("/jobs", status_code=202)
    def start_run(body: RunRequest, response: Response) -> Any:
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
