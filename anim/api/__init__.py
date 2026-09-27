"""The pipeline API feature - FastAPI over `work/`, on :8720.

A thin HTTP layer over the same stage functions the `anim` CLI calls, so the
studio (the web pixel editor feature and its Concept / Views / Animate tabs)
drive the same `work/<job>/` files the CLI writes. There is no
database: a job IS its directory, a run IS a subprocess of the CLI plus a log
file, and the project settings are one JSON file.

    anim/api/app.py       the FastAPI app factory and every route
    anim/api/jobs.py      reading work/<job>/ - listing, stage status, frames
    anim/api/runs.py      running a stage as a background job, the GPU rule, SSE
    anim/api/settings.py  work/project.json
    anim/api/pages.py     the minimal jobs + review pages served until web/dist exists

Runs as the `anim-api` compose service:

    docker compose up -d anim-api
    # then, from the Mac:  http://192.168.1.158:8720/
"""
