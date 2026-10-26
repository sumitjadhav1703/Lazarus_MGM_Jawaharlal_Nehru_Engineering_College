"""Local research API with serialized accelerated historical playback."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class ReplayRequest(BaseModel):
    action: Literal["start", "pause", "reset", "step"]
    steps: int = Field(default=1, ge=1, le=120)


class ReviewRequest(BaseModel):
    status: Literal["pending", "reviewed", "dismissed"]
    note: str = Field(default="", max_length=2000)


def create_app(service, auto_tick=True, playback_seconds=1.0):
    if playback_seconds <= 0:
        raise ValueError("playback_seconds must be positive")

    async def replay_loop():
        while True:
            await asyncio.sleep(playback_seconds)
            try:
                await asyncio.to_thread(service.tick)
            except Exception:
                logging.exception("Replay halted on processing error")
                for patient in service.patients():
                    if patient["replay_status"] == "running":
                        service.control(patient["patient_id"], "pause")

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(replay_loop()) if auto_tick else None
        yield
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="CardioTwin research replay", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def call(fn, *args):
        try:
            return fn(*args)
        except KeyError as error:
            raise HTTPException(404, "Patient or warning not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "historical_simulation", "research_only": True}

    @app.get("/api/patients")
    def patients():
        return service.patients()

    @app.get("/api/patients/{pid}/state")
    def state(pid: str):
        return call(service.state, pid)

    @app.get("/api/patients/{pid}/forecast")
    def forecast(pid: str):
        return call(service.state, pid)["forecast"]

    @app.post("/api/patients/{pid}/replay")
    def replay(pid: str, request: ReplayRequest):
        return (
            call(service.step, pid, request.steps)
            if request.action == "step"
            else call(service.control, pid, request.action)
        )

    @app.get("/api/patients/{pid}/warnings")
    def warnings(pid: str, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return call(service.warnings, pid, limit, offset)

    @app.post("/api/patients/{pid}/warnings/{warning_id}/review")
    def review(pid: str, warning_id: str, request: ReviewRequest):
        return call(service.review, pid, warning_id, request.status, request.note)

    @app.get("/api/metrics")
    def metrics():
        return service.metrics()

    dashboard = Path(__file__).parents[2] / "dashboard/dist"
    if dashboard.is_dir():
        if (dashboard / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dashboard / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dashboard / "index.html")

    return app
