"""Slim public server: the prompt-first DJ on the pre-rendered library, with no GPU and no torch.

Only the listener page and /api/dj-live/* are served. The planner key comes from this process's
environment. Limits (all overridable): AUTODJ_MAX_SESSIONS=3 concurrent listeners,
AUTODJ_MAX_STARTS_PER_HOUR=6 session starts per client, AUTODJ_MAX_STEERS and AUTODJ_MAX_TRACKS
per session. Set AUTODJ_TRUST_PROXY=1 only behind a proxy that sets X-Forwarded-For.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from autodj_agent import AgentError, AgentRunner  # noqa: E402
from autodj_bank import BankAudio  # noqa: E402
from autodj_gemini import GeminiError, configured_model, require_planner  # noqa: E402
from autodj_live_agent import LiveDJ  # noqa: E402
from autodj_stable_audio import StableAudioError  # noqa: E402

HERE = Path(__file__).resolve().parent


class Request1000(BaseModel):
    request: str = Field(max_length=1000)


class Ack(BaseModel):
    seq: int


class StartLimiter:
    """Per-client sliding window on session starts. Thread-safe, in memory."""

    def __init__(self, per_hour: int, clock=time.monotonic):
        self.per_hour, self.clock = per_hour, clock
        self.starts: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, client: str) -> float:
        """Returns 0 if the start is allowed (and records it), else seconds until one is."""
        now = self.clock()
        with self.lock:
            window = self.starts[client]
            while window and now - window[0] >= 3600:
                window.popleft()
            if len(window) >= self.per_hour:
                return 3600 - (now - window[0])
            window.append(now)
            if len(self.starts) > 10000:  # bound memory under address churn
                for key in [k for k, v in self.starts.items() if not v or now - v[-1] >= 3600]:
                    del self.starts[key]
            return 0.0


def create_public_app(agent_dir: str | None = None, *, starts_per_hour: int | None = None) -> FastAPI:
    app = FastAPI(title="Neural Deck", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/deck-assets", StaticFiles(directory=HERE / "autodj_web"), name="deck-assets")
    runner = AgentRunner(agent_dir or os.environ.get("AUTODJ_AGENT_DIR", str(HERE / "dj_agent_runs")), None)
    live_dj = LiveDJ(str(runner.root / "live"), None, runner.log, threading.Lock())
    live_dj.music_backend = "bank"  # always the library: there is no GPU or renderer here
    live_dj.MAX_SESSIONS = int(os.environ.get("AUTODJ_MAX_SESSIONS", "3"))
    app.state.live_dj = live_dj
    limiter = StartLimiter(starts_per_hour or int(os.environ.get("AUTODJ_MAX_STARTS_PER_HOUR", "6")))
    trust_proxy = os.environ.get("AUTODJ_TRUST_PROXY") == "1"

    def client_id(request: Request) -> str:
        if trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            if forwarded:
                return forwarded
        return request.client.host if request.client else "unknown"

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return (HERE / "autodj_frontend.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "tracks": len(BankAudio(live_dj.sr).entries) if (Path(os.environ.get("AUTODJ_BANK_DIR", "bank")) / "bank.json").is_file() else 0}

    @app.post("/api/dj-live/start")
    def start(req: Request1000, request: Request) -> dict[str, Any]:
        try:
            BankAudio(live_dj.sr).check()
            require_planner()
        except StableAudioError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GeminiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        wait = limiter.allow(client_id(request))
        if wait:
            raise HTTPException(status_code=429, detail="Too many sessions started from your connection. Try again later.",
                                headers={"Retry-After": str(int(wait) + 1)})
        try:
            return live_dj.start(req.request, configured_model(), True)
        except AgentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/dj-live/{sid}/status")
    def status(sid: str) -> dict[str, Any]:
        try: return live_dj.status(sid)
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/steer")
    def steer(sid: str, req: Request1000) -> dict[str, str]:
        try: live_dj.steer(sid, req.request); return {"status": "accepted"}
        except AgentError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/stop")
    def stop(sid: str) -> dict[str, str]:
        try: live_dj.stop(sid); return {"status": "stopping"}
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/ack")
    def ack(sid: str, req: Ack) -> dict[str, str]:
        try: live_dj.ack(sid, req.seq); return {"status": "ok"}
        except AgentError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dj-live/{sid}/audio/{seq}")
    def audio(sid: str, seq: int) -> FileResponse:
        try: return FileResponse(str(live_dj.segment_path(sid, seq)), media_type="audio/wav")
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8300")))
    args = parser.parse_args()
    uvicorn.run(create_public_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
