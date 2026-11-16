"""Slim public server, multi-session limits, and the library backend end to end."""
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import autodj_live_agent as live_agent
from autodj_agent import AgentError
from autodj_live_agent import LiveDJ
from autodj_public import StartLimiter, create_public_app
from test_bank import make_bank
from test_live_dj import BASE, Producer, wait_for

REAL_BANK = Path(__file__).resolve().parent.parent / "bank"


def test_start_limiter_window():
    now = [0.0]
    limiter = StartLimiter(2, clock=lambda: now[0])
    assert limiter.allow("a") == 0 and limiter.allow("a") == 0
    wait = limiter.allow("a")
    assert 3500 < wait <= 3600
    assert limiter.allow("b") == 0  # other clients are independent
    now[0] = 3601
    assert limiter.allow("a") == 0  # the window slides


def test_concurrent_session_cap_and_gate(tmp_path):
    gate = threading.Lock()
    dj = LiveDJ(str(tmp_path), Producer(), lambda *e: None, gate, planner=lambda *a: {**BASE}, sample_rate=8000)
    dj.MAX_SESSIONS = 2
    a = dj.start("one", "test", False)["session_id"]
    b = dj.start("two", "test", False)["session_id"]
    assert a != b and not gate.locked()  # several sessions do not take the exclusion gate
    with pytest.raises(AgentError, match="busy"):
        dj.start("three", "test", False)
    dj.stop(a)
    wait_for(lambda: dj.status(a)["state"] == "stopped")
    c = dj.start("four", "test", False)["session_id"]  # a freed slot is reusable
    assert dj.status(b)["state"] != "stopped"  # the other listener was not disturbed
    for sid in (b, c):
        dj.stop(sid)
        wait_for(lambda s=sid: dj.status(s)["state"] == "stopped")


def test_single_session_mode_still_uses_gate(tmp_path):
    gate = threading.Lock()
    dj = LiveDJ(str(tmp_path), Producer(), lambda *e: None, gate, planner=lambda *a: {**BASE}, sample_rate=8000)
    dj.MAX_SESSIONS = 1
    sid = dj.start("one", "test", False)["session_id"]
    assert gate.locked()
    with pytest.raises(AgentError, match="already active"):
        dj.start("two", "test", False)
    dj.stop(sid)
    wait_for(lambda: not gate.locked())


@pytest.fixture
def public(tmp_path, monkeypatch):
    bank = tmp_path / "bank"; bank.mkdir()
    make_bank(bank)
    monkeypatch.setenv("AUTODJ_BANK_DIR", str(bank))
    monkeypatch.setenv("AUTODJ_GEMINI_TRANSPORT", "api")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    app = create_public_app(str(tmp_path / "runs"), starts_per_hour=1)
    app.state.live_dj.planner = lambda *a: {**BASE}
    client = TestClient(app)
    yield client
    for sid in list(app.state.live_dj.active):
        app.state.live_dj.stop(sid)
    wait_for(lambda: not app.state.live_dj.active, timeout=10)


def test_public_app_serves_only_listener_surface(public):
    assert "Start listening" in public.get("/").text
    assert public.get("/healthz").json()["ok"] is True
    for path in ("/api/state", "/api/export/wav", "/docs", "/openapi.json", "/studio"):
        assert public.get(path).status_code == 404, path
    assert public.post("/api/library/scan", json={"path": "/etc"}).status_code in (404, 405)
    assert public.post("/api/dj-live/start", json={"request": "x" * 1001}).status_code == 422


def test_public_start_is_rate_limited_per_client(public):
    first = public.post("/api/dj-live/start", json={"request": "soft piano"})
    assert first.status_code == 200
    second = public.post("/api/dj-live/start", json={"request": "soft piano"})
    assert second.status_code == 429 and int(second.headers["retry-after"]) > 0


def test_public_start_needs_planner_key(public, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = public.post("/api/dj-live/start", json={"request": "soft piano"})
    assert response.status_code == 503 and "GEMINI_API_KEY" in response.json()["detail"]


@pytest.mark.skipif(not (REAL_BANK / "bank.json").is_file(), reason="bank not built (tools/build_bank.py)")
def test_real_bank_session_plays_and_hands_off(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODJ_BANK_DIR", str(REAL_BANK))
    monkeypatch.setenv("AUTODJ_LISTENING", "off")
    records, plans, handoffs = [], [], []
    real_handoff = live_agent.prepare_handoff
    monkeypatch.setattr(live_agent, "prepare_handoff", lambda *a: handoffs.append(1) or real_handoff(*a))
    plan = {**BASE, "bars": 16, "style": "Heavy melodic dubstep with a grand piano hook", "bpm": 140,
            "camelot": "8A", "overlap_seconds": 2, "transition": "equal_power"}
    dj = LiveDJ(str(tmp_path), None, lambda *e: records.append(e), threading.Lock(),
                planner=lambda *a: plans.append(1) or plan, sample_rate=44100)
    dj.MAX_SESSIONS = 2
    dj.music_backend = "bank"
    sid = dj.start("dubstep with piano", "test", True)["session_id"]
    segments = wait_for(lambda: (x if len(x := dj.status(sid)["segments"]) >= 2 else None), timeout=60)
    assert [s["seq"] for s in segments] == [0, 1]
    assert segments[0]["backend"].startswith("Pre-rendered") and segments[1]["transition"] == "equal_power"
    assert segments[1]["duration_seconds"] > 30 and segments[1]["listening"]["state"] == "listening off"
    assert dj.segment_path(sid, 0).is_file()
    dj.stop(sid)
    wait_for(lambda: dj.status(sid)["state"] == "stopped", timeout=15)
    # Listening is off, so only the winning take needs a handoff: at most one per track after the first.
    assert 1 <= len(handoffs) <= len(plans) - 1
