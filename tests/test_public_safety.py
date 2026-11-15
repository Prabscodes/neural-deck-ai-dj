"""Safety limits for a shared deployment: route lockdown, session caps, API retry."""
import io
import threading
from email.message import Message
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import autodj_gemini
from autodj_agent import AgentError
from autodj_live_agent import LiveDJ
from autodj_server import create_app
from test_live_dj import BASE, Producer, wait_for


LEGACY_POSTS = ["/api/library/scan", "/api/host-drop", "/api/pipewire-play", "/api/produce", "/api/deck/load"]


def test_legacy_routes_hidden_by_default(tmp_path):
    client = TestClient(create_app(cache_dir=str(tmp_path)))
    for path in LEGACY_POSTS:
        assert client.post(path, json={"path": "/etc"}).status_code == 404, path
    assert client.get("/api/state").status_code == 404
    assert client.get("/api/export/wav").status_code == 404
    # Prompt-first routes still resolve to the app's own handler, not the lockdown 404.
    missing = client.get("/api/dj-live/0000/status")
    assert missing.status_code == 404 and missing.json()["detail"] == "Live session not found"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/telemetry"):
            pass


def test_legacy_routes_available_when_enabled(tmp_path):
    client = TestClient(create_app(cache_dir=str(tmp_path), enable_legacy_studio=True))
    assert client.get("/api/state").status_code == 200
    assert client.post("/api/library/scan", json={"path": str(tmp_path / "nope")}).status_code == 404
    assert client.post("/api/library/scan", json={"path": str(tmp_path)}).status_code != 404


def test_steer_limit(tmp_path):
    gate = threading.Lock()
    dj = LiveDJ(str(tmp_path), Producer(), lambda *e: None, gate,
                planner=lambda *a: {**BASE}, sample_rate=8000)
    dj.MAX_STEERS = 2
    sid = dj.start("soft start", "test", False)["session_id"]
    dj.steer(sid, "one")
    dj.steer(sid, "two")
    with pytest.raises(AgentError, match="limit"):
        dj.steer(sid, "three")
    dj.stop(sid)
    wait_for(lambda: dj.status(sid)["state"] == "stopped")


def test_track_limit_stops_session(tmp_path):
    records = []
    gate = threading.Lock()
    dj = LiveDJ(str(tmp_path), Producer(), lambda *e: records.append(e), gate,
                planner=lambda *a: {**BASE}, sample_rate=8000)
    dj.MAX_TRACKS = 2
    sid = dj.start("soft start", "test", False)["session_id"]
    wait_for(lambda: dj.status(sid)["state"] == "stopped")
    assert [e[1:3] for e in records].count(("live_generate", "completed")) == 2
    assert any(e[1] == "live" and e[3].get("reason") == "track limit reached" for e in records)
    assert not gate.locked()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return HTTPError("https://example.invalid", code, "err", headers, io.BytesIO(b"{}"))


def run_post(monkeypatch, outcomes):
    calls, sleeps = [], []

    def fake_urlopen(req, timeout):
        calls.append(1)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr(autodj_gemini, "urlopen", fake_urlopen)
    return calls, sleeps, lambda: autodj_gemini._post_with_retry(object(), 5, sleep=sleeps.append)


def test_api_retries_rate_limit_then_succeeds(monkeypatch):
    calls, sleeps, run = run_post(monkeypatch, [http_error(429, retry_after=3), http_error(503), b'{"ok": 1}'])
    assert run() == {"ok": 1}
    assert len(calls) == 3 and sleeps == [3.0, 2.0]


def test_api_retry_is_capped_and_reraises(monkeypatch):
    calls, sleeps, run = run_post(monkeypatch, [http_error(429, retry_after=999)] * 3)
    with pytest.raises(HTTPError):
        run()
    assert len(calls) == 3 and sleeps == [autodj_gemini.MAX_BACKOFF_SECONDS] * 2


def test_api_does_not_retry_client_errors(monkeypatch):
    calls, sleeps, run = run_post(monkeypatch, [http_error(403)])
    with pytest.raises(HTTPError):
        run()
    assert len(calls) == 1 and sleeps == []
