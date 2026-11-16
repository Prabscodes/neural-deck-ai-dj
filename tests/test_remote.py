"""OpenRouter/Lyria generation with the bank as fallback. No network: urlopen is replaced by a fake."""
import base64
import io
import json
import threading
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import autodj_remote
from autodj_bank import BACKEND as BANK_BACKEND, BankAudio
from autodj_live_agent import LiveDJ
from autodj_remote import REMOTE_BACKEND, HybridAudio, lyria_prompt
from test_bank import make_bank, never, plan
from test_live_dj import BASE, wait_for

REAL_BANK = Path(__file__).resolve().parent.parent / "bank"
SR = 8000


def flac_b64(seconds=25, sr=SR):
    wave = (np.sin(2 * np.pi * 220 * np.arange(int(sr * seconds)) / sr) * .3).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([wave, wave], axis=1), sr, format="FLAC")
    return base64.b64encode(buffer.getvalue()).decode()


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def sse(*events):
    return FakeResponse(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n")


def audio_events(b64):
    half = len(b64) // 2
    return [{"choices": [{"delta": {"audio": {"data": b64[:half]}}}]},
            {"choices": [{"delta": {"audio": {"data": b64[half:]}}}]}]


@pytest.fixture
def hybrid(tmp_path):
    return HybridAudio(SR, BankAudio(SR, str(make_bank(tmp_path))), key="test-key", daily_cap=3)


def fake(monkeypatch, handler):
    seen = []
    def urlopen(request, timeout=0):
        seen.append(json.loads(request.data))
        return handler()
    monkeypatch.setattr(autodj_remote.urllib.request, "urlopen", urlopen)
    return seen


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b""))


def test_prompt_carries_style_tempo_and_key_name():
    text = lyria_prompt({"style": "Deep house with Rhodes.", "bpm": 124, "camelot": "8A"})
    assert "Deep house with Rhodes." in text and "124 BPM" in text and "A minor" in text and "no vocals" in text


def test_generated_track_is_used_and_streamed_chunks_are_joined(hybrid, monkeypatch):
    seen = fake(monkeypatch, lambda: sse(*audio_events(flac_b64())))
    wave = hybrid.render(plan(), 0, never)
    assert wave.dtype == np.float32 and wave.shape[1] == 2 and len(wave) == 25 * SR
    assert hybrid.last_backend == REMOTE_BACKEND and hybrid.takes == 1 and hybrid.catalog_text() == ""
    assert seen[0]["stream"] is True and seen[0]["modalities"] == ["text", "audio"]


def test_empty_reply_is_retried_once_then_served_from_the_bank(hybrid, monkeypatch):
    hybrid.daily_cap = 99
    seen = fake(monkeypatch, lambda: sse({"choices": [{"delta": {"content": "no audio"}}]}))
    wave = hybrid.render(plan(), 0, never)
    assert len(seen) == 2  # one retry, not more
    assert wave.shape == (SR, 2) and hybrid.last_backend == BANK_BACKEND  # the bank test tracks are 1 s long
    assert hybrid.remote_available()  # one failed slot is not a cooldown


def test_empty_reply_then_audio_uses_the_generator(hybrid, monkeypatch):
    replies = iter([sse({"choices": [{"delta": {"content": "no audio"}}]}), sse(*audio_events(flac_b64()))])
    seen = fake(monkeypatch, lambda: next(replies))
    hybrid.render(plan(), 0, never)
    assert len(seen) == 2 and hybrid.last_backend == REMOTE_BACKEND


def test_three_failures_start_a_cooldown_and_planning_goes_to_the_bank_catalog(hybrid, monkeypatch):
    hybrid.daily_cap = 99
    fake(monkeypatch, lambda: (_ for _ in ()).throw(http_error(500)))
    for _ in range(3):
        hybrid.render(plan(), 0, never)
    assert not hybrid.remote_available() and hybrid.takes == 3 and "BPM" in hybrid.catalog_text()


def test_credit_or_key_problem_stops_remote_at_once(hybrid, monkeypatch):
    seen = fake(monkeypatch, lambda: (_ for _ in ()).throw(http_error(402)))
    hybrid.render(plan(), 0, never)
    assert not hybrid.remote_available()
    hybrid.render(plan(), 0, never)
    assert len(seen) == 1  # no further paid attempts during the cooldown


def test_daily_cap_and_missing_key_use_the_bank_without_calling(hybrid, monkeypatch):
    seen = fake(monkeypatch, lambda: sse(*audio_events(flac_b64())))
    for _ in range(3):
        hybrid.render(plan(), 0, never)
    hybrid.render(plan(), 0, never)
    assert len(seen) == 3 and hybrid.last_backend == BANK_BACKEND
    hybrid.key, hybrid.daily_cap = "", 99
    hybrid.calls.clear()
    hybrid.render(plan(), 0, never)
    assert len(seen) == 3


def test_short_or_silent_audio_is_rejected(hybrid, monkeypatch):
    fake(monkeypatch, lambda: sse(*audio_events(flac_b64(seconds=5))))
    hybrid.render(plan(), 0, never)
    assert hybrid.last_backend == BANK_BACKEND and hybrid.remote_available()


@pytest.mark.skipif(not (REAL_BANK / "bank.json").is_file(), reason="bank not built")
@pytest.mark.parametrize("remote_works", [True, False])
def test_live_loop_uses_generator_then_falls_back(tmp_path, monkeypatch, remote_works):
    monkeypatch.setenv("AUTODJ_BANK_DIR", str(REAL_BANK))
    monkeypatch.setenv("AUTODJ_LISTENING", "off")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    track = base64.b64encode((REAL_BANK / BankAudio(44100).entries[0]["file"]).read_bytes()).decode()
    fake(monkeypatch, (lambda: sse(*audio_events(track))) if remote_works else (lambda: (_ for _ in ()).throw(http_error(500))))
    plan_ = {**BASE, "bars": 16, "style": "Heavy melodic dubstep with a grand piano hook", "bpm": 140,
             "camelot": "8A", "overlap_seconds": 2, "transition": "equal_power"}
    dj = LiveDJ(str(tmp_path), None, lambda *e: None, threading.Lock(), planner=lambda *a: plan_, sample_rate=44100)
    dj.MAX_SESSIONS, dj.music_backend = 2, "hybrid"
    sid = dj.start("dubstep with piano", "test", True)["session_id"]
    segments = wait_for(lambda: (x if len(x := dj.status(sid)["segments"]) >= 1 else None), timeout=90)
    if remote_works:
        assert segments[0]["backend"] == REMOTE_BACKEND and segments[0]["generation_attempts"] == 1
    else:
        assert segments[0]["backend"] == BANK_BACKEND
    dj.stop(sid)
    wait_for(lambda: dj.status(sid)["state"] == "stopped", timeout=15)
