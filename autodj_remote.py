"""Live music generation through OpenRouter (Google Lyria), with the pre-rendered bank as the fallback.

HybridAudio is a drop-in for BankAudio/StableAudio in the live loop. It asks OpenRouter for one track per
slot. If the key is missing, the daily cap is reached, credit runs out, or a call fails or is blocked,
that slot (and, after repeated failures, a cooldown period) is served from the bank instead.
The hard spending limit is the one set on the OpenRouter key; AUTODJ_REMOTE_MAX_PER_DAY is a second guard.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal

from autodj_bank import BACKEND as BANK_BACKEND, BankAudio
from autodj_stable_audio import StableAudioError

REMOTE_BACKEND = "Google Lyria via OpenRouter (live generation)"
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/lyria-3-pro-preview"
CAMELOT_KEYS = {
    "1A": "Ab minor", "2A": "Eb minor", "3A": "Bb minor", "4A": "F minor", "5A": "C minor", "6A": "G minor",
    "7A": "D minor", "8A": "A minor", "9A": "E minor", "10A": "B minor", "11A": "F# minor", "12A": "C# minor",
    "1B": "B major", "2B": "F# major", "3B": "Db major", "4B": "Ab major", "5B": "Eb major", "6B": "Bb major",
    "7B": "F major", "8B": "C major", "9B": "G major", "10B": "D major", "11B": "A major", "12B": "E major",
}
FAILURES_BEFORE_COOLDOWN = 3
COOLDOWN_SECONDS = 600
CREDIT_COOLDOWN_SECONDS = 3600   # 401/402/403: the key or credit is the problem, so stop asking for a while
MIN_SECONDS = 20


def lyria_prompt(plan: dict) -> str:
    key = CAMELOT_KEYS.get(plan["camelot"], "")
    return (f"{plan['style'].strip().rstrip('.')}. Instrumental only, no vocals. {int(plan['bpm'])} BPM"
            f"{', ' + key if key else ''}. A clear steady beat from the very first bar. About 70 seconds long.")


class HybridAudio:
    def __init__(self, sr: int, bank: BankAudio | None = None, key: str | None = None, model: str | None = None,
                 daily_cap: int | None = None, clock=time.monotonic):
        self.sr = sr
        self.bank = bank or BankAudio(sr)
        self.key = (key if key is not None else os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.model = model or os.environ.get("AUTODJ_REMOTE_MODEL", DEFAULT_MODEL)
        self.daily_cap = daily_cap if daily_cap is not None else int(os.environ.get("AUTODJ_REMOTE_MAX_PER_DAY", "60"))
        self.clock = clock
        self.calls: deque[float] = deque()
        self.failures = 0
        self.blocked_until = 0.0
        self.last_backend = BANK_BACKEND
        self._lock = threading.Lock()

    def remote_available(self) -> bool:
        with self._lock:
            now = self.clock()
            while self.calls and now - self.calls[0] >= 86400:
                self.calls.popleft()
            return bool(self.key) and now >= self.blocked_until and len(self.calls) < self.daily_cap

    @property
    def takes(self) -> int:
        """Candidate takes per slot: one paid take from the generator, three free ones from the bank."""
        return 1 if self.remote_available() else 3

    def catalog_text(self) -> str:
        """Free-form planning while the generator is live; the bank's styles once it is not."""
        return "" if self.remote_available() else self.bank.catalog_text()

    def check(self) -> None:
        self.bank.check()

    def render(self, plan: dict, seed: int, stopped) -> np.ndarray:
        for attempt in range(2):
            if not self.remote_available():
                break
            try:
                wave = self._generate(plan, stopped)
                with self._lock:
                    self.failures = 0
                self.last_backend = REMOTE_BACKEND
                return wave
            except StableAudioError as exc:
                if stopped():
                    raise
                if attempt == 0 and getattr(exc, "retryable", False):
                    continue  # the generator sometimes returns nothing for a prompt it accepts on the next try
                self._record_failure(exc)
                break
        return self.rescue(plan, seed, stopped)

    def rescue(self, plan: dict, seed: int, stopped) -> np.ndarray:
        self.last_backend = BANK_BACKEND
        return self.bank.render(plan, seed, stopped)

    def _record_failure(self, exc: Exception) -> None:
        with self._lock:
            self.failures += 1
            credit = getattr(exc, "credit_problem", False)
            if credit or self.failures >= FAILURES_BEFORE_COOLDOWN:
                self.blocked_until = self.clock() + (CREDIT_COOLDOWN_SECONDS if credit else COOLDOWN_SECONDS)
                self.failures = 0

    def _generate(self, plan: dict, stopped) -> np.ndarray:
        with self._lock:
            self.calls.append(self.clock())  # counted when sent: a call that fails may still be billed
        body = {"model": self.model, "messages": [{"role": "user", "content": lyria_prompt(plan)}],
                "modalities": ["text", "audio"], "audio": {"format": "mp3"}, "stream": True}
        request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        chunks: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                for raw in response:
                    if stopped():
                        raise StableAudioError("Track generation stopped")
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    if "error" in event:
                        raise StableAudioError("Generator error: " + str(event["error"])[:200])
                    for choice in event.get("choices", []):
                        audio = (choice.get("delta") or {}).get("audio")
                        if isinstance(audio, dict) and audio.get("data"):
                            chunks.append(audio["data"])
        except urllib.error.HTTPError as exc:
            error = StableAudioError(f"Generator HTTP {exc.code}")
            error.credit_problem = exc.code in (401, 402, 403)
            raise error from exc
        except (OSError, ValueError) as exc:  # timeouts, resets, bad base64 later
            raise StableAudioError(f"Generator request failed: {type(exc).__name__}") from exc
        if not chunks:
            error = StableAudioError("Generator returned no audio (blocked or empty)")
            error.retryable = True
            raise error
        try:
            wave, sr = sf.read(io.BytesIO(base64.b64decode("".join(chunks))), dtype="float32", always_2d=True)
        except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
            raise StableAudioError("Generator audio could not be decoded") from exc
        if wave.shape[1] == 1:
            wave = np.repeat(wave, 2, axis=1)
        if wave.shape[1] != 2 or sr <= 0 or len(wave) < MIN_SECONDS * sr or float(np.sqrt(np.mean(wave ** 2))) < 0.005:
            raise StableAudioError("Generator audio was too short or silent")
        peak = float(np.abs(wave).max())
        if peak > 0.98:  # decoded mp3 can overshoot full scale; keep it clean for the mixer
            wave = (wave * (0.98 / peak)).astype("float32")
        if sr != self.sr:
            common = gcd(self.sr, sr)
            wave = signal.resample_poly(wave, self.sr // common, sr // common, axis=0).astype("float32")
        return wave
