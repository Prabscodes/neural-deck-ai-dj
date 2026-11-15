"""Pre-rendered track bank: a drop-in replacement for StableAudio.render on hosts without a GPU.

The bank is a folder holding bank.json plus audio files rendered offline with Stable Audio
(see tools/build_bank.py). For a plan it returns the best-matching stored track, so the live
loop's existing cue detection, handoffs and listening run unchanged.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from autodj_stable_audio import StableAudioError

BACKEND = "Pre-rendered Stable Audio 3 Small-Music bank"
TOP_CHOICES = 3          # takes per slot differ by picking among the best few matches
RECENT_RENDERS = 6       # recently returned tracks are penalized so sessions do not loop
RECENT_PENALTY = 6.0
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FILE = re.compile(r"[A-Za-z0-9_-]{1,64}\.(?:ogg|flac|wav)")
_CAMELOT = re.compile(r"(1[0-2]|[1-9])([AB])")


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _key_distance(a: str, b: str) -> int | None:
    """Camelot wheel distance: 0 same, 1 neighbor or relative major/minor, else None (clash)."""
    ma, mb = _CAMELOT.fullmatch(a), _CAMELOT.fullmatch(b)
    if not ma or not mb:
        return None
    na, nb = int(ma[1]), int(mb[1])
    step = min((na - nb) % 12, (nb - na) % 12)
    if ma[2] == mb[2]:
        return step if step <= 1 else None
    return 1 if step == 0 else None


class BankAudio:
    def __init__(self, sr: int, directory: str | None = None):
        self.sr = sr
        self.root = Path(directory or os.environ.get("AUTODJ_BANK_DIR", "") or "bank").resolve()
        self.entries = self._load()
        self._recent: deque[str] = deque(maxlen=RECENT_RENDERS)
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        try:
            data = json.loads((self.root / "bank.json").read_text(encoding="utf-8"))
            raw = data["tracks"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StableAudioError("The track bank is missing or unreadable. Set AUTODJ_BANK_DIR.") from exc
        entries, seen = [], set()
        for item in raw:
            try:
                ok = (_ID.fullmatch(item["id"]) and item["id"] not in seen and _FILE.fullmatch(item["file"])
                      and _CAMELOT.fullmatch(item["camelot"]) and 40 <= float(item["bpm"]) <= 220
                      and isinstance(item["tags"], list) and (self.root / item["file"]).is_file())
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:
                continue  # Skip malformed entries; file names are plain names, never paths.
            seen.add(item["id"])
            entries.append({"id": item["id"], "file": item["file"], "camelot": item["camelot"],
                            "bpm": float(item["bpm"]), "tags": [str(t).lower() for t in item["tags"]]})
        return entries

    def check(self) -> None:
        if not self.entries:
            raise StableAudioError("The track bank has no playable tracks.")

    def _score(self, entry: dict, style: str, bpm: float, camelot: str) -> float:
        score = 3.0 * sum(1 for tag in entry["tags"] if _words(tag) in style)
        score -= min(abs(entry["bpm"] - bpm), 40) / 5
        distance = _key_distance(entry["camelot"], camelot)
        score += {0: 2.0, 1: 1.0}.get(distance, 0.0)
        return score

    def rank(self, plan: dict) -> list[dict]:
        style, bpm, camelot = _words(plan["style"]), float(plan["bpm"]), plan["camelot"]
        with self._lock:
            recent = set(self._recent)
        scored = [(self._score(e, style, bpm, camelot) - (RECENT_PENALTY if e["id"] in recent else 0), e["id"], e)
                  for e in self.entries]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored]

    def render(self, plan: dict, seed: int, stopped) -> np.ndarray:
        if stopped():
            raise StableAudioError("Track selection stopped")
        self.check()
        ranked = self.rank(plan)
        entry = ranked[seed % min(TOP_CHOICES, len(ranked))]
        with self._lock:
            self._recent.append(entry["id"])
        wave, sr = sf.read(str(self.root / entry["file"]), dtype="float32", always_2d=True)
        if wave.shape[1] == 1:
            wave = np.repeat(wave, 2, axis=1)
        if wave.shape[1] != 2 or sr <= 0:
            raise StableAudioError("Bank track is not stereo audio")
        if sr != self.sr:
            common = gcd(self.sr, sr)
            wave = signal.resample_poly(wave, self.sr // common, sr // common, axis=0).astype("float32")
        return wave
