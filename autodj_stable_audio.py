"""Local Stable Audio 3 generation and preparation for the live DJ."""
from __future__ import annotations

from dataclasses import dataclass, replace
import io
import json
import os
from math import gcd
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import soundfile as sf
from scipy import signal
from websocket import WebSocketTimeoutException, create_connection

from autodj_beatmatch import BeatGrid, estimate_grid

KEYS = {
    "A": ("A-flat", "E-flat", "B-flat", "F", "C", "G", "D", "A", "E", "B", "F-sharp", "C-sharp"),
    "B": ("B", "F-sharp", "D-flat", "A-flat", "E-flat", "B-flat", "F", "C", "G", "D", "A", "E"),
}
BACKEND = "Stable Audio 3 Small-Music (local)"
CHECKPOINT = "stable_audio_3_small_music.safetensors"
ENCODER = "t5gemma_b_b_ul2.safetensors"
SECONDS = 75.0


class StableAudioError(RuntimeError):
    pass


@dataclass
class Deck:
    wave: np.ndarray
    opening: BeatGrid | None
    closing: BeatGrid | None
    trimmed: float
    gain_db: float
    score: float
    cue_seconds: float
    entry_has_kick: bool


def _kick_envelope(wave: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    mono = wave[::4].mean(axis=1)
    fs = sr / 4
    band = signal.sosfilt(signal.butter(3, (38, 155), btype="bandpass", fs=fs, output="sos"), mono)
    hop = round(fs * .025)
    blocks = band[:len(band)//hop*hop].reshape(-1, hop)
    return np.sqrt(np.square(blocks).mean(axis=1)), hop / fs


def _bass_pulse(block: np.ndarray, hop: float, threshold: float) -> bool:
    if len(block) < round(3.8 / hop):
        return False
    peaks, _ = signal.find_peaks(np.log(block + .001), distance=round(.18 / hop), prominence=.55)
    # A constant sub or sustained piano note can be loud. Require multiple distinct low-end attacks.
    return float(np.mean(block)) >= threshold and np.count_nonzero(block[peaks] >= 1.6 * threshold) >= 4


def prepare_deck(wave: np.ndarray, sr: int, bpm: float, *, incoming: bool = False) -> Deck:
    """Find a mix-out cue before a generator's fade and match active music levels."""
    if len(wave) < 50 * sr or not np.isfinite(wave).all():
        raise StableAudioError("Stable Audio returned a short or invalid track")
    half_second = sr // 2
    blocks = wave[:len(wave) // half_second * half_second].reshape(-1, half_second, 2)
    levels = np.sqrt(np.square(blocks).mean(axis=(1, 2)))
    reference = float(np.quantile(levels, .75))
    active = np.flatnonzero(levels >= max(.004, reference * .32))
    if len(active) < 90 or (active[-1] + 1) * half_second < 50 * sr:
        raise StableAudioError("The generated track had less than 45 seconds of active music")
    # Keep a little of the final decay to avoid a hard cut when taking a new deck live.
    end = min(len(wave), (active[-1] + 1) * half_second + sr // 8)
    # The first piano phrase can be much quieter than the club drop.
    audible = np.flatnonzero(levels >= .004)
    start = max(0, int(audible[0]) * half_second - sr // 4)
    trimmed = (len(wave) - end) / sr
    wave = wave[start:end]
    peak = float(np.abs(wave).max())
    gain = min(.126 / reference, .76 / peak)
    wave = np.asarray(wave * gain, dtype=np.float32)
    bass, hop = _kick_envelope(wave, sr)
    threshold = max(.022, .52 * float(np.quantile(bass, .75)))

    def groove(second: float) -> bool:
        index = round(second / hop)
        return _bass_pulse(bass[index:index+round(4/hop)], hop, threshold)

    cue = 0.0
    opening = estimate_grid(wave[:14 * sr], sr, bpm, tempo_window=.10)
    entry_has_kick = groove(0)
    if incoming and (not entry_has_kick or opening is None):
        for second in range(2, 21, 2):
            if not groove(second):
                continue
            offset = second * sr
            grid = estimate_grid(wave[offset:offset + 12 * sr], sr, bpm, tempo_window=.10)
            if grid is not None and grid.confidence >= .4:
                cue, opening, entry_has_kick = float(second), grid, True
                break
    # Use the last rhythmic phrase when a fade or rising sweep has replaced the drums.
    end_seconds = len(wave) / sr
    closing_has_kick = groove(end_seconds - 4)
    if not closing_has_kick:
        for back in (2, 4, 6, 8):
            candidate = end_seconds - back
            if candidate - cue >= 48 and groove(candidate - 4):
                wave = wave[:round(candidate * sr)]
                trimmed += back
                closing_has_kick = True
                break
    wave = wave[round(cue * sr):]
    window = max(0, len(wave) - 14 * sr)
    closing = estimate_grid(wave[window:], sr, opening.bpm if opening else bpm, tempo_window=.10)
    if closing is not None:
        closing = replace(closing, phase=(closing.phase + window / sr) % closing.period)
    score = sum(grid.confidence for grid in (opening, closing) if grid is not None)
    score += 0.4 * min(len(wave) / (70 * sr), 1) + float(closing_has_kick)
    if incoming:
        score += 1.4 * float(entry_has_kick) - .015 * cue
    return Deck(wave, opening, closing, trimmed, float(20 * np.log10(gain)), score, cue, entry_has_kick)


class StableAudio:
    def __init__(self, sr: int):
        self.sr = sr
        url = urlsplit(os.environ.get("AUTODJ_COMFY_URL", "http://127.0.0.1:8296"))
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.username or url.path not in ("", "/") or url.query or url.fragment:
            raise StableAudioError("Set AUTODJ_COMFY_URL to a local ComfyUI HTTP address")
        self.url = f"http://{url.netloc}"

    def _request(self, path: str, payload: dict | None = None, *, limit: int = 32 * 1024 * 1024) -> bytes:
        req = Request(self.url + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=12) as response:
            result = response.read(limit + 1)
        if len(result) > limit:
            raise StableAudioError("ComfyUI reply exceeded the size limit")
        return result

    def check(self) -> None:
        try:
            checkpoints = json.loads(self._request("/models/checkpoints"))
            encoders = json.loads(self._request("/models/text_encoders"))
        except (OSError, ValueError) as exc:
            raise StableAudioError(f"Start your local Stable Audio ComfyUI server at {self.url}") from exc
        if CHECKPOINT not in checkpoints or ENCODER not in encoders:
            raise StableAudioError("ComfyUI is missing the Stable Audio Small-Music checkpoint or T5Gemma encoder")

    def render(self, plan: dict, seed: int, stopped) -> np.ndarray:
        key = plan["camelot"]
        note = KEYS[key[-1]][int(key[:-1])-1]
        mode = "minor" if key[-1] == "A" else "major"
        prompt = (
            f"{plan['style']} {plan['bpm']} BPM in {note} {mode}. "
            "DJ-ready instrumental club track. Begin with a distinct melodic hook over steady drums "
            "and a clean bass pulse. Develop a strong build, a heavy sustained main drop and a short "
            "rhythmic mix-out. Keep the drum groove active to the last phrase. No vocals or long fade."
        )
        style = plan["style"].lower()
        if "dubstep" in style and "piano" in style:
            prompt = (
                "Heavy modern melodic dubstep club record with huge half-time kick and cracking snare, "
                "deep clean sub, and pitched syncopated call-and-response growl bass. "
                "A real acoustic grand piano plays a recognizable, rhythmic recurring hook. "
                "The piano returns as an answer in the gaps of the bass drop. "
                + prompt
            )
        def node(kind, **inputs):
            return {"class_type": kind, "inputs": inputs}
        graph = {
            "1": node("CheckpointLoaderSimple", ckpt_name=CHECKPOINT),
            "2": node("CLIPLoader", clip_name=ENCODER, type="stable_audio", device="default"),
            "3": node("CLIPTextEncode", clip=["2", 0], text=prompt),
            "4": node("CLIPTextEncode", clip=["2", 0], text=""),
            "5": node("EmptyLatentAudio", seconds=SECONDS, batch_size=1),
            "6": node("KSampler", model=["1", 0], positive=["3", 0], negative=["4", 0],
                      latent_image=["5", 0], seed=seed, steps=8, cfg=1, sampler_name="lcm", scheduler="simple", denoise=1),
            "7": node("VAEDecodeAudio", samples=["6", 0], vae=["1", 2]),
            "8": node("PreviewAudio", audio=["7", 0]),
        }
        client = uuid4().hex
        socket = create_connection(self.url.replace("http:", "ws:", 1) + "/ws?clientId=" + client, timeout=5)
        try:
            job = json.loads(self._request("/prompt", {"client_id": client, "prompt": graph}))["prompt_id"]
            deadline = time.monotonic() + 110
            socket.settimeout(1)
            while time.monotonic() < deadline and not stopped():
                try:
                    message = socket.recv()
                except WebSocketTimeoutException:
                    continue
                if not message:
                    raise StableAudioError("Stable Audio connection closed during generation")
                if not isinstance(message, str):
                    continue
                event = json.loads(message)
                data = event.get("data", {})
                if data.get("prompt_id") != job:
                    continue
                if event["type"] in {"execution_error", "execution_interrupted"}:
                    raise StableAudioError("ComfyUI could not generate this track")
                if event["type"] == "executing" and data.get("node") is None:
                    break
            else:
                raise StableAudioError("Stable Audio generation stopped or timed out")
            history = json.loads(self._request("/history/" + job))[job]
            item = history["outputs"]["8"]["audio"][0]
            audio = self._request("/view?" + urlencode({k: item[k] for k in ("filename", "subfolder", "type")}))
        finally:
            socket.close()
        wave, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        if wave.shape[1] != 2 or sr <= 0:
            raise StableAudioError("Expected stereo audio from Stable Audio")
        if sr != self.sr:
            common = gcd(self.sr, sr)
            wave = signal.resample_poly(wave, self.sr // common, sr // common, axis=0)
        return wave
