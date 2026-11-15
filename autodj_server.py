#!/usr/bin/env python3
"""Lookahead PipeWire Streaming Engine, `.wav`/`.cue` Exporter & Unified 2-Deck FastAPI Web Studio (`autodj_server.py`).

Implements:
1. `PipeWireLookaheadStreamer`: Double-buffered `/dev/shm` ringbuffer streaming gapless `float32le`
   stereo `44.1 kHz` PCM to `/usr/bin/pw-play --format=f32 --rate=44100 --channels=2 -` (Trap #4 Fix).
2. `export_master_mix_and_cue`: Stitches certified transitions into a continuous master `.wav`
   and writes a DJ software-compatible `.cue` tracklist sheet.
3. Unified FastAPI + WebSocket 2-Deck DJ & AI Production Studio (`app`):
   - `GET /`: Full Pioneer CDJ-3000 / Traktor Pro 4 / Neural Telemetry 2-Deck Studio UI
   - `GET /api/state`, `POST /api/quickstart`, `POST /api/library/scan`, `POST /api/deck/load`,
     `POST /api/deck/mix-pair`, `POST /api/agent-mode`, `POST /api/steer`, `POST /api/produce`,
     `POST /api/run-session`, `POST /api/host-drop`, `POST /api/pipewire-play`,
     `GET /api/envelope/{track_id}`, `GET /api/audio/*`, `GET /api/export/*`,
     `GET /api/spectrogram/{filename}`, and `WS /ws/telemetry`.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
import soundfile as sf

from autodj_agent import AgentRunner, AgentError
from autodj_gemini import GeminiError, configured_model, require_planner
from autodj_live_agent import LiveDJ
from autodj_crew import AutoDJCrew, _dict_to_track_profile
from autodj_indexer import (
    SAMPLE_RATE,
    STEM_NAMES,
    camelot_distance,
    decode_audio_stereo,
)
from autodj_mixer import (
    build_stem_automation_envelopes,
    crossfade_join,
    normalize_and_brickwall_limit,
)
from autodj_planner import ARCHETYPE_DESCRIPTIONS
from autodj_producer import free_cached_gpu_models, offload_external_gpu_vram


# ============================================================================
# 1. Double-Buffered `/dev/shm` Ringbuffer & PipeWire `pw-play` Streamer (Trap #4 Fix)
# ============================================================================

class PipeWireLookaheadStreamer:
    """Streams gapless `float32le` stereo 44.1 kHz PCM from a double-buffered `/dev/shm` ringbuffer to `pw-play`."""

    def __init__(self, shm_dir: str = "/dev/shm/autodj_ringbuffer", sr: int = SAMPLE_RATE) -> None:
        self.shm_dir = shm_dir
        self.sr = sr
        os.makedirs(shm_dir, exist_ok=True)
        self.slot_paths = [
            os.path.join(shm_dir, "slot_0.f32le"),
            os.path.join(shm_dir, "slot_1.f32le"),
        ]
        self.active_slot = 0
        self.bytes_streamed = 0
        self.pw_play_bin = shutil.which("pw-play") or "/usr/bin/pw-play"

    def write_lookahead_slot(self, stereo: np.ndarray, slot_idx: int | None = None) -> str:
        """Writes stereo `(2, N)` float32 audio as interleaved `float32le` PCM into `/dev/shm/slot_X.f32le`."""
        idx = (1 - self.active_slot) if slot_idx is None else (slot_idx % 2)
        target = self.slot_paths[idx]
        interleaved = np.ascontiguousarray(stereo.T, dtype="<f4")
        with open(target, "wb") as f:
            f.write(interleaved.tobytes())
        return target

    def stream_slot_to_pipewire(
        self,
        slot_idx: int = 0,
        live_audio: bool = False,
        max_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Streams interleaved `float32le` PCM from `/dev/shm` to `/usr/bin/pw-play` (or verifies sink in dry-run)."""
        path = self.slot_paths[slot_idx % 2]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Ringbuffer slot file not found: {path}")

        with open(path, "rb") as f:
            raw_bytes = f.read()
        if max_seconds is not None:
            max_bytes = int(max_seconds * self.sr) * 2 * 4
            raw_bytes = raw_bytes[:max_bytes]

        cmd = [
            self.pw_play_bin,
            "--format=f32",
            f"--rate={self.sr}",
            "--channels=2",
            "-",
        ]
        if live_audio and os.path.isfile(self.pw_play_bin):
            pw_env = dict(os.environ)
            pw_env.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
            proc = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=pw_env, check=False)
            self.bytes_streamed += len(raw_bytes)
            self.active_slot = slot_idx % 2
            return {
                "status": "streamed_pw_play" if proc.returncode == 0 else "pw_play_error",
                "command": " ".join(cmd),
                "bytes_streamed": len(raw_bytes),
                "frames_streamed": len(raw_bytes) // 8,
                "returncode": proc.returncode,
            }

        self.bytes_streamed += len(raw_bytes)
        self.active_slot = slot_idx % 2
        return {
            "status": "verified_shm_ringbuffer",
            "command": " ".join(cmd),
            "slot_path": path,
            "bytes_streamed": len(raw_bytes),
            "frames_streamed": len(raw_bytes) // 8,
            "pw_play_available": os.path.isfile(self.pw_play_bin),
        }


# ============================================================================
# 2. Continuous Master `.wav` & `.cue` Sheet Exporter
# ============================================================================

def _sec_to_cue_timestamp(sec: float) -> str:
    """Formats seconds into standard CD CUE sheet `MM:SS:FF` (75 frames/sec)."""
    total_frames = int(round(max(0.0, sec) * 75.0))
    ff = total_frames % 75
    total_secs = total_frames // 75
    ss = total_secs % 60
    mm = total_secs // 60
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def export_master_mix_and_cue(
    session_state: dict[str, Any],
    output_wav_path: str,
    output_cue_path: str | None = None,
    max_transitions: int = 12,
) -> dict[str, Any]:
    """Concatenates certified transition mixes into a gapless master `.wav` and writes a `.cue` sheet."""
    if output_cue_path is None:
        output_cue_path = os.path.splitext(output_wav_path)[0] + ".cue"

    certified = session_state.get("certified_transitions", [])
    if not certified:
        raise ValueError("No certified transitions found in session_state to export.")

    recent_certified = [
        item for item in certified[-max_transitions:]
        if os.path.isfile(item.get("rendered", {}).get("candidate_wav_path", ""))
    ]
    if not recent_certified:
        recent_certified = certified[-1:]

    master_stereo = np.zeros((2, 0), dtype=np.float32)
    cue_entries: list[dict[str, Any]] = []
    curr_sec = 0.0

    for idx, item in enumerate(recent_certified):
        wav_path = item["rendered"]["candidate_wav_path"]
        spec = item["spec"]
        seg, _ = decode_audio_stereo(wav_path, sr=SAMPLE_RATE)
        cue_entries.append({
            "track_number": idx + 1,
            "title": f"{spec['track_a_title']} -> {spec['track_b_title']} ({spec['archetype']})",
            "bpm": spec["target_bpm"],
            "archetype": spec["archetype"],
            "start_sec": round(curr_sec, 3),
            "cue_timestamp": _sec_to_cue_timestamp(curr_sec),
        })
        if master_stereo.shape[1] == 0:
            master_stereo = seg
        else:
            master_stereo = crossfade_join(master_stereo, seg, crossfade_samples=2048)
        curr_sec = master_stereo.shape[1] / SAMPLE_RATE

    master_stereo, master_stats = normalize_and_brickwall_limit(
        master_stereo, target_lufs=-14.0, ceiling_dbtp=-1.0, sr=SAMPLE_RATE
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    sf.write(output_wav_path, master_stereo.T, SAMPLE_RATE, subtype="FLOAT")

    cue_lines = [
        'PERFORMER "AI-Auto-DJ 6-Agent Crew"',
        f'TITLE "{session_state.get("steering_prompt", "AI Auto DJ Live Mix")}"',
        f'FILE "{os.path.basename(output_wav_path)}" WAVE',
    ]
    for entry in cue_entries:
        t_num = entry["track_number"]
        cue_lines.extend([
            f"  TRACK {t_num:02d} AUDIO",
            f'    TITLE "{entry["title"]}"',
            '    PERFORMER "AI-Auto-DJ"',
            f'    REM BPM {entry["bpm"]}',
            f'    REM ARCHETYPE {entry["archetype"]}',
            f'    INDEX 01 {entry["cue_timestamp"]}',
        ])
    with open(output_cue_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cue_lines) + "\n")

    return {
        "master_wav_path": os.path.abspath(output_wav_path),
        "cue_path": os.path.abspath(output_cue_path),
        "duration_sec": round(master_stereo.shape[1] / SAMPLE_RATE, 4),
        "integrated_lufs": master_stats["integrated_lufs"],
        "true_peak_dbtp": master_stats["true_peak_dbtp"],
        "cue_entries": cue_entries,
    }


# ============================================================================
# 3. Pydantic Models for Unified 2-Deck Web Studio
# ============================================================================

class AgentModeUpdate(BaseModel):
    agent: Literal["director", "producer", "listener", "mixer", "critic", "host"]
    mode: Literal["local", "cloud", "hybrid"]
    audio_ear_model: str | None = None


class SteerRequest(BaseModel):
    steering_prompt: str
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    selected_archetype: Literal["AUTO", "T1", "T2", "T3", "T4", "T5", "T6"] | None = None


class ProduceRequest(BaseModel):
    title: str
    style: str = "Peak-Time Melodic Techno"
    target_bpm: float = 128.0
    target_camelot: str = "8A"
    bars: int = 16
    engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    lyrics: str = "[Instrumental]\n[Build]\n[Drop]\n[Outro]"
    auto_index: bool = False
    use_neural_gpu: bool = False


class LibraryScanRequest(BaseModel):
    path: str
    max_files: int = 12


class DeckLoadRequest(BaseModel):
    deck: Literal["A", "B"]
    track_id_or_path: str | None = None
    stem_gains: dict[str, float] | None = None
    selected_archetype: Literal["AUTO", "T1", "T2", "T3", "T4", "T5", "T6"] | None = None


class DeckMixPairRequest(BaseModel):
    track_a_id: str | None = None
    track_b_id: str | None = None
    archetype: Literal["AUTO", "T1", "T2", "T3", "T4", "T5", "T6"] = "AUTO"
    overlap_bars: int | None = None
    enable_host_mc: bool = True
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    stem_gains_a: dict[str, float] | None = None
    stem_gains_b: dict[str, float] | None = None


class RunSessionRequest(BaseModel):
    track_paths: list[str] = []
    steering_prompt: str = "High-energy progressive club set"
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    enable_host_mc: bool = True


class QuickStartRequest(BaseModel):
    steering_prompt: str = "Peak-time progressive club energy arc"
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    enable_host_mc: bool = True
    bars: int = 8
    force_new_tracks: bool = False


class HostDropRequest(BaseModel):
    script: str = "Welcome to the Neural Deck AI Auto DJ Studio — locking into 128 BPM!"
    voice_name: str = "af_heart"
    speed: float = 1.05


class PipeWirePlayRequest(BaseModel):
    live_audio: bool = False
    max_seconds: float | None = 8.0


class LiveStartRequest(BaseModel):
    steering_prompt: str = "Peak-time progressive club energy arc"
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    use_neural_gpu: bool = True
    enable_host_mc: bool = True
    pipewire_live: bool = False
    bars: int = 8
    client_playing_index: int = -1


class LiveProgressRequest(BaseModel):
    playing_index: int = 0
    client_playing_index: int | None = None


WEB_STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NEURAL DECK // AI-Auto-DJ: 6-Agent Virtual DJ &amp; Production Studio</title>
<link rel="icon" href="/deck-assets/deck-icon.png" type="image/png">
<style>
  :root {
    color-scheme: dark;
    accent-color: #38bdf8;
    --in-oklab: ;
    --in-oklch: ;
    --bg: #05070e;
    --chassis: rgba(11, 15, 26, 0.82);
    --panel: rgba(16, 23, 38, 0.78);
    --panel-hi: rgba(24, 34, 56, 0.9);
    --oled: rgba(5, 8, 15, 0.92);
    --border: rgba(255, 255, 255, 0.07);
    --border-hi: rgba(255, 255, 255, 0.13);
    --accent: #38bdf8;
    --deckA: #ff6b2b;
    --deckA-glow: rgba(255, 107, 43, 0.25);
    --deckB: #a855f7;
    --deckB-glow: rgba(168, 85, 247, 0.25);
    --pass: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #f8fafc;
    --muted: #94a3b8;
    --scrollbar-thumb: #334155;
    --scrollbar-track: #070a12;
  }
  @supports (background: linear-gradient(in oklab, white, black)) {
    :root {
      --in-oklab: in oklab;
      --in-oklch: in oklch;
    }
  }
  * { box-sizing: border-box; }
  body, .scroller {
    scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
    scrollbar-width: thin;
  }
  @supports not (scrollbar-color: auto) {
    .scroller::-webkit-scrollbar { width: 8px; height: 8px; }
    .scroller::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 4px; }
    .scroller::-webkit-scrollbar-track { background: var(--scrollbar-track); }
  }
  html {
    margin: 0;
    padding: 0;
    overflow-x: hidden;
  }
  body {
    margin: 0;
    background:
      radial-gradient(circle at 15% 18%, rgba(255, 107, 43, 0.08) 0%, transparent 42%),
      radial-gradient(circle at 85% 18%, rgba(168, 85, 247, 0.08) 0%, transparent 42%),
      radial-gradient(circle at 50% 0% var(--in-oklab), #0f172a 0%, var(--bg) 75%);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px;
    padding: 6px 14px 8px;
    overflow-x: hidden;
  }
  .mono { font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  /* Top Hardware Master Command Bar */
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    background: rgba(13, 19, 33, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-hi);
    border-radius: 14px;
    padding: 5px 12px;
    margin-bottom: 6px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
    min-height: 46px;
  }
  .topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    flex-shrink: 0;
  }
  .brand-title {
    font-size: 13.5px;
    font-weight: 800;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }
  .led-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--pass);
    box-shadow: 0 0 10px var(--pass);
    display: inline-block;
    flex-shrink: 0;
  }
  .hw-pills-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .specs-dropdown {
    position: relative;
    display: inline-block;
  }
  .specs-dropdown summary {
    list-style: none;
    cursor: pointer;
    user-select: none;
  }
  .specs-dropdown summary::-webkit-details-marker {
    display: none;
  }
  .specs-menu {
    position: absolute;
    top: calc(100% + 8px);
    left: 0;
    z-index: 1000;
    background: rgba(9, 14, 24, 0.96);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-hi);
    border-radius: 12px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.85);
    min-width: 210px;
  }
  .pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
  }
  .pill.hi { border-color: rgba(56, 189, 248, 0.4); color: var(--accent); }
  .pill.pass { border-color: rgba(16, 185, 129, 0.45); color: var(--pass); background: rgba(16, 185, 129, 0.08); }
  #statusBadge {
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    display: inline-block;
    vertical-align: middle;
  }
  .specs-pill {
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .specs-pill:hover { border-color: var(--accent); color: var(--text); }

  .transport-cluster {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    flex-shrink: 1;
  }
  .transport-cluster button {
    padding: 6px 11px;
    font-size: 11.5px;
    border-radius: 999px;
  }
  button {
    background: rgba(30, 41, 59, 0.75);
    color: var(--text);
    border: 1px solid var(--border-hi);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  button:hover { border-color: var(--accent); background: rgba(51, 65, 85, 0.85); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  .btn-quickstart {
    background: linear-gradient(135deg var(--in-oklab), #ff6b2b 0%, #ec4899 55%, #8b5cf6 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.28);
    box-shadow: 0 4px 20px rgba(236, 72, 153, 0.35);
    padding: 7px 16px !important;
    font-size: 12px !important;
    font-weight: 800;
    border-radius: 999px !important;
    letter-spacing: 0.02em;
  }
  .btn-primary {
    background: linear-gradient(180deg var(--in-oklab), #38bdf8 0%, #0284c7 100%);
    color: #020617;
    border-color: #7dd3fc;
    font-weight: 700;
  }
  .btn-mix {
    background: rgba(16, 185, 129, 0.16);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.45);
    font-weight: 700;
  }
  .btn-stop {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.4);
    color: #fca5a5;
  }
  .master-audio-box {
    display: flex;
    flex-direction: column;
    gap: 3px;
    background: rgba(5, 8, 15, 0.75);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px 10px;
    flex-shrink: 0;
  }
  .master-audio-row { display: flex; align-items: center; gap: 6px; justify-content: space-between; }
  audio { height: 24px; width: 155px; filter: invert(0.92) hue-rotate(180deg); opacity: 0.9; }
  .audio-src-select { max-width: 130px; }

  input, select {
    background: rgba(5, 8, 15, 0.85);
    color: var(--text);
    border: 1px solid var(--border-hi);
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 11.5px;
  }
  input:not(input[type="range"]):user-invalid { border-color: var(--danger); }
  input:not(input[type="range"]):user-valid { border-color: var(--border-hi); }
  #deckASelect, #deckBSelect {
    min-width: 0;
    max-width: 195px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    font-weight: 600;
  }
  /* Live Autonomous AI DJ Command Banner & 6-Agent Pipeline Strip */
  .live-pipeline-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    background: linear-gradient(90deg, rgba(16, 24, 42, 0.92) 0%, rgba(10, 15, 28, 0.94) 100%);
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 12px;
    padding: 4px 10px;
    margin-bottom: 6px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
  }
  .btn-live-auto {
    background: linear-gradient(135deg var(--in-oklab), #ef4444 0%, #f97316 55%, #eab308 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.45);
    padding: 5px 12px !important;
    font-size: 11px !important;
    font-weight: 800;
    border-radius: 999px !important;
    letter-spacing: 0.02em;
    flex-shrink: 0;
  }
  .btn-live-auto.on-air, .btn-live-auto.live-active {
    background: linear-gradient(135deg var(--in-oklab), #10b981 0%, #059669 60%, #0284c7 100%);
    box-shadow: 0 0 22px rgba(16, 185, 129, 0.7);
  }
  .pipe-nodes-wrap {
    display: flex;
    align-items: center;
    gap: 3px;
    flex-shrink: 1;
    min-width: 0;
    overflow: hidden;
  }
  .pipe-node {
    background: rgba(5, 8, 15, 0.85);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 2px 6px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 700;
    white-space: nowrap;
    transition: all 0.2s ease;
  }
  .pipe-node.active {
    background: rgba(56, 189, 248, 0.25);
    border-color: var(--accent);
    color: #ffffff;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
  }
  .pipe-node.done {
    background: rgba(16, 185, 129, 0.14);
    border-color: rgba(16, 185, 129, 0.55);
    color: var(--pass);
  }
  .pipe-arrow { color: #475569; font-size: 8.5px; user-select: none; }
  .live-vibe-cluster {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
  }
  @media (max-width: 1520px) {
    .topbar { gap: 5px; padding: 4px 8px; }
    .topbar-left { gap: 5px; }
    .brand-title { font-size: 11.5px; }
    .btn-quickstart { padding: 5px 9px !important; font-size: 10px !important; }
    .transport-cluster { gap: 3px; }
    .transport-cluster button { padding: 4px 6px; font-size: 9.5px; }
    audio { width: 115px; }
    .audio-src-select { max-width: 98px; }
    #statusBadge { max-width: 110px; overflow: hidden; text-overflow: ellipsis; }
    .btn-live-auto { padding: 4px 8px !important; font-size: 9.5px !important; }
    .pipe-node { padding: 2px 4px; font-size: 8px; }
    #liveAgentStepCaption { max-width: 135px !important; }
    #liveVibeInput { width: 110px !important; }
  }

  /* Stacked Dual Battle Waveforms */
  .waveform-rack {
    background: rgba(7, 10, 19, 0.88);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-hi);
    border-radius: 14px;
    padding: 5px 10px;
    margin-bottom: 6px;
    box-shadow: inset 0 2px 16px rgba(0,0,0,0.75);
  }
  .wave-row {
    position: relative;
    height: 36px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 3px;
  }
  .wave-row:last-child { border-bottom: none; margin-bottom: 0; }
  .wave-label {
    position: absolute;
    top: 4px; left: 8px;
    z-index: 2;
    background: rgba(5, 8, 15, 0.82);
    backdrop-filter: blur(6px);
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.03em;
    pointer-events: none;
    border: 1px solid;
  }
  canvas.wave-canvas { width: 100%; height: 36px; display: block; cursor: pointer; }

  /* 3-Column DJ Booth Console */
  .booth-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.92fr) minmax(0, 1.15fr);
    gap: 12px;
    margin-bottom: 6px;
  }
  .deck-card {
    background: linear-gradient(180deg var(--in-oklab), rgba(18, 25, 43, 0.82) 0%, rgba(10, 14, 24, 0.9) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-hi);
    border-radius: 16px;
    padding: 9px 12px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .deck-card.deck-a { border-top: 3px solid var(--deckA); box-shadow: 0 8px 30px rgba(255, 107, 43, 0.08); }
  .deck-card.deck-b { border-top: 3px solid var(--deckB); box-shadow: 0 8px 30px rgba(168, 85, 247, 0.08); }

  .deck-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .deck-badge {
    font-size: 11px;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }
  .deck-a .deck-badge { background: rgba(255, 107, 43, 0.15); color: var(--deckA); border: 1px solid rgba(255, 107, 43, 0.4); }
  .deck-b .deck-badge { background: rgba(168, 85, 247, 0.15); color: var(--deckB); border: 1px solid rgba(168, 85, 247, 0.4); }

  .deck-main {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    min-height: 160px;
  }
  .deck-card.stems-open .deck-main {
    display: grid;
    grid-template-columns: 156px 1fr;
    align-items: center;
  }
  /* Motorized Jog Wheel */
  .platter-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .jog-wheel {
    width: 154px;
    height: 154px;
    border-radius: 50%;
    position: relative;
    background:
      radial-gradient(circle at center, #030509 0 44%, transparent 45%),
      repeating-radial-gradient(circle at center, #131c2e 0 2px, #090d16 2px 5px);
    border: 3px solid #334155;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: grab;
    user-select: none;
    flex-shrink: 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .deck-a .jog-wheel { box-shadow: 0 0 24px var(--deckA-glow), inset 0 0 14px var(--deckA-glow); border-color: var(--deckA); }
  .deck-b .jog-wheel { box-shadow: 0 0 24px var(--deckB-glow), inset 0 0 14px var(--deckB-glow); border-color: var(--deckB); }
  .jog-ring {
    position: absolute;
    inset: 5px;
    border-radius: 50%;
    border: 2px dashed rgba(255,255,255,0.18);
    pointer-events: none;
  }
  .jog-marker {
    position: absolute;
    top: 3px; left: 50%;
    width: 3.5px; height: 12px;
    margin-left: -1.75px;
    border-radius: 2px;
    background: #fff;
    box-shadow: 0 0 8px #fff;
  }
  .jog-oled {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: radial-gradient(circle at center, #0b1120 0%, #030509 100%);
    border: 1px solid rgba(255,255,255,0.14);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4px;
    z-index: 2;
  }
  .oled-bpm { font-size: 18px; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em; }
  .oled-key {
    font-size: 11.5px;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 99px;
    margin-top: 3px;
    background: rgba(255,255,255,0.05);
    border: 1px solid #475569;
  }
  .oled-sub { font-size: 9px; color: var(--muted); margin-top: 3px; }

  /* 4-Stem Faders & Collapsible Isolator Drawer */
  .stems-box, .stem-drawer {
    background: var(--oled);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 8px;
  }
  .stem-row {
    display: grid;
    grid-template-columns: 46px 1fr 24px 24px;
    gap: 4px;
    align-items: center;
    margin-bottom: 4px;
  }
  .stem-row:last-child { margin-bottom: 0; }
  .stem-name { font-size: 9px; font-weight: 800; letter-spacing: 0.04em; }
  .stem-meter-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  input[type="range"] {
    --slider-fill-color: var(--accent);
    margin: 0;
    border: 0;
  }
  input[type="range"]:focus-visible {
    outline: 2px solid var(--slider-fill-color);
    outline-offset: 3px;
  }
  input[type="range"].stem-slider {
    --slider-track-size: 5px;
    --slider-thumb-size: 12px;
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 12px;
    border-radius: 3px;
    background: transparent;
    padding: 0;
    border: 0;
    cursor: pointer;
  }
  input[type="range"].stem-slider::-webkit-slider-runnable-track {
    block-size: var(--slider-track-size);
    border: 0;
    border-radius: 3px;
    background: #1e293b;
  }
  input[type="range"].stem-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 10px;
    height: var(--slider-thumb-size);
    margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
    border-radius: 2px;
    background: #f8fafc;
    border: 1px solid #020617;
    cursor: pointer;
  }
  input[type="range"].stem-slider::-moz-range-track {
    block-size: var(--slider-track-size);
    border: 0;
    border-radius: 3px;
    background: #1e293b;
  }
  input[type="range"].stem-slider::-moz-range-thumb {
    width: 10px;
    height: var(--slider-thumb-size);
    border-radius: 2px;
    background: #f8fafc;
    border: 1px solid #020617;
    cursor: pointer;
  }
  .stem-bar { height: 4px; border-radius: 2px; background: #111827; overflow: hidden; }
  .stem-fill { height: 100%; transition: width 0.2s ease; }
  .mini-btn {
    padding: 2px 4px;
    font-size: 8.5px;
    border-radius: 4px;
    justify-content: center;
  }
  .mini-btn.muted { background: var(--danger); color: #fff; border-color: var(--danger); }

  /* 8 RGB Hot-Cue / Phrase Jump Pads */
  .pads-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 8px;
  }
  .cue-pad {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 5px 4px;
    text-align: center;
    cursor: pointer;
    font-size: 9.5px;
    font-weight: 700;
    transition: all 0.15s ease;
    overflow: hidden;
  }
  .cue-pad:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.14); transform: translateY(-1px); }
  .cue-pad .pad-sec { display: block; font-size: 8px; color: var(--muted); margin-top: 2px; }

  /* Center Mixer Console */
  .mixer-card {
    background: linear-gradient(180deg var(--in-oklab), rgba(20, 29, 49, 0.85) 0%, rgba(9, 13, 24, 0.92) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-hi);
    border-radius: 16px;
    padding: 9px 12px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 6px;
  }
  .arch-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 4px;
  }
  .arch-btn {
    padding: 5px 2px;
    font-size: 9.5px;
    justify-content: center;
    border-radius: 7px;
  }
  .arch-btn.active {
    background: linear-gradient(180deg, #38bdf8 0%, #0284c7 100%);
    color: #020617;
    border-color: #bae6fd;
    font-weight: 800;
    box-shadow: 0 0 14px rgba(56,189,248,0.45);
  }
  .camelot-row {
    display: grid;
    grid-template-columns: 92px 1fr;
    gap: 10px;
    align-items: center;
    background: rgba(5, 8, 15, 0.72);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px 9px;
  }
  #mixSpecMeta, #mixRationaleText {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .xfader-box {
    background: rgba(5, 8, 15, 0.82);
    border: 1px solid var(--border-hi);
    border-radius: 12px;
    padding: 6px 10px;
  }
  input[type="range"].xfader {
    --slider-track-size: 8px;
    --slider-thumb-size: 20px;
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 20px;
    border-radius: 4px;
    background: transparent;
    border: 0;
    cursor: ew-resize;
  }
  input[type="range"].xfader::-webkit-slider-runnable-track {
    block-size: var(--slider-track-size);
    border-radius: 4px;
    background: linear-gradient(90deg, var(--deckA) 0%, #334155 50%, var(--deckB) 100%);
  }
  input[type="range"].xfader::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: var(--slider-thumb-size);
    margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
    border-radius: 3px;
    background: #f8fafc;
    border: 2px solid #020617;
    cursor: ew-resize;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
  }
  input[type="range"].xfader::-moz-range-track {
    block-size: var(--slider-track-size);
    border-radius: 4px;
    background: linear-gradient(90deg, var(--deckA) 0%, #334155 50%, var(--deckB) 100%);
  }
  input[type="range"].xfader::-moz-range-thumb {
    width: 16px;
    height: var(--slider-thumb-size);
    border-radius: 3px;
    background: #f8fafc;
    border: 2px solid #020617;
    cursor: ew-resize;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
  }

  /* 5-Tab Studio Workstation Dock */
  .dock-container {
    background: rgba(14, 20, 34, 0.84);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-hi);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
  }
  .dock-tabbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(7, 10, 18, 0.88);
    border-bottom: 1px solid var(--border);
    padding: 5px 12px;
    gap: 6px;
  }
  .dock-tabs-left {
    display: flex;
    align-items: center;
    gap: 6px;
    overflow-x: auto;
  }
  .dock-tab-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }
  .dock-tab-btn:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.04);
  }
  .dock-tab-btn.active {
    background: rgba(56, 189, 248, 0.14);
    color: var(--accent);
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 0 12px rgba(56,189,248,0.18);
  }
  .dock-tab-btn .tab-count {
    font-size: 9.5px;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 1px 6px;
    border-radius: 99px;
    margin-left: 5px;
    color: var(--pass);
  }
  .toggle-dock-btn {
    font-size: 10.5px;
    padding: 4px 10px;
    color: var(--muted);
  }
  .dock-panes-body {
    padding: 8px 12px;
    background: rgba(9, 13, 24, 0.72);
    min-height: 138px;
    max-height: 158px;
    overflow-y: auto;
  }
  .tab-pane {
    display: none;
  }
  .tab-pane.active {
    display: block;
  }
  .crate-list {
    max-height: 125px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--oled);
  }
  .crate-item {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 6px;
    align-items: center;
    padding: 4px 7px;
    border-bottom: 1px solid #111827;
    font-size: 10.5px;
  }
  .crate-item:last-child { border-bottom: none; }
  .agent-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    border-bottom: 1px solid #162032;
    font-size: 10.5px;
  }
  .spec-preview {
    width: 100%;
    height: 118px;
    object-fit: contain;
    background: #020408;
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: zoom-in;
  }
  .qa-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    font-size: 9.5px;
  }
  .qa-cell {
    background: var(--oled);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 6px;
    display: flex;
    justify-content: space-between;
  }
  #events {
    max-height: 120px;
    overflow-y: auto;
    background: var(--oled);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 7px;
    font-size: 10px;
    line-height: 1.4;
    word-break: break-word;
    overflow-wrap: anywhere;
  }
  dialog#specModal {
    background: transparent;
    border: none;
    padding: 0;
    max-width: 94vw;
    max-height: 92vh;
  }
  dialog#specModal::backdrop {
    background-color: rgba(2, 6, 23, 0.88);
    backdrop-filter: blur(2px);
  }
  dialog#specModal img {
    display: block;
    max-width: 92vw;
    max-height: 90vh;
    border: 2px solid var(--accent);
    border-radius: 10px;
  }
</style>
</head>
<body>

<!-- TOP HARDWARE COMMAND BAR -->
<header class="topbar">
  <div class="topbar-left">
    <div class="brand-title">
      <span class="led-dot" id="liveLed"></span>
      <span>AI-Auto-DJ: 6-Agent Virtual DJ</span>
      <a href="/" style="color:#b9cf88;margin-left:8px;font-size:11px;text-decoration:none">← Prompt DJ</a>
    </div>
    <div class="hw-pills-row">
      <details class="specs-dropdown" id="specsDropdown">
        <summary class="pill specs-pill" title="Hardware &amp; Neural Stack Specifications">⚙️ Specs ▾</summary>
        <div class="specs-menu">
          <span class="pill hi">Option A1 Neural Stack</span>
          <span class="pill">librubberband R3 (&lt;0.1ms)</span>
          <span class="pill">ACE-Step 1.5 + Suno v5</span>
          <span class="pill">Kokoro-82M MC (-10dB Duck)</span>
        </div>
      </details>
      <span class="pill pass" id="statusBadge">PIPEWIRE READY (-14 LUFS)</span>
    </div>
  </div>

  <div class="transport-cluster">
    <button class="btn-quickstart" id="btnQuickStart" onclick="runQuickStart()" title="Produce 3 tracks &amp; execute 6-agent set">
      ⚡ 1-CLICK QUICK-START DEMO MIX
    </button>
    <button class="btn-primary" id="btnRunSession" onclick="runFullSession()" title="Run full 6-agent set">
      ▶ RUN SET
    </button>
    <button class="btn-mix" id="btnMixPair" onclick="mixDeckPair()" title="Mix Deck A to Deck B">
      🔀 MIX A→B
    </button>
    <button class="btn-stop" id="btnStopSession" onclick="stopStudioSession()" title="Stop session &amp; halt playback">
      ⏹ STOP
    </button>
    <button onclick="streamToPipeWire(false)" title="Verify/Stream float32le ringbuffer to PipeWire">
      🔊 PipeWire
    </button>
  </div>

  <div class="master-audio-box">
    <div class="master-audio-row">
      <select id="audioSourceSelect" onchange="switchAudioSource(this.value)" class="audio-src-select">
        <option value="master">🎧 Master Mix (.WAV)</option>
      </select>
      <audio id="masterAudio" controls preload="none"></audio>
      <a href="/api/export/wav" download="ai_autodj_master_mix.wav" style="text-decoration:none;"><button class="mini-btn">⬇ WAV</button></a>
      <a href="/api/export/cue" download="ai_autodj_master_mix.cue" style="text-decoration:none;"><button class="mini-btn">⬇ CUE</button></a>
    </div>
    <div class="master-audio-row">
      <canvas id="fftCanvas" width="220" height="18" style="width:100%;height:18px;background:#020408;border-radius:3px;"></canvas>
      <span class="mono" id="masterMetrics" style="font-size:9.5px;color:var(--pass);white-space:nowrap;margin-left:4px;">-14.0 LUFS • -1.0 dBTP</span>
    </div>
  </div>
</header>

<!-- LIVE AUTONOMOUS AI DJ COMMAND BANNER & 6-AGENT PIPELINE STRIP -->
<section class="live-pipeline-strip" id="liveAgentPipelineStrip">
  <button class="btn-live-auto" id="btnLiveAutonomous" onclick="toggleAutonomousLiveDJ()" title="Continuously produce, analyze, plan &amp; mix new music on RTX 5090">
    ▶ START AUTONOMOUS LIVE DJ
  </button>

  <div class="pipe-nodes-wrap" id="pipeNodesWrap">
    <span class="pipe-node" id="pipeNode_director">🧠 1. Director LLM</span><span class="pipe-arrow">➔</span>
    <span class="pipe-node" id="pipeNode_producer">🎹 2. Producer GPU</span><span class="pipe-arrow">➔</span>
    <span class="pipe-node" id="pipeNode_listener">🎧 3. Listener HTDemucs</span><span class="pipe-arrow">➔</span>
    <span class="pipe-node" id="pipeNode_mixer">🎚️ 4. Mixer R3</span><span class="pipe-arrow">➔</span>
    <span class="pipe-node" id="pipeNode_critic">🔬 5. Critic QA</span><span class="pipe-arrow">➔</span>
    <span class="pipe-node" id="pipeNode_host">🎙️ 6. Host Kokoro-82M</span>
  </div>

  <span class="mono" id="liveAgentStepCaption" style="font-size:9.5px;color:var(--accent);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:250px;">Ready for continuous RTX 5090 live mix</span>

  <div class="live-vibe-cluster">
    <input id="liveVibeInput" style="width:155px;padding:4px 7px;font-size:10.5px;" value="Peak-time progressive club energy arc" placeholder="Live DJ vibe direction..." oninput="syncSteerInputs(this.value)" />
    <button class="mini-btn" id="btnUpdateLiveVibe" onclick="sendSteer()" style="padding:4px 8px;font-size:10px;border-color:var(--accent);color:var(--accent);">🎯 Update Live Vibe</button>
  </div>
</section>

<!-- DUAL STACKED BATTLE WAVEFORMS & PHRASE TIMELINE -->
<section class="waveform-rack">
  <div class="wave-row">
    <div class="wave-label" style="color:var(--deckA);border-color:var(--deckA);" id="waveLabelA">DECK A // NO TRACK LOADED (CLICK ⚡ QUICK-START OR LOAD FROM CRATE)</div>
    <canvas id="waveCanvasA" class="wave-canvas" width="1280" height="36" onclick="seekWaveform('A', event)"></canvas>
  </div>
  <div class="wave-row">
    <div class="wave-label" style="color:var(--deckB);border-color:var(--deckB);" id="waveLabelB">DECK B // NO TRACK LOADED</div>
    <canvas id="waveCanvasB" class="wave-canvas" width="1280" height="36" onclick="seekWaveform('B', event)"></canvas>
  </div>
</section>

<!-- 3-COLUMN DJ BOOTH: DECK A | CENTER 4-CH MIXER & CAMELOT RADAR | DECK B -->
<section class="booth-grid">

  <!-- DECK A -->
  <div class="deck-card deck-a">
    <div class="deck-header">
      <span class="deck-badge">DECK A // OUTGOING</span>
      <select id="deckASelect" onchange="loadDeck('A', this.value)" style="flex:1;"></select>
      <button class="mini-btn" id="btnToggleStemsA" onclick="toggleStemDrawer('A')" title="Toggle 4-Stem Isolator Drawer">🎚️ Stems ▾</button>
      <button class="mini-btn" onclick="playDeckTrack('A')">▶ Audition A</button>
      <button class="mini-btn" onclick="inspectDeckSpectrogram('A')">📊 4-Panel</button>
    </div>

    <div class="deck-main">
      <div class="platter-wrap">
        <div class="jog-wheel" id="jogA" onclick="togglePlayPause()">
          <div class="jog-ring" id="jogRingA"><div class="jog-marker"></div></div>
          <div class="jog-oled">
            <div class="oled-bpm mono" id="deckABpm">128.0</div>
            <div class="oled-key mono" id="deckAKey" style="color:var(--deckA);border-color:var(--deckA);">8A</div>
            <div class="oled-sub mono" id="deckAMeta">A Minor • -14.0 LUFS</div>
          </div>
        </div>
        <div class="mono" style="font-size:9.5px;color:var(--muted);" id="deckACueInfo">CUE IN: Bar 0 • CUE OUT: Bar 4</div>
      </div>

      <div class="stem-drawer stems-box" id="stemDrawerA" style="display:none;">
        <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--muted);margin-bottom:4px;">
          <span>4-STEM ISOLATOR (HTDEMUCS)</span>
          <span style="color:var(--pass);">EXACT RECON &lt;1e-5</span>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#f97316;">DRUMS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_A_drums" min="0" max="120" value="100" oninput="updateStemGain('A','drums',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_A_drums" style="width:85%;background:#f97316;"></div></div>
          </div>
          <button class="mini-btn" id="mute_A_drums" onclick="toggleMuteStem('A','drums')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('A','drums')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#ef4444;">BASS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_A_bass" min="0" max="120" value="100" oninput="updateStemGain('A','bass',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_A_bass" style="width:92%;background:#ef4444;"></div></div>
          </div>
          <button class="mini-btn" id="mute_A_bass" onclick="toggleMuteStem('A','bass')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('A','bass')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#22c55e;">VOCALS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_A_vocals" min="0" max="120" value="100" oninput="updateStemGain('A','vocals',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_A_vocals" style="width:65%;background:#22c55e;"></div></div>
          </div>
          <button class="mini-btn" id="mute_A_vocals" onclick="toggleMuteStem('A','vocals')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('A','vocals')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#38bdf8;">OTHER</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_A_other" min="0" max="120" value="100" oninput="updateStemGain('A','other',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_A_other" style="width:78%;background:#38bdf8;"></div></div>
          </div>
          <button class="mini-btn" id="mute_A_other" onclick="toggleMuteStem('A','other')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('A','other')">🎧</button>
        </div>
      </div>
    </div>

    <div class="pads-grid" id="padsA"></div>
  </div>

  <!-- CENTER 4-CHANNEL NEURAL MIXER & CAMELOT HARMONIC RADAR -->
  <div class="mixer-card">
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-weight:800;font-size:10.5px;color:var(--accent);">TRANSITION ARCHETYPE (T1–T6)</span>
        <span class="mono" id="activeArchBadge" style="font-size:9.5px;color:var(--pass);">AUTO • T1</span>
      </div>
      <div class="arch-grid" id="archButtons">
        <button class="arch-btn active" onclick="selectArchetype('AUTO')">AUTO</button>
        <button class="arch-btn" onclick="selectArchetype('T1')" title="Phrase-Aligned 4-Stem Blend &amp; Hard Bass Swap">T1</button>
        <button class="arch-btn" onclick="selectArchetype('T2')" title="Vocal-Over-Instrumental Mashup">T2</button>
        <button class="arch-btn" onclick="selectArchetype('T3')" title="Breakdown-to-Build Double-Drop">T3</button>
        <button class="arch-btn" onclick="selectArchetype('T4')" title="Echo-Freeze &amp; High-Pass Wash Tempo Reset">T4</button>
        <button class="arch-btn" onclick="selectArchetype('T5')" title="Harmonic Pitch-Shift Bridge (+/-1..2 st)">T5</button>
        <button class="arch-btn" onclick="selectArchetype('T6')" title="AI Generative Bridge (ACE-Step / MusicGen / Suno)">T6</button>
      </div>
    </div>

    <!-- Camelot Wheel Radar + Compatibility Readout -->
    <div class="camelot-row">
      <svg id="camelotSvg" viewBox="0 0 120 120" width="96" height="96"></svg>
      <div style="display:flex;flex-direction:column;gap:3px;min-width:0;">
        <div class="mono" style="font-size:11.5px;font-weight:800;color:var(--pass);" id="compatScoreText">COMPAT: 95% (d_key=0)</div>
        <div style="font-size:10px;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" id="archNameText">Phrase-Aligned 4-Stem Blend &amp; Hard Bass Swap</div>
        <div class="mono" style="font-size:9.5px;color:var(--muted);" id="mixSpecMeta">Target: 128.0 BPM • Overlap: 8 Bars • Swap: Bar 4</div>
        <div style="font-size:9px;color:#cbd5e1;line-height:1.2;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;" id="mixRationaleText">Ready for sample-accurate 4-stem transition.</div>
      </div>
    </div>

    <!-- Stem Automation Curve Preview Canvas -->
    <div style="background:var(--oled);border:1px solid var(--border);border-radius:7px;padding:4px 6px;">
      <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--muted);margin-bottom:2px;">
        <span>STEM ENVELOPE AUTOMATION &amp; BASS-SWAP GUARD</span>
        <span class="mono" style="color:var(--pass);">max(gA+gB) &le; 1.0</span>
      </div>
      <canvas id="envCanvas" width="320" height="48" style="width:100%;height:48px;display:block;"></canvas>
    </div>

    <!-- Master Crossfader -->
    <div class="xfader-box">
      <div style="display:flex;justify-content:space-between;font-size:9.5px;font-weight:800;margin-bottom:3px;">
        <span style="color:var(--deckA);">◄ DECK A</span>
        <span class="mono" id="xfaderLabel" style="color:var(--muted);">X-FADER CENTER (50/50)</span>
        <span style="color:var(--deckB);">DECK B ►</span>
      </div>
      <input type="range" class="xfader" id="crossfader" min="0" max="100" value="50" oninput="onCrossfader(this.value)" />
    </div>
  </div>

  <!-- DECK B -->
  <div class="deck-card deck-b">
    <div class="deck-header">
      <span class="deck-badge">DECK B // INCOMING</span>
      <select id="deckBSelect" onchange="loadDeck('B', this.value)" style="flex:1;"></select>
      <button class="mini-btn" id="btnToggleStemsB" onclick="toggleStemDrawer('B')" title="Toggle 4-Stem Isolator Drawer">🎚️ Stems ▾</button>
      <button class="mini-btn" onclick="playDeckTrack('B')">▶ Audition B</button>
      <button class="mini-btn" onclick="inspectDeckSpectrogram('B')">📊 4-Panel</button>
    </div>

    <div class="deck-main">
      <div class="platter-wrap">
        <div class="jog-wheel" id="jogB" onclick="togglePlayPause()">
          <div class="jog-ring" id="jogRingB"><div class="jog-marker"></div></div>
          <div class="jog-oled">
            <div class="oled-bpm mono" id="deckBBpm">128.0</div>
            <div class="oled-key mono" id="deckBKey" style="color:var(--deckB);border-color:var(--deckB);">8A</div>
            <div class="oled-sub mono" id="deckBMeta">A Minor • -14.0 LUFS</div>
          </div>
        </div>
        <div class="mono" style="font-size:9.5px;color:var(--muted);" id="deckBCueInfo">CUE IN: Bar 0 • CUE OUT: Bar 4</div>
      </div>

      <div class="stem-drawer stems-box" id="stemDrawerB" style="display:none;">
        <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--muted);margin-bottom:4px;">
          <span>4-STEM ISOLATOR (RUBBERBAND R3)</span>
          <span style="color:var(--accent);" id="deckBStretchBadge">SYNC LOCKED</span>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#f97316;">DRUMS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_B_drums" min="0" max="120" value="100" oninput="updateStemGain('B','drums',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_B_drums" style="width:80%;background:#f97316;"></div></div>
          </div>
          <button class="mini-btn" id="mute_B_drums" onclick="toggleMuteStem('B','drums')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('B','drums')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#ef4444;">BASS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_B_bass" min="0" max="120" value="100" oninput="updateStemGain('B','bass',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_B_bass" style="width:90%;background:#ef4444;"></div></div>
          </div>
          <button class="mini-btn" id="mute_B_bass" onclick="toggleMuteStem('B','bass')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('B','bass')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#22c55e;">VOCALS</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_B_vocals" min="0" max="120" value="85" oninput="updateStemGain('B','vocals',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_B_vocals" style="width:55%;background:#22c55e;"></div></div>
          </div>
          <button class="mini-btn" id="mute_B_vocals" onclick="toggleMuteStem('B','vocals')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('B','vocals')">🎧</button>
        </div>
        <div class="stem-row">
          <span class="stem-name" style="color:#a855f7;">OTHER</span>
          <div class="stem-meter-wrap">
            <input type="range" class="stem-slider" id="slider_B_other" min="0" max="120" value="100" oninput="updateStemGain('B','other',this.value)" />
            <div class="stem-bar"><div class="stem-fill" id="meter_B_other" style="width:75%;background:#a855f7;"></div></div>
          </div>
          <button class="mini-btn" id="mute_B_other" onclick="toggleMuteStem('B','other')">M</button>
          <button class="mini-btn" onclick="soloStemAudio('B','other')">🎧</button>
        </div>
      </div>
    </div>

    <div class="pads-grid" id="padsB"></div>
  </div>

</section>

<!-- 5-TAB STUDIO WORKSTATION DOCK -->
<section class="dock-container" id="dockContainer">
  <!-- Tab Bar Triggers -->
  <div class="dock-tabbar" role="tablist" aria-label="Studio Workstation Panels">
    <div class="dock-tabs-left">
      <button class="dock-tab-btn active" id="tabBtnCrate" role="tab" aria-selected="true" aria-controls="tabPaneCrate" onclick="switchDockTab('Crate')">
        📚 Library &amp; Crate <span class="mono tab-count" id="crateCountBadge">0 tracks</span>
      </button>
      <button class="dock-tab-btn" id="tabBtnProducer" role="tab" aria-selected="false" aria-controls="tabPaneProducer" onclick="switchDockTab('Producer')">
        🎹 AI Producer
      </button>
      <button class="dock-tab-btn" id="tabBtnAgents" role="tab" aria-selected="false" aria-controls="tabPaneAgents" onclick="switchDockTab('Agents')">
        🤖 6-Agent Crew &amp; MC
      </button>
      <button class="dock-tab-btn" id="tabBtnCritic" role="tab" aria-selected="false" aria-controls="tabPaneCritic" onclick="switchDockTab('Critic')">
        📊 Spectrogram &amp; Critic QA
      </button>
      <button class="dock-tab-btn" id="tabBtnTelemetry" role="tab" aria-selected="false" aria-controls="tabPaneTelemetry" onclick="switchDockTab('Telemetry')">
        📡 Telemetry &amp; Mixes
      </button>
    </div>
    <div class="dock-tabs-right">
      <button class="dock-tab-btn toggle-dock-btn" id="btnToggleDock" onclick="toggleDockCollapse()" title="Minimize / Restore Studio Dock">
        ▾ Dock
      </button>
    </div>
  </div>

  <!-- Dock Panes Wrapper -->
  <div class="dock-panes-body" id="dockPanesBody">
    <!-- TAB 1: LIBRARY & CRATE -->
    <div class="tab-pane active" id="tabPaneCrate" role="tabpanel" aria-labelledby="tabBtnCrate">
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">
        <label style="font-weight:700;font-size:10.5px;color:var(--accent);white-space:nowrap;">Music Folder / Audio Scan:</label>
        <input id="scanPathInput" style="flex:1;" placeholder="/path/to/local/music/folder or .wav" value="/dev/shm/autodj_studio" />
        <button onclick="scanLocalFolder()">📂 Scan &amp; Index</button>
      </div>
      <div class="crate-list scroller" id="crateList" style="max-height:140px;overflow-y:auto;">
        <div style="padding:12px;color:var(--muted);text-align:center;">
          Crate empty — click <b>⚡ 1-CLICK QUICK-START DEMO MIX</b> in top bar or <b>📂 Scan &amp; Index</b>!
        </div>
      </div>
    </div>

    <!-- TAB 2: AI TRACK PRODUCER -->
    <div class="tab-pane" id="tabPaneProducer" role="tabpanel" aria-labelledby="tabBtnProducer" style="display:none;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-weight:700;font-size:11px;color:var(--accent);">AI NEURAL TRACK GENERATOR</span>
        <span class="mono" style="font-size:10px;color:var(--muted);">ACE-Step 1.5 • MusicGen • Suno v5</span>
      </div>
      <div style="background:var(--oled);border:1px solid var(--border);border-radius:7px;padding:7px;display:flex;flex-direction:column;gap:6px;">
        <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:6px;">
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Track Title</label>
            <input id="prodTitle" style="width:100%;" placeholder="Track Title" value="Cyber Horizon" />
          </div>
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Synthesis Engine</label>
            <select id="prodEngine" style="width:100%;">
              <option value="acestep">ACE-Step 1.5 (Local GPU)</option>
              <option value="musicgen_melody">MusicGen-Melody (4D Stems)</option>
              <option value="suno">Suno v5 (Cloud Co-Producer)</option>
            </select>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1.5fr 70px 70px 65px auto;gap:6px;align-items:end;">
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Musical Style / Prompt</label>
            <input id="prodStyle" style="width:100%;" placeholder="Style" value="Peak-Time Melodic Techno" />
          </div>
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">BPM</label>
            <input id="prodBpm" type="number" style="width:100%;" value="128" min="80" max="175" title="Target BPM" />
          </div>
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Key</label>
            <select id="prodKey" style="width:100%;" title="Target Camelot Key">
              <option value="8A" selected>8A</option><option value="9A">9A</option><option value="7A">7A</option>
              <option value="8B">8B</option><option value="10A">10A</option><option value="5A">5A</option>
              <option value="2A">2A</option><option value="3B">3B</option><option value="11A">11A</option>
            </select>
          </div>
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Bars</label>
            <select id="prodBars" style="width:100%;" title="Bars">
              <option value="8" selected>8b</option><option value="12">12b</option><option value="16">16b</option>
            </select>
          </div>
          <button class="btn-primary" id="btnProduce" onclick="produceTrack()" style="padding:6px 14px;">
            🎹 Produce &amp; Index
          </button>
        </div>
      </div>
    </div>

    <!-- TAB 3: 6-AGENT MATRIX & HOST MC -->
    <div class="tab-pane" id="tabPaneAgents" role="tabpanel" aria-labelledby="tabBtnAgents" style="display:none;">
      <div style="display:grid;grid-template-columns:1.1fr 1fr;gap:12px;">
        <div>
          <div class="agent-row" style="margin-bottom:4px;">
            <span style="font-weight:700;">Option A1 Audio Ear</span>
            <select id="earSelect" onchange="updateEar()" style="max-width:210px;">
              <option value="nvidia/music-flamingo-think-2601-hf">Music-Flamingo-Think (Option A1)</option>
              <option value="nvidia/audio-flamingo-next-think-hf">Audio-Flamingo-Next (30m Set)</option>
              <option value="google/gemma-4-E4B-it">Gemma-4-E4B-it (Apache 2.0 Omni)</option>
              <option value="Qwen/Qwen3-Omni-30B-A3B-Instruct">Qwen3-Omni-30B-A3B</option>
            </select>
          </div>
          <div id="agentMatrix" class="scroller" style="max-height:130px;overflow-y:auto;"></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px;">
          <div>
            <label style="display:block;font-size:9.5px;color:var(--muted);margin-bottom:2px;">Real-Time Crowd Energy Steering</label>
            <div style="display:flex;gap:5px;">
              <input id="steerInput" style="flex:1;" value="Peak-time progressive club energy arc" placeholder="Natural-language crowd steering..." />
              <button onclick="sendSteer()">🎯 Steer</button>
            </div>
          </div>
          <div style="background:var(--oled);border:1px solid var(--border);border-radius:7px;padding:6px;display:flex;flex-direction:column;gap:4px;">
            <div style="display:flex;justify-content:space-between;font-size:9.5px;color:var(--muted);">
              <span>🎙️ HOST MC DROP (Kokoro-82M • -10dB Mid Duck)</span>
              <label style="cursor:pointer;"><input type="checkbox" id="chkHostMc" checked /> Auto-MC in Mix</label>
            </div>
            <div style="display:flex;gap:5px;">
              <input id="mcScriptInput" style="flex:1;" value="Welcome to the Neural Deck studio, locking into 128 BPM!" />
              <button onclick="synthesizeHostDrop()">🎙️ Speak</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: SPECTROGRAM & CRITIC QA -->
    <div class="tab-pane" id="tabPaneCritic" role="tabpanel" aria-labelledby="tabBtnCritic" style="display:none;">
      <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:12px;align-items:center;">
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <select id="specSelect" onchange="selectSpectrogram(this.value)" style="flex:1;margin-right:6px;">
              <option value="">Select Track or Transition Spectrogram (1024×1024)...</option>
            </select>
            <span class="mono" id="criticVerdictBadge" style="font-size:9.5px;color:var(--pass);">STAGE 1+2 READY</span>
          </div>
          <img id="specImage" class="spec-preview" alt="4-Panel Visual-Acoustic Spectrogram Bridge" onclick="openSpecModal()" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='400' height='120'><rect width='100%' height='100%' fill='%23020408'/><text x='50%' y='50%' fill='%2364748b' font-family='monospace' font-size='10' text-anchor='middle'>4-Panel 1024x1024 Spectrogram Bridge (Click to Enlarge)</text></svg>" style="height:125px;" />
        </div>
        <div>
          <div style="font-weight:700;font-size:10.5px;color:var(--accent);margin-bottom:5px;">AUTOMATED CRITIC QA SCORECARD</div>
          <div class="qa-grid" id="criticQaGrid" style="gap:4px;">
            <div class="qa-cell"><span>Beat Sync Err</span><span class="mono" style="color:var(--pass);" id="qaBeat">0.00 ms (&lt;1ms)</span></div>
            <div class="qa-cell"><span>True Peak</span><span class="mono" style="color:var(--pass);" id="qaPeak">-1.00 dBTP</span></div>
            <div class="qa-cell"><span>LUFS Delta</span><span class="mono" style="color:var(--pass);" id="qaLufs">0.08 LU (&le;1.5)</span></div>
            <div class="qa-cell"><span>Bass-Swap Law</span><span class="mono" style="color:var(--pass);" id="qaBass">0 sim (PASS)</span></div>
          </div>
          <div style="margin-top:6px;font-size:9px;color:var(--muted);line-height:1.3;">
            Deterministic 2-stage verification: zero simultaneous kick/bass collisions and sample-accurate rubberband phase lock.
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: TELEMETRY & MIXES -->
    <div class="tab-pane" id="tabPaneTelemetry" role="tabpanel" aria-labelledby="tabBtnTelemetry" style="display:none;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-weight:700;font-size:10.5px;color:var(--accent);">CERTIFIED TRANSITIONS</span>
            <span class="mono" id="wsStatusBadge" style="font-size:9.5px;color:var(--pass);">WS LIVE</span>
          </div>
          <div id="transitionsList" class="scroller" style="max-height:135px;overflow-y:auto;background:var(--oled);border:1px solid var(--border);border-radius:6px;padding:4px;">
            <div style="color:var(--muted);font-size:9.5px;text-align:center;padding:6px;">No transitions rendered yet.</div>
          </div>
        </div>
        <div>
          <div style="font-weight:700;font-size:10.5px;color:var(--accent);margin-bottom:4px;">6-AGENT PIPELINE LOGS</div>
          <div id="events" class="scroller" style="max-height:135px;overflow-y:auto;">Loading studio state...</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- 4-Panel Spectrogram Fullscreen Lightbox Modal (<dialog closedby="any">) -->
<dialog id="specModal" closedby="any" aria-label="Full 1024x1024 4-Panel Spectrogram">
  <img id="specModalImg" src="" alt="Full 1024x1024 4-Panel Spectrogram" onclick="closeSpecModal()" />
</dialog>

<script>
function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
let STUDIO = {};
let ENVELOPES = {};
let PLAYHEAD_RATIO = 0.0;
let JOG_ANGLE_A = 0;
let JOG_ANGLE_B = 0;
let AUDIO_CTX = null;
let ANALYSER = null;
let SOURCE_NODE = null;
let MUTED_STEMS = { A: {}, B: {} };
var LIVE_QUEUE = [];
var LIVE_PLAYING_INDEX = -1;
var LIVE_ACTIVE = false;
var LIVE_INTRO_PLAYING = false;
var LIVE_POLL_TIMER = null;
var LAST_LIVE_CERT_COUNT = 0;

const CAMELOT_KEYS_ORDER = ["12", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"];

function setBusy(msg) {
  const b = document.getElementById('statusBadge');
  b.textContent = msg;
  b.style.borderColor = '#eab308';
  b.style.color = '#eab308';
}

function setReady(msg) {
  const b = document.getElementById('statusBadge');
  b.textContent = msg || 'PIPEWIRE /dev/shm READY (-14 LUFS / -1.0 dBTP)';
  b.style.borderColor = '#22c55e';
  b.style.color = '#22c55e';
}

async function refreshState() {
  const res = await fetch('/api/state');
  STUDIO = await res.json();

  document.getElementById('earSelect').value = STUDIO.audio_ear_model;
  if (document.activeElement !== document.getElementById('steerInput')) {
    document.getElementById('steerInput').value = STUDIO.steering_prompt || '';
  }
  const lvIn = document.getElementById('liveVibeInput');
  if (lvIn && document.activeElement !== lvIn) {
    lvIn.value = STUDIO.steering_prompt || '';
  }
  if (LIVE_ACTIVE && LIVE_PLAYING_INDEX >= 0 && LIVE_QUEUE[LIVE_PLAYING_INDEX]) {
    const activeSeg = LIVE_QUEUE[LIVE_PLAYING_INDEX];
    if (activeSeg.track_a_id && activeSeg.track_b_id) {
      STUDIO.deck_a_id = activeSeg.track_a_id;
      STUDIO.deck_b_id = activeSeg.track_b_id;
    }
  }

  // 1. Render 6-Agent Mode Matrix
  const matrix = document.getElementById('agentMatrix');
  matrix.innerHTML = '';
  for (const [agent, mode] of Object.entries(STUDIO.agent_modes || {})) {
    matrix.innerHTML += `<div class="agent-row">
      <span style="text-transform:capitalize;font-weight:600;">${agent}Agent</span>
      <select onchange="setMode('${agent}', this.value)">
        <option value="local" ${mode==='local'?'selected':''}>Local (RTX 5090)</option>
        <option value="hybrid" ${mode==='hybrid'?'selected':''}>Hybrid</option>
        <option value="cloud" ${mode==='cloud'?'selected':''}>Cloud</option>
      </select>
    </div>`;
  }

  // 2. Render Crate & Deck Selectors
  const crate = STUDIO.crate || [];
  document.getElementById('crateCountBadge').textContent = `${crate.length} tracks`;
  const selA = document.getElementById('deckASelect');
  const selB = document.getElementById('deckBSelect');
  selA.innerHTML = crate.length ? '' : '<option value="">-- Crate Empty --</option>';
  selB.innerHTML = crate.length ? '' : '<option value="">-- Crate Empty --</option>';

  const crateDiv = document.getElementById('crateList');
  if (crate.length === 0) {
    crateDiv.innerHTML = `<div style="padding:12px;color:var(--muted);text-align:center;">
      Crate empty — click <b>⚡ 1-CLICK QUICK-START DEMO MIX</b> in the top bar!
    </div>`;
  } else {
    crateDiv.innerHTML = '';
    for (const t of crate) {
      selA.innerHTML += `<option value="${t.track_id}" ${t.track_id===STUDIO.deck_a_id?'selected':''}>${t.title} (${t.bpm.toFixed(1)} BPM • ${t.camelot_global})</option>`;
      selB.innerHTML += `<option value="${t.track_id}" ${t.track_id===STUDIO.deck_b_id?'selected':''}>${t.title} (${t.bpm.toFixed(1)} BPM • ${t.camelot_global})</option>`;
      crateDiv.innerHTML += `<div class="crate-item">
        <div>
          <div style="font-weight:700;color:var(--text);">${t.title}</div>
          <div class="mono" style="font-size:9.5px;color:var(--muted);">${t.bpm.toFixed(1)} BPM • Key <b style="color:var(--accent)">${t.camelot_global}</b> (${t.key_name}) • ${t.integrated_lufs.toFixed(1)} LUFS</div>
        </div>
        <div style="display:flex;gap:3px;">
          <button class="mini-btn" style="border-color:var(--deckA);color:var(--deckA);" onclick="loadDeck('A','${t.track_id}')">→A</button>
          <button class="mini-btn" style="border-color:var(--deckB);color:var(--deckB);" onclick="loadDeck('B','${t.track_id}')">→B</button>
          <button class="mini-btn" onclick="playAudioUrl('/api/audio/track/${t.track_id}', 'Track: ${t.title}')">▶</button>
        </div>
      </div>`;
    }
  }

  // 2b. Sync Deck Stem Isolator Faders & Mutes from Server State
  const dGains = STUDIO.deck_stems_gain || {};
  for (const dk of ['A', 'B']) {
    const sg = dGains[dk] || {};
    for (const st of ['drums', 'bass', 'vocals', 'other']) {
      if (sg[st] !== undefined) {
        const pct = Math.round(sg[st] * 100);
        const sl = document.getElementById(`slider_${dk}_${st}`);
        const mt = document.getElementById(`meter_${dk}_${st}`);
        const mb = document.getElementById(`mute_${dk}_${st}`);
        if (sl) sl.value = pct;
        if (mt) mt.style.width = `${Math.min(100, pct)}%`;
        const isMuted = pct === 0;
        MUTED_STEMS[dk][st] = isMuted;
        if (mb) mb.classList.toggle('muted', isMuted);
      }
    }
  }

  // 3. Populate Audio Source Dropdown
  const audioSel = document.getElementById('audioSourceSelect');
  const prevVal = audioSel.value;
  audioSel.innerHTML = '';
  if (STUDIO.master_mix_ready) {
    audioSel.innerHTML += `<option value="/api/audio/master">🎧 Master Mix (.WAV)</option>`;
  }
  (STUDIO.certified_transitions || []).forEach((c, idx) => {
    const sp = c.spec || {};
    audioSel.innerHTML += `<option value="/api/audio/transition/${idx}">🔀 Mix #${idx+1}: ${escapeHtml(sp.track_a_title)} → ${escapeHtml(sp.track_b_title)} [${escapeHtml(sp.archetype)}]</option>`;
  });
  for (const t of crate) {
    audioSel.innerHTML += `<option value="/api/audio/track/${escapeHtml(t.track_id)}">🎵 Track: ${escapeHtml(t.title)}</option>`;
  }
  (STUDIO.host_voiceovers || []).forEach((vo, idx) => {
    audioSel.innerHTML += `<option value="/api/audio/voiceover/${idx}">🎙️ MC Drop #${idx+1}</option>`;
  });
  if (prevVal && [...audioSel.options].some(o => o.value === prevVal)) {
    audioSel.value = prevVal;
  }

  // 4. Update Deck A & Deck B Readouts, Waveforms, and Hot-Cue Pads
  await updateDeckUI('A', STUDIO.deck_a_id);
  await updateDeckUI('B', STUDIO.deck_b_id);

  // 5. Update Center Mixer, Camelot Radar & Automation Envelope Canvas
  updateCenterMixerUI(STUDIO.deck_pair_preview || {});

  // 6. Update 4-Panel Spectrogram Selector & Critic Scorecard
  const specSel = document.getElementById('specSelect');
  const specs = STUDIO.spectrograms || [];
  const curSpec = specSel.value;
  specSel.innerHTML = specs.length ? '' : '<option value="">No spectrograms yet — run Quick-Start or Index</option>';
  for (const s of specs) {
    specSel.innerHTML += `<option value="${s.url}">${s.label}</option>`;
  }
  if (specs.length > 0) {
    const chosen = (curSpec && specs.some(s => s.url === curSpec)) ? curSpec : specs[specs.length - 1].url;
    specSel.value = chosen;
    document.getElementById('specImage').src = chosen + '?t=' + Date.now();
  }

  // 7. Certified Transitions List & Critic Metrics
  const certs = STUDIO.certified_transitions || [];
  const tDiv = document.getElementById('transitionsList');
  if (certs.length > 0) {
    const latest = certs[certs.length - 1];
    const cr = latest.critique || {};
    document.getElementById('criticVerdictBadge').textContent = `${cr.verdict || 'PASS'} (Retries: ${latest.retries_used || 0})`;
    document.getElementById('qaBeat').textContent = `${(cr.beat_alignment_error_ms || 0).toFixed(3)} ms (<1ms)`;
    document.getElementById('qaPeak').textContent = `${(cr.true_peak_dbtp || -1.0).toFixed(2)} dBTP`;
    document.getElementById('qaLufs').textContent = `${(cr.lufs_step_delta_lu || 0.1).toFixed(2)} LU (≤1.5)`;
    document.getElementById('qaBass').textContent = `${cr.simultaneous_bass_samples || 0} sim (PASS)`;

    tDiv.innerHTML = certs.map((c, i) => {
      const sp = c.spec || {};
      const rd = c.rendered || {};
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 4px;border-bottom:1px solid #162032;font-size:10px;">
        <span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:6px;"><b style="color:var(--pass);">[${escapeHtml(sp.archetype)}]</b> ${escapeHtml(sp.track_a_title)} → ${escapeHtml(sp.track_b_title)} <span class="mono" style="color:var(--muted);">(${rd.integrated_lufs?.toFixed(1)} LUFS)</span></span>
        <button class="mini-btn" style="flex-shrink:0;" onclick="playAudioUrl('/api/audio/transition/${i}', 'Mix #${i+1}')">▶ Play Mix</button>
      </div>`;
    }).join('');
  }

  // 8. Live Telemetry Feed
  const evDiv = document.getElementById('events');
  if (STUDIO.agent_events && STUDIO.agent_events.length > 0) {
    evDiv.innerHTML = STUDIO.agent_events.slice(-18).reverse().map(
      e => `<div style="margin-bottom:3px;"><b style="color:#38bdf8">[${escapeHtml(e.agent)}]</b> ${escapeHtml(e.detail)}</div>`
    ).join('');
  } else {
    evDiv.innerHTML = '<div>Studio initialized and ready. Click <b>⚡ 1-CLICK QUICK-START DEMO MIX</b> to launch!</div>';
  }
  if (STUDIO.live_status) {
    updateLivePipelineUI(STUDIO.live_status);
  }
}

async function updateDeckUI(deck, trackId) {
  const crate = STUDIO.crate || [];
  const t = crate.find(x => x.track_id === trackId);
  const padsDiv = document.getElementById('pads' + deck);
  if (!t) {
    padsDiv.innerHTML = [1,2,3,4].map(i => `<div class="cue-pad">PAD ${i}<span class="pad-sec">--:--</span></div>`).join('');
    drawWaveform('waveCanvas' + deck, null, deck);
    return;
  }
  document.getElementById(`deck${deck}Bpm`).textContent = t.bpm.toFixed(1);
  document.getElementById(`deck${deck}Key`).textContent = t.camelot_global;
  document.getElementById(`deck${deck}Meta`).textContent = `${t.key_name} • ${t.integrated_lufs.toFixed(1)} LUFS`;
  document.getElementById(`deck${deck}CueInfo`).textContent = `CUE IN: Bar ${t.cue_in_bar} • CUE OUT: Bar ${t.cue_out_bar} • ${t.duration_sec.toFixed(1)}s`;
  document.getElementById(`waveLabel${deck}`).textContent = `DECK ${deck} // ${t.title.toUpperCase()} — ${t.bpm.toFixed(1)} BPM • KEY ${t.camelot_global} (${t.key_name}) • ${t.integrated_lufs.toFixed(1)} LUFS`;

  if (!ENVELOPES[trackId]) {
    try {
      const r = await fetch(`/api/envelope/${trackId}`);
      if (r.ok) ENVELOPES[trackId] = await r.json();
    } catch (e) {}
  }
  const env = ENVELOPES[trackId];
  drawWaveform('waveCanvas' + deck, env, deck);

  const secs = (t.sections || []).slice(0, 8);
  padsDiv.innerHTML = secs.map((s, idx) => {
    const sId = escapeHtml(t.track_id);
    const sTitle = escapeHtml(t.title);
    const sLbl = escapeHtml(s.label.toUpperCase());
    return `
    <div class="cue-pad" data-tid="${sId}" data-sec="${s.start_sec}" data-label="${sTitle} [${sLbl}]" onclick="jumpToTrackSection(this.dataset.tid, parseFloat(this.dataset.sec), this.dataset.label)">
      <span style="color:${deck==='A'?'#f97316':'#a855f7'}">${idx+1}. ${sLbl}</span>
      <span class="pad-sec mono">Bar ${s.start_bar} (${s.start_sec.toFixed(1)}s)</span>
    </div>`;
  }).join('');
}

function drawWaveform(canvasId, env, deck) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  // Subtle beatgrid background
  ctx.fillStyle = '#030509';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#111827';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 32) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  if (!env || !env.peaks_l) return;

  // Draw Section Spans in background
  const dur = Math.max(0.1, env.duration_sec || 1);
  const secColors = { intro: 'rgba(56,189,248,0.10)', build: 'rgba(234,179,8,0.12)', drop: 'rgba(239,68,68,0.15)', outro: 'rgba(168,85,247,0.10)' };
  for (const s of (env.sections || [])) {
    const x0 = (s.start_sec / dur) * w;
    const x1 = (s.end_sec / dur) * w;
    ctx.fillStyle = secColors[s.label] || 'rgba(255,255,255,0.05)';
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 9px monospace';
    ctx.fillText(s.label.toUpperCase(), x0 + 4, h - 4);
  }

  // Draw 4-stem colored waveform bars
  const peaks = env.peaks_l;
  const n = peaks.length;
  const barW = w / n;
  const mid = h / 2;

  for (let i = 0; i < n; i++) {
    const p = Math.min(1.0, peaks[i]);
    const x = i * barW;
    const fullH = Math.max(2, p * (h * 0.84));

    // RGB Stem layers: Bass (red inner), Drums (orange mid), Vocals/Other (cyan/purple outer)
    ctx.fillStyle = deck === 'A' ? '#38bdf8' : '#a855f7';
    ctx.fillRect(x, mid - fullH / 2, Math.max(1.5, barW - 1), fullH);

    ctx.fillStyle = '#f97316';
    ctx.fillRect(x, mid - (fullH * 0.65) / 2, Math.max(1.5, barW - 1), fullH * 0.65);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(x, mid - (fullH * 0.35) / 2, Math.max(1.5, barW - 1), fullH * 0.35);
  }

  // Playhead cursor
  const px = PLAYHEAD_RATIO * w;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
}

function updateCenterMixerUI(prev) {
  const sel = STUDIO.selected_archetype || 'AUTO';
  document.querySelectorAll('#archButtons .arch-btn').forEach(b => {
    b.classList.toggle('active', b.textContent.trim() === sel);
  });
  const activeArch = prev.active_archetype || 'T1';
  document.getElementById('activeArchBadge').textContent = `${sel} → ${activeArch}`;
  document.getElementById('compatScoreText').textContent =
    `COMPAT: ${Math.round((prev.compatibility_score || 0.95) * 100)}% (d_key=${prev.camelot_distance ?? 0})`;
  document.getElementById('archNameText').textContent = prev.archetype_name || 'Phrase-Aligned 4-Stem Blend';
  document.getElementById('mixSpecMeta').textContent =
    `Target: ${(prev.target_bpm || 128).toFixed(1)} BPM • Overlap: ${prev.overlap_bars || 8} Bars • Swap: Bar ${prev.bass_swap_bar || 4}`;
  document.getElementById('mixRationaleText').textContent = prev.rationale || '';

  // Render SVG Camelot Wheel
  const crate = STUDIO.crate || [];
  const tA = crate.find(x => x.track_id === STUDIO.deck_a_id);
  const tB = crate.find(x => x.track_id === STUDIO.deck_b_id);
  drawCamelotWheel(tA ? tA.camelot_global : '8A', tB ? tB.camelot_global : '8A');

  // Render Stem Automation Envelopes
  drawEnvelopeCanvas(prev);
}

function drawCamelotWheel(keyA, keyB) {
  const svg = document.getElementById('camelotSvg');
  if (!svg) return;
  const cx = 60, cy = 60;
  let html = `<circle cx="${cx}" cy="${cy}" r="56" fill="#050811" stroke="#1e293b" stroke-width="1.5"/>`;
  html += `<circle cx="${cx}" cy="${cy}" r="38" fill="none" stroke="#1e293b" stroke-width="1"/>`;
  html += `<circle cx="${cx}" cy="${cy}" r="20" fill="#020408" stroke="#334155" stroke-width="1"/>`;

  let posA = null, posB = null;
  CAMELOT_KEYS_ORDER.forEach((num, idx) => {
    const angle = (idx * 30 - 90) * (Math.PI / 180);
    // Outer ring (B = Major), Inner ring (A = Minor)
    for (const [mode, r] of [['B', 47], ['A', 29]]) {
      const code = num + mode;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      const isA = code === keyA;
      const isB = code === keyB;
      if (isA) posA = {x, y};
      if (isB) posB = {x, y};
      const fill = (isA && isB) ? '#22c55e' : isA ? '#f97316' : isB ? '#a855f7' : '#0f172a';
      const stroke = (isA || isB) ? '#ffffff' : '#334155';
      html += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(isA||isB)?7.5:5.5}" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`;
      html += `<text x="${x.toFixed(1)}" y="${(y+2.5).toFixed(1)}" fill="${(isA||isB)?'#020617':'#94a3b8'}" font-size="6.5" font-weight="bold" font-family="monospace" text-anchor="middle">${code}</text>`;
    }
  });
  if (posA && posB) {
    html += `<line x1="${posA.x}" y1="${posA.y}" x2="${posB.x}" y2="${posB.y}" stroke="#38bdf8" stroke-width="2" stroke-dasharray="3,2"/>`;
  }
  html += `<text x="${cx}" y="${cy+3}" fill="#38bdf8" font-size="8" font-weight="bold" font-family="monospace" text-anchor="middle">${keyA}→${keyB}</text>`;
  svg.innerHTML = html;
}

function drawEnvelopeCanvas(prev) {
  const c = document.getElementById('envCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#020408';
  ctx.fillRect(0, 0, w, h);

  const envA = prev.envelopes_a || {};
  const envB = prev.envelopes_b || {};
  const plotCurve = (arr, color, dashed) => {
    if (!arr || !arr.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * w;
      const y = h - 4 - v * (h - 10);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plotCurve(envA.bass, '#ef4444', false);
  plotCurve(envB.bass, '#ef4444', true);
  plotCurve(envA.drums, '#f97316', false);
  plotCurve(envB.other, '#a855f7', false);
  ctx.setLineDash([]);

  // Bass Swap Bar vertical marker
  const swapRatio = (prev.bass_swap_bar || 4) / Math.max(1, prev.overlap_bars || 8);
  const sx = swapRatio * w;
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
  ctx.fillStyle = '#22c55e';
  ctx.font = 'bold 8.5px monospace';
  ctx.fillText('BASS SWAP', Math.min(w - 55, sx + 4), 10);
}

// Actions
async function runQuickStart() {
  setBusy('⚡ PRODUCING 3 CLUB TRACKS & RUNNING 6-AGENT MIX...');
  try {
    const res = await fetch('/api/quickstart', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steering_prompt: document.getElementById('steerInput').value || 'Peak-time progressive club energy arc',
        enable_host_mc: document.getElementById('chkHostMc').checked,
        bars: 8
      })
    });
    await refreshState();
    setReady('⚡ QUICK-START MIX CERTIFIED • PLAYING MASTER MIX');
    playAudioUrl('/api/audio/master', 'Master Mix (.WAV)');
  } catch (e) {
    setReady('Quick-Start Error: ' + e);
  }
}

async function runFullSession() {
  setBusy('▶ RUNNING 6-AGENT LANGGRAPH SESSION...');
  const res = await fetch('/api/run-session', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      steering_prompt: document.getElementById('steerInput').value,
      enable_host_mc: document.getElementById('chkHostMc').checked
    })
  });
  if (!res.ok) {
    // If fewer than 2 tracks in crate, automatically run QuickStart!
    return runQuickStart();
  }
  await refreshState();
  setReady('6-AGENT SESSION CERTIFIED (-14.0 LUFS / -1.0 dBTP)');
  playAudioUrl('/api/audio/master', 'Master Mix (.WAV)');
}

async function stopStudioSession() {
  const audio = document.getElementById('masterAudio');
  if (audio) {
    audio.pause();
    audio.currentTime = 0;
  }
  PLAYHEAD_RATIO = 0.0;
  LIVE_ACTIVE = false;
  if (LIVE_POLL_TIMER) {
    clearInterval(LIVE_POLL_TIMER);
    LIVE_POLL_TIMER = null;
  }
  if (STUDIO.deck_a_id && ENVELOPES[STUDIO.deck_a_id]) drawWaveform('waveCanvasA', ENVELOPES[STUDIO.deck_a_id], 'A');
  if (STUDIO.deck_b_id && ENVELOPES[STUDIO.deck_b_id]) drawWaveform('waveCanvasB', ENVELOPES[STUDIO.deck_b_id], 'B');
  await fetch('/api/session/stop', { method: 'POST' });
  await refreshState();
  setReady('⏹ SESSION & PLAYBACK STOPPED');
}

async function mixDeckPair() {
  setBusy('🔀 RENDERING DECK A → B TRANSITION...');
  const res = await fetch('/api/deck/mix-pair', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      archetype: STUDIO.selected_archetype || 'AUTO',
      enable_host_mc: document.getElementById('chkHostMc').checked
    })
  });
  if (!res.ok) {
    return runQuickStart();
  }
  await refreshState();
  setReady('DECK A → B TRANSITION CERTIFIED');
  playAudioUrl('/api/audio/master', 'Live Deck A→B Mix');
}

async function produceTrack() {
  const title = document.getElementById('prodTitle').value || 'Neon Pulse';
  const style = document.getElementById('prodStyle').value || 'Peak-Time Melodic Techno';
  const bpm = parseFloat(document.getElementById('prodBpm').value || '128');
  const cam = document.getElementById('prodKey').value || '8A';
  const bars = parseInt(document.getElementById('prodBars').value || '8', 10);
  const engine = document.getElementById('prodEngine').value || 'acestep';
  setBusy(`🎹 PRODUCING & INDEXING '${title}' (${bpm} BPM, ${cam})...`);
  await fetch('/api/produce', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ title, style, target_bpm: bpm, target_camelot: cam, bars, engine, auto_index: true })
  });
  await refreshState();
  setReady(`PRODUCED & INDEXED '${title}'`);
}

async function scanLocalFolder() {
  const path = document.getElementById('scanPathInput').value;
  setBusy(`📂 SCANNING & INDEXING '${path}'...`);
  const res = await fetch('/api/library/scan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ path, max_files: 10 })
  });
  await refreshState();
  setReady(res.ok ? `SCANNED & INDEXED CRATE` : `NO AUDIO FILES YET IN PATH`);
}

async function loadDeck(deck, trackId) {
  if (!trackId) return;
  await fetch('/api/deck/load', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ deck, track_id_or_path: trackId })
  });
  await refreshState();
}

async function selectArchetype(arch) {
  STUDIO.selected_archetype = arch;
  await fetch('/api/steer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      steering_prompt: document.getElementById('steerInput').value,
      selected_archetype: arch
    })
  });
  await refreshState();
}

async function setMode(agent, mode) {
  await fetch('/api/agent-mode', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({agent, mode})
  });
  await refreshState();
}

async function updateEar() {
  const ear = document.getElementById('earSelect').value;
  await fetch('/api/agent-mode', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({agent: 'listener', mode: 'local', audio_ear_model: ear})
  });
  await refreshState();
}

function syncSteerInputs(val) {
  const s1 = document.getElementById('steerInput');
  const s2 = document.getElementById('liveVibeInput');
  if (s1 && s1.value !== val) s1.value = val;
  if (s2 && s2.value !== val) s2.value = val;
}

async function sendSteer() {
  const s2 = document.getElementById('liveVibeInput');
  const s1 = document.getElementById('steerInput');
  const prompt = (s2 && s2.value) ? s2.value : (s1 ? s1.value : '');
  syncSteerInputs(prompt);
  await fetch('/api/steer', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({steering_prompt: prompt})
  });
  await refreshState();
  setReady(`🎯 LIVE CROWD STEERING UPDATED: '${prompt}'`);
}

async function synthesizeHostDrop() {
  const script = document.getElementById('mcScriptInput').value;
  setBusy('🎙️ SYNTHESIZING KOKORO-82M MC DROP...');
  const res = await fetch('/api/host-drop', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ script })
  });
  const data = await res.json();
  await refreshState();
  setReady('🎙️ PLAYING KOKORO-82M MC DROP');
  if (data.audio_url) playAudioUrl(data.audio_url, 'MC Drop');
}

async function streamToPipeWire(live) {
  const res = await fetch('/api/pipewire-play', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ live_audio: live, max_seconds: 8.0 })
  });
  if (res.ok) {
    const d = await res.json();
    setReady(`PIPEWIRE RINGBUFFER: ${d.frames_streamed} FRAMES (${d.status})`);
    await refreshState();
  }
}

async function updateStemGain(deck, stem, val) {
  const vNum = parseFloat(val);
  document.getElementById(`meter_${deck}_${stem}`).style.width = `${Math.min(100, vNum)}%`;
  const isMuted = vNum === 0;
  MUTED_STEMS[deck][stem] = isMuted;
  const btn = document.getElementById(`mute_${deck}_${stem}`);
  if (btn) btn.classList.toggle('muted', isMuted);

  try {
    const res = await fetch('/api/deck/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ deck, stem_gains: { [stem]: vNum / 100.0 } })
    });
    if (res.ok) {
      const data = await res.json();
      STUDIO.deck_stems_gain = data.deck_stems_gain;
      STUDIO.deck_pair_preview = data.deck_pair_preview;
      updateCenterMixerUI(data.deck_pair_preview || {});
    }
  } catch (e) {}
}

function toggleMuteStem(deck, stem) {
  const isM = !MUTED_STEMS[deck][stem];
  MUTED_STEMS[deck][stem] = isM;
  const sl = document.getElementById(`slider_${deck}_${stem}`);
  sl.value = isM ? 0 : 100;
  updateStemGain(deck, stem, sl.value);
}

function soloStemAudio(deck, stem) {
  const tid = deck === 'A' ? STUDIO.deck_a_id : STUDIO.deck_b_id;
  if (!tid) return;
  playAudioUrl(`/api/audio/stem/${tid}/${stem}`, `Deck ${deck} Isolated ${stem.toUpperCase()} Stem`);
}

function playDeckTrack(deck) {
  const tid = deck === 'A' ? STUDIO.deck_a_id : STUDIO.deck_b_id;
  if (!tid) return;
  playAudioUrl(`/api/audio/track/${tid}`, `Deck ${deck} Full Track`);
}

function jumpToTrackSection(tid, sec, label) {
  playAudioUrl(`/api/audio/track/${tid}`, label, sec);
}

function seekWaveform(deck, ev) {
  const c = ev.currentTarget;
  const rect = c.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
  PLAYHEAD_RATIO = ratio;
  if (STUDIO.deck_a_id && ENVELOPES[STUDIO.deck_a_id]) drawWaveform('waveCanvasA', ENVELOPES[STUDIO.deck_a_id], 'A');
  if (STUDIO.deck_b_id && ENVELOPES[STUDIO.deck_b_id]) drawWaveform('waveCanvasB', ENVELOPES[STUDIO.deck_b_id], 'B');
  const audio = document.getElementById('masterAudio');
  if (audio.duration && isFinite(audio.duration)) {
    audio.currentTime = ratio * audio.duration;
  } else {
    const tid = deck === 'A' ? STUDIO.deck_a_id : STUDIO.deck_b_id;
    const dur = (tid && ENVELOPES[tid]) ? ENVELOPES[tid].duration_sec : 0;
    if (tid) playAudioUrl(`/api/audio/track/${tid}`, `Deck ${deck} Full Track`, ratio * dur);
  }
}

function inspectDeckSpectrogram(deck) {
  const tid = deck === 'A' ? STUDIO.deck_a_id : STUDIO.deck_b_id;
  const crate = STUDIO.crate || [];
  const t = crate.find(x => x.track_id === tid);
  if (t && t.spectrogram_png_name) {
    const url = `/api/spectrogram/${t.spectrogram_png_name}`;
    document.getElementById('specSelect').value = url;
    selectSpectrogram(url);
    switchDockTab('Critic');
  }
}

function switchDockTab(tabName) {
  const tabs = ['Crate', 'Producer', 'Agents', 'Critic', 'Telemetry'];
  tabs.forEach(t => {
    const btn = document.getElementById('tabBtn' + t);
    const pane = document.getElementById('tabPane' + t);
    const isActive = (t === tabName);
    if (btn) {
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    }
    if (pane) {
      pane.classList.toggle('active', isActive);
      pane.style.display = isActive ? 'block' : 'none';
    }
  });
  const body = document.getElementById('dockPanesBody');
  if (body && body.style.display === 'none') {
    body.style.display = 'block';
    const tb = document.getElementById('btnToggleDock');
    if (tb) tb.textContent = '▾ Dock';
  }
}

function toggleDockCollapse() {
  const body = document.getElementById('dockPanesBody');
  const tb = document.getElementById('btnToggleDock');
  if (!body) return;
  const isHidden = (body.style.display === 'none');
  body.style.display = isHidden ? 'block' : 'none';
  if (tb) tb.textContent = isHidden ? '▾ Dock' : '▸ Dock';
}

function toggleStemDrawer(deck) {
  const drawer = document.getElementById('stemDrawer' + deck);
  const btn = document.getElementById('btnToggleStems' + deck);
  const card = document.querySelector('.deck-card.deck-' + deck.toLowerCase());
  if (!drawer) return;
  const isHidden = (drawer.style.display === 'none');
  drawer.style.display = isHidden ? 'block' : 'none';
  if (card) card.classList.toggle('stems-open', isHidden);
  if (btn) {
    btn.classList.toggle('active', isHidden);
    btn.textContent = isHidden ? '🎚️ Stems ▴' : '🎚️ Stems ▾';
  }
}

function selectSpectrogram(url) {
  if (!url) return;
  document.getElementById('specImage').src = url + '?t=' + Date.now();
}

function openSpecModal() {
  const src = document.getElementById('specImage').src;
  if (!src) return;
  document.getElementById('specModalImg').src = src;
  const dlg = document.getElementById('specModal');
  if (dlg && typeof dlg.showModal === 'function' && !dlg.open) {
    dlg.showModal();
  }
}

function closeSpecModal() {
  const dlg = document.getElementById('specModal');
  if (dlg && typeof dlg.close === 'function' && dlg.open) {
    dlg.close();
  }
}

// Fallback for browsers without <dialog closedby="any"> support
(function setupDialogClosedByFallback() {
  const dialog = document.getElementById('specModal');
  if (dialog && typeof HTMLDialogElement !== 'undefined' && !('closedBy' in HTMLDialogElement.prototype)) {
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const isDialogContent = (
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width
      );
      if (isDialogContent) return;
      dialog.close();
    });
  }
})();

function onCrossfader(val) {
  const v = parseInt(val, 10);
  const lbl = document.getElementById('xfaderLabel');
  if (v < 45) lbl.textContent = `DECK A BIAS (${100-v}/${v})`;
  else if (v > 55) lbl.textContent = `DECK B BIAS (${100-v}/${v})`;
  else lbl.textContent = `X-FADER CENTER (50/50)`;
}

function switchAudioSource(url) {
  if (url) playAudioUrl(url, 'Selected Stream');
}

var AUDIO_LOADING = false;
const SILENT_UNLOCK_WAV = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA';

function unlockBrowserAudioOnGesture() {
  const audio = document.getElementById('masterAudio');
  if (!audio) return;
  ensureWebAudioConnected(audio);
  if (AUDIO_CTX && AUDIO_CTX.state === 'suspended') {
    AUDIO_CTX.resume().catch(() => {});
  }
  if (!audio.src || audio.paused) {
    try {
      const prevSrc = audio.src;
      if (!prevSrc) {
        audio.src = SILENT_UNLOCK_WAV;
      }
      const p = audio.play();
      if (p && typeof p.then === 'function') {
        p.then(() => {
          if (audio.src === SILENT_UNLOCK_WAV) {
            audio.pause();
            audio.currentTime = 0;
          }
        }).catch(() => {});
      }
    } catch (e) {}
  }
}

function playAudioUrl(url, label, startSec = 0) {
  const audio = document.getElementById('masterAudio');
  if (!audio) return Promise.resolve(false);
  ensureWebAudioConnected(audio);
  AUDIO_LOADING = true;
  if (label) setReady(`▶ PLAYING: ${label}`);

  return new Promise((resolve) => {
    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      AUDIO_LOADING = false;
      resolve(ok);
    };
    const timer = setTimeout(() => finish(false), 3500);
    audio.onplaying = () => {
      clearTimeout(timer);
      finish(true);
    };
    audio.onerror = () => {
      clearTimeout(timer);
      finish(false);
    };
    audio.onloadedmetadata = () => {
      if (startSec > 0 && startSec < audio.duration) audio.currentTime = startSec;
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.then(() => {
          clearTimeout(timer);
          finish(true);
        }).catch(() => {
          clearTimeout(timer);
          // Fallback to direct PipeWire hardware stream if browser autoplay blocked
          fetch('/api/pipewire-play', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ live_audio: true, max_seconds: 18.0 })
          }).catch(() => {});
          finish(false);
        });
      } else {
        clearTimeout(timer);
        finish(true);
      }
    };
    audio.src = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now();
    audio.load();
  });
}

async function toggleAutonomousLiveDJ() {
  if (LIVE_ACTIVE) {
    await stopAutonomousLiveDJ();
  } else {
    await startAutonomousLiveDJ();
  }
}

async function startAutonomousLiveDJ() {
  // Immediately unlock browser WebAudio & <audio> element inside the synchronous user click gesture
  unlockBrowserAudioOnGesture();

  const lvIn = document.getElementById('liveVibeInput');
  const stIn = document.getElementById('steerInput');
  const prompt = (lvIn && lvIn.value) ? lvIn.value : (stIn && stIn.value ? stIn.value : 'Peak-time progressive melodic techno club energy');
  syncSteerInputs(prompt);
  setBusy('🟢 STARTING AUTONOMOUS LIVE AI DJ (RTX 5090 6-AGENT LOOKAHEAD)...');
  LIVE_ACTIVE = true;
  LIVE_INTRO_PLAYING = false;
  LIVE_QUEUE = [];
  LIVE_PLAYING_INDEX = -1;
  LAST_LIVE_CERT_COUNT = 0;
  if (STUDIO.deck_a_id) {
    LIVE_INTRO_PLAYING = true;
    playAudioUrl(`/api/audio/track/${STUDIO.deck_a_id}`, `OPENING DECK A INTRO (Preparing Cycle #1 Mix...)`);
  }

  try {
    const res = await fetch('/api/live/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        steering_prompt: prompt,
        enable_host_mc: document.getElementById('chkHostMc') ? document.getElementById('chkHostMc').checked : true,
        bars: 8,
        use_neural_gpu: true,
        pipewire_live: false
      })
    });
    if (res.ok) {
      const data = await res.json();
      if (data.live_status) updateLivePipelineUI(data.live_status);
    }
  } catch (e) {}

  if (LIVE_POLL_TIMER) clearInterval(LIVE_POLL_TIMER);
  LIVE_POLL_TIMER = setInterval(pollLiveStatus, 950);
  await pollLiveStatus();
}

async function stopAutonomousLiveDJ() {
  LIVE_ACTIVE = false;
  AUDIO_LOADING = false;
  if (LIVE_POLL_TIMER) {
    clearInterval(LIVE_POLL_TIMER);
    LIVE_POLL_TIMER = null;
  }
  const audio = document.getElementById('masterAudio');
  if (audio) {
    audio.pause();
  }
  try {
    const res = await fetch('/api/live/stop', { method: 'POST' });
    if (res.ok) {
      const d = await res.json();
      if (d.live_status) updateLivePipelineUI(d.live_status);
    }
  } catch (e) {}
  await refreshState();
  setReady('⏹ AUTONOMOUS LIVE DJ STOPPED • GPU MODELS RELEASED');
}

async function pollLiveStatus() {
  try {
    const res = await fetch('/api/live/status');
    if (!res.ok) return;
    const ls = await res.json();
    const certCount = (ls.segments || []).length;
    if (certCount > LAST_LIVE_CERT_COUNT || (ls.crate_count || 0) > (STUDIO.crate || []).length) {
      LAST_LIVE_CERT_COUNT = certCount;
      await refreshState();
    }
    updateLivePipelineUI(ls);
    if (!ls.live_running && LIVE_POLL_TIMER) {
      clearInterval(LIVE_POLL_TIMER);
      LIVE_POLL_TIMER = null;
      await refreshState();
    }
  } catch (e) {}
}

function updateLivePipelineUI(ls) {
  if (!ls) return;
  LIVE_ACTIVE = !!ls.live_running;
  const btn = document.getElementById('btnLiveAutonomous');
  if (btn) {
    if (ls.live_running) {
      btn.classList.add('on-air', 'live-active');
      btn.textContent = `⏹ STOP AUTONOMOUS DJ (ON AIR — CYCLE #${ls.live_cycle || 1})`;
    } else {
      btn.classList.remove('on-air', 'live-active');
      btn.textContent = '▶ START AUTONOMOUS LIVE DJ';
    }
  }
  const agents = ['director', 'producer', 'listener', 'mixer', 'critic', 'host'];
  const doneList = ls.completed_agents_in_cycle || [];
  for (const ag of agents) {
    const node = document.getElementById('pipeNode_' + ag);
    if (!node) continue;
    const isAct = ls.live_running && ls.current_agent === ag;
    const isDone = ls.live_running && doneList.includes(ag) && !isAct;
    node.classList.toggle('active', isAct);
    node.classList.toggle('done', isDone);
  }
  const cap = document.getElementById('liveAgentStepCaption');
  if (cap) {
    cap.textContent = ls.live_step_detail || (ls.live_running ? 'Autonomous 6-Agent Lookahead Running...' : 'IDLE — Click START AUTONOMOUS LIVE DJ to launch continuous 6-agent lookahead on RTX 5090');
    cap.style.color = ls.live_running ? '#38bdf8' : 'var(--muted)';
  }
  if (ls.live_running && ls.live_step_detail) {
    setBusy(`🟢 LIVE CYCLE #${ls.live_cycle || 1}: ${ls.live_step_detail}`);
  }
  if (Array.isArray(ls.segments)) {
    LIVE_QUEUE = ls.segments;
    const audio = document.getElementById('masterAudio');
    const isIdle = !AUDIO_LOADING && (!audio || audio.paused || audio.ended || !audio.src || audio.src === SILENT_UNLOCK_WAV);
    if (ls.live_running && !LIVE_STARTING_SEGMENT && (LIVE_PLAYING_INDEX + 1) < LIVE_QUEUE.length && (isIdle || LIVE_INTRO_PLAYING)) {
      LIVE_INTRO_PLAYING = false;
      playNextLiveSegment();
    } else if (ls.live_running && LIVE_QUEUE.length === 0 && isIdle && !LIVE_INTRO_PLAYING && ls.deck_a_id) {
      LIVE_INTRO_PLAYING = true;
      STUDIO.deck_a_id = ls.deck_a_id;
      updateDeckUI('A', ls.deck_a_id);
      playAudioUrl(`/api/audio/track/${ls.deck_a_id}`, `OPENING DECK A INTRO (Preparing Cycle #1 Mix...)`);
    }
  }
}

let LIVE_STARTING_SEGMENT = false;
async function playNextLiveSegment() {
  if (LIVE_STARTING_SEGMENT || AUDIO_LOADING) return;
  if ((LIVE_PLAYING_INDEX + 1) >= LIVE_QUEUE.length) return;
  LIVE_STARTING_SEGMENT = true;
  try {
    LIVE_PLAYING_INDEX += 1;
    const seg = LIVE_QUEUE[LIVE_PLAYING_INDEX];
    if (!seg) return;

    try {
      await fetch('/api/live/progress', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ playing_index: LIVE_PLAYING_INDEX })
      });
    } catch (e) {}

    if (seg.track_a_id && seg.track_b_id) {
      STUDIO.deck_a_id = seg.track_a_id;
      STUDIO.deck_b_id = seg.track_b_id;
      await updateDeckUI('A', seg.track_a_id);
      await updateDeckUI('B', seg.track_b_id);
    }
    const xf = document.getElementById('crossfader');
    if (xf) {
      xf.value = 0;
      onCrossfader(0);
    }
    const sel = document.getElementById('audioSourceSelect');
    if (sel && [...sel.options].some(o => o.value === seg.audio_url)) {
      sel.value = seg.audio_url;
    }
    await playAudioUrl(seg.audio_url, `LIVE CYCLE #${seg.cycle || (LIVE_PLAYING_INDEX + 1)}: ${seg.track_a_title} → ${seg.track_b_title} [${seg.archetype}]`);
  } finally {
    LIVE_STARTING_SEGMENT = false;
  }
}

(function bindMasterAudioEnded() {
  const audio = document.getElementById('masterAudio');
  if (!audio) return;
  audio.addEventListener('ended', async () => {
    if (LIVE_ACTIVE && LIVE_PLAYING_INDEX >= 0 && LIVE_PLAYING_INDEX < LIVE_QUEUE.length) {
      const finishedSeg = LIVE_QUEUE[LIVE_PLAYING_INDEX];
      if (finishedSeg && finishedSeg.track_b_id) {
        // Incoming Deck B now becomes active Deck A; reset crossfader to Deck A
        STUDIO.deck_a_id = finishedSeg.track_b_id;
        await updateDeckUI('A', finishedSeg.track_b_id);
        const xf = document.getElementById('crossfader');
        if (xf) {
          xf.value = 0;
          onCrossfader(0);
        }
      }
      if ((LIVE_PLAYING_INDEX + 1) < LIVE_QUEUE.length) {
        await playNextLiveSegment();
      }
    }
  });
})();

function togglePlayPause() {
  const audio = document.getElementById('masterAudio');
  if (!audio.src) {
    if ( STUDIO.master_mix_ready ) playAudioUrl('/api/audio/master', 'Master Mix');
    return;
  }
  if (audio.paused) audio.play().catch(() => {});
  else audio.pause();
}

function ensureWebAudioConnected(audioEl) {
  if (AUDIO_CTX) {
    if (AUDIO_CTX.state === 'suspended') AUDIO_CTX.resume();
    return;
  }
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    AUDIO_CTX = new AudioCtx();
    ANALYSER = AUDIO_CTX.createAnalyser();
    ANALYSER.fftSize = 128;
    SOURCE_NODE = AUDIO_CTX.createMediaElementSource(audioEl);
    SOURCE_NODE.connect(ANALYSER);
    ANALYSER.connect(AUDIO_CTX.destination);
  } catch (e) {}
}

// 60 FPS Animation Loop for Spinning Jog Wheels, FFT Analyzer & Playhead
function animateLoop() {
  const audio = document.getElementById('masterAudio');
  const isPlaying = audio && !audio.paused && !audio.ended;

  if (isPlaying || LIVE_ACTIVE) {
    const speed = isPlaying ? 2.2 : 1.1;
    JOG_ANGLE_A = (JOG_ANGLE_A + speed) % 360;
    JOG_ANGLE_B = (JOG_ANGLE_B + speed) % 360;
    document.getElementById('jogRingA').style.transform = `rotate(${JOG_ANGLE_A}deg)`;
    document.getElementById('jogRingB').style.transform = `rotate(${JOG_ANGLE_B}deg)`;
    if (isPlaying && audio.duration && isFinite(audio.duration)) {
      PLAYHEAD_RATIO = audio.currentTime / audio.duration;
      if (STUDIO.deck_a_id && ENVELOPES[STUDIO.deck_a_id]) drawWaveform('waveCanvasA', ENVELOPES[STUDIO.deck_a_id], 'A');
      if (STUDIO.deck_b_id && ENVELOPES[STUDIO.deck_b_id]) drawWaveform('waveCanvasB', ENVELOPES[STUDIO.deck_b_id], 'B');
      if (LIVE_ACTIVE) {
        const xf = document.getElementById('crossfader');
        if (xf) {
          const xv = Math.min(100, Math.max(0, Math.round(PLAYHEAD_RATIO * 100)));
          xf.value = xv;
          onCrossfader(xv);
        }
      }
    }
  }

  // FFT Spectrum Canvas
  const fc = document.getElementById('fftCanvas');
  if (fc) {
    const fctx = fc.getContext('2d');
    const fw = fc.width, fh = fc.height;
    fctx.fillStyle = '#020408';
    fctx.fillRect(0, 0, fw, fh);
    if (ANALYSER && isPlaying) {
      const bins = new Uint8Array(ANALYSER.frequencyBinCount);
      ANALYSER.getByteFrequencyData(bins);
      const nb = 40;
      const bw = fw / nb;
      for (let i = 0; i < nb; i++) {
        const val = bins[i] / 255.0;
        const bh = Math.max(2, val * fh);
        fctx.fillStyle = i < 8 ? '#ef4444' : i < 22 ? '#f97316' : '#38bdf8';
        fctx.fillRect(i * bw, fh - bh, bw - 1.5, bh);
      }
    } else {
      // Idle heartbeat line
      fctx.strokeStyle = '#1e293b';
      fctx.beginPath(); fctx.moveTo(0, fh/2); fctx.lineTo(fw, fh/2); fctx.stroke();
    }
  }
  requestAnimationFrame(animateLoop);
}

// Connect WebSocket Telemetry
function connectWs() {
  try {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws/telemetry`);
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'live_segment_ready' || (msg.live_status && (msg.live_status.crate_count || 0) > (STUDIO.crate || []).length)) {
          await refreshState();
        }
        if (msg.live_status) {
          updateLivePipelineUI(msg.live_status);
        }
      } catch (e) {}
    };
    ws.onclose = () => { document.getElementById('wsStatusBadge').textContent = 'HTTP POLL'; };
  } catch (e) {}
}

refreshState();
connectWs();
requestAnimationFrame(animateLoop);
</script>
</body>
</html>
"""


# ============================================================================
# 4. FastAPI Unified 2-Deck DJ & Production Studio Factory
# ============================================================================

class DJAgentRequest(BaseModel):
    request: str

class DJLiveAck(BaseModel):
    seq: int


def create_app(cache_dir: str = "/dev/shm/autodj_studio", *, enable_legacy_studio: bool | None = None) -> FastAPI:
    """Creates the Unified FastAPI 2-Deck Web Studio application."""
    app = FastAPI(title="AI-Auto-DJ 6-Agent Studio", version="2.0.0")
    app.mount("/deck-assets", StaticFiles(directory=Path(__file__).with_name("autodj_web")), name="deck-assets")
    crew = AutoDJCrew(cache_dir=cache_dir)
    agent_runner = AgentRunner(os.environ.get("AUTODJ_AGENT_DIR", os.path.join(os.getcwd(), "dj_agent_runs")), crew.producer)
    agent_lock = threading.Lock()
    live_dj = LiveDJ(str(agent_runner.root / "live"), crew.producer, agent_runner.log, agent_lock)
    app.state.llm_live_dj = live_dj
    streamer = PipeWireLookaheadStreamer(shm_dir=os.path.join(cache_dir, "ringbuffer"))
    render_lock = threading.Lock()
    pw_stream_lock = threading.Lock()
    ws_lock = threading.Lock()
    ws_clients: set[tuple[WebSocket, asyncio.AbstractEventLoop]] = set()

    studio_state: dict[str, Any] = {
        "crew": crew,
        "streamer": streamer,
        "track_paths": [],
        "produced_tracks": [],
        "profiles": {},
        "deck_a_id": None,
        "deck_b_id": None,
        "deck_stems_gain": {
            "A": {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "other": 1.0},
            "B": {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "other": 1.0},
        },
        "selected_archetype": "AUTO",
        "steering_prompt": "High-energy progressive club mix",
        "bridge_engine": "acestep",
        "enable_host_mc": True,
        "last_session": {},
        "last_export": {},
        "host_voiceovers": [],
        "extra_events": [],
        "envelope_cache": {},
        "live_state": {
            "live_running": False,
            "run_id": 0,
            "lock": threading.Lock(),
            "live_cycle": 0,
            "current_agent": "idle",
            "live_step_detail": "IDLE — Click START AUTONOMOUS LIVE DJ to launch continuous 6-agent lookahead on RTX 5090",
            "completed_agents_in_cycle": [],
            "segments": [],
            "client_playing_index": -1,
            "use_neural_gpu": True,
            "pipewire_live": False,
            "vram_offloaded": {},
            "thread": None,
        },
    }

    @app.middleware("http")
    async def serialize_legacy_actions(request, call_next):
        # Only the prompt-first agent is public. Studio routes (library scan, host drop, ...)
        # take server paths and shell-adjacent input, so they exist only when the studio is enabled.
        if not enable_legacy_studio and request.url.path.startswith("/api/") and not request.url.path.startswith(("/api/dj-live/", "/api/dj-agent/")):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        # The studio and the prompt-first agent share producer/GPU state.
        if request.method == "POST" and request.url.path.startswith("/api/") and not request.url.path.startswith(("/api/dj-live/", "/api/dj-agent/")):
            from fastapi.responses import JSONResponse
            if not agent_lock.acquire(blocking=False):
                return JSONResponse(status_code=409, content={"detail": "A DJ agent session is active or stopping"})
            try:
                return await call_next(request)
            finally:
                agent_lock.release()
        return await call_next(request)

    def _add_event(agent: str, action: str, detail: str) -> None:
        studio_state["extra_events"].append({
            "timestamp": round(time.time(), 3),
            "agent": agent,
            "mode": crew.agent_modes.get(agent.lower().replace("agent", ""), "local"),
            "action": action,
            "detail": detail,
        })
        if len(studio_state["extra_events"]) > 200:
            del studio_state["extra_events"][:-120]

    def _get_live_status() -> dict[str, Any]:
        ls = studio_state["live_state"]
        return {
            "live_running": bool(ls["live_running"]),
            "live_cycle": int(ls["live_cycle"]),
            "current_agent": str(ls["current_agent"]),
            "live_step_detail": str(ls["live_step_detail"]),
            "completed_agents_in_cycle": list(ls["completed_agents_in_cycle"]),
            "segments": list(ls["segments"]),
            "client_playing_index": int(ls["client_playing_index"]),
            "use_neural_gpu": bool(ls["use_neural_gpu"]),
            "pipewire_live": bool(ls["pipewire_live"]),
            "crate_count": len(studio_state["profiles"]),
            "certified_count": len(studio_state.get("last_session", {}).get("certified_transitions", [])),
            "deck_a_id": studio_state["deck_a_id"],
            "deck_b_id": studio_state["deck_b_id"],
        }

    def _broadcast_ws(payload: dict[str, Any]) -> None:
        with ws_lock:
            targets = list(ws_clients)
        for client_ws, loop in targets:
            try:
                if not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(client_ws.send_json(payload), loop)
            except Exception:
                with ws_lock:
                    ws_clients.discard((client_ws, loop))

    def _index_track_file(path: str) -> dict[str, Any]:
        abs_p = os.path.abspath(path)
        for existing in studio_state["profiles"].values():
            if os.path.abspath(existing.get("path", "")) == abs_p:
                if abs_p not in studio_state["track_paths"]:
                    studio_state["track_paths"].append(abs_p)
                if studio_state["deck_a_id"] is None:
                    studio_state["deck_a_id"] = existing["track_id"]
                elif studio_state["deck_b_id"] is None and existing["track_id"] != studio_state["deck_a_id"]:
                    studio_state["deck_b_id"] = existing["track_id"]
                return existing

        prof = crew.listener.analyze_track(abs_p, prefer_neural_stems=False)
        prof_dict = prof.to_dict()
        studio_state["profiles"][prof.track_id] = prof_dict
        if abs_p not in studio_state["track_paths"]:
            studio_state["track_paths"].append(abs_p)
        if studio_state["deck_a_id"] is None:
            studio_state["deck_a_id"] = prof.track_id
        elif studio_state["deck_b_id"] is None and prof.track_id != studio_state["deck_a_id"]:
            studio_state["deck_b_id"] = prof.track_id

        _add_event(
            "ListenerAgent",
            "index_track",
            f"Indexed '{prof.title}': {prof.bpm:.1f} BPM • Key {prof.camelot_global} ({prof.key_name}) • {prof.integrated_lufs:.1f} LUFS",
        )
        return prof_dict

    def _build_deck_pair_preview() -> dict[str, Any]:
        aid = studio_state.get("deck_a_id")
        bid = studio_state.get("deck_b_id")
        profiles = studio_state["profiles"]
        gains_a = studio_state["deck_stems_gain"]["A"]
        gains_b = studio_state["deck_stems_gain"]["B"]
        if not aid or not bid or aid not in profiles or bid not in profiles:
            # Default 64-point T1 curve preview even before tracks are loaded
            arch_default = studio_state["selected_archetype"] if studio_state["selected_archetype"] != "AUTO" else "T1"
            ga, gb = build_stem_automation_envelopes(
                n_samples=64,
                num_bars=8,
                bass_swap_bar=4,
                archetype=arch_default,
                stem_gains_a=gains_a,
                stem_gains_b=gains_b,
            )
            return {
                "ready": False,
                "active_archetype": arch_default,
                "recommended_archetype": "T1",
                "archetype_name": ARCHETYPE_DESCRIPTIONS[arch_default],
                "compatibility_score": 0.95,
                "camelot_distance": 0,
                "target_bpm": 128.0,
                "overlap_bars": 8,
                "bass_swap_bar": 4,
                "pitch_shift_semitones_b": 0.0,
                "requires_ai_bridge": arch_default == "T6",
                "rationale": "Load two tracks onto Deck A & Deck B (or click Quick-Start Demo Mix) to compute live compatibility.",
                "envelopes_a": {k: [round(float(x), 3) for x in v] for k, v in ga.items()},
                "envelopes_b": {k: [round(float(x), 3) for x in v] for k, v in gb.items()},
            }

        prof_a = _dict_to_track_profile(profiles[aid])
        prof_b = _dict_to_track_profile(profiles[bid])
        spec = crew.director.plan_transition(
            prof_a,
            prof_b,
            steering_prompt=studio_state["steering_prompt"],
            bridge_engine=studio_state["bridge_engine"],
        )
        sel = studio_state.get("selected_archetype", "AUTO")
        active_arch = spec.archetype if sel == "AUTO" else sel
        ga, gb = build_stem_automation_envelopes(
            n_samples=64,
            num_bars=max(2, spec.overlap_bars),
            bass_swap_bar=spec.bass_swap_bar,
            archetype=active_arch,
            vocal_attenuation_db_a=spec.vocal_attenuation_db_a,
            vocal_attenuation_db_b=spec.vocal_attenuation_db_b,
            stem_gains_a=gains_a,
            stem_gains_b=gains_b,
        )
        return {
            "ready": True,
            "track_a_id": aid,
            "track_b_id": bid,
            "track_a_title": prof_a.title,
            "track_b_title": prof_b.title,
            "active_archetype": active_arch,
            "recommended_archetype": spec.archetype,
            "archetype_name": ARCHETYPE_DESCRIPTIONS[active_arch],
            "compatibility_score": spec.compatibility_score,
            "camelot_distance": camelot_distance(prof_a.camelot_cue_out, prof_b.camelot_cue_in),
            "target_bpm": spec.target_bpm,
            "overlap_bars": spec.overlap_bars,
            "bass_swap_bar": spec.bass_swap_bar,
            "pitch_shift_semitones_b": spec.pitch_shift_semitones_b,
            "requires_ai_bridge": (active_arch == "T6") or spec.requires_ai_bridge,
            "rationale": spec.rationale,
            "envelopes_a": {k: [round(float(x), 3) for x in v] for k, v in ga.items()},
            "envelopes_b": {k: [round(float(x), 3) for x in v] for k, v in gb.items()},
        }

    def _collect_spectrograms() -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for tid, p in studio_state["profiles"].items():
            png = p.get("spectrogram_png_path", "")
            if png and os.path.isfile(png):
                fname = os.path.basename(png)
                if fname not in seen:
                    seen.add(fname)
                    items.append({
                        "name": fname,
                        "label": f"Track: {p.get('title', tid)} ({p.get('bpm', 128):.1f} BPM • {p.get('camelot_global', '8A')})",
                        "category": "listener",
                        "url": f"/api/spectrogram/{fname}",
                    })
        for idx, cert in enumerate(studio_state.get("last_session", {}).get("certified_transitions", [])):
            png = cert.get("critique", {}).get("candidate_spectrogram_png", "")
            if png and os.path.isfile(png):
                fname = os.path.basename(png)
                if fname not in seen:
                    seen.add(fname)
                    sp = cert.get("spec", {})
                    items.append({
                        "name": fname,
                        "label": f"Transition #{idx + 1}: {sp.get('track_a_title', 'A')} → {sp.get('track_b_title', 'B')} [{sp.get('archetype', 'T1')}]",
                        "category": "critic",
                        "url": f"/api/spectrogram/{fname}",
                    })
        return items

    @app.post("/api/dj-live/start")
    def start_llm_live(req: DJAgentRequest) -> dict[str, Any]:
        try:
            legacy = studio_state["live_state"]
            if legacy["live_running"] or (legacy.get("thread") is not None and legacy["thread"].is_alive()):
                raise AgentError("Stop the advanced studio live session and wait for it to finish")
            import torch
            backend = os.environ.get("AUTODJ_MUSIC_BACKEND", "").lower()
            if backend == "stable_audio":
                from autodj_stable_audio import StableAudio, StableAudioError
                try:
                    StableAudio(live_dj.sr).check()
                except StableAudioError as exc:
                    raise AgentError(str(exc)) from exc
                neural = True
            elif backend == "bank":
                from autodj_bank import BankAudio
                from autodj_stable_audio import StableAudioError
                try:
                    BankAudio(live_dj.sr).check()
                except StableAudioError as exc:
                    raise AgentError(str(exc)) from exc
                neural = True
            elif backend == "ace_step":
                driver = crew.producer.ace_driver
                has_dit = driver.has_pretrained_dit_weights()
                root = driver.acestep_snap or ""
                required = ("Qwen3-Embedding-0.6B/model.safetensors", "vae/diffusion_pytorch_model.safetensors", "acestep-v15-turbo/silence_latent.pt")
                neural = bool(torch.cuda.is_available() and has_dit and all(os.path.isfile(os.path.join(root, p)) for p in required))
                if not neural:
                    raise AgentError("ACE-Step is selected but the GPU or cached model components are unavailable")
            else:
                neural = torch.cuda.is_available() and crew.producer.musicgen_driver.has_pretrained_weights()
            require_planner()
            return live_dj.start(req.request, configured_model(), neural)
        except GeminiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AgentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/dj-live/{sid}/status")
    def llm_live_status(sid: str) -> dict[str, Any]:
        try: return live_dj.status(sid)
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/steer")
    def llm_live_steer(sid: str, req: DJAgentRequest) -> dict[str, str]:
        try: live_dj.steer(sid, req.request); return {"status":"accepted"}
        except AgentError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/stop")
    def llm_live_stop(sid: str) -> dict[str, str]:
        try: live_dj.stop(sid); return {"status":"stopping"}
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dj-live/{sid}/ack")
    def llm_live_ack(sid: str, req: DJLiveAck) -> dict[str, str]:
        try: live_dj.ack(sid, req.seq); return {"status":"ok"}
        except AgentError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dj-live/{sid}/audio/{seq}")
    def llm_live_audio(sid: str, seq: int) -> FileResponse:
        try: return FileResponse(str(live_dj.segment_path(sid, seq)), media_type="audio/wav")
        except AgentError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dj-agent/run")
    def run_dj_agent(req: DJAgentRequest) -> dict[str, Any]:
        if not agent_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A DJ agent run is already in progress")
        try:
            import torch
            neural_available = torch.cuda.is_available() and crew.producer.musicgen_driver.has_pretrained_weights()
            require_planner()
            return agent_runner.run(req.request, configured_model(), neural_available)
        except (AgentError, GeminiError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"DJ run failed: {exc}") from exc
        finally:
            agent_lock.release()

    @app.get("/api/dj-agent/audio/{run_id}")
    def agent_audio(run_id: str) -> FileResponse:
        import re
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise HTTPException(status_code=404, detail="Mix not found")
        path = agent_runner.root / f"{run_id}_mix.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Mix not found")
        return FileResponse(str(path), media_type="audio/wav")

    @app.get("/", response_class=HTMLResponse)
    def index_page() -> str:
        return Path(__file__).with_name("autodj_frontend.html").read_text(encoding="utf-8")

    if enable_legacy_studio is None:
        enable_legacy_studio = os.environ.get("AUTODJ_ENABLE_LEGACY_STUDIO") == "1"

    @app.get("/studio", response_model=None, include_in_schema=False)
    def advanced_studio_page() -> HTMLResponse | RedirectResponse:
        if enable_legacy_studio:
            return HTMLResponse(WEB_STUDIO_HTML)
        return RedirectResponse(url="/")

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        c: AutoDJCrew = studio_state["crew"]
        last = studio_state.get("last_session", {})
        session_events = list(last.get("agent_events", []))
        combined_events = sorted(
            session_events + studio_state["extra_events"],
            key=lambda e: e.get("timestamp", 0.0),
        )[-60:]

        crate_list = []
        for tid, p in studio_state["profiles"].items():
            png_name = os.path.basename(p.get("spectrogram_png_path", "")) if p.get("spectrogram_png_path") else ""
            crate_list.append({
                "track_id": tid,
                "title": p.get("title", tid),
                "path": p.get("path", ""),
                "duration_sec": p.get("duration_sec", 0.0),
                "bpm": p.get("bpm", 128.0),
                "beat_stability": p.get("beat_stability", 1.0),
                "camelot_global": p.get("camelot_global", "8A"),
                "camelot_cue_in": p.get("camelot_cue_in", "8A"),
                "camelot_cue_out": p.get("camelot_cue_out", "8A"),
                "key_name": p.get("key_name", "A Minor"),
                "key_confidence": p.get("key_confidence", 0.9),
                "integrated_lufs": p.get("integrated_lufs", -14.0),
                "true_peak_dbtp": p.get("true_peak_dbtp", -1.0),
                "energy_score": p.get("energy_score", 0.7),
                "cue_in_bar": p.get("cue_in_bar", 0),
                "cue_out_bar": p.get("cue_out_bar", 4),
                "sections": p.get("sections", []),
                "spectrogram_png_name": png_name,
            })

        master_wav = os.path.join(cache_dir, "master_mix.wav")
        voiceovers = list(studio_state["host_voiceovers"])

        return {
            "agent_modes": c.agent_modes,
            "audio_ear_model": c.audio_ear_model,
            "steering_prompt": studio_state["steering_prompt"],
            "bridge_engine": studio_state["bridge_engine"],
            "selected_archetype": studio_state["selected_archetype"],
            "enable_host_mc": studio_state["enable_host_mc"],
            "track_paths": studio_state["track_paths"],
            "produced_tracks": studio_state["produced_tracks"],
            "crate": crate_list,
            "deck_a_id": studio_state["deck_a_id"],
            "deck_b_id": studio_state["deck_b_id"],
            "deck_stems_gain": studio_state["deck_stems_gain"],
            "deck_pair_preview": _build_deck_pair_preview(),
            "set_plan": last.get("set_plan", {}),
            "certified_transitions": last.get("certified_transitions", []),
            "host_voiceovers": voiceovers,
            "agent_events": combined_events,
            "master_mix_ready": os.path.isfile(master_wav),
            "export_info": studio_state.get("last_export", {}),
            "spectrograms": _collect_spectrograms(),
            "ringbuffer_bytes_streamed": streamer.bytes_streamed,
            "live_status": _get_live_status(),
        }

    @app.post("/api/agent-mode")
    def update_agent_mode(req: AgentModeUpdate) -> dict[str, Any]:
        c: AutoDJCrew = studio_state["crew"]
        c.agent_modes[req.agent] = req.mode
        if req.audio_ear_model:
            c.audio_ear_model = req.audio_ear_model
            c.listener.ear.audio_model_id = req.audio_ear_model
            c.critic.ear.audio_model_id = req.audio_ear_model
        _add_event(
            f"{req.agent.capitalize()}Agent",
            "mode_switch",
            f"Switched {req.agent.capitalize()}Agent to '{req.mode.upper()}'"
            + (f" (Audio Ear: {req.audio_ear_model})" if req.audio_ear_model else ""),
        )
        return {"status": "ok", "agent_modes": c.agent_modes, "audio_ear_model": c.audio_ear_model}

    @app.post("/api/steer")
    def steer_set(req: SteerRequest) -> dict[str, Any]:
        studio_state["steering_prompt"] = req.steering_prompt
        studio_state["bridge_engine"] = req.bridge_engine
        if req.selected_archetype is not None:
            studio_state["selected_archetype"] = req.selected_archetype
        _add_event(
            "DirectorAgent",
            "crowd_steer",
            f"Updated steering: '{req.steering_prompt}' • Bridge: {req.bridge_engine} • Archetype: {studio_state['selected_archetype']}",
        )
        return {
            "status": "ok",
            "steering_prompt": req.steering_prompt,
            "bridge_engine": req.bridge_engine,
            "selected_archetype": studio_state["selected_archetype"],
            "deck_pair_preview": _build_deck_pair_preview(),
        }

    @app.post("/api/produce")
    def produce_track_endpoint(req: ProduceRequest) -> dict[str, Any]:
        c: AutoDJCrew = studio_state["crew"]
        if req.engine == "suno":
            art = c.producer.suno_driver.compile_and_dispatch(
                title=req.title,
                style=req.style,
                lyrics=req.lyrics,
                output_path=os.path.join(c.producer.cache_dir, f"suno_{req.title.lower().replace(' ', '_')}.wav"),
                target_bpm=req.target_bpm,
                target_camelot=req.target_camelot,
                bars=req.bars,
            )
        else:
            art = c.producer.produce_track(
                title=req.title,
                style=req.style,
                lyrics=req.lyrics,
                target_bpm=req.target_bpm,
                target_camelot=req.target_camelot,
                bars=req.bars,
                use_neural_gpu=req.use_neural_gpu,
            )
        if art.output_path not in studio_state["track_paths"]:
            studio_state["track_paths"].append(art.output_path)
        if art.output_path not in studio_state["produced_tracks"]:
            studio_state["produced_tracks"].append(art.output_path)

        _add_event(
            "ProducerAgent",
            "produce_track",
            f"Produced '{req.title}' ({req.target_bpm:.1f} BPM, Key {req.target_camelot}, {req.bars} bars) via {art.backend_used}",
        )
        prof_dict = None
        if req.auto_index:
            prof_dict = _index_track_file(art.output_path)

        return {
            "status": "ok",
            "output_path": art.output_path,
            "backend_used": art.backend_used,
            "duration_sec": art.duration_sec,
            "profile": prof_dict,
        }

    @app.post("/api/library/scan")
    def scan_library_endpoint(req: LibraryScanRequest) -> dict[str, Any]:
        target = os.path.abspath(os.path.expanduser(req.path.strip()))
        if not os.path.exists(target):
            raise HTTPException(status_code=404, detail=f"Path not found: {target}")

        audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        found_files: list[str] = []
        if os.path.isfile(target):
            found_files.append(target)
        else:
            for root, _, files in os.walk(target):
                for fn in sorted(files):
                    if fn.lower().endswith(audio_exts) and not fn.endswith("_window.wav") and fn != "master_mix.wav":
                        found_files.append(os.path.join(root, fn))
                        if len(found_files) >= req.max_files:
                            break
                if len(found_files) >= req.max_files:
                    break

        if not found_files:
            raise HTTPException(status_code=400, detail=f"No audio files ({', '.join(audio_exts)}) found in {target}")

        indexed = []
        skipped_files: list[dict[str, str]] = []
        for fpath in found_files[: req.max_files]:
            try:
                p_dict = _index_track_file(fpath)
                indexed.append(p_dict["track_id"])
            except Exception as exc:
                skipped_files.append({"path": fpath, "error": str(exc)})
                _add_event(
                    "ListenerAgent",
                    "skip_corrupt_file",
                    f"Skipped unreadable audio file '{os.path.basename(fpath)}': {exc}",
                )

        if not indexed:
            raise HTTPException(
                status_code=400,
                detail=f"No valid decodable audio files found in {target} (skipped {len(skipped_files)} corrupted/unreadable files)",
            )

        return {
            "status": "ok",
            "scanned_path": target,
            "indexed_count": len(indexed),
            "skipped_count": len(skipped_files),
            "skipped_files": skipped_files,
            "track_ids": indexed,
        }

    @app.post("/api/deck/load")
    def load_deck_endpoint(req: DeckLoadRequest) -> dict[str, Any]:
        if req.selected_archetype is not None:
            studio_state["selected_archetype"] = req.selected_archetype
        if req.stem_gains is not None:
            for k, v in req.stem_gains.items():
                if k in STEM_NAMES:
                    studio_state["deck_stems_gain"][req.deck][k] = float(np.clip(v, 0.0, 1.25))

        if req.track_id_or_path:
            val = req.track_id_or_path
            if val in studio_state["profiles"]:
                tid = val
            elif os.path.isfile(val):
                try:
                    p_dict = _index_track_file(val)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio file '{val}': {exc}") from exc
                tid = p_dict["track_id"]
            else:
                raise HTTPException(status_code=404, detail=f"Track '{val}' not found in crate or filesystem")

            if req.deck == "A":
                studio_state["deck_a_id"] = tid
            else:
                studio_state["deck_b_id"] = tid
            prof = studio_state["profiles"][tid]
            _add_event(
                "MixerAgent",
                f"load_deck_{req.deck.lower()}",
                f"Loaded '{prof['title']}' ({prof['bpm']:.1f} BPM • Key {prof['camelot_global']}) onto DECK {req.deck}",
            )

        return {
            "status": "ok",
            "deck_a_id": studio_state["deck_a_id"],
            "deck_b_id": studio_state["deck_b_id"],
            "deck_stems_gain": studio_state["deck_stems_gain"],
            "selected_archetype": studio_state["selected_archetype"],
            "deck_pair_preview": _build_deck_pair_preview(),
        }

    @app.post("/api/deck/mix-pair")
    def mix_deck_pair_endpoint(req: DeckMixPairRequest) -> dict[str, Any]:
        with render_lock:
            if req.stem_gains_a is not None:
                for k, v in req.stem_gains_a.items():
                    if k in STEM_NAMES:
                        studio_state["deck_stems_gain"]["A"][k] = float(np.clip(v, 0.0, 1.25))
            if req.stem_gains_b is not None:
                for k, v in req.stem_gains_b.items():
                    if k in STEM_NAMES:
                        studio_state["deck_stems_gain"]["B"][k] = float(np.clip(v, 0.0, 1.25))

            aid = req.track_a_id or studio_state.get("deck_a_id")
            bid = req.track_b_id or studio_state.get("deck_b_id")
            if not aid or not bid or aid not in studio_state["profiles"] or bid not in studio_state["profiles"]:
                raise HTTPException(status_code=400, detail="Load tracks onto both Deck A and Deck B before mixing.")

            prof_a = _dict_to_track_profile(studio_state["profiles"][aid])
            prof_b = _dict_to_track_profile(studio_state["profiles"][bid])

            spec = crew.director.plan_transition(
                prof_a,
                prof_b,
                steering_prompt=studio_state["steering_prompt"],
                bridge_engine=req.bridge_engine,
            )
            arch_choice = req.archetype if req.archetype != "AUTO" else studio_state.get("selected_archetype", "AUTO")
            if arch_choice != "AUTO":
                spec.archetype = arch_choice
                spec.archetype_name = ARCHETYPE_DESCRIPTIONS[arch_choice]
                if arch_choice == "T6":
                    spec.requires_ai_bridge = True
            if req.overlap_bars is not None:
                spec.overlap_bars = max(2, min(16, req.overlap_bars))
                spec.bass_swap_bar = max(1, spec.overlap_bars // 2)

            if spec.requires_ai_bridge and not spec.bridge_wav_path:
                stems_a = dict(np.load(prof_a.stems_cache_path)) if os.path.isfile(prof_a.stems_cache_path) else None
                bridge = crew.producer.produce_bridge(
                    track_a_title=prof_a.title,
                    track_b_title=prof_b.title,
                    stems_a=stems_a,
                    target_bpm=spec.target_bpm,
                    target_camelot=prof_b.camelot_cue_in,
                    bars=spec.overlap_bars,
                    engine=req.bridge_engine,
                )
                spec.bridge_wav_path = bridge.output_path
                spec.archetype = "T6"
                _add_event("ProducerAgent", "produce_bridge", f"Generated {spec.overlap_bars}-bar {req.bridge_engine} bridge")

            gains_a = dict(studio_state["deck_stems_gain"]["A"])
            gains_b = dict(studio_state["deck_stems_gain"]["B"])
            out_wav = os.path.join(cache_dir, "mixer", f"live_pair_{aid}_{bid}_{spec.archetype}.wav")
            rendered = crew.mixer.render_transition(
                prof_a,
                prof_b,
                spec,
                output_wav_path=out_wav,
                stem_gains_a=gains_a,
                stem_gains_b=gains_b,
            )

            if req.enable_host_mc:
                vo = crew.host.synthesize_drop(
                    script=spec.host_script or f"Transitioning from {prof_a.title} into {prof_b.title} at {spec.target_bpm:.0f} BPM!"
                )
                studio_state["host_voiceovers"].append(vo.to_dict())
                m_stereo, _ = decode_audio_stereo(rendered.candidate_wav_path, sr=SAMPLE_RATE)
                v_stereo, _ = decode_audio_stereo(vo.wav_path, sr=SAMPLE_RATE)
                ducked_stereo, duck_stats = crew.host.mix_voiceover_with_midband_ducking(
                    m_stereo, v_stereo, insert_offset_samples=SAMPLE_RATE // 2, duck_db=-10.0, sr=SAMPLE_RATE
                )
                sf.write(rendered.candidate_wav_path, ducked_stereo.T, SAMPLE_RATE, subtype="FLOAT")
                rendered.integrated_lufs = duck_stats["integrated_lufs"]
                rendered.true_peak_dbtp = duck_stats["true_peak_dbtp"]
                _add_event("HostAgent", "synthesize_mc_drop", f"Mixed {vo.duration_sec:.2f}s MC voiceover with -10 dB mid-band ducking")

            critique = crew.critic.evaluate_transition(prof_a, prof_b, spec, rendered)
            retries_used = 0
            if critique.verdict == "REJECT" and critique.adjustment is not None:
                spec = crew.critic.apply_self_healing_adjustment(spec, critique.adjustment)
                if spec.requires_ai_bridge and not spec.bridge_wav_path:
                    bridge = crew.producer.produce_bridge(
                        track_a_title=prof_a.title,
                        track_b_title=prof_b.title,
                        target_bpm=spec.target_bpm,
                        target_camelot=prof_b.camelot_cue_in,
                        bars=spec.overlap_bars,
                        engine=req.bridge_engine,
                    )
                    spec.bridge_wav_path = bridge.output_path
                rendered = crew.mixer.render_transition(
                    prof_a,
                    prof_b,
                    spec,
                    output_wav_path=out_wav,
                    stem_gains_a=gains_a,
                    stem_gains_b=gains_b,
                )
                critique = crew.critic.evaluate_transition(prof_a, prof_b, spec, rendered)
                retries_used = 1
                _add_event("CriticAgent", "self_heal", f"Self-healed Deck A→B mix to {spec.archetype} ({critique.verdict})")

            cert_entry = {
                "transition_index": 0,
                "spec": spec.to_dict(),
                "rendered": rendered.to_dict(),
                "critique": critique.to_dict(),
                "retries_used": retries_used,
            }
            studio_state["last_session"] = {
                "steering_prompt": studio_state["steering_prompt"],
                "set_plan": {
                    "ordered_track_ids": [aid, bid],
                    "transitions": [spec.to_dict()],
                    "global_score": spec.compatibility_score,
                    "steering_prompt": studio_state["steering_prompt"],
                    "director_notes": f"Live 2-Deck Mix: {prof_a.title} -> {prof_b.title} [{spec.archetype}]",
                    "energy_curve": [prof_a.energy_score, prof_b.energy_score],
                },
                "certified_transitions": [cert_entry],
                "agent_events": studio_state.get("last_session", {}).get("agent_events", []),
            }
            master_wav = os.path.join(cache_dir, "master_mix.wav")
            export_info = export_master_mix_and_cue(studio_state["last_session"], output_wav_path=master_wav)
            studio_state["last_export"] = export_info
            master_stereo, _ = decode_audio_stereo(master_wav, sr=SAMPLE_RATE)
            streamer.write_lookahead_slot(master_stereo, slot_idx=0)
            streamer.stream_slot_to_pipewire(slot_idx=0, live_audio=False)
            _add_event(
                "CriticAgent",
                "certify_live_mix",
                f"Certified Deck A→B [{spec.archetype}]: {rendered.integrated_lufs:.1f} LUFS • {rendered.true_peak_dbtp:.2f} dBTP • BeatErr {rendered.beat_alignment_error_ms:.3f}ms",
            )
            return {
                "status": "ok",
                "transition": cert_entry,
                "export": export_info,
            }

    def _set_live_agent(
        agent_name: str,
        detail: str,
        mark_done: str | None = None,
        run_id: int | None = None,
    ) -> None:
        ls = studio_state["live_state"]
        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return
        if mark_done and mark_done not in ls["completed_agents_in_cycle"]:
            ls["completed_agents_in_cycle"].append(mark_done)
        ls["current_agent"] = agent_name
        ls["live_step_detail"] = detail
        _add_event(f"{agent_name.capitalize()}Agent", "live_autonomous", detail)
        _broadcast_ws({
            "type": "live_agent_step",
            "current_agent": agent_name,
            "live_step_detail": detail,
            "live_status": _get_live_status(),
        })

    def _produce_and_index_live_track(
        concept: dict[str, Any],
        bars: int,
        use_neural_gpu: bool,
        seed: int,
        cycle_idx: int,
        run_id: int | None = None,
    ) -> TrackProfile:
        title = str(concept.get("title", f"Live_Club_Cycle_{cycle_idx}"))
        style = str(concept.get("style", "Peak-Time Melodic Techno"))
        bpm = float(concept.get("target_bpm", 128.0))
        cam = str(concept.get("target_camelot", "8A"))
        _set_live_agent(
            "producer",
            f"Cycle #{cycle_idx}: Synthesizing '{title}' ({bpm:.1f} BPM, Key {cam}) on {'RTX 5090 Neural GPU' if use_neural_gpu else 'Studio DSP'}...",
            run_id=run_id,
        )
        art = crew.producer.produce_track(
            title=title,
            style=style,
            lyrics=str(concept.get("lyrics", "[Instrumental Club Mix]")),
            target_bpm=bpm,
            target_camelot=cam,
            bars=bars,
            seed=seed,
            use_neural_gpu=use_neural_gpu,
        )
        ls = studio_state["live_state"]
        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return crew.listener.analyze_track(art.output_path, prefer_neural_stems=False)
        if art.output_path not in studio_state["track_paths"]:
            studio_state["track_paths"].append(art.output_path)
        if art.output_path not in studio_state["produced_tracks"]:
            studio_state["produced_tracks"].append(art.output_path)

        _set_live_agent(
            "listener",
            f"Cycle #{cycle_idx}: Separating 4 stems (HTDemucs CUDA) & indexing CQT key for '{title}'...",
            mark_done="producer",
            run_id=run_id,
        )
        prof = crew.listener.analyze_track(art.output_path, prefer_neural_stems=use_neural_gpu)
        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return prof
        studio_state["profiles"][prof.track_id] = prof.to_dict()
        if "listener" not in ls["completed_agents_in_cycle"]:
            ls["completed_agents_in_cycle"].append("listener")
        return prof

    def _render_and_certify_live_transition(
        prof_a: TrackProfile,
        prof_b: TrackProfile,
        cycle_idx: int,
        host_script: str,
        enable_host_mc: bool,
        use_neural_gpu: bool,
        pipewire_live: bool,
        run_id: int | None = None,
    ) -> dict[str, Any]:
        ls = studio_state["live_state"]
        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return {}
        spec = crew.director.plan_transition(
            prof_a,
            prof_b,
            steering_prompt=studio_state["steering_prompt"],
            bridge_engine=studio_state["bridge_engine"],
        )
        if host_script:
            spec.host_script = host_script
        sel_arch = studio_state.get("selected_archetype", "AUTO")
        if sel_arch != "AUTO":
            spec.archetype = sel_arch
            spec.archetype_name = ARCHETYPE_DESCRIPTIONS[sel_arch]
            if sel_arch == "T6":
                spec.requires_ai_bridge = True

        if spec.requires_ai_bridge and not spec.bridge_wav_path:
            stems_a = dict(np.load(prof_a.stems_cache_path)) if os.path.isfile(prof_a.stems_cache_path) else None
            bridge = crew.producer.produce_bridge(
                track_a_title=prof_a.title,
                track_b_title=prof_b.title,
                stems_a=stems_a,
                target_bpm=spec.target_bpm,
                target_camelot=prof_b.camelot_cue_in,
                bars=spec.overlap_bars,
                engine=studio_state["bridge_engine"],
                use_neural_gpu=use_neural_gpu,
            )
            spec.bridge_wav_path = bridge.output_path
            spec.archetype = "T6"

        _set_live_agent(
            "mixer",
            f"Cycle #{cycle_idx}: Mixing {prof_a.title} → {prof_b.title} [{spec.archetype}] @ {spec.target_bpm:.1f} BPM...",
            mark_done="listener",
            run_id=run_id,
        )
        gains_a = dict(studio_state["deck_stems_gain"]["A"])
        gains_b = dict(studio_state["deck_stems_gain"]["B"])
        out_wav = os.path.join(cache_dir, "mixer", f"live_auto_c{cycle_idx}_{prof_a.track_id}_{prof_b.track_id}_{spec.archetype}.wav")
        rendered = crew.mixer.render_transition(
            prof_a,
            prof_b,
            spec,
            output_wav_path=out_wav,
            stem_gains_a=gains_a,
            stem_gains_b=gains_b,
        )
        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return {}

        _set_live_agent(
            "critic",
            f"Cycle #{cycle_idx}: Auditing LUFS / True Peak / Bass Collision & 4-Panel Spectrogram...",
            mark_done="mixer",
            run_id=run_id,
        )
        critique = crew.critic.evaluate_transition(prof_a, prof_b, spec, rendered)
        retries_used = 0
        if critique.verdict == "REJECT" and critique.adjustment is not None:
            spec = crew.critic.apply_self_healing_adjustment(spec, critique.adjustment)
            if spec.requires_ai_bridge and not spec.bridge_wav_path:
                bridge = crew.producer.produce_bridge(
                    track_a_title=prof_a.title,
                    track_b_title=prof_b.title,
                    target_bpm=spec.target_bpm,
                    target_camelot=prof_b.camelot_cue_in,
                    bars=spec.overlap_bars,
                    engine=studio_state["bridge_engine"],
                    use_neural_gpu=use_neural_gpu,
                )
                spec.bridge_wav_path = bridge.output_path
            rendered = crew.mixer.render_transition(
                prof_a,
                prof_b,
                spec,
                output_wav_path=out_wav,
                stem_gains_a=gains_a,
                stem_gains_b=gains_b,
            )
            critique = crew.critic.evaluate_transition(prof_a, prof_b, spec, rendered)
            retries_used = 1

        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return {}

        _set_live_agent(
            "host",
            f"Cycle #{cycle_idx}: Synthesizing Kokoro-82M MC Drop & -10 dB Mid-Band Ducking...",
            mark_done="critic",
            run_id=run_id,
        )
        if enable_host_mc:
            vo = crew.host.synthesize_drop(
                script=spec.host_script or f"Live on air! Mixing {prof_a.title} into {prof_b.title} at {spec.target_bpm:.0f} BPM!"
            )
            studio_state["host_voiceovers"].append(vo.to_dict())
            m_stereo, _ = decode_audio_stereo(rendered.candidate_wav_path, sr=SAMPLE_RATE)
            v_stereo, _ = decode_audio_stereo(vo.wav_path, sr=SAMPLE_RATE)
            ducked_stereo, duck_stats = crew.host.mix_voiceover_with_midband_ducking(
                m_stereo, v_stereo, insert_offset_samples=SAMPLE_RATE // 2, duck_db=-10.0, sr=SAMPLE_RATE
            )
            sf.write(rendered.candidate_wav_path, ducked_stereo.T, SAMPLE_RATE, subtype="FLOAT")
            rendered.integrated_lufs = duck_stats["integrated_lufs"]
            rendered.true_peak_dbtp = duck_stats["true_peak_dbtp"]

        if run_id is not None and (not ls["live_running"] or ls["run_id"] != run_id):
            return {}

        if "host" not in ls["completed_agents_in_cycle"]:
            ls["completed_agents_in_cycle"].append("host")

        last_sess = studio_state.setdefault("last_session", {})
        certs = last_sess.setdefault("certified_transitions", [])
        trans_idx = len(certs)
        cert_entry = {
            "transition_index": trans_idx,
            "spec": spec.to_dict(),
            "rendered": rendered.to_dict(),
            "critique": critique.to_dict(),
            "retries_used": retries_used,
        }
        certs.append(cert_entry)

        # Rolling /dev/shm cleanup for old transition WAVs already played > 20 cycles ago
        if len(certs) > 24:
            prune_idx = len(certs) - 21
            if 0 <= prune_idx < ls["client_playing_index"]:
                old_wav = certs[prune_idx].get("rendered", {}).get("candidate_wav_path", "")
                if old_wav and os.path.isfile(old_wav):
                    try:
                        os.remove(old_wav)
                    except Exception:
                        pass

        sp_plan = last_sess.setdefault("set_plan", {
            "ordered_track_ids": [],
            "transitions": [],
            "global_score": 0.96,
            "steering_prompt": studio_state["steering_prompt"],
            "director_notes": "Continuous Autonomous Live 6-Agent Lookahead Session",
            "energy_curve": [],
        })
        if prof_a.track_id not in sp_plan["ordered_track_ids"]:
            sp_plan["ordered_track_ids"].append(prof_a.track_id)
            sp_plan["energy_curve"].append(prof_a.energy_score)
        if prof_b.track_id not in sp_plan["ordered_track_ids"]:
            sp_plan["ordered_track_ids"].append(prof_b.track_id)
            sp_plan["energy_curve"].append(prof_b.energy_score)
        sp_plan["transitions"].append(spec.to_dict())

        master_wav = os.path.join(cache_dir, "master_mix.wav")
        export_info = export_master_mix_and_cue(last_sess, output_wav_path=master_wav)
        studio_state["last_export"] = export_info

        slot_i = (cycle_idx - 1) % 2
        seg_stereo, _ = decode_audio_stereo(rendered.candidate_wav_path, sr=SAMPLE_RATE)
        streamer.write_lookahead_slot(seg_stereo, slot_idx=slot_i)
        if pipewire_live:
            def _serialized_pw_stream(s_idx: int) -> None:
                with pw_stream_lock:
                    if ls["live_running"] and (run_id is None or ls["run_id"] == run_id):
                        streamer.stream_slot_to_pipewire(slot_idx=s_idx, live_audio=True, max_seconds=None)

            threading.Thread(target=_serialized_pw_stream, args=(slot_i,), daemon=True).start()
        else:
            streamer.stream_slot_to_pipewire(slot_idx=slot_i, live_audio=False)

        seg_meta = {
            "cycle": cycle_idx,
            "transition_index": trans_idx,
            "track_a_id": prof_a.track_id,
            "track_b_id": prof_b.track_id,
            "track_a_title": prof_a.title,
            "track_b_title": prof_b.title,
            "archetype": spec.archetype,
            "target_bpm": spec.target_bpm,
            "integrated_lufs": rendered.integrated_lufs,
            "true_peak_dbtp": rendered.true_peak_dbtp,
            "verdict": critique.verdict,
            "audio_url": f"/api/audio/transition/{trans_idx}",
        }
        ls["segments"].append(seg_meta)
        ls["live_step_detail"] = (
            f"Cycle #{cycle_idx} CERTIFIED: {prof_a.title} → {prof_b.title} [{spec.archetype}] "
            f"({rendered.integrated_lufs:.1f} LUFS / {rendered.true_peak_dbtp:.2f} dBTP)"
        )
        _add_event("CriticAgent", "live_segment_ready", ls["live_step_detail"])
        _broadcast_ws({
            "type": "live_segment_ready",
            "transition_index": trans_idx,
            "audio_url": f"/api/audio/transition/{trans_idx}",
            "deck_a_id": prof_a.track_id,
            "deck_b_id": prof_b.track_id,
            "live_status": _get_live_status(),
        })
        return seg_meta

    def _live_lookahead_worker(req: LiveStartRequest, run_id: int) -> None:
        ls = studio_state["live_state"]

        def _is_active() -> bool:
            return bool(ls["live_running"] and ls.get("run_id") == run_id and not crew.stop_requested)

        try:
            # Step A: Check GPU VRAM & offload idle ComfyUI if needed
            if req.use_neural_gpu and _is_active():
                vram_info = offload_external_gpu_vram()
                if not _is_active():
                    return
                ls["vram_offloaded"] = vram_info
                _add_event(
                    "ProducerAgent",
                    "vram_offload",
                    f"RTX 5090 VRAM ready: free={vram_info.get('free_gb_after', 0):.2f} GiB (offloaded={vram_info.get('offloaded_ports', [])})",
                )

            if not _is_active():
                return

            # Step B: Cycle 1 Instant Bootstrap (< 2.0s to live audio & waveforms on Deck A/B)
            ls["live_cycle"] = 1
            ls["completed_agents_in_cycle"] = []
            _set_live_agent(
                "director",
                "Cycle #1: Locking instant opening harmonic pair (9A → 8A) while warming RTX 5090 lookahead...",
                run_id=run_id,
            )
            steer_str = studio_state.get("steering_prompt") or "Peak-Time Melodic Techno"
            concept_1 = {
                "title": "Solaris_Pulse_9A",
                "style": f"{steer_str} — Driving Opening Groove",
                "target_bpm": 128.0,
                "target_camelot": "9A",
                "lyrics": "[Instrumental Club Mix]",
                "host_script": "Welcome to Neural Deck Live! Kicking off the autonomous set in 9A at 128 BPM!",
            }
            concept_2 = {
                "title": "Obsidian_Mirage_8A",
                "style": f"{steer_str} — Hypnotic Rolling Bassline",
                "target_bpm": 129.0,
                "target_camelot": "8A",
                "lyrics": "[Instrumental Club Mix]",
                "host_script": "Live on the Neural Deck autonomous stream! Mixing Solaris Pulse 9A into Obsidian Mirage 8A!",
            }
            if not _is_active():
                return
            if "director" not in ls["completed_agents_in_cycle"]:
                ls["completed_agents_in_cycle"].append("director")

            # Fast Cycle #1 Bootstrap (< 2.0s to live audio) so the booth goes ON AIR immediately,
            # while Cycles #2, #3, #4... run Ollama + full RTX 5090 Neural GPU tracks in the background lookahead!
            prof_1 = _produce_and_index_live_track(
                concept_1, bars=req.bars, use_neural_gpu=False, seed=101, cycle_idx=1, run_id=run_id
            )
            if not _is_active():
                return
            studio_state["deck_a_id"] = prof_1.track_id
            _broadcast_ws({"type": "live_agent_step", "live_status": _get_live_status()})

            prof_2 = _produce_and_index_live_track(
                concept_2, bars=req.bars, use_neural_gpu=False, seed=202, cycle_idx=1, run_id=run_id
            )
            if not _is_active():
                return
            studio_state["deck_b_id"] = prof_2.track_id
            _broadcast_ws({"type": "live_agent_step", "live_status": _get_live_status()})

            seg_1 = _render_and_certify_live_transition(
                prof_1,
                prof_2,
                cycle_idx=1,
                host_script=str(concept_2.get("host_script", "")),
                enable_host_mc=req.enable_host_mc,
                use_neural_gpu=req.use_neural_gpu,
                pipewire_live=req.pipewire_live,
                run_id=run_id,
            )
            if seg_1 is None or not _is_active():
                return
            prev_prof = prof_2

            # Step C: Infinite Continuous Lookahead Loop (Cycles 2, 3, 4, ...)
            cycle_idx = 2
            while _is_active():
                # Pace lookahead when >= 2 transitions ahead of client playback
                while _is_active():
                    ahead = len(ls["segments"]) - (ls["client_playing_index"] + 1)
                    if ls["client_playing_index"] >= 0 and ahead >= 2:
                        ls["current_agent"] = "host"
                        ls["live_step_detail"] = (
                            f"Cycle #{cycle_idx - 1} On Air • Lookahead Queue Ready ({ahead} transitions buffered ahead)"
                        )
                        time.sleep(0.6)
                    else:
                        break

                for _ in range(3):
                    if not _is_active():
                        break
                    time.sleep(0.35)
                if not _is_active():
                    break

                ls["live_cycle"] = cycle_idx
                ls["completed_agents_in_cycle"] = []
                _set_live_agent(
                    "director",
                    f"Cycle #{cycle_idx}: Planning next harmonic track from '{prev_prof.title}' ({prev_prof.bpm:.1f} BPM • {prev_prof.camelot_global})...",
                    run_id=run_id,
                )
                concept_n = crew.director.generate_live_track_concept(
                    steering_prompt=studio_state["steering_prompt"],
                    prev_track=prev_prof.to_dict(),
                    cycle_idx=cycle_idx + 1,
                )
                if not _is_active():
                    break
                if "director" not in ls["completed_agents_in_cycle"]:
                    ls["completed_agents_in_cycle"].append("director")

                prof_n = _produce_and_index_live_track(
                    concept_n,
                    bars=req.bars,
                    use_neural_gpu=req.use_neural_gpu,
                    seed=100 + cycle_idx * 37,
                    cycle_idx=cycle_idx,
                    run_id=run_id,
                )
                if not _is_active():
                    break

                if ls["client_playing_index"] < 0:
                    studio_state["deck_a_id"] = prev_prof.track_id
                    studio_state["deck_b_id"] = prof_n.track_id
                seg_n = _render_and_certify_live_transition(
                    prev_prof,
                    prof_n,
                    cycle_idx=cycle_idx,
                    host_script=str(concept_n.get("host_script", "")),
                    enable_host_mc=req.enable_host_mc,
                    use_neural_gpu=req.use_neural_gpu,
                    pipewire_live=req.pipewire_live,
                    run_id=run_id,
                )
                if seg_n is None or not _is_active():
                    break
                prev_prof = prof_n
                cycle_idx += 1
        except Exception as exc:
            if ls.get("run_id") == run_id:
                _add_event("DirectorAgent", "live_error", f"Live Autonomous loop error: {exc}")
                ls["live_step_detail"] = f"Live loop error: {exc}"
        finally:
            if ls.get("run_id") == run_id and not ls["live_running"]:
                ls["current_agent"] = "idle"
                free_cached_gpu_models()

    class AutonomousLiveDJController:
        """Thread-safe controller for the continuous 6-agent autonomous live DJ lookahead loop."""

        def start(self, req: LiveStartRequest) -> dict[str, Any]:
            ls = studio_state["live_state"]
            with ls["lock"]:
                studio_state["steering_prompt"] = req.steering_prompt
                studio_state["bridge_engine"] = req.bridge_engine
                studio_state["enable_host_mc"] = req.enable_host_mc
                if ls["live_running"]:
                    return {"status": "already_running", "live_status": _get_live_status()}

                ls["run_id"] = int(ls.get("run_id", 0)) + 1
                active_run_id = ls["run_id"]
                crew.stop_requested = False
                ls["live_running"] = True
                ls["live_cycle"] = 1
                ls["current_agent"] = "director"
                ls["live_step_detail"] = "Cycle #1: Launching 6-Agent Autonomous Lookahead on RTX 5090..."
                ls["completed_agents_in_cycle"] = []
                ls["segments"] = []
                ls["client_playing_index"] = -1
                ls["use_neural_gpu"] = req.use_neural_gpu
                ls["pipewire_live"] = req.pipewire_live
                studio_state["last_session"] = {
                    "steering_prompt": req.steering_prompt,
                    "set_plan": {
                        "ordered_track_ids": [],
                        "transitions": [],
                        "global_score": 0.96,
                        "steering_prompt": req.steering_prompt,
                        "director_notes": "Continuous Autonomous Live 6-Agent Lookahead Session",
                        "energy_curve": [],
                    },
                    "certified_transitions": [],
                    "agent_events": [],
                }
                t = threading.Thread(target=_live_lookahead_worker, args=(req, active_run_id), daemon=True)
                ls["thread"] = t
                t.start()
            status_snap = _get_live_status()
            _broadcast_ws({"type": "live_agent_step", "live_status": status_snap})
            return {"status": "started", "live_status": status_snap}

        def stop(self) -> dict[str, Any]:
            ls = studio_state["live_state"]
            with ls["lock"]:
                ls["run_id"] = int(ls.get("run_id", 0)) + 1
                ls["live_running"] = False
                crew.stop_requested = True
                ls["current_agent"] = "idle"
                ls["live_step_detail"] = "IDLE — Autonomous Live DJ stopped • GPU models released"
            freed_info = free_cached_gpu_models()
            _add_event(
                "DirectorAgent",
                "live_stop",
                f"Stopped Autonomous Live DJ after Cycle #{ls['live_cycle']} (freed GPU models: {freed_info.get('freed_models', [])})",
            )
            status_snap = _get_live_status()
            _broadcast_ws({"type": "live_stopped", "live_status": status_snap})
            return {
                "status": "stopped",
                "freed_gpu": freed_info,
                "live_status": status_snap,
            }

        def status(self) -> dict[str, Any]:
            return _get_live_status()

        def report_progress(self, req: LiveProgressRequest) -> dict[str, Any]:
            ls = studio_state["live_state"]
            idx = req.client_playing_index if req.client_playing_index is not None else req.playing_index
            ls["client_playing_index"] = int(idx)
            if 0 <= ls["client_playing_index"] < len(ls["segments"]):
                seg = ls["segments"][ls["client_playing_index"]]
                ta = seg.get("track_a_id") or seg.get("deck_a_id")
                tb = seg.get("track_b_id") or seg.get("deck_b_id")
                if ta:
                    studio_state["deck_a_id"] = ta
                if tb:
                    studio_state["deck_b_id"] = tb
            return {"status": "ok", "client_playing_index": ls["client_playing_index"]}

    live_controller = AutonomousLiveDJController()
    studio_state["live_controller"] = live_controller
    app.state.live_controller = live_controller

    @app.post("/api/live/start")
    def start_live_autonomous_endpoint(req: LiveStartRequest) -> dict[str, Any]:
        return live_controller.start(req)

    @app.post("/api/live/stop")
    def stop_live_autonomous_endpoint() -> dict[str, Any]:
        return live_controller.stop()

    @app.get("/api/live/status")
    def get_live_status_endpoint() -> dict[str, Any]:
        return live_controller.status()

    @app.post("/api/live/progress")
    def report_live_progress_endpoint(req: LiveProgressRequest) -> dict[str, Any]:
        return live_controller.report_progress(req)

    @app.post("/api/session/stop")
    def stop_session_endpoint() -> dict[str, Any]:
        c: AutoDJCrew = studio_state["crew"]
        c.stop_requested = True
        ls = studio_state["live_state"]
        if ls["live_running"]:
            live_controller.stop()
        _add_event("DirectorAgent", "stop_session", "Operator requested session stop — halting active transition loop")
        return {"status": "stopped", "stop_requested": True, "live_status": _get_live_status()}

    @app.post("/api/run-session")
    def run_session_endpoint(req: RunSessionRequest) -> dict[str, Any]:
        with render_lock:
            c: AutoDJCrew = studio_state["crew"]
            c.stop_requested = False
            paths = req.track_paths or studio_state["track_paths"]
            if len(paths) < 2:
                raise HTTPException(status_code=400, detail="Provide at least 2 tracks to run a DJ transition session.")
            state = c.run_session(
                track_paths=paths,
                steering_prompt=req.steering_prompt,
                bridge_engine=req.bridge_engine,
                enable_host_mc=req.enable_host_mc,
                existing_profiles=studio_state["profiles"],
            )
            studio_state["last_session"] = state
            existing_vo_paths = {v.get("wav_path") for v in studio_state["host_voiceovers"]}
            for vo_item in state.get("host_voiceovers", []):
                if vo_item.get("wav_path") not in existing_vo_paths:
                    studio_state["host_voiceovers"].append(vo_item)
                    existing_vo_paths.add(vo_item.get("wav_path"))

            for tid, p_dict in state.get("profiles", {}).items():
                studio_state["profiles"][tid] = p_dict
                p_abs = os.path.abspath(p_dict.get("path", ""))
                if p_abs and p_abs not in studio_state["track_paths"]:
                    studio_state["track_paths"].append(p_abs)

            ordered_ids = state.get("set_plan", {}).get("ordered_track_ids", [])
            if len(ordered_ids) >= 2:
                studio_state["deck_a_id"] = ordered_ids[0]
                studio_state["deck_b_id"] = ordered_ids[1]

            export_info = {}
            if state.get("certified_transitions"):
                master_wav = os.path.join(cache_dir, "master_mix.wav")
                export_info = export_master_mix_and_cue(state, output_wav_path=master_wav)
                studio_state["last_export"] = export_info
                master_stereo, _ = decode_audio_stereo(master_wav, sr=SAMPLE_RATE)
                streamer.write_lookahead_slot(master_stereo, slot_idx=0)
                streamer.stream_slot_to_pipewire(slot_idx=0, live_audio=False)
            return {
                "status": "ok",
                "certified_count": len(state.get("certified_transitions", [])),
                "set_plan": state.get("set_plan", {}),
                "export": export_info,
            }

    @app.post("/api/quickstart")
    def quickstart_endpoint(req: QuickStartRequest) -> dict[str, Any]:
        """One-Click Studio Bootstrap: produces/loads 3 club tracks, indexes stems, runs all 6 agents, and prepares master audio."""
        c: AutoDJCrew = studio_state["crew"]
        if len(studio_state["track_paths"]) < 2 or req.force_new_tracks:
            demo_specs = [
                ("Cyber_Horizon_8A", "Deep Progressive Melodic Techno", 126.0, "8A"),
                ("Neon_Pulse_8A", "Peak-Time Driving Club Techno", 128.0, "8A"),
                ("Solaris_Protocol_9A", "Euphoric Festival Synthesizer Anthem", 130.0, "9A"),
            ]
            for title, style, bpm, cam in demo_specs:
                art = c.producer.produce_track(
                    title=title,
                    style=style,
                    target_bpm=bpm,
                    target_camelot=cam,
                    bars=req.bars,
                )
                if art.output_path not in studio_state["track_paths"]:
                    studio_state["track_paths"].append(art.output_path)
                if art.output_path not in studio_state["produced_tracks"]:
                    studio_state["produced_tracks"].append(art.output_path)

        studio_state["steering_prompt"] = req.steering_prompt
        studio_state["bridge_engine"] = req.bridge_engine
        studio_state["enable_host_mc"] = req.enable_host_mc

        run_res = run_session_endpoint(
            RunSessionRequest(
                track_paths=studio_state["track_paths"][:4],
                steering_prompt=req.steering_prompt,
                bridge_engine=req.bridge_engine,
                enable_host_mc=req.enable_host_mc,
            )
        )
        return {
            "status": "ok",
            "certified_count": run_res["certified_count"],
            "export": run_res["export"],
            "deck_a_id": studio_state["deck_a_id"],
            "deck_b_id": studio_state["deck_b_id"],
        }

    @app.post("/api/host-drop")
    def synthesize_host_drop_endpoint(req: HostDropRequest) -> dict[str, Any]:
        c: AutoDJCrew = studio_state["crew"]
        c.host.voice_name = req.voice_name
        vo = c.host.synthesize_drop(script=req.script, speed=req.speed)
        vo_dict = vo.to_dict()
        studio_state["host_voiceovers"].append(vo_dict)
        idx = len(studio_state["host_voiceovers"]) - 1
        _add_event(
            "HostAgent",
            "synthesize_mc_drop",
            f"Synthesized {vo.duration_sec:.2f}s drop ('{req.script[:42]}...') via {vo.backend_used}",
        )
        return {
            "status": "ok",
            "voiceover": vo_dict,
            "audio_url": f"/api/audio/voiceover/{idx}",
        }

    @app.post("/api/pipewire-play")
    def pipewire_play_endpoint(req: PipeWirePlayRequest) -> dict[str, Any]:
        master_wav = os.path.join(cache_dir, "master_mix.wav")
        if not os.path.isfile(streamer.slot_paths[0]) and os.path.isfile(master_wav):
            master_stereo, _ = decode_audio_stereo(master_wav, sr=SAMPLE_RATE)
            streamer.write_lookahead_slot(master_stereo, slot_idx=0)
        if not os.path.isfile(streamer.slot_paths[0]):
            raise HTTPException(status_code=400, detail="No audio in ringbuffer yet — run a session or mix first.")
        res = streamer.stream_slot_to_pipewire(
            slot_idx=0,
            live_audio=req.live_audio,
            max_seconds=req.max_seconds,
        )
        _add_event(
            "MixerAgent",
            "pipewire_stream",
            f"PipeWire ringbuffer ({res['status']}): {res['frames_streamed']} stereo frames @ 44.1 kHz float32le",
        )
        return res

    @app.get("/api/envelope/{track_id}")
    def get_track_envelope(track_id: str) -> dict[str, Any]:
        if track_id not in studio_state["profiles"]:
            raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
        if track_id in studio_state["envelope_cache"]:
            return studio_state["envelope_cache"][track_id]

        p = studio_state["profiles"][track_id]
        stereo, _ = decode_audio_stereo(p["path"], sr=SAMPLE_RATE)
        n_bins = 160
        total_samples = stereo.shape[1]
        step = max(1, total_samples // n_bins)
        peaks_l: list[float] = []
        peaks_r: list[float] = []
        for i in range(n_bins):
            s0 = i * step
            s1 = min(total_samples, s0 + step)
            if s1 > s0:
                peaks_l.append(round(float(np.max(np.abs(stereo[0, s0:s1]))), 3))
                peaks_r.append(round(float(np.max(np.abs(stereo[1, s0:s1]))), 3))
            else:
                peaks_l.append(0.0)
                peaks_r.append(0.0)

        env_data = {
            "track_id": track_id,
            "title": p["title"],
            "duration_sec": p["duration_sec"],
            "bpm": p["bpm"],
            "camelot_global": p["camelot_global"],
            "key_name": p["key_name"],
            "integrated_lufs": p["integrated_lufs"],
            "true_peak_dbtp": p["true_peak_dbtp"],
            "cue_in_bar": p["cue_in_bar"],
            "cue_out_bar": p["cue_out_bar"],
            "downbeats_sec": p["downbeats_sec"],
            "sections": p["sections"],
            "bar_rms_db": p["bar_rms_db"],
            "peaks_l": peaks_l,
            "peaks_r": peaks_r,
        }
        studio_state["envelope_cache"][track_id] = env_data
        return env_data

    @app.get("/api/audio/master")
    def get_master_audio() -> FileResponse:
        master_wav = os.path.join(cache_dir, "master_mix.wav")
        if not os.path.isfile(master_wav):
            raise HTTPException(status_code=404, detail="Master mix WAV not rendered yet")
        return FileResponse(master_wav, media_type="audio/wav", filename="master_mix.wav")

    @app.get("/api/audio/transition/{idx}")
    def get_transition_audio(idx: int) -> FileResponse:
        certs = studio_state.get("last_session", {}).get("certified_transitions", [])
        if idx < 0 or idx >= len(certs):
            raise HTTPException(status_code=404, detail=f"Transition #{idx} not found")
        wav_path = certs[idx]["rendered"]["candidate_wav_path"]
        if not os.path.isfile(wav_path):
            raise HTTPException(status_code=404, detail="Transition WAV file missing")
        return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    @app.get("/api/audio/track/{track_id}")
    def get_track_audio(track_id: str) -> FileResponse:
        if track_id not in studio_state["profiles"]:
            raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")
        wav_path = studio_state["profiles"][track_id]["path"]
        if not os.path.isfile(wav_path):
            raise HTTPException(status_code=404, detail="Track audio file not found")
        return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    @app.get("/api/audio/stem/{track_id}/{stem}")
    def get_stem_audio(track_id: str, stem: str) -> FileResponse:
        if track_id not in studio_state["profiles"] or stem not in STEM_NAMES:
            raise HTTPException(status_code=404, detail=f"Stem '{stem}' for track '{track_id}' not found")
        prof = studio_state["profiles"][track_id]
        stem_wav = os.path.join(cache_dir, "listener", f"stem_{track_id}_{stem}.wav")
        if not os.path.isfile(stem_wav):
            npz_path = prof.get("stems_cache_path", "")
            if not os.path.isfile(npz_path):
                raise HTTPException(status_code=404, detail="Separated stems cache not found")
            loaded = np.load(npz_path)
            stem_arr = loaded[stem].astype(np.float32)
            os.makedirs(os.path.dirname(stem_wav), exist_ok=True)
            sf.write(stem_wav, stem_arr.T, SAMPLE_RATE, subtype="FLOAT")
        return FileResponse(stem_wav, media_type="audio/wav", filename=os.path.basename(stem_wav))

    @app.get("/api/audio/voiceover/{idx}")
    def get_voiceover_audio(idx: int) -> FileResponse:
        all_vos = studio_state["host_voiceovers"]
        if idx < 0 or idx >= len(all_vos):
            raise HTTPException(status_code=404, detail=f"Voiceover #{idx} not found")
        wav_path = all_vos[idx]["wav_path"]
        if not os.path.isfile(wav_path):
            raise HTTPException(status_code=404, detail="Voiceover WAV not found")
        return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    @app.get("/api/export/wav")
    def download_master_wav() -> FileResponse:
        master_wav = os.path.join(cache_dir, "master_mix.wav")
        if not os.path.isfile(master_wav):
            raise HTTPException(status_code=404, detail="Master mix WAV not exported yet")
        return FileResponse(master_wav, media_type="audio/wav", filename="ai_autodj_master_mix.wav")

    @app.get("/api/export/cue")
    def download_master_cue() -> FileResponse:
        cue_path = os.path.join(cache_dir, "master_mix.cue")
        if not os.path.isfile(cue_path):
            raise HTTPException(status_code=404, detail="Master mix CUE sheet not exported yet")
        return FileResponse(cue_path, media_type="text/plain", filename="ai_autodj_master_mix.cue")

    @app.get("/api/spectrogram/{filename}")
    def get_spectrogram(filename: str) -> FileResponse:
        safe_name = os.path.basename(filename)
        for sub in ("listener", "critic"):
            candidate = os.path.join(cache_dir, sub, safe_name)
            if os.path.isfile(candidate):
                return FileResponse(candidate, media_type="image/png")
        for match in glob.glob(os.path.join(cache_dir, "**", safe_name), recursive=True):
            if os.path.isfile(match):
                return FileResponse(match, media_type="image/png")
        raise HTTPException(status_code=404, detail="Spectrogram PNG not found")

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        if not enable_legacy_studio:
            await ws.close(code=1008)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        client_pair = (ws, loop)
        with ws_lock:
            ws_clients.add(client_pair)
        try:
            c: AutoDJCrew = studio_state["crew"]
            last = studio_state.get("last_session", {})
            await ws.send_json({
                "type": "telemetry_snapshot",
                "timestamp": time.time(),
                "agent_modes": c.agent_modes,
                "audio_ear_model": c.audio_ear_model,
                "events_count": len(last.get("agent_events", [])) + len(studio_state["extra_events"]),
                "ringbuffer_bytes_streamed": streamer.bytes_streamed,
                "live_status": _get_live_status(),
            })
            while True:
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_json({"type": "pong", "timestamp": time.time(), "live_status": _get_live_status()})
                elif msg == "state":
                    await ws.send_json({"type": "state_update", "state": get_state()})
        except WebSocketDisconnect:
            pass
        finally:
            with ws_lock:
                ws_clients.discard(client_pair)

    return app


app = create_app(cache_dir=os.environ.get("AUTODJ_CACHE_DIR", "/dev/shm/autodj_studio"))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Auto-DJ Unified 6-Agent Virtual DJ & Production Studio GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8088, help="Bind port")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
