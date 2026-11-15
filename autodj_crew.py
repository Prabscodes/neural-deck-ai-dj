#!/usr/bin/env python3
"""6-Agent Virtual DJ & Production Studio Crew (`autodj_crew.py`) built on `langgraph==1.2.10`.

Implements:
1. `HostAgent` (Agent 6): Cached local `hexgrad/Kokoro-82M` (`EspeakG2P` + `KModel`) DJ Radio MC
   synthesis with Broadcast EQ (`120 Hz` HPF + `+2.5 dB` presence boost) and `-10 dB` Mid-Band
   (`300–3500 Hz`) sidechain ducking (leaving sub-bass `< 300 Hz` untouched) (Trap #8 Fix).
2. Modification 3 & Trap #16 Helpers:
   - `AudioProjectorGraft`: 2-layer MLP projector + `inputs_embeds.masked_scatter` audio-token
     injection matching `Qwen3_5MoeModel` / `Qwen3_5Model` 3-axis M-RoPE (`mm_token_type_ids == 0`).
   - `defuse_moe_3d_experts`: Converts 3D `nn.Parameter` tensors (`gate_up_proj`, `down_proj`) into
     `nn.Linear` modules so `bitsandbytes` 4-bit NF4 quantizes 100% of MoE expert weights (Trap #16 Fix).
3. `DJCrewState` & `AutoDJCrew`: Full 6-Agent `LangGraph` state machine (`ListenerAgent` ->
   `DirectorAgent` -> `ProducerAgent` -> `HostAgent` -> `MixerAgent` -> `CriticAgent` with
   conditional self-healing retry edges).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import math
import os
import sys
import threading
import time
import types
from typing import Any, Literal, TypedDict

sys.modules.setdefault("torchao", None)
# Provide lightweight dummy spacy module so `kokoro/__init__.py` loads `KModel` without requiring spacy (Trap #8)
sys.modules.setdefault("spacy", types.ModuleType("spacy"))

from langgraph.graph import END, START, StateGraph
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import torch
import torch.nn as nn

from autodj_critic import CriticAgent, CritiqueReport, MixAdjustment
from autodj_indexer import (
    ListenerAgent,
    SAMPLE_RATE,
    SectionSpan,
    TrackProfile,
    decode_audio_stereo,
)
from autodj_mixer import MixerAgent, RenderedTransition, normalize_and_brickwall_limit
from autodj_planner import DirectorAgent, SetPlan, TransitionSpec
from autodj_producer import ProducerAgent


# ============================================================================
# 1. Agent 6: `HostAgent` (`Kokoro-82M` + Broadcast EQ + `-10 dB` Sidechain Ducking)
# ============================================================================

@dataclass
class HostVoiceover:
    """Synthesized DJ radio MC drop from `HostAgent`."""

    script: str
    voice_name: str
    wav_path: str
    sample_rate: int
    duration_sec: float
    peak_dbfs: float
    backend_used: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HostAgent:
    """Agent 6 (`HostAgent`): Synthesizes DJ radio hype drops via cached `Kokoro-82M` and sidechain-ducks mid-band."""

    _kmodel: Any = None
    _g2p: Any = None
    _snap_dir: str | None = None
    _kokoro_lock = threading.Lock()

    def __init__(
        self,
        mode: Literal["local", "cloud", "hybrid"] = "local",
        voice_name: str = "af_heart",
        cache_dir: str = "/dev/shm/autodj_host",
    ) -> None:
        self.mode = mode
        self.voice_name = voice_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @classmethod
    def _ensure_kokoro_loaded(cls) -> tuple[Any, Any, str] | None:
        if cls._kmodel is not None and cls._g2p is not None and cls._snap_dir is not None:
            return cls._kmodel, cls._g2p, cls._snap_dir

        snaps = glob.glob("/home/prab/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*")
        if not snaps:
            return None
        snap = snaps[0]
        cfg_path = os.path.join(snap, "config.json")
        pth_path = os.path.join(snap, "kokoro-v1_0.pth")
        if not (os.path.isfile(cfg_path) and os.path.isfile(pth_path)):
            return None

        try:
            from misaki.espeak import EspeakG2P
            from kokoro.model import KModel

            cls._g2p = EspeakG2P(language="en-us")
            cls._kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg_path, model=pth_path).eval()
            cls._snap_dir = snap
            return cls._kmodel, cls._g2p, cls._snap_dir
        except Exception:
            return None

    @staticmethod
    def apply_broadcast_eq(stereo: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Applies broadcast radio EQ: 120 Hz 4th-order high-pass filter + +2.5 dB presence boost (2.5-4.5 kHz)."""
        sos_hp = signal.butter(4, 120.0 / (0.5 * sr), btype="high", output="sos")
        y = signal.sosfiltfilt(sos_hp, stereo, axis=1).astype(np.float32)

        sos_pres = signal.butter(2, [2400.0 / (0.5 * sr), 4600.0 / (0.5 * sr)], btype="bandpass", output="sos")
        pres_band = signal.sosfiltfilt(sos_pres, y, axis=1).astype(np.float32)
        y = y + 0.33 * pres_band  # ~+2.5 dB presence lift
        peak = float(np.max(np.abs(y)))
        if peak > 1e-6:
            y = y * (0.78 / peak)
        return np.ascontiguousarray(y, dtype=np.float32)

    def synthesize_drop(
        self,
        script: str,
        output_wav_path: str | None = None,
        speed: float = 1.05,
    ) -> HostVoiceover:
        """Synthesizes a stereo 44.1 kHz broadcast-EQ'd DJ voiceover from `script` using cached `Kokoro-82M`."""
        if output_wav_path is None:
            output_wav_path = os.path.join(self.cache_dir, f"mc_drop_{abs(hash(script)) % 100000}.wav")

        with self._kokoro_lock:
            loaded = self._ensure_kokoro_loaded()
            backend = "Kokoro-82M (Local Neural TTS)"
            if loaded is not None:
                kmodel, g2p, snap = loaded
                phonemes, _ = g2p(script)
                if not phonemes:
                    phonemes = "dˌiːʤˈA"
                voice_file = os.path.join(snap, "voices", f"{self.voice_name}.pt")
                if not os.path.isfile(voice_file):
                    voice_files = glob.glob(os.path.join(snap, "voices", "*.pt"))
                    voice_file = voice_files[0]
                voice_pack = torch.load(voice_file, map_location="cpu", weights_only=True)
                ref_s = voice_pack[min(max(0, len(phonemes) - 1), voice_pack.shape[0] - 1)]
                with torch.no_grad():
                    wav_24k = kmodel(phonemes, ref_s, speed=speed).float().cpu().numpy().reshape(-1)
                # Resample 24000 Hz -> 44100 Hz (up=147, down=80)
                mono_44k = signal.resample_poly(wav_24k, up=147, down=80).astype(np.float32)
            else:
                backend = "Formant-Synth-Fallback"
                dur = max(1.5, min(4.0, len(script) * 0.05))
                t = np.arange(int(dur * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
                mono_44k = (0.3 * np.sin(2.0 * np.pi * 190.0 * t) * np.sin(2.0 * np.pi * 4.0 * t)).astype(np.float32)

        stereo = np.stack([mono_44k, mono_44k], axis=0)
        stereo = self.apply_broadcast_eq(stereo, sr=SAMPLE_RATE)
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
        sf.write(output_wav_path, stereo.T, SAMPLE_RATE, subtype="FLOAT")
        peak_db = 20.0 * math.log10(max(float(np.max(np.abs(stereo))), 1e-9))

        return HostVoiceover(
            script=script,
            voice_name=self.voice_name,
            wav_path=os.path.abspath(output_wav_path),
            sample_rate=SAMPLE_RATE,
            duration_sec=round(stereo.shape[1] / SAMPLE_RATE, 4),
            peak_dbfs=round(peak_db, 2),
            backend_used=backend,
        )

    @staticmethod
    def mix_voiceover_with_midband_ducking(
        music_stereo: np.ndarray,
        voice_stereo: np.ndarray,
        insert_offset_samples: int = 0,
        duck_db: float = -10.0,
        sr: int = SAMPLE_RATE,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Mixes `voice_stereo` over `music_stereo` with `-10 dB` Mid-Band (`300-3500 Hz`) sidechain ducking.

        Leaves sub-bass (`< 300 Hz`) and air (`> 3500 Hz`) at 100% gain so club low-end never drops when the MC speaks.
        """
        out = music_stereo.copy().astype(np.float32)
        s0 = max(0, min(out.shape[1], insert_offset_samples))
        v_len = min(voice_stereo.shape[1], out.shape[1] - s0)
        if v_len <= 0:
            return out, {"midband_attenuation_db": 0.0, "lowband_attenuation_db": 0.0}

        seg_music = out[:, s0 : s0 + v_len]
        seg_voice = voice_stereo[:, :v_len].copy()

        # Taper voiceover & ducking envelope at start/end boundaries to prevent step clicks
        taper_len = min(1024, v_len // 4)
        boundary_win = np.ones(v_len, dtype=np.float32)
        if taper_len > 8:
            theta = np.linspace(0.0, 0.5 * math.pi, taper_len, dtype=np.float32)
            boundary_win[:taper_len] = np.sin(theta) ** 2
            boundary_win[-taper_len:] = np.cos(theta) ** 2
            seg_voice *= boundary_win[None, :]

        # Split music segment into Mid-Band (300-3500 Hz) and Remainder (Low <300 Hz + High >3500 Hz)
        sos_mid = signal.butter(2, [300.0 / (0.5 * sr), 3500.0 / (0.5 * sr)], btype="bandpass", output="sos")
        music_mid = signal.sosfiltfilt(sos_mid, seg_music, axis=1).astype(np.float32)
        music_low_high = (seg_music - music_mid).astype(np.float32)

        # Compute smooth sidechain envelope from voiceover RMS
        voice_abs = np.max(np.abs(seg_voice), axis=0)
        voice_env = uniform_filter1d(voice_abs, size=max(1, int(0.06 * sr)))
        env_norm = np.clip(voice_env / max(float(np.percentile(voice_env, 90)), 1e-4), 0.0, 1.0) * boundary_win

        min_gain = float(10.0 ** (duck_db / 20.0))  # -10 dB -> 0.3162
        duck_curve = (1.0 - (1.0 - min_gain) * env_norm).astype(np.float32)

        ducked_mid = music_mid * duck_curve[None, :]
        pre_mid_rms = float(np.sqrt(np.mean(music_mid ** 2) + 1e-12))
        post_mid_rms = float(np.sqrt(np.mean(ducked_mid ** 2) + 1e-12))
        measured_mid_duck_db = 20.0 * math.log10(post_mid_rms / pre_mid_rms)

        out[:, s0 : s0 + v_len] = music_low_high + ducked_mid + seg_voice * 0.52
        out, post_stats = normalize_and_brickwall_limit(out, target_lufs=-14.0, ceiling_dbtp=-1.0, sr=sr)
        return out, {
            "midband_attenuation_db": round(measured_mid_duck_db, 2),
            "target_duck_db": duck_db,
            "lowband_preserved": True,
            "integrated_lufs": post_stats["integrated_lufs"],
            "true_peak_dbtp": post_stats["true_peak_dbtp"],
        }


# ============================================================================
# 2. Modification 3 & Trap #16: Neural Audio-Tower Graft & 3D MoE De-Fuser
# ============================================================================

class AudioProjectorGraft(nn.Module):
    """Modification 3: 2-Layer MLP Projector grafting a frozen Audio Tower (`d_audio=1280` or `2048`)
    into `Qwen3_5MoeModel` / `Qwen3_5Model` via `inputs_embeds.masked_scatter` (`mm_token_type_ids == 0`).
    """

    def __init__(self, audio_dim: int = 1280, llm_hidden_size: int = 2048, audio_token_id: int = 151646) -> None:
        super().__init__()
        self.audio_token_id = audio_token_id
        self.proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(audio_hidden)

    def splice_into_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        audio_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Splices projected audio embeddings into `inputs_embeds` at `audio_token_id` positions."""
        audio_embeds = self.forward(audio_hidden).to(inputs_embeds.device, inputs_embeds.dtype)
        audio_mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        n_mask_tokens = int((input_ids == self.audio_token_id).sum().item())
        flat_embeds = audio_embeds.reshape(-1, inputs_embeds.shape[-1])[:n_mask_tokens]
        return inputs_embeds.masked_scatter(audio_mask, flat_embeds)


def defuse_moe_3d_experts(moe_block: nn.Module) -> nn.Module:
    """Trap #16 Fix: Converts fused 3D `nn.Parameter` tensors (`gate_up_proj`, `down_proj`) into `nn.ModuleList` of `nn.Linear`.

    In `Qwen3_5MoeExperts` and `Qwen3OmniMoeExperts`, routed experts are stored as 3D `nn.Parameter`
    tensors (`(num_experts, 2*intermediate, hidden)`), which `bitsandbytes` skips during 4-bit NF4 quantization.
    Converting them into `nn.Linear` allows 100% 4-bit quantization.
    """
    if hasattr(moe_block, "gate_up_proj") and isinstance(moe_block.gate_up_proj, nn.Parameter) and moe_block.gate_up_proj.ndim == 3:
        n_exp, two_inter, hidden = moe_block.gate_up_proj.shape
        _, out_dim, inter = moe_block.down_proj.shape
        gate_up_list = nn.ModuleList()
        down_list = nn.ModuleList()
        for e in range(n_exp):
            l1 = nn.Linear(hidden, two_inter, bias=False, device=moe_block.gate_up_proj.device, dtype=moe_block.gate_up_proj.dtype)
            l1.weight = nn.Parameter(moe_block.gate_up_proj.data[e])
            l2 = nn.Linear(inter, out_dim, bias=False, device=moe_block.down_proj.device, dtype=moe_block.down_proj.dtype)
            l2.weight = nn.Parameter(moe_block.down_proj.data[e])
            gate_up_list.append(l1)
            down_list.append(l2)
        delattr(moe_block, "gate_up_proj")
        delattr(moe_block, "down_proj")
        moe_block.gate_up_experts = gate_up_list
        moe_block.down_experts = down_list
    return moe_block


# ============================================================================
# 3. `LangGraph` 6-Agent State Machine (`DJCrewState` & `AutoDJCrew`)
# ============================================================================

class DJCrewState(TypedDict, total=False):
    """Typed state shared across the 6-Agent LangGraph Virtual DJ Studio."""

    track_paths: list[str]
    steering_prompt: str
    agent_modes: dict[str, Literal["local", "cloud", "hybrid"]]
    audio_ear_model: str
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"]
    enable_host_mc: bool
    max_critic_retries: int
    inject_test_defect_on_first_try: bool

    profiles: dict[str, dict[str, Any]]
    set_plan: dict[str, Any]
    current_transition_idx: int
    retry_count: int
    current_spec: dict[str, Any]
    current_rendered: dict[str, Any]
    current_critique: dict[str, Any]
    certified_transitions: list[dict[str, Any]]
    host_voiceovers: list[dict[str, Any]]
    agent_events: list[dict[str, Any]]


def _dict_to_track_profile(d: dict[str, Any]) -> TrackProfile:
    cp = dict(d)
    cp["sections"] = [SectionSpan(**s) if isinstance(s, dict) else s for s in cp.get("sections", [])]
    return TrackProfile(**cp)


class AutoDJCrew:
    """Orchestrates the 6-Agent Virtual DJ & Production Studio via `langgraph==1.2.10`."""

    def __init__(
        self,
        cache_dir: str = "/dev/shm/autodj_studio",
        audio_ear_model: str = "nvidia/music-flamingo-think-2601-hf",
        agent_modes: dict[str, Literal["local", "cloud", "hybrid"]] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.audio_ear_model = audio_ear_model
        self.agent_modes: dict[str, Literal["local", "cloud", "hybrid"]] = {
            "director": "local",
            "producer": "local",
            "listener": "local",
            "mixer": "local",
            "critic": "local",
            "host": "local",
            **(agent_modes or {}),
        }

        self.listener = ListenerAgent(
            mode=self.agent_modes["listener"],
            audio_ear_model=audio_ear_model,
            cache_dir=os.path.join(cache_dir, "listener"),
        )
        self.producer = ProducerAgent(
            mode=self.agent_modes["producer"],
            cache_dir=os.path.join(cache_dir, "producer"),
        )
        self.director = DirectorAgent(mode=self.agent_modes["director"])
        self.mixer = MixerAgent(cache_dir=os.path.join(cache_dir, "mixer"))
        self.critic = CriticAgent(
            mode=self.agent_modes["critic"],
            audio_ear_model=audio_ear_model,
            cache_dir=os.path.join(cache_dir, "critic"),
        )
        self.host = HostAgent(
            mode=self.agent_modes["host"],
            cache_dir=os.path.join(cache_dir, "host"),
        )
        self.graph = self._build_graph()

    def _log_event(self, state: DJCrewState, agent: str, action: str, detail: str) -> list[dict[str, Any]]:
        events = list(state.get("agent_events", []))
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": agent,
            "mode": self.agent_modes.get(agent.lower(), "local"),
            "action": action,
            "detail": detail,
        })
        return events

    def _listener_node(self, state: DJCrewState) -> dict[str, Any]:
        profiles = dict(state.get("profiles", {}))
        events = list(state.get("agent_events", []))
        for path in state.get("track_paths", []):
            abs_p = os.path.abspath(path)
            if not any(p.get("path") == abs_p for p in profiles.values()):
                prof = self.listener.analyze_track(abs_p, prefer_neural_stems=False)
                profiles[prof.track_id] = prof.to_dict()
                events.append({
                    "timestamp": round(time.time(), 3),
                    "agent": "ListenerAgent",
                    "mode": self.agent_modes["listener"],
                    "action": "index_track",
                    "detail": f"Indexed '{prof.title}': {prof.bpm:.1f} BPM, Key {prof.camelot_global} ({prof.key_name}), {prof.integrated_lufs:.1f} LUFS",
                })
        return {"profiles": profiles, "agent_events": events}

    def _director_node(self, state: DJCrewState) -> dict[str, Any]:
        wanted_paths = {os.path.abspath(p) for p in state.get("track_paths", [])}
        profiles_map = {
            tid: _dict_to_track_profile(d)
            for tid, d in state["profiles"].items()
            if not wanted_paths or os.path.abspath(d.get("path", "")) in wanted_paths
        }
        plan = state.get("set_plan")
        events = list(state.get("agent_events", []))

        if not plan:
            set_plan = self.director.plan_setlist_beam_search(
                tracks=list(profiles_map.values()),
                steering_prompt=state.get("steering_prompt", "Progressive club energy arc"),
                bridge_engine=state.get("bridge_engine", "acestep"),
            )
            plan = set_plan.to_dict()
            events.append({
                "timestamp": round(time.time(), 3),
                "agent": "DirectorAgent",
                "mode": self.agent_modes["director"],
                "action": "plan_setlist",
                "detail": f"Planned {len(plan['ordered_track_ids'])} tracks ({len(plan['transitions'])} transitions), global_score={plan['global_score']:.2f}",
            })

        idx = state.get("current_transition_idx", 0)
        transitions = plan.get("transitions", [])
        if idx >= len(transitions):
            return {"set_plan": plan, "agent_events": events}

        spec = dict(transitions[idx])
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": "DirectorAgent",
            "mode": self.agent_modes["director"],
            "action": "select_transition",
            "detail": f"Transition #{idx + 1}: {spec['track_a_title']} -> {spec['track_b_title']} via {spec['archetype']} ({spec['archetype_name']})",
        })
        return {
            "set_plan": plan,
            "current_transition_idx": idx,
            "retry_count": 0,
            "current_spec": spec,
            "agent_events": events,
        }

    def _producer_node(self, state: DJCrewState) -> dict[str, Any]:
        spec = dict(state.get("current_spec", {}))
        events = list(state.get("agent_events", []))
        if not spec or not spec.get("requires_ai_bridge"):
            return {"current_spec": spec, "agent_events": events}

        prof_a = _dict_to_track_profile(state["profiles"][spec["track_a_id"]])
        prof_b = _dict_to_track_profile(state["profiles"][spec["track_b_id"]])
        stems_a = dict(np.load(prof_a.stems_cache_path)) if os.path.isfile(prof_a.stems_cache_path) else None

        bridge = self.producer.produce_bridge(
            track_a_title=prof_a.title,
            track_b_title=prof_b.title,
            stems_a=stems_a,
            target_bpm=spec["target_bpm"],
            target_camelot=prof_b.camelot_cue_in,
            bars=spec["overlap_bars"],
            engine=spec.get("bridge_engine", "acestep"),
        )
        spec["bridge_wav_path"] = bridge.output_path
        spec["archetype"] = "T6"
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": "ProducerAgent",
            "mode": self.agent_modes["producer"],
            "action": "produce_bridge",
            "detail": f"Generated {spec['overlap_bars']}-bar bridge via {bridge.backend_used} -> {os.path.basename(bridge.output_path)}",
        })
        return {"current_spec": spec, "agent_events": events}

    def _host_node(self, state: DJCrewState) -> dict[str, Any]:
        spec = dict(state.get("current_spec", {}))
        events = list(state.get("agent_events", []))
        voiceovers = list(state.get("host_voiceovers", []))
        if not spec or not state.get("enable_host_mc", True) or state.get("retry_count", 0) > 0:
            return {"host_voiceovers": voiceovers, "agent_events": events}

        script = spec.get("host_script") or f"Mixing into {spec.get('track_b_title', 'next track')}!"
        vo = self.host.synthesize_drop(script=script)
        voiceovers.append(vo.to_dict())
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": "HostAgent",
            "mode": self.agent_modes["host"],
            "action": "synthesize_mc_drop",
            "detail": f"Synthesized {vo.duration_sec:.2f}s MC voiceover via {vo.backend_used}",
        })
        return {"host_voiceovers": voiceovers, "agent_events": events}

    def _mixer_node(self, state: DJCrewState) -> dict[str, Any]:
        spec_dict = dict(state["current_spec"])
        spec = TransitionSpec(**spec_dict)
        prof_a = _dict_to_track_profile(state["profiles"][spec.track_a_id])
        prof_b = _dict_to_track_profile(state["profiles"][spec.track_b_id])
        retry = state.get("retry_count", 0)

        out_wav = os.path.join(
            self.cache_dir,
            "mixer",
            f"candidate_{ state.get('current_transition_idx', 0) }_try{retry}_{spec.archetype}.wav",
        )
        rendered = self.mixer.render_transition(prof_a, prof_b, spec, output_wav_path=out_wav)

        # Mix in HostAgent MC voiceover with -10 dB mid-band sidechain ducking if enabled
        voiceovers = state.get("host_voiceovers", [])
        if state.get("enable_host_mc", True) and voiceovers:
            latest_vo = voiceovers[-1]
            if os.path.isfile(latest_vo["wav_path"]):
                m_stereo, _ = decode_audio_stereo(rendered.candidate_wav_path, sr=SAMPLE_RATE)
                v_stereo, _ = decode_audio_stereo(latest_vo["wav_path"], sr=SAMPLE_RATE)
                ducked_stereo, duck_stats = self.host.mix_voiceover_with_midband_ducking(
                    m_stereo, v_stereo, insert_offset_samples=SAMPLE_RATE // 2, duck_db=-10.0, sr=SAMPLE_RATE
                )
                sf.write(rendered.candidate_wav_path, ducked_stereo.T, SAMPLE_RATE, subtype="FLOAT")
                if "integrated_lufs" in duck_stats:
                    rendered.integrated_lufs = duck_stats["integrated_lufs"]
                    rendered.true_peak_dbtp = duck_stats["true_peak_dbtp"]

        # Optional test hook for verifying CriticAgent's self-healing loop on first try
        if state.get("inject_test_defect_on_first_try") and retry == 0:
            rendered.simultaneous_bass_samples = 2048
            rendered.max_bass_gain_sum = 1.65

        events = list(state.get("agent_events", []))
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": "MixerAgent",
            "mode": self.agent_modes["mixer"],
            "action": "render_transition",
            "detail": (
                f"Rendered {spec.archetype} (try #{retry + 1}): {rendered.integrated_lufs:.1f} LUFS, "
                f"{rendered.true_peak_dbtp:.2f} dBTP, beat_err={rendered.beat_alignment_error_ms:.3f}ms"
            ),
        })
        return {"current_rendered": rendered.to_dict(), "agent_events": events}

    def _critic_node(self, state: DJCrewState) -> dict[str, Any]:
        spec = TransitionSpec(**state["current_spec"])
        rendered = RenderedTransition(**state["current_rendered"])
        prof_a = _dict_to_track_profile(state["profiles"][spec.track_a_id])
        prof_b = _dict_to_track_profile(state["profiles"][spec.track_b_id])
        retry = state.get("retry_count", 0)
        max_retries = state.get("max_critic_retries", 2)

        report = self.critic.evaluate_transition(prof_a, prof_b, spec, rendered)
        events = list(state.get("agent_events", []))

        if report.verdict == "REJECT" and retry < max_retries and report.adjustment is not None:
            healed_spec = self.critic.apply_self_healing_adjustment(spec, report.adjustment)
            events.append({
                "timestamp": round(time.time(), 3),
                "agent": "CriticAgent",
                "mode": self.agent_modes["critic"],
                "action": "reject_and_self_heal",
                "detail": f"REJECT (try #{retry + 1}): {report.adjustment.reason} -> Upgrading to {healed_spec.archetype}",
            })
            return {
                "current_critique": report.to_dict(),
                "current_spec": healed_spec.to_dict(),
                "retry_count": retry + 1,
                "agent_events": events,
            }

        # Certified PASS (or max retries reached): record certified transition, reset retry_count, and advance index
        certified = list(state.get("certified_transitions", []))
        certified.append({
            "transition_index": state.get("current_transition_idx", 0),
            "spec": spec.to_dict(),
            "rendered": rendered.to_dict(),
            "critique": report.to_dict(),
            "retries_used": retry,
        })
        events.append({
            "timestamp": round(time.time(), 3),
            "agent": "CriticAgent",
            "mode": self.agent_modes["critic"],
            "action": "certify_pass",
            "detail": (
                f"CERTIFIED {report.verdict} after {retry + 1} attempt(s): "
                f"Stage1={report.stage1_dsp_pass}, Stage2={report.stage2_perceptual_pass}"
            ),
        })
        return {
            "current_critique": report.to_dict(),
            "certified_transitions": certified,
            "current_transition_idx": state.get("current_transition_idx", 0) + 1,
            "retry_count": 0,
            "agent_events": events,
        }

    def _route_after_director(self, state: DJCrewState) -> str:
        if getattr(self, "stop_requested", False):
            return "end"
        plan = state.get("set_plan", {})
        transitions = plan.get("transitions", [])
        idx = state.get("current_transition_idx", 0)
        if idx >= len(transitions):
            return "end"
        spec = state.get("current_spec", {})
        if spec.get("requires_ai_bridge"):
            return "producer"
        return "host"

    def _route_after_critic(self, state: DJCrewState) -> str:
        if getattr(self, "stop_requested", False):
            return "end"
        if state.get("retry_count", 0) > 0:
            spec_now = state.get("current_spec", {})
            if spec_now.get("requires_ai_bridge") and not spec_now.get("bridge_wav_path"):
                return "producer"
            return "mixer"

        plan = state.get("set_plan", {})
        transitions = plan.get("transitions", [])
        if state.get("current_transition_idx", 0) < len(transitions):
            return "director"
        return "end"

    def _build_graph(self) -> Any:
        workflow = StateGraph(DJCrewState)
        workflow.add_node("listener", self._listener_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("producer", self._producer_node)
        workflow.add_node("host", self._host_node)
        workflow.add_node("mixer", self._mixer_node)
        workflow.add_node("critic", self._critic_node)

        workflow.add_edge(START, "listener")
        workflow.add_edge("listener", "director")
        workflow.add_conditional_edges(
            "director",
            self._route_after_director,
            {"producer": "producer", "host": "host", "end": END},
        )
        workflow.add_edge("producer", "host")
        workflow.add_edge("host", "mixer")
        workflow.add_edge("mixer", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"producer": "producer", "mixer": "mixer", "director": "director", "end": END},
        )
        return workflow.compile()

    def run_session(
        self,
        track_paths: list[str],
        steering_prompt: str = "High-energy progressive club mix",
        bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep",
        enable_host_mc: bool = True,
        max_critic_retries: int = 2,
        inject_test_defect_on_first_try: bool = False,
        existing_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> DJCrewState:
        """Executes the full 6-Agent LangGraph DJ session across `track_paths`."""
        self.stop_requested = False
        initial_state: DJCrewState = {
            "track_paths": [os.path.abspath(p) for p in track_paths],
            "steering_prompt": steering_prompt,
            "agent_modes": dict(self.agent_modes),
            "audio_ear_model": self.audio_ear_model,
            "bridge_engine": bridge_engine,
            "enable_host_mc": enable_host_mc,
            "max_critic_retries": max_critic_retries,
            "inject_test_defect_on_first_try": inject_test_defect_on_first_try,
            "profiles": dict(existing_profiles or {}),
            "set_plan": {},
            "current_transition_idx": 0,
            "retry_count": 0,
            "certified_transitions": [],
            "host_voiceovers": [],
            "agent_events": [],
        }
        return self.graph.invoke(initial_state)
