#!/usr/bin/env python3
"""DirectorAgent (`autodj_planner.py`) — Multi-Objective Compatibility Graph, Energy Arc Beam Search & Set Brain.

Implements:
1. Multi-Objective Track Compatibility Scoring $S(A, B) \\in [0, 1]$ across Camelot key distance,
   BPM stretch feasibility, Energy Arc alignment, and Vocal Clash avoidance.
2. Automatic selection of the 6 Professional DJ Transition Archetypes (`T1`-`T6`):
   - `T1`: Phrase-Aligned 4-Stem Blend & Hard Bass Swap
   - `T2`: Vocal-Over-Instrumental Mashup Handshake
   - `T3`: Breakdown-to-Build "Double Drop" Switch
   - `T4`: Echo-Freeze / High-Pass Wash Tempo Reset (`>= 8%` BPM jump)
   - `T5`: Harmonic Pitch-Shift Bridge (`+/- 1` or `2` semitones via `librubberband.so.2`)
   - `T6`: AI Generative Bridge (`ACE-Step 1.5` / `MusicGen-Stereo-Melody` / `Suno` commissioned bridge)
3. Beam Search (`K=8`) global setlist optimizer & natural-language crowd steering (Gemini Developer API).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal

import numpy as np

from autodj_gemini import GeminiError, configured_model, generate_text
from autodj_indexer import CAMELOT_MAJOR, CAMELOT_MINOR, PITCH_CLASSES, TrackProfile, camelot_distance


ArchetypeId = Literal["T1", "T2", "T3", "T4", "T5", "T6"]

ARCHETYPE_DESCRIPTIONS: dict[ArchetypeId, str] = {
    "T1": "Phrase-Aligned 4-Stem Blend & Hard Bass Swap",
    "T2": "Vocal-Over-Instrumental Mashup Handshake",
    "T3": "Breakdown-to-Build Double-Drop Switch",
    "T4": "Echo-Freeze & High-Pass Wash Tempo Reset",
    "T5": "Harmonic Pitch-Shift Bridge (+/-1..2 Semitones)",
    "T6": "AI Generative Bridge (ACE-Step 1.5 / MusicGen-Melody / Suno)",
}

# Reverse lookup from Camelot code to pitch class index (0=C .. 11=B) and scale mode
CAMELOT_TO_PITCH_INDEX: dict[str, tuple[int, str]] = {}
for _i, _p in enumerate(PITCH_CLASSES):
    CAMELOT_TO_PITCH_INDEX[CAMELOT_MAJOR[_p]] = (_i, "B")
    CAMELOT_TO_PITCH_INDEX[CAMELOT_MINOR[_p]] = (_i, "A")


@dataclass
class TransitionSpec:
    """Complete deterministic mixing instructions from `DirectorAgent` to `MixerAgent`."""

    track_a_id: str
    track_b_id: str
    track_a_title: str
    track_b_title: str
    archetype: ArchetypeId
    archetype_name: str
    compatibility_score: float
    overlap_bars: int
    cue_out_bar_a: int
    cue_in_bar_b: int
    bass_swap_bar: int
    target_bpm: float
    pitch_shift_semitones_b: float
    vocal_attenuation_db_a: float
    vocal_attenuation_db_b: float
    requires_ai_bridge: bool
    bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep"
    bridge_wav_path: str | None = None
    host_script: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetPlan:
    """Ordered DJ set plan produced by `DirectorAgent`."""

    ordered_track_ids: list[str]
    transitions: list[TransitionSpec]
    global_score: float
    steering_prompt: str
    director_notes: str
    energy_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_minimal_semitone_shift(cam_a: str, cam_b: str) -> int:
    """Finds if shifting Track B by -1, +1, -2, or +2 semitones aligns Track B's Camelot clock number with Track A."""
    d_key = camelot_distance(cam_a, cam_b)
    if d_key <= 1 or d_key >= 6:
        return 0
    if cam_b not in CAMELOT_TO_PITCH_INDEX:
        return 0
    try:
        num_a = int(cam_a[:-1])
    except Exception:
        return 0
    pc_b, mode_b = CAMELOT_TO_PITCH_INDEX[cam_b]
    table = CAMELOT_MINOR if mode_b == "A" else CAMELOT_MAJOR
    for shift in (-1, 1, -2, 2):
        shifted_pc = PITCH_CLASSES[(pc_b + shift) % 12]
        shifted_cam = table[shifted_pc]
        if int(shifted_cam[:-1]) == num_a and camelot_distance(cam_a, shifted_cam) <= 1:
            return shift
    return 0


def score_pair_compatibility(
    track_a: TrackProfile,
    track_b: TrackProfile,
    target_energy_delta: float = 0.05,
    steering_prompt: str = "",
) -> dict[str, Any]:
    """Computes multi-objective compatibility score $S(A, B)$ and selects optimal archetype `T1`-`T6`."""
    cam_a = track_a.camelot_cue_out or track_a.camelot_global
    cam_b = track_b.camelot_cue_in or track_b.camelot_global
    d_key = camelot_distance(cam_a, cam_b)
    semi_shift = find_minimal_semitone_shift(cam_a, cam_b)

    if d_key == 0:
        s_key = 1.00
    elif d_key == 1:
        s_key = 0.92
    elif semi_shift != 0:
        s_key = 0.78
    elif d_key == 2:
        s_key = 0.62
    elif d_key >= 6:
        s_key = 0.15
    else:
        s_key = 0.25

    bpm_delta_pct = abs(track_b.bpm - track_a.bpm) / max(track_a.bpm, 1.0) * 100.0
    if bpm_delta_pct <= 4.0:
        s_bpm = 1.0 - 0.03 * bpm_delta_pct
    elif bpm_delta_pct <= 8.0:
        s_bpm = 0.88 - 0.06 * (bpm_delta_pct - 4.0)
    else:
        s_bpm = max(0.20, 0.55 - 0.03 * (bpm_delta_pct - 8.0))

    actual_energy_delta = track_b.energy_score - track_a.energy_score
    s_energy = float(max(0.2, 1.0 - abs(actual_energy_delta - target_energy_delta) * 1.2))

    # Check vocal overlap across 8-bar transition window
    n_bars_a = len(track_a.downbeats_sec)
    n_bars_b = len(track_b.downbeats_sec)
    overlap_bars = min(8, max(2, min(n_bars_a // 2, n_bars_b // 2)))

    out_bar_a = min(max(0, track_a.cue_out_bar), max(0, n_bars_a - overlap_bars))
    in_bar_b = min(max(0, track_b.cue_in_bar), max(0, n_bars_b - overlap_bars))

    voc_a = track_a.vocal_active_bars[out_bar_a : out_bar_a + overlap_bars]
    voc_b = track_b.vocal_active_bars[in_bar_b : in_bar_b + overlap_bars]
    both_vocal_bars = sum(1 for a, b in zip(voc_a, voc_b) if a and b)
    either_vocal_only = sum(1 for a, b in zip(voc_a, voc_b) if (a ^ b))

    s_vocal = 1.0 - 0.12 * both_vocal_bars
    raw_score = float(np.clip(0.38 * s_key + 0.32 * s_bpm + 0.18 * s_energy + 0.12 * s_vocal, 0.0, 1.0))

    prompt_lower = steering_prompt.lower()
    force_bridge = any(k in prompt_lower for k in ("ai bridge", "t6", "generate bridge", "morph"))
    force_double_drop = "double drop" in prompt_lower or "t3" in prompt_lower
    force_mashup = "mashup" in prompt_lower or "vocal" in prompt_lower or "t2" in prompt_lower

    pitch_shift_b = 0.0
    requires_ai_bridge = False

    if (
        force_bridge
        or raw_score < 0.55
        or d_key >= 6
        or (d_key >= 3 and semi_shift == 0)
        or (bpm_delta_pct >= 8.0 and d_key >= 2)
    ):
        archetype: ArchetypeId = "T6"
        requires_ai_bridge = True
    elif bpm_delta_pct >= 8.0:
        archetype = "T4"
    elif d_key > 1 and semi_shift != 0:
        archetype = "T5"
        pitch_shift_b = float(semi_shift)
    elif force_double_drop or (
        track_a.energy_score >= 0.72 and track_b.energy_score >= 0.72 and bpm_delta_pct <= 5.0 and d_key <= 1
    ):
        archetype = "T3"
    elif force_mashup or (either_vocal_only >= overlap_bars // 2 and both_vocal_bars == 0 and d_key <= 1):
        archetype = "T2"
    else:
        archetype = "T1"

    # Configure vocal attenuation if both tracks have vocals in the transition window
    voc_atten_a = 0.0
    voc_atten_b = -12.0 if both_vocal_bars > 0 else 0.0

    # Target transition BPM: smooth compromise when within 6%, else lock to Track B after reset
    target_bpm = round(0.5 * (track_a.bpm + track_b.bpm), 2) if bpm_delta_pct < 8.0 else track_b.bpm
    bass_swap_bar = max(1, overlap_bars // 2)

    return {
        "score": round(raw_score, 4),
        "archetype": archetype,
        "overlap_bars": overlap_bars,
        "cue_out_bar_a": out_bar_a,
        "cue_in_bar_b": in_bar_b,
        "bass_swap_bar": bass_swap_bar,
        "target_bpm": target_bpm,
        "pitch_shift_semitones_b": pitch_shift_b,
        "vocal_attenuation_db_a": voc_atten_a,
        "vocal_attenuation_db_b": voc_atten_b,
        "requires_ai_bridge": requires_ai_bridge,
        "d_key": d_key,
        "bpm_delta_pct": round(bpm_delta_pct, 2),
    }


class DirectorAgent:
    """Agent 1 (`DirectorAgent`): Plans setlist order, energy arc, transition archetypes `T1`-`T6` & MC cues."""

    def __init__(self, mode: Literal["local", "cloud", "hybrid"] = "local") -> None:
        self.mode = mode

    def _request_track_note(self, prompt: str) -> str | None:
        try:
            return generate_text(configured_model(), "Write a short note for the DJ.", prompt)
        except GeminiError:
            return None

    def generate_live_track_concept(
        self,
        steering_prompt: str = "Peak-time progressive club energy arc",
        prev_track: TrackProfile | dict[str, Any] | None = None,
        cycle_idx: int = 1,
    ) -> dict[str, Any]:
        """Generates the next autonomous live DJ track concept (title, style_prompt, target_bpm, target_camelot, mc_script)."""
        creative_titles = [
            "Solaris_Pulse",
            "Obsidian_Mirage",
            "Hyperion_Drift",
            "Valkyrie_Sub",
            "Aether_Resonance",
            "Quantum_Strobe",
            "Velvet_Horizon",
            "Krypton_Echo",
            "Chronos_Drive",
            "Nebula_Ignition",
            "Midnight_Laser",
            "Helios_Ascent",
        ]
        if isinstance(prev_track, dict):
            prev_cam = str(prev_track.get("camelot_global") or prev_track.get("target_camelot") or "8A").upper()
            prev_bpm = float(prev_track.get("bpm") or prev_track.get("target_bpm") or 126.0)
            prev_title = str(prev_track.get("title") or "the opening set")
        elif prev_track is not None:
            prev_cam = prev_track.camelot_global.upper()
            prev_bpm = float(prev_track.bpm)
            prev_title = prev_track.title
        else:
            prev_cam = "8A"
            prev_bpm = 126.0
            prev_title = "the opening set"

        # Pick a harmonically adjacent Camelot key (d_key <= 1) on the Camelot wheel
        try:
            num = int(prev_cam[:-1])
            letter = prev_cam[-1]
        except Exception:
            num, letter = 8, "A"

        adjacent_keys = [
            f"{num}{letter}",
            f"{(num % 12) + 1}{letter}",
            f"{((num - 2) % 12) + 1}{letter}",
            f"{num}{'B' if letter == 'A' else 'A'}",
        ]
        next_camelot = adjacent_keys[cycle_idx % len(adjacent_keys)]

        # Smooth BPM progression within +/- 2 BPM of prev_track.bpm (clamped 124..132 BPM)
        bpm_steps = [0.0, 2.0, 1.0, -1.0, 2.0, -2.0]
        raw_bpm = prev_bpm + bpm_steps[cycle_idx % len(bpm_steps)]
        next_bpm = float(round(min(132.0, max(124.0, raw_bpm))))

        base_title = f"{creative_titles[(cycle_idx - 1) % len(creative_titles)]}_{next_camelot}"
        style_prompt = (
            f"{steering_prompt}, driving analog bassline, hypnotic melodic synth arpeggio, "
            f"crisp club percussion, {next_bpm:.0f} BPM, Camelot key {next_camelot}"
        )
        mc_script = (
            f"Live on the Neural Deck autonomous stream! Transitioning out of {prev_title} "
            f"into {base_title.replace('_', ' ')} at {next_bpm:.0f} BPM in key {next_camelot}!"
        )

        llm_raw = self._request_track_note(
            f"You are an autonomous AI club DJ Director. Crowd vibe: '{steering_prompt}'. "
            f"Previous track: '{prev_title}' ({prev_bpm:.0f} BPM, {prev_cam}). "
            f"Reply with ONLY a short 2-3 word futuristic track title and a 1-sentence DJ MC hype shoutout separated by '|'.",
        )
        llm_used = "Deterministic Camelot-Graph Planner"
        if llm_raw and "|" in llm_raw:
            parts = [p.strip().strip('"').strip("'") for p in llm_raw.split("|", 1)]
            if parts[0] and len(parts[0]) <= 36:
                clean_t = "".join(c if (c.isalnum() or c in " _-") else "" for c in parts[0]).strip().replace(" ", "_")
                if clean_t:
                    base_title = f"{clean_t}_{next_camelot}"
            if len(parts) > 1 and len(parts[1]) >= 12:
                mc_script = parts[1][:200]
            llm_used = f"Gemini API ({configured_model()})"

        return {
            "title": base_title,
            "style": style_prompt,
            "style_prompt": style_prompt,
            "target_bpm": next_bpm,
            "target_camelot": next_camelot,
            "host_script": mc_script,
            "mc_script": mc_script,
            "llm_used": llm_used,
        }


    def plan_transition(
        self,
        track_a: TrackProfile,
        track_b: TrackProfile,
        steering_prompt: str = "",
        bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep",
    ) -> TransitionSpec:
        """Creates a single `TransitionSpec` between `track_a` and `track_b`."""
        comp = score_pair_compatibility(track_a, track_b, steering_prompt=steering_prompt)
        arch: ArchetypeId = comp["archetype"]
        arch_name = ARCHETYPE_DESCRIPTIONS[arch]

        rationale = (
            f"Selected {arch} ({arch_name}): Camelot {track_a.camelot_cue_out}->{track_b.camelot_cue_in} "
            f"(d={comp['d_key']}), BPM {track_a.bpm:.1f}->{track_b.bpm:.1f} ({comp['bpm_delta_pct']:.1f}%), "
            f"compatibility={comp['score']:.2f}."
        )
        host_script = (
            f"You are locked into the AI Auto DJ studio! Blending out of {track_a.title} "
            f"and dropping straight into {track_b.title} at {comp['target_bpm']:.0f} BPM!"
        )

        return TransitionSpec(
            track_a_id=track_a.track_id,
            track_b_id=track_b.track_id,
            track_a_title=track_a.title,
            track_b_title=track_b.title,
            archetype=arch,
            archetype_name=arch_name,
            compatibility_score=comp["score"],
            overlap_bars=comp["overlap_bars"],
            cue_out_bar_a=comp["cue_out_bar_a"],
            cue_in_bar_b=comp["cue_in_bar_b"],
            bass_swap_bar=comp["bass_swap_bar"],
            target_bpm=comp["target_bpm"],
            pitch_shift_semitones_b=comp["pitch_shift_semitones_b"],
            vocal_attenuation_db_a=comp["vocal_attenuation_db_a"],
            vocal_attenuation_db_b=comp["vocal_attenuation_db_b"],
            requires_ai_bridge=comp["requires_ai_bridge"],
            bridge_engine=bridge_engine,
            host_script=host_script,
            rationale=rationale,
        )

    def plan_setlist_beam_search(
        self,
        tracks: list[TrackProfile],
        steering_prompt: str = "Progressive club energy arc",
        beam_width: int = 8,
        bridge_engine: Literal["acestep", "musicgen_melody", "suno"] = "acestep",
        query_llm: bool = False,
    ) -> SetPlan:
        """Optimizes global track sequence via Beam Search (`K=8`) and builds all `TransitionSpec` steps."""
        if not tracks:
            raise ValueError("Cannot plan setlist on an empty track list")
        if len(tracks) == 1:
            return SetPlan(
                ordered_track_ids=[tracks[0].track_id],
                transitions=[],
                global_score=1.0,
                steering_prompt=steering_prompt,
                director_notes=f"Single-track showcase: {tracks[0].title}",
                energy_curve=[tracks[0].energy_score],
            )

        by_id = {t.track_id: t for t in tracks}
        n = len(tracks)

        # Initialize beam with each track as a candidate opener (prefer moderate warm-up energy)
        beams: list[tuple[float, list[str]]] = []
        for t in tracks:
            opener_score = 1.0 - 0.35 * abs(t.energy_score - 0.55)
            beams.append((opener_score, [t.track_id]))
        beams.sort(key=lambda x: x[0], reverse=True)
        beams = beams[:beam_width]

        for step in range(1, n):
            # Target energy rises through first 75% of set, then cools slightly at final track
            target_delta = 0.08 if step < max(1, int(0.75 * n)) else -0.05
            candidates: list[tuple[float, list[str]]] = []
            for cum_score, seq in beams:
                last_track = by_id[seq[-1]]
                for nxt in tracks:
                    if nxt.track_id in seq:
                        continue
                    pair = score_pair_compatibility(
                        last_track, nxt, target_energy_delta=target_delta, steering_prompt=steering_prompt
                    )
                    candidates.append((cum_score + pair["score"], seq + [nxt.track_id]))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        best_total_score, best_seq = beams[0]
        transitions: list[TransitionSpec] = []
        for i in range(len(best_seq) - 1):
            ta = by_id[best_seq[i]]
            tb = by_id[best_seq[i + 1]]
            transitions.append(
                self.plan_transition(ta, tb, steering_prompt=steering_prompt, bridge_engine=bridge_engine)
            )

        avg_score = round(best_total_score / max(1, len(transitions)), 4)
        llm_note = self._request_track_note(
            f"Summarize this DJ set in one sentence: {[by_id[tid].title for tid in best_seq]}, prompt={steering_prompt}"
        ) if query_llm else None
        notes = llm_note or (
            f"Beam-search optimized {len(best_seq)}-track set ({avg_score:.2f} compatibility) "
            f"steered for '{steering_prompt}'."
        )

        return SetPlan(
            ordered_track_ids=best_seq,
            transitions=transitions,
            global_score=avg_score,
            steering_prompt=steering_prompt,
            director_notes=notes,
            energy_curve=[by_id[tid].energy_score for tid in best_seq],
        )
