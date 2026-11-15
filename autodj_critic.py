#!/usr/bin/env python3
"""CriticAgent (`autodj_critic.py`) — Booth Monitor & 2-Stage QA Gate with Self-Healing Re-Mix Loop.

Implements:
1. Stage 1 (Deterministic Automated DSP QC):
   - Beat alignment error `< 1.0 ms`
   - 4x-oversampled True Peak `<= -0.5 dBTP`
   - Loudness step delta across transition `<= 1.5 LU`
   - Bass-Swap Law compliance (`simultaneous_bass_samples == 0`, `max_bass_gain_sum <= 1.05`)
   - Zero boundary splice clicks (`max_splice_discontinuity <= 0.35`)
2. Stage 2 (Dual Acoustic Ear + 4-Panel Visual-Acoustic Spectrogram Bridge):
   - Pluggable Native Audio Ear (`nvidia/music-flamingo-think-2601-hf`, `nvidia/audio-flamingo-next-think-hf`,
     `google/gemma-4-E4B-it`, or `Qwen/Qwen3-Omni-30B-A3B-Instruct`)
   - Renders the 4-panel `1024x1024` Visual-Acoustic Spectrogram PNG of `/dev/shm/candidate_mix.wav`
     for `Qwen3.6-35B-A3B`'s 446M Vision Tower
   - Emits structured `PASS` / `REJECT` verdict + actionable `MixAdjustment` for `MixerAgent`'s self-healing loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Literal

import numpy as np

from autodj_indexer import (
    AudioEarDriver,
    SAMPLE_RATE,
    TrackProfile,
    camelot_distance,
    decode_audio_stereo,
    measure_loudness_and_peak,
    render_4panel_spectrogram_png,
    segment_structure_and_relabel,
    separate_stems_htdemucs,
    shift_camelot_key,
    track_beats_and_downbeats,
)
from autodj_mixer import RenderedTransition
from autodj_planner import ARCHETYPE_DESCRIPTIONS, ArchetypeId, TransitionSpec, find_minimal_semitone_shift


@dataclass
class MixAdjustment:
    """Corrective parameters emitted by `CriticAgent` when a transition is rejected."""

    recommended_archetype: ArchetypeId
    bass_swap_bar: int
    vocal_attenuation_db_b: float
    pitch_shift_semitones_b: float
    requires_ai_bridge: bool
    reason: str


@dataclass
class CritiqueReport:
    """2-Stage QA verification record from `CriticAgent`."""

    verdict: Literal["PASS", "REJECT"]
    stage1_dsp_pass: bool
    stage2_perceptual_pass: bool
    beat_alignment_error_ms: float
    true_peak_dbtp: float
    lufs_step_delta_lu: float
    max_bass_gain_sum: float
    simultaneous_bass_samples: int
    max_splice_discontinuity: float
    harmonic_clash_score: float
    vocal_clash_score: float
    candidate_spectrogram_png: str
    failed_checks: list[str] = field(default_factory=list)
    stage2_notes: dict[str, Any] = field(default_factory=dict)
    adjustment: MixAdjustment | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CriticAgent:
    """Agent 5 (`CriticAgent`): Audits candidate transitions in `/dev/shm` via Stage-1 DSP + Stage-2 Audio/Vision."""

    def __init__(
        self,
        mode: Literal["local", "cloud", "hybrid"] = "local",
        audio_ear_model: str = "nvidia/music-flamingo-think-2601-hf",
        vision_ollama_model: str = "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
        cache_dir: str = "/dev/shm/autodj_critic",
    ) -> None:
        self.mode = mode
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.ear = AudioEarDriver(
            audio_model_id=audio_ear_model,
            vision_ollama_model=vision_ollama_model,
            mode=mode,
        )

    def evaluate_transition(
        self,
        track_a: TrackProfile,
        track_b: TrackProfile,
        spec: TransitionSpec,
        rendered: RenderedTransition,
        query_ollama: bool = False,
    ) -> CritiqueReport:
        """Runs Stage-1 Deterministic DSP QC and Stage-2 Dual Acoustic + 4-Panel Spectrogram Critique."""
        stereo, sr = decode_audio_stereo(rendered.candidate_wav_path, sr=SAMPLE_RATE)
        mid_idx = stereo.shape[1] // 2
        stats_first = measure_loudness_and_peak(stereo[:, :mid_idx], sr=sr)
        stats_second = measure_loudness_and_peak(stereo[:, mid_idx:], sr=sr)
        stats_full = measure_loudness_and_peak(stereo, sr=sr)

        lufs_delta = round(abs(stats_second["integrated_lufs"] - stats_first["integrated_lufs"]), 3)
        tp_dbtp = stats_full["true_peak_dbtp"]

        # Stage 1 Deterministic Checks
        failed: list[str] = []
        if rendered.beat_alignment_error_ms >= 1.0:
            failed.append(f"beat_alignment_error_ms={rendered.beat_alignment_error_ms:.3f}ms >= 1.0ms")
        if tp_dbtp > -0.5:
            failed.append(f"true_peak_dbtp={tp_dbtp:.2f}dBTP > -0.5dBTP")
        if lufs_delta > 1.5:
            failed.append(f"lufs_step_delta_lu={lufs_delta:.2f}LU > 1.5LU")
        if rendered.simultaneous_bass_samples > 0 or rendered.max_bass_gain_sum > 1.05:
            failed.append(
                f"bass_swap_violation (sim_samples={rendered.simultaneous_bass_samples}, max_sum={rendered.max_bass_gain_sum:.3f})"
            )
        if rendered.max_splice_discontinuity > 0.35:
            failed.append(f"splice_click_discontinuity={rendered.max_splice_discontinuity:.3f} > 0.35")

        stage1_pass = len(failed) == 0

        # Stage 2: Render 4-Panel Spectrogram PNG of the candidate transition & run Acoustic + Vision critique
        stems_cand, _ = separate_stems_htdemucs(
            stereo, sr=sr, cache_dir=os.path.join(self.cache_dir, "stems"), prefer_neural=False
        )
        rhythm_cand = track_beats_and_downbeats(stereo, stems_cand, sr=sr)
        secs_cand, _, _, _, _ = segment_structure_and_relabel(
            stereo, stems_cand, rhythm_cand["downbeats_sec"], stereo.shape[1] / sr, sr=sr
        )
        spec_png = os.path.join(
            self.cache_dir, f"critic_4panel_{track_a.track_id}_{track_b.track_id}_{spec.archetype}.png"
        )
        render_4panel_spectrogram_png(
            stereo=stereo,
            stems=stems_cand,
            beats_sec=rhythm_cand["beats_sec"],
            downbeats_sec=rhythm_cand["downbeats_sec"],
            sections=secs_cand,
            output_png_path=spec_png,
            sr=sr,
            title=f"Candidate Transition {track_a.title} -> {track_b.title} [{spec.archetype}]",
        )

        # Evaluate harmonic compatibility, tempo jump rules & vocal collision in Stage 2
        raw_d_key = camelot_distance(track_a.camelot_cue_out, track_b.camelot_cue_in)
        effective_cam_b = track_b.camelot_cue_in
        if spec.archetype == "T5" and abs(spec.pitch_shift_semitones_b) > 0 and raw_d_key < 6:
            shifted_cam_b = shift_camelot_key(effective_cam_b, int(round(spec.pitch_shift_semitones_b)))
            if int(shifted_cam_b[:-1]) == int(track_a.camelot_cue_out[:-1]):
                effective_cam_b = shifted_cam_b
        d_key = camelot_distance(track_a.camelot_cue_out, effective_cam_b)
        if spec.archetype == "T6" and spec.bridge_wav_path:
            d_key = min(1, d_key)

        harmonic_clash = round(max(0.0, (d_key - 1) * 0.35), 3)
        bpm_delta_pct = abs(track_b.bpm - track_a.bpm) / max(track_a.bpm, 1.0) * 100.0
        tempo_jump_violation = bpm_delta_pct >= 12.0 and spec.archetype not in ("T4", "T6")

        # Check if unattenuated vocal collision occurred
        voc_a_bars = track_a.vocal_active_bars[spec.cue_out_bar_a : spec.cue_out_bar_a + spec.overlap_bars]
        voc_b_bars = track_b.vocal_active_bars[spec.cue_in_bar_b : spec.cue_in_bar_b + spec.overlap_bars]
        both_voc = sum(1 for a, b in zip(voc_a_bars, voc_b_bars) if a and b)
        vocal_clash = 0.0
        if both_voc > 0 and spec.vocal_attenuation_db_b > -6.0 and spec.archetype not in ("T1", "T4", "T6"):
            vocal_clash = round(min(1.0, both_voc * 0.25), 3)

        if harmonic_clash >= 0.50:
            failed.append(f"stage2_harmonic_clash={harmonic_clash:.2f} (Camelot {track_a.camelot_cue_out}->{track_b.camelot_cue_in})")
        if tempo_jump_violation:
            failed.append(f"stage2_tempo_jump={bpm_delta_pct:.1f}% requires T4 or T6 (got {spec.archetype})")
        if vocal_clash >= 0.50:
            failed.append(f"stage2_vocal_clash={vocal_clash:.2f} (simultaneous unattenuated vocals)")

        stage2_pass = (harmonic_clash < 0.50) and (not tempo_jump_violation) and (vocal_clash < 0.50)
        stage2_notes = self.ear.analyze_vibe_and_mix(
            stereo=stereo,
            sr=sr,
            spectrogram_png_path=spec_png,
            dsp_summary={
                "bpm": spec.target_bpm,
                "camelot_global": track_b.camelot_global,
                "integrated_lufs": stats_full["integrated_lufs"],
                "vocal_active_ratio": 0.0,
            },
            query_ollama=query_ollama,
        )

        verdict: Literal["PASS", "REJECT"] = "PASS" if (stage1_pass and stage2_pass) else "REJECT"
        adjustment: MixAdjustment | None = None
        if verdict == "REJECT":
            semi_fix = find_minimal_semitone_shift(track_a.camelot_cue_out, track_b.camelot_cue_in)
            if harmonic_clash >= 0.50 and semi_fix != 0 and bpm_delta_pct < 8.0 and spec.archetype != "T5":
                rec_arch: ArchetypeId = "T5"
                req_bridge = False
                pitch_b = float(semi_fix)
            elif harmonic_clash >= 0.50:
                rec_arch = "T6"
                req_bridge = True
                pitch_b = 0.0
            elif bpm_delta_pct >= 8.0 and spec.archetype not in ("T4", "T6"):
                if d_key <= 1:
                    rec_arch = "T4"
                    req_bridge = False
                else:
                    rec_arch = "T6"
                    req_bridge = True
                pitch_b = 0.0
            else:
                rec_arch = spec.archetype
                req_bridge = spec.requires_ai_bridge
                pitch_b = spec.pitch_shift_semitones_b

            adjustment = MixAdjustment(
                recommended_archetype=rec_arch,
                bass_swap_bar=max(1, spec.overlap_bars // 2),
                vocal_attenuation_db_b=-15.0,
                pitch_shift_semitones_b=pitch_b,
                requires_ai_bridge=req_bridge,
                reason="; ".join(failed),
            )

        return CritiqueReport(
            verdict=verdict,
            stage1_dsp_pass=stage1_pass,
            stage2_perceptual_pass=stage2_pass,
            beat_alignment_error_ms=rendered.beat_alignment_error_ms,
            true_peak_dbtp=tp_dbtp,
            lufs_step_delta_lu=lufs_delta,
            max_bass_gain_sum=rendered.max_bass_gain_sum,
            simultaneous_bass_samples=rendered.simultaneous_bass_samples,
            max_splice_discontinuity=rendered.max_splice_discontinuity,
            harmonic_clash_score=harmonic_clash,
            vocal_clash_score=vocal_clash,
            candidate_spectrogram_png=spec_png,
            failed_checks=failed,
            stage2_notes=stage2_notes,
            adjustment=adjustment,
        )

    @staticmethod
    def apply_self_healing_adjustment(spec: TransitionSpec, adjustment: MixAdjustment) -> TransitionSpec:
        """Applies `CriticAgent`'s `MixAdjustment` to `TransitionSpec` for the self-healing retry loop."""
        spec.archetype = adjustment.recommended_archetype
        spec.archetype_name = ARCHETYPE_DESCRIPTIONS[adjustment.recommended_archetype]
        spec.bass_swap_bar = adjustment.bass_swap_bar
        spec.vocal_attenuation_db_b = adjustment.vocal_attenuation_db_b
        spec.pitch_shift_semitones_b = adjustment.pitch_shift_semitones_b
        spec.requires_ai_bridge = adjustment.requires_ai_bridge
        spec.rationale += f" | Self-healed by CriticAgent ({adjustment.reason})"
        return spec
