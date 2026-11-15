#!/usr/bin/env python3
"""End-to-End Automated Verification Suite (`tests/test_autodj.py`) for `ai-auto-dj`.

Verifies all 6 agents, all 17 technical trap fixes, Stage-1 & Stage-2 audio/visual QC gates,
`librubberband.so.2` R3 sub-ms beat warping, `AutoencoderOobleck` 1D overlap-add chunked decoding
on CUDA, `MusicgenMelodyFeatureExtractor` 4D stem tensor extraction, `Kokoro-82M` voiceover
synthesis + `-10 dB` mid-band sidechain ducking, `LangGraph` self-healing retry loop, and
FastAPI + WebSocket 2-Deck Web Studio endpoints.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image
import pytest
from scipy import signal
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autodj_critic import CriticAgent
from autodj_crew import AudioProjectorGraft, AutoDJCrew, HostAgent, defuse_moe_3d_experts
from autodj_indexer import (
    AudioEarDriver,
    ListenerAgent,
    SAMPLE_RATE,
    camelot_distance,
    measure_loudness_and_peak,
    natten1dav,
    natten1dqkrpb,
    natten2dav,
    natten2dqkrpb,
    separate_stems_htdemucs,
)
from autodj_mixer import (
    MixerAgent,
    build_stem_automation_envelopes,
    crossfade_join,
    normalize_and_brickwall_limit,
    rubberband_stretch_and_pitch,
)
from autodj_planner import DirectorAgent, find_minimal_semitone_shift
from autodj_producer import (
    LocalMusicGenMelodyDriver,
    ProducerAgent,
    SunoCloudDriver,
    chunked_oobleck_decode,
    synthesize_club_audio,
)
from autodj_server import PipeWireLookaheadStreamer, create_app, export_master_mix_and_cue


# ============================================================================
# 1. Pure-PyTorch `allin1` Neighborhood Attention Shim & AudioEar 16k Guard
# ============================================================================

def test_natten_1d_and_2d_unfold_shims() -> None:
    """Verifies Trap #2 Fix: pure-PyTorch `F.unfold` 1D and 2D neighborhood attention shims."""
    q1 = torch.randn(2, 4, 16, 8)
    k1 = torch.randn(2, 4, 16, 8)
    v1 = torch.randn(2, 4, 16, 8)
    rpb1 = torch.randn(4, 9)
    attn1 = natten1dqkrpb(q1, k1, rpb1, kernel_size=5, dilation=1)
    assert attn1.shape == (2, 4, 16, 5)
    out1 = natten1dav(attn1.softmax(dim=-1), v1, kernel_size=5, dilation=1)
    assert out1.shape == (2, 4, 16, 8)
    assert torch.isfinite(out1).all()

    q2 = torch.randn(1, 2, 6, 6, 8)
    k2 = torch.randn(1, 2, 6, 6, 8)
    v2 = torch.randn(1, 2, 6, 6, 8)
    rpb2 = torch.randn(2, 5, 5)
    attn2 = natten2dqkrpb(q2, k2, rpb2, kernel_size=3, dilation=1)
    assert attn2.shape == (1, 2, 6, 6, 9)
    out2 = natten2dav(attn2.softmax(dim=-1), v2, kernel_size=3, dilation=1)
    assert out2.shape == (1, 2, 6, 6, 8)
    assert torch.isfinite(out2).all()


def test_audio_ear_16k_mono_numpy_guard() -> None:
    """Verifies Trap #13 Fix: `prepare_16k_mono_numpy` always outputs a 1D `np.ndarray` at 16 kHz."""
    stereo_44k = np.random.randn(2, SAMPLE_RATE * 2).astype(np.float32)
    wav_16k = AudioEarDriver.prepare_16k_mono_numpy(stereo_44k, sr=SAMPLE_RATE)
    assert isinstance(wav_16k, np.ndarray)
    assert wav_16k.ndim == 1
    assert wav_16k.dtype == np.float32
    assert len(wav_16k) == 32000


# ============================================================================
# 2. `ListenerAgent`: 4-Stem Exact Reconstruction, Downbeat-Snapped EDM Roles,
#    CQT Camelot Key Detection & 4-Panel `1024x1024` Spectrogram Bridge
# ============================================================================

def test_listener_agent_stems_key_sections_and_4panel_bridge(tmp_path) -> None:
    """Verifies `ListenerAgent` 4-stem exact reconstruction, downbeat-snapped roles, Camelot key, and 1024x1024 PNG."""
    prod = ProducerAgent(cache_dir=str(tmp_path))
    art = prod.produce_track(
        title="Test_Club_8A",
        style="Melodic Techno",
        target_bpm=128.0,
        target_camelot="8A",
        bars=16,
        output_path=str(tmp_path / "club_8a.wav"),
    )

    stereo = synthesize_club_audio(bpm=128.0, camelot="8A", bars=8, sr=SAMPLE_RATE)
    stems, _ = separate_stems_htdemucs(stereo, sr=SAMPLE_RATE, cache_dir=str(tmp_path / "stems"), prefer_neural=False)

    # Verify exact master reconstruction: sum(drums + bass + other + vocals) == stereo (< 1e-5 error)
    recon = stems["drums"] + stems["bass"] + stems["other"] + stems["vocals"]
    max_recon_err = float(np.max(np.abs(recon - stereo)))
    assert max_recon_err < 1e-5, f"Stem reconstruction error too high: {max_recon_err}"

    # Also verify neural HTDemucs 4-stem separation with cached HDEMUCS_HIGH_MUSDB_PLUS checkpoint
    stems_neural, _ = separate_stems_htdemucs(
        stereo[:, : SAMPLE_RATE * 2], sr=SAMPLE_RATE, cache_dir=str(tmp_path / "stems_neural"), prefer_neural=True
    )
    recon_neural = stems_neural["drums"] + stems_neural["bass"] + stems_neural["other"] + stems_neural["vocals"]
    assert float(np.max(np.abs(recon_neural - stereo[:, : SAMPLE_RATE * 2]))) < 1e-5

    listener = ListenerAgent(cache_dir=str(tmp_path / "listener"))
    prof = listener.analyze_track(art.output_path, prefer_neural_stems=False)
    assert prof.vibe_profile.get("pretrained_weights_present") is True

    assert abs(prof.bpm - 128.0) <= 1.0
    # Verify Camelot key is 8A (or harmonically adjacent distance <= 1 on Camelot wheel)
    assert camelot_distance(prof.camelot_global, "8A") <= 1
    assert len(prof.sections) >= 3
    labels = {s.label for s in prof.sections}
    assert "intro" in labels and "outro" in labels
    assert ("drop" in labels) or ("build" in labels)

    # Verify every section start_sec is snapped strictly to a downbeat (0.0 ms offset, Trap #3 Fix)
    downbeat_set = set(prof.downbeats_sec)
    for sec in prof.sections:
        assert any(abs(sec.start_sec - db) < 1e-4 for db in downbeat_set), f"Section {sec} not snapped to downbeat"

    # Verify 4-panel Visual-Acoustic Spectrogram PNG is exactly 1024x1024 RGB/RGBA
    assert os.path.isfile(prof.spectrogram_png_path)
    with Image.open(prof.spectrogram_png_path) as img:
        assert img.size == (1024, 1024)


# ============================================================================
# 3. `ProducerAgent`: `AutoencoderOobleck` 1D Overlap-Add Chunked Decoder (Trap #10),
#    `MusicGen-Melody` 4D Stem Tensor Guard (Trap #11), and `Suno` Bridge (Trap #7)
# ============================================================================

def test_oobleck_chunked_vae_decoder_on_gpu() -> None:
    """Verifies Trap #10 Fix: `chunked_oobleck_decode` decodes long `AutoencoderOobleck` latents on RTX 5090."""
    from autodj_producer import LocalAceStepDriver

    ace = LocalAceStepDriver()
    assert ace.has_pretrained_dit_weights() is True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = ace.load_pretrained_vae(device=device)

    # Confirm vae.enable_tiling() raises NotImplementedError (Trap #10)
    with pytest.raises(NotImplementedError):
        vae.enable_tiling()

    # Decode 300 frames (12 seconds @ 25 Hz -> 300 * 1920 = 576,000 stereo samples) using chunked overlap-add
    latents = torch.randn(1, 64, 300, device=device, dtype=dtype)
    wav = chunked_oobleck_decode(vae, latents, chunk_frames=120, overlap_frames=20, hop_samples=1920)
    assert wav.shape == (1, 2, 300 * 1920)
    assert torch.isfinite(wav).all()


def test_musicgen_melody_4d_stem_tensor_guard(tmp_path) -> None:
    """Verifies Trap #11 Fix: `LocalMusicGenMelodyDriver` accepts 4D `(1, 4, 2, N)` stems and rejects 2D `(2, N)`."""
    mg = LocalMusicGenMelodyDriver()
    assert mg.has_pretrained_weights() is True
    stems_4d = torch.randn(1, 4, 2, SAMPLE_RATE * 2)
    feats = mg.extract_4d_stem_chroma(stems_4d, sr=SAMPLE_RATE)
    assert feats.ndim == 3 and feats.shape[0] == 1 and feats.shape[2] == 12

    # Passing a 2D stereo tensor (2, N) must raise ValueError (Trap #11)
    with pytest.raises(ValueError, match="Trap #11"):
        mg.extract_4d_stem_chroma(torch.randn(2, SAMPLE_RATE * 2), sr=SAMPLE_RATE)

    # Test generating a chroma bridge from a stems dict
    stereo = synthesize_club_audio(bpm=126.0, camelot="8A", bars=4, sr=SAMPLE_RATE)
    stems_dict, _ = separate_stems_htdemucs(stereo, sr=SAMPLE_RATE, cache_dir=str(tmp_path), prefer_neural=False)
    bridge_out = mg.generate_chroma_bridge(
        stems_a=stems_dict,
        prompt="Melodic techno chroma bridge",
        output_path=str(tmp_path / "mg_bridge.wav"),
        target_bpm=128.0,
        target_camelot="8A",
        bars=4,
    )
    assert os.path.isfile(bridge_out.output_path)
    assert bridge_out.duration_sec > 5.0
    assert bridge_out.metadata.get("pretrained_weights_present") is True


def test_suno_cloud_driver_enforces_no_loopback(tmp_path) -> None:
    """Verifies Trap #7 Fix: `SunoCloudDriver` compiles custom payload and enforces `fallback_loopback=False`."""
    driver = SunoCloudDriver(cdp_url="http://127.0.0.1:19999")  # closed port -> clean hybrid fallback
    res = driver.compile_and_dispatch(
        title="Midnight Protocol",
        style="Peak-Time Techno, 130 BPM, Key 8A",
        lyrics="[Intro]\n[Drop]\nSystem online\n[Outro]",
        output_path=str(tmp_path / "suno_hybrid.wav"),
        target_bpm=130.0,
        target_camelot="8A",
        bars=4,
    )
    assert os.path.isfile(res.output_path)
    assert res.metadata.get("fallback_loopback_enforced_false") is True
    assert res.metadata["suno_payload"]["valid"] is True


# ============================================================================
# 4. `MixerAgent`: `librubberband.so.2` Sub-ms Stretch, Bass-Swap Law,
#    2048-Sample Crossfade, and All 6 Archetypes (`T1`-`T6`)
# ============================================================================

def test_librubberband_subms_stretch_and_pitch() -> None:
    """Verifies Trap #1 Fix: `librubberband.so.2` R3 Finer Engine stretches to the exact target sample count."""
    sr = SAMPLE_RATE
    t = np.arange(sr * 2, dtype=np.float32) / sr
    stereo = np.stack([np.sin(2 * np.pi * 220 * t), np.cos(2 * np.pi * 220 * t)], axis=0)
    target_samples = int(round(stereo.shape[1] * 1.0625))
    stretched = rubberband_stretch_and_pitch(
        stereo,
        time_ratio=1.0625,
        pitch_semitones=1.0,
        sr=sr,
        exact_target_samples=target_samples,
    )
    assert stretched.shape == (2, target_samples)
    assert np.isfinite(stretched).all()


@pytest.mark.parametrize("archetype", ["T1", "T2", "T3", "T4", "T5", "T6"])
def test_all_six_transition_archetypes_obey_bass_swap_law(archetype: str) -> None:
    """Verifies `T1`-`T6` stem gain envelopes strictly obey the Bass-Swap Law."""
    n_samples = SAMPLE_RATE * 8
    gains_a, gains_b = build_stem_automation_envelopes(
        n_samples=n_samples,
        num_bars=8,
        bass_swap_bar=4,
        archetype=archetype,
    )
    bass_sum = gains_a["bass"] + gains_b["bass"]
    assert float(np.max(bass_sum)) <= 1.0001
    sim_active = int(np.sum((gains_a["bass"] > 0.5) & (gains_b["bass"] > 0.5)))
    assert sim_active == 0


# ============================================================================
# 5. `HostAgent`: Cached `Kokoro-82M` Synthesis & `-10 dB` Mid-Band Ducking
# ============================================================================

def test_host_agent_kokoro_synthesis_and_midband_sidechain_ducking(tmp_path) -> None:
    """Verifies Trap #8 Fix: `HostAgent` synthesizes neural speech via cached `Kokoro-82M` and ducks 300-3500 Hz."""
    host = HostAgent(cache_dir=str(tmp_path / "host"))
    drop = host.synthesize_drop(
        script="Welcome to the AI Auto DJ studio, locking into 128 BPM!",
        output_wav_path=str(tmp_path / "mc_drop.wav"),
    )
    assert os.path.isfile(drop.wav_path)
    assert drop.backend_used == "Kokoro-82M (Local Neural TTS)"
    assert drop.duration_sec > 1.5

    # Test -10 dB mid-band sidechain ducking while preserving sub-bass (< 300 Hz)
    music = synthesize_club_audio(bpm=128.0, camelot="8A", bars=4, sr=SAMPLE_RATE)
    voice = np.ones((2, SAMPLE_RATE * 2), dtype=np.float32) * 0.4
    ducked, duck_stats = host.mix_voiceover_with_midband_ducking(
        music_stereo=music,
        voice_stereo=voice,
        insert_offset_samples=0,
        duck_db=-10.0,
        sr=SAMPLE_RATE,
    )
    assert ducked.shape == music.shape
    assert duck_stats["midband_attenuation_db"] < -5.0
    assert duck_stats["lowband_preserved"] is True


# ============================================================================
# 6. Modification 3 (`AudioProjectorGraft`) & Trap #16 (`defuse_moe_3d_experts`)
# ============================================================================

def test_modification3_audio_projector_graft_and_trap16_moe_defuser() -> None:
    """Verifies Modification 3 `masked_scatter` audio token splicing and Trap #16 3D MoE expert de-fuser."""
    graft = AudioProjectorGraft(audio_dim=1280, llm_hidden_size=2048, audio_token_id=151646)
    input_ids = torch.tensor([[101, 151646, 151646, 151646, 102]])
    inputs_embeds = torch.zeros(1, 5, 2048)
    audio_hidden = torch.randn(1, 3, 1280)
    spliced = graft.splice_into_inputs_embeds(input_ids, inputs_embeds, audio_hidden)
    assert spliced.shape == (1, 5, 2048)
    assert torch.all(spliced[:, 0, :] == 0.0)
    assert not torch.all(spliced[:, 1:4, :] == 0.0)

    class DummyFusedMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 32, 16))
            self.down_proj = nn.Parameter(torch.randn(4, 16, 16))

    moe = DummyFusedMoE()
    defused = defuse_moe_3d_experts(moe)
    assert not hasattr(defused, "gate_up_proj")
    assert len(defused.gate_up_experts) == 4
    assert isinstance(defused.gate_up_experts[0], nn.Linear)


# ============================================================================
# 7. `LangGraph` 6-Agent State Machine & Critic Self-Healing Re-Mix Loop
# ============================================================================

def test_langgraph_6agent_crew_and_critic_self_healing_loop(tmp_path) -> None:
    """Verifies the full 6-Agent `LangGraph` workflow and `CriticAgent`'s self-healing retry loop."""
    crew = AutoDJCrew(cache_dir=str(tmp_path / "studio"))
    t1 = crew.producer.produce_track(
        title="Track_Alpha_8A",
        target_bpm=126.0,
        target_camelot="8A",
        bars=12,
        output_path=str(tmp_path / "alpha_8a.wav"),
    )
    t2 = crew.producer.produce_track(
        title="Track_Beta_8A",
        target_bpm=128.0,
        target_camelot="8A",
        bars=12,
        output_path=str(tmp_path / "beta_8a.wav"),
    )
    t3 = crew.producer.produce_track(
        title="Track_Gamma_3B",
        target_bpm=130.0,
        target_camelot="3B",
        bars=12,
        output_path=str(tmp_path / "gamma_3b.wav"),
    )

    # Run with inject_test_defect_on_first_try=True so CriticAgent rejects attempt #1 and self-heals on attempt #2
    state = crew.run_session(
        track_paths=[t1.output_path, t2.output_path, t3.output_path],
        steering_prompt="Peak-time progressive club energy arc",
        enable_host_mc=True,
        max_critic_retries=2,
        inject_test_defect_on_first_try=True,
    )

    certified = state["certified_transitions"]
    assert len(certified) == 2
    for item in certified:
        crit = item["critique"]
        assert crit["verdict"] == "PASS"
        assert crit["stage1_dsp_pass"] is True
        assert crit["stage2_perceptual_pass"] is True
        assert crit["beat_alignment_error_ms"] < 1.0
        assert crit["true_peak_dbtp"] <= -0.5
        assert crit["lufs_step_delta_lu"] <= 1.5
        assert crit["simultaneous_bass_samples"] == 0
        # Confirm self-healing retry occurred on each transition
        assert item["retries_used"] == 1

    # Export gapless master WAV + CUE and stream to /dev/shm PipeWire lookahead ringbuffer
    export_res = export_master_mix_and_cue(
        state,
        output_wav_path=str(tmp_path / "master_set.wav"),
    )
    assert os.path.isfile(export_res["master_wav_path"])
    assert os.path.isfile(export_res["cue_path"])
    assert export_res["true_peak_dbtp"] <= -0.5

    streamer = PipeWireLookaheadStreamer(shm_dir=str(tmp_path / "ringbuffer"))
    slot_file = streamer.write_lookahead_slot(np.zeros((2, SAMPLE_RATE), dtype=np.float32), slot_idx=0)
    assert os.path.isfile(slot_file)
    stream_info = streamer.stream_slot_to_pipewire(slot_idx=0, live_audio=False)
    assert stream_info["frames_streamed"] == SAMPLE_RATE


# ============================================================================
# 8. FastAPI + WebSocket 2-Deck Web Studio Endpoints (`autodj_server.py`)
# ============================================================================

def test_fastapi_2deck_web_studio_endpoints(tmp_path) -> None:
    """Verifies FastAPI REST & WebSocket endpoints for the 2-Deck Web Studio."""
    from fastapi.testclient import TestClient

    app = create_app(cache_dir=str(tmp_path / "web_studio"), enable_legacy_studio=True)
    client = TestClient(app)

    r_home = client.get("/studio")
    assert r_home.status_code == 200
    assert "AI-Auto-DJ: 6-Agent Virtual DJ" in r_home.text

    # Toggle Audio Ear between Music-Flamingo-Think and Gemma-4-E4B-it
    r_mode = client.post(
        "/api/agent-mode",
        json={"agent": "listener", "mode": "hybrid", "audio_ear_model": "google/gemma-4-E4B-it"},
    )
    assert r_mode.status_code == 200
    assert r_mode.json()["audio_ear_model"] == "google/gemma-4-E4B-it"
    assert r_mode.json()["agent_modes"]["listener"] == "hybrid"

    # Produce two tracks via API and execute a full session
    p1 = client.post("/api/produce", json={"title": "DeckA_Track", "target_bpm": 128.0, "target_camelot": "8A", "bars": 8}).json()
    p2 = client.post("/api/produce", json={"title": "DeckB_Track", "target_bpm": 128.0, "target_camelot": "8A", "bars": 8}).json()
    assert os.path.isfile(p1["output_path"]) and os.path.isfile(p2["output_path"])

    r_run = client.post(
        "/api/run-session",
        json={"track_paths": [p1["output_path"], p2["output_path"]], "steering_prompt": "Club energy", "enable_host_mc": False},
    )
    assert r_run.status_code == 200
    assert r_run.json()["certified_count"] == 1

    with client.websocket_connect("/ws/telemetry") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "telemetry_snapshot"
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"


# ============================================================================
# 9. Adversarial Edge-Case Verification: Extreme Tempo Jumps, Tritone Clashes,
#    LangGraph Max-Retry Termination, and Traps #12, #16, #17
# ============================================================================

def test_extreme_tempo_jumps_and_tritone_clashes_force_t4_or_t6(tmp_path) -> None:
    """Verifies that extreme tempo jumps (>25% BPM) and tritone clashes (d_key==6) always force T4 or T6."""
    assert camelot_distance("8A", "2A") == 6
    assert camelot_distance("8A", "2B") == 6
    assert find_minimal_semitone_shift("8A", "2A") == 0
    assert find_minimal_semitone_shift("8A", "2B") == 0
    # Semitone neighbors on Circle of Fifths (ring_dist == 5) still resolve cleanly for T5
    assert find_minimal_semitone_shift("8A", "3A") == -1
    assert find_minimal_semitone_shift("8A", "1A") == 1

    crew = AutoDJCrew(cache_dir=str(tmp_path / "adversarial"))
    t_100_8a = crew.producer.produce_track("Track_100_8A", target_bpm=100.0, target_camelot="8A", bars=8, output_path=str(tmp_path / "t100_8a.wav"))
    t_140_8a = crew.producer.produce_track("Track_140_8A", target_bpm=140.0, target_camelot="8A", bars=8, output_path=str(tmp_path / "t140_8a.wav"))
    t_128_2a = crew.producer.produce_track("Track_128_2A", target_bpm=128.0, target_camelot="2A", bars=8, output_path=str(tmp_path / "t128_2a.wav"))
    t_128_2b = crew.producer.produce_track("Track_128_2B", target_bpm=128.0, target_camelot="2B", bars=8, output_path=str(tmp_path / "t128_2b.wav"))

    p_100_8a = crew.listener.analyze_track(t_100_8a.output_path, prefer_neural_stems=False)
    p_140_8a = crew.listener.analyze_track(t_140_8a.output_path, prefer_neural_stems=False)
    p_128_2a = crew.listener.analyze_track(t_128_2a.output_path, prefer_neural_stems=False)
    p_128_2b = crew.listener.analyze_track(t_128_2b.output_path, prefer_neural_stems=False)

    # 1. 100 BPM -> 140 BPM (40% BPM jump, same key 8A) -> forces T4 (Echo-Freeze Tempo Reset)
    spec_tempo = crew.director.plan_transition(p_100_8a, p_140_8a)
    assert spec_tempo.archetype == "T4"

    # If a caller forces T1 on a 40% BPM jump, CriticAgent rejects with stage2_tempo_jump and recommends T4
    spec_bad_tempo = crew.director.plan_transition(p_100_8a, p_140_8a)
    spec_bad_tempo.archetype = "T1"
    rend_bad_tempo = crew.mixer.render_transition(p_100_8a, p_140_8a, spec_bad_tempo)
    crit_bad_tempo = crew.critic.evaluate_transition(p_100_8a, p_140_8a, spec_bad_tempo, rend_bad_tempo)
    assert crit_bad_tempo.verdict == "REJECT"
    assert any("stage2_tempo_jump" in f for f in crit_bad_tempo.failed_checks)
    assert crit_bad_tempo.adjustment is not None and crit_bad_tempo.adjustment.recommended_archetype == "T4"

    # 2. Same-mode tritone clash (8A -> 2A) and cross-mode tritone clash (8A -> 2B) -> force T6 (AI Generative Bridge)
    spec_tritone_same = crew.director.plan_transition(p_100_8a, p_128_2a)
    assert spec_tritone_same.archetype == "T6" and spec_tritone_same.requires_ai_bridge is True

    spec_tritone_cross = crew.director.plan_transition(p_140_8a, p_128_2b)
    assert spec_tritone_cross.archetype == "T6" and spec_tritone_cross.requires_ai_bridge is True

    # If a caller forces T5 with an invalid -1 semitone shift on a tritone clash (8A -> 2A), CriticAgent rejects and upgrades to T6
    spec_bad_t5 = crew.director.plan_transition(p_140_8a, p_128_2a)
    spec_bad_t5.archetype = "T5"
    spec_bad_t5.pitch_shift_semitones_b = -1.0
    spec_bad_t5.requires_ai_bridge = False
    rend_bad_t5 = crew.mixer.render_transition(p_140_8a, p_128_2a, spec_bad_t5)
    crit_bad_t5 = crew.critic.evaluate_transition(p_140_8a, p_128_2a, spec_bad_t5, rend_bad_t5)
    assert crit_bad_t5.verdict == "REJECT"
    assert crit_bad_t5.adjustment is not None
    assert crit_bad_t5.adjustment.recommended_archetype == "T6"
    assert crit_bad_t5.adjustment.requires_ai_bridge is True


def test_langgraph_max_critic_retries_terminates_without_infinite_loop(tmp_path) -> None:
    """Verifies that hitting `max_critic_retries=0` on a rejected transition advances cleanly without looping."""
    crew = AutoDJCrew(cache_dir=str(tmp_path / "max_retry_test"))
    t1 = crew.producer.produce_track("T1_8A", target_bpm=128.0, target_camelot="8A", bars=8, output_path=str(tmp_path / "t1.wav"))
    t2 = crew.producer.produce_track("T2_8A", target_bpm=128.0, target_camelot="8A", bars=8, output_path=str(tmp_path / "t2.wav"))
    t3 = crew.producer.produce_track("T3_8A", target_bpm=128.0, target_camelot="8A", bars=8, output_path=str(tmp_path / "t3.wav"))

    state = crew.run_session(
        track_paths=[t1.output_path, t2.output_path, t3.output_path],
        enable_host_mc=False,
        max_critic_retries=0,
        inject_test_defect_on_first_try=True,
    )
    assert len(state["certified_transitions"]) == 2
    assert state["retry_count"] == 0


def test_trap12_trap16_and_trap17_audio_ear_configs() -> None:
    """Verifies Trap #12, Trap #16, and Trap #17 helper configurations on `AudioEarDriver`."""
    omni_cfg = AudioEarDriver.get_hf_load_and_generate_kwargs("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert omni_cfg["from_pretrained_kwargs"]["enable_audio_output"] is False
    assert omni_cfg["from_pretrained_kwargs"]["load_in_4bit_direct_safe"] is False
    assert omni_cfg["generate_kwargs"]["return_audio"] is False

    mf_cfg = AudioEarDriver.get_hf_load_and_generate_kwargs("nvidia/music-flamingo-think-2601-hf")
    assert mf_cfg["from_pretrained_kwargs"]["load_in_4bit_direct_safe"] is True

    mtmd = AudioEarDriver.build_llama_server_mtmd_env_and_cmd("model.gguf", "mmproj.gguf", port=11435)
    assert mtmd["binary"] == "/usr/local/lib/ollama/llama-server"
    assert "/usr/local/lib/ollama" in mtmd["env"]["LD_LIBRARY_PATH"]
    assert "--mmproj" in mtmd["cmd"]




def test_unified_autodj_entrypoint_and_full_gui_studio_workflows(tmp_path) -> None:
    """Verifies the single `autodj.py` launcher and all end-to-end GUI studio endpoints, audio streams, and edge cases."""
    from fastapi.testclient import TestClient
    import autodj

    # 1. Verify single unified CLI entrypoint check-only mode
    rc = autodj.main(["--cache-dir", str(tmp_path / "cli_studio"), "--check-only", "--no-browser"])
    assert rc == 0

    app = create_app(cache_dir=str(tmp_path / "full_gui_studio"), enable_legacy_studio=True)
    client = TestClient(app)

    # 2. Verify GUI HTML contains all DJ booth components (dual waveforms, jog wheels, Camelot SVG, FFT, stem solo)
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    for marker in ("waveCanvasA", "waveCanvasB", "camelotSvg", "envCanvas", "fftCanvas", "runQuickStart", "soloStemAudio"):
        assert marker in r_ui.text

    # 3. Verify error edge cases before tracks are loaded
    assert client.get("/api/audio/master").status_code == 404
    assert client.post("/api/deck/mix-pair", json={}).status_code == 400
    assert client.post("/api/library/scan", json={"path": str(tmp_path / "nonexistent_dir")}).status_code == 404
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()
    assert client.post("/api/library/scan", json={"path": str(empty_dir)}).status_code == 400

    # 4. Run 1-Click QuickStart Bootstrap from GUI
    r_qs = client.post("/api/quickstart", json={"bars": 6, "enable_host_mc": False})
    assert r_qs.status_code == 200
    qs_data = r_qs.json()
    assert qs_data["certified_count"] >= 2
    assert qs_data["deck_a_id"] is not None and qs_data["deck_b_id"] is not None

    # 5. Verify full state, deck pair preview, and 64-point stem automation envelopes
    st = client.get("/api/state").json()
    assert st["master_mix_ready"] is True
    assert len(st["crate"]) == 3
    assert st["deck_pair_preview"]["ready"] is True
    assert len(st["deck_pair_preview"]["envelopes_a"]["bass"]) == 64
    assert len(st["spectrograms"]) >= 3

    tid_a = st["deck_a_id"]
    tid_b = st["deck_b_id"]

    # 6. Verify 160-bin waveform & stem envelope endpoint for HTML5 Canvas rendering
    r_env = client.get(f"/api/envelope/{tid_a}")
    assert r_env.status_code == 200
    env_json = r_env.json()
    assert len(env_json["peaks_l"]) == 160
    assert "drums" in env_json["bar_rms_db"]

    # 7. Verify audio streaming endpoints (Master WAV, Transition WAV, Track WAV, Separated Stem WAV, CUE export)
    assert client.get("/api/audio/master").status_code == 200
    assert client.get("/api/audio/transition/0").status_code == 200
    assert client.get(f"/api/audio/track/{tid_a}").status_code == 200
    r_stem = client.get(f"/api/audio/stem/{tid_a}/bass")
    assert r_stem.status_code == 200 and len(r_stem.content) > 1000
    assert client.get("/api/export/wav").status_code == 200
    assert client.get("/api/export/cue").status_code == 200

    # 8. Verify spectrogram PNG serving
    spec_url = st["spectrograms"][0]["url"]
    assert client.get(spec_url).status_code == 200

    # 9. Verify Deck Load, Stem Gain adjustment, and Live Deck A -> B Mix with archetype override
    r_load = client.post(
        "/api/deck/load",
        json={"deck": "A", "track_id_or_path": tid_a, "stem_gains": {"drums": 0.0, "bass": 1.0}, "selected_archetype": "T3"},
    )
    assert r_load.status_code == 200
    load_json = r_load.json()
    assert load_json["selected_archetype"] == "T3"
    assert max(load_json["deck_pair_preview"]["envelopes_a"]["drums"]) == 0.0

    # Restore drums and test live mix with stem_gains_a / stem_gains_b
    r_mix = client.post(
        "/api/deck/mix-pair",
        json={
            "track_a_id": tid_a,
            "track_b_id": tid_b,
            "archetype": "T2",
            "enable_host_mc": True,
            "stem_gains_a": {"drums": 0.85, "bass": 1.0, "vocals": 0.75, "other": 1.0},
            "stem_gains_b": {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "other": 0.9},
        },
    )
    assert r_mix.status_code == 200
    assert r_mix.json()["transition"]["spec"]["archetype"] == "T2"
    assert r_mix.json()["transition"]["critique"]["simultaneous_bass_samples"] == 0

    # 10. Verify Host MC Drop synthesis & audio_url index alignment after session/mix with enable_host_mc=True
    r_mc = client.post("/api/host-drop", json={"script": "Hands up in the club!"})
    assert r_mc.status_code == 200
    vo_url = r_mc.json()["audio_url"]
    vo_idx = int(vo_url.rsplit("/", 1)[-1])
    st_after_mc = client.get("/api/state").json()
    assert st_after_mc["host_voiceovers"][vo_idx]["script"] == "Hands up in the club!"
    assert client.get(vo_url).status_code == 200

    r_pw = client.post("/api/pipewire-play", json={"live_audio": False, "max_seconds": 2.0})
    assert r_pw.status_code == 200
    assert r_pw.json()["frames_streamed"] > 0

    # 11. Verify scanning a directory containing both valid WAVs and a corrupted 0-byte WAV file
    scan_dir = tmp_path / "full_gui_studio" / "producer"
    corrupt_wav = scan_dir / "00_corrupt_empty.wav"
    corrupt_wav.write_bytes(b"")
    assert client.post("/api/deck/load", json={"deck": "A", "track_id_or_path": str(corrupt_wav)}).status_code == 400

    r_scan = client.post("/api/library/scan", json={"path": str(scan_dir), "max_files": 6})
    assert r_scan.status_code == 200
    assert r_scan.json()["indexed_count"] >= 1
    assert r_scan.json()["skipped_count"] == 1

    only_corrupt_dir = tmp_path / "only_corrupt"
    only_corrupt_dir.mkdir()
    (only_corrupt_dir / "broken.wav").write_bytes(b"not a valid riff wav header")
    assert client.post("/api/library/scan", json={"path": str(only_corrupt_dir)}).status_code == 400

    # 12. Verify Session Stop endpoint and concurrent mix-pair thread safety
    r_stop = client.post("/api/session/stop")
    assert r_stop.status_code == 200
    assert r_stop.json()["stop_requested"] is True

    from concurrent.futures import ThreadPoolExecutor

    def _concurrent_mix(arch: str) -> int:
        resp = client.post(
            "/api/deck/mix-pair",
            json={"track_a_id": tid_a, "track_b_id": tid_b, "archetype": arch, "enable_host_mc": False},
        )
        return resp.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(_concurrent_mix, ["T1", "T2"]))
    assert statuses == [200, 200]


def test_live_autonomous_dj_lookahead_and_neural_gpu_track(tmp_path: Path) -> None:
    """Verifies DirectorAgent live concept generation, real RTX 5090 Neural GPU track synthesis, and /api/live/* continuous lookahead."""
    import time
    from fastapi.testclient import TestClient
    from autodj_planner import DirectorAgent, camelot_distance
    from autodj_producer import ProducerAgent, free_cached_gpu_models

    # 1. Verify DirectorAgent live harmonic concept progression
    director = DirectorAgent(mode="local")
    c1 = director.generate_live_track_concept("Peak-time melodic techno", prev_track=None, cycle_idx=1)
    c2 = director.generate_live_track_concept(
        "Peak-time melodic techno",
        prev_track={"title": c1["title"], "bpm": c1["target_bpm"], "camelot_global": c1["target_camelot"]},
        cycle_idx=2,
    )
    assert abs(float(c2["target_bpm"]) - float(c1["target_bpm"])) <= 4.0
    assert camelot_distance(str(c1["target_camelot"]), str(c2["target_camelot"])) <= 1

    # 2. Verify real GPU neural track synthesis on RTX 5090 (MusicGen-Stereo-Melody + Oobleck VAE + Qwen3-Emb)
    producer = ProducerAgent(mode="local", cache_dir=str(tmp_path / "prod_gpu"))
    gpu_wav = tmp_path / "prod_gpu" / "neural_gpu_test.wav"
    art = producer.generate_neural_gpu_track(
        prompt="Peak-Time Melodic Techno club anthem",
        output_path=str(gpu_wav),
        target_bpm=128.0,
        target_camelot="8A",
        bars=4,
        seed=42,
    )
    assert os.path.isfile(art.output_path)
    assert "Neural GPU" in art.backend_used
    assert art.metadata.get("backend_used") == (
        "Neural GPU (MusicGen-Stereo-Melody + ACE-Step 1.5 Oobleck VAE + Qwen3-Emb on RTX 5090)"
    )
    assert abs(art.duration_sec - 7.5) < 0.25

    # 3. Verify /api/live/start -> /api/live/status (2 continuous cycles) -> /api/live/stop
    app = create_app(cache_dir=str(tmp_path / "live_studio"), enable_legacy_studio=True)
    assert hasattr(app.state, "live_controller")
    client = TestClient(app)

    # Verify POST /api/produce with engine="suno" does not raise AttributeError
    r_prod_suno = client.post(
        "/api/produce",
        json={
            "title": "Suno_Fallback_Test",
            "style": "Melodic Techno",
            "target_bpm": 128.0,
            "target_camelot": "8A",
            "bars": 4,
            "engine": "suno",
            "use_neural_gpu": False,
        },
    )
    assert r_prod_suno.status_code == 200

    # Verify rapid start -> stop -> start does not spawn duplicate zombie workers
    r_first = client.post(
        "/api/live/start",
        json={"steering_prompt": "Aborted run", "enable_host_mc": False, "bars": 4, "use_neural_gpu": False, "pipewire_live": False},
    )
    assert r_first.status_code == 200
    r_abort = client.post("/api/live/stop")
    assert r_abort.status_code == 200
    assert r_abort.json()["live_status"]["live_running"] is False

    r_start = client.post(
        "/api/live/start",
        json={
            "steering_prompt": "Hypnotic progressive club groove",
            "enable_host_mc": False,
            "bars": 4,
            "use_neural_gpu": False,
            "pipewire_live": False,
        },
    )
    assert r_start.status_code == 200
    assert r_start.json()["status"] in ("started", "already_running")

    # Wait until at least 2 continuous lookahead transitions are certified
    deadline = time.time() + 45.0
    status_data = {}
    while time.time() < deadline:
        r_st = client.get("/api/live/status")
        assert r_st.status_code == 200
        status_data = r_st.json()
        if len(status_data.get("segments", [])) >= 2:
            break
        time.sleep(0.5)

    assert len(status_data.get("segments", [])) >= 2
    assert status_data["live_cycle"] >= 2
    assert status_data["crate_count"] >= 3

    # Report client progress and verify active segment deck synchronization
    r_prog = client.post("/api/live/progress", json={"playing_index": 0})
    assert r_prog.status_code == 200

    r_stop = client.post("/api/live/stop")
    assert r_stop.status_code == 200
    assert r_stop.json()["live_status"]["live_running"] is False
    assert "freed_models" in r_stop.json()["freed_gpu"]
    free_cached_gpu_models()

