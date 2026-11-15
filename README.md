# NEURAL DECK // AI-Auto-DJ (`ai-auto-dj`)

> For the LLM-directed two-track demo and verified backend limitations, see [AGENT_DEMO.md](AGENT_DEMO.md). The original feature descriptions below include aspirational and unverified claims; the procedural fallback is not ACE-Step inference.

A single-GUI, local-first / hybrid-capable **6-Agent Virtual DJ & Music Production Studio** built on **`LangGraph 1.2.10`**, **`PyTorch 2.9.1+cu128`**, **`Transformers 5.14.1`**, **`Diffusers 0.36.0`**, **`librubberband.so.2` (Rubber Band R3 Finer Engine)**, **`Kokoro-82M`**, and **PipeWire (`pw-play`)**.

---

## 1. Launch Everything from a Single Command & Single GUI

Start the entire studio from the unified entry point [`autodj.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj.py):

```bash
/home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj.py
```

Then open **`http://127.0.0.1:8088`** (opens automatically when a desktop display is active).

### What You Can Do Inside the Single GUI (`http://127.0.0.1:8088`)
- **⚡ 1-Click Quick-Start Demo Mix**: Generates 3 club tracks (`126` / `128` / `130` BPM, Keys `8A` / `9A`), separates 4 stems (`drums`, `bass`, `other`, `vocals`), plans the harmonic energy arc, renders sample-accurate transitions + `Kokoro-82M` DJ MC voiceovers, certifies via 2-Stage `CriticAgent`, and immediately plays the master mix in your browser.
- **🎧 Dual Motorized CDJ-3000 Decks (Deck A & Deck B)**:
  - Spinning platters with central OLED readouts (`BPM`, `Camelot Key`, `LUFS`, `Cue In/Out Bars`).
  - Stacked multi-color **4-Stem RGB Battle Waveforms** (`#waveCanvasA`, `#waveCanvasB`) with downbeat-snapped EDM section banners (`INTRO`, `BUILD`, `DROP`, `BREAKDOWN`, `OUTRO`) and 8 clickable RGB Hot-Cue Jump Pads per deck.
  - Per-deck **4-Stem Isolator Faders (`DRUMS`, `BASS`, `VOCALS`, `OTHER`)**, Mute buttons (`[M]`), and **Isolated Stem Solo Audition (`[🎧]`)** streaming directly from `/dev/shm`.
- **🎛️ Center 4-Channel Neural Mixer & Camelot Harmonic Wheel**:
  - Interactive **24-Key SVG Camelot Wheel (`1A`–`12B`)** with live Deck A → Deck B compatibility scoring $S(A,B)$.
  - **Transition Archetype Selector (`AUTO`, `T1`–`T6`)** with live mathematical stem automation curves (`#envCanvas`) and `Bass-Swap Law` guard visualization.
  - **Live Mix Deck A → B Now** button + interactive hardware Crossfader.
- **🗂️ Crate, Local Folder Scanner & AI Music Producer**:
  - Scan & index any local folder or `.wav`/`.mp3`/`.flac` file (`POST /api/library/scan`).
  - Produce new club tracks or bridges via **`ACE-Step 1.5` (Local GPU)**, **`MusicGen-Stereo-Melody` (4D Stems)**, or **`Suno v5` (Cloud Co-Producer)** (`POST /api/produce`).
- **🤖 6-Agent Mode Matrix, Crowd Steering, Host MC & 4-Panel Spectrogram Inspector**:
  - Per-agent `Local (RTX 5090)` / `Hybrid` / `Cloud` toggles and Option A1 Audio Ear selector (`nvidia/music-flamingo-think-2601-hf`, `nvidia/audio-flamingo-next-think-hf`, `google/gemma-4-E4B-it`, `Qwen/Qwen3-Omni-30B-A3B-Instruct`).
  - Live `Kokoro-82M` DJ Radio MC hype drop synthesizer (`POST /api/host-drop`) with `-10 dB` mid-band (`300–3500 Hz`) sidechain ducking.
  - Built-in **4-Panel `1024x1024` Visual-Acoustic Spectrogram Bridge** viewer & Stage-1/Stage-2 `CriticAgent` QA scorecard.
  - Built-in HTML5/WebAudio player with **48-Band Real-Time FFT Spectrum Analyzer**, `.WAV`/`.CUE` downloads, and 1-click **PipeWire (`pw-play`)** hardware output.

---

## 2. Internal 6-Agent Crew Modules (Orchestrated Automatically Behind the GUI)

| # | Agent | Module | Role |
|---|---|---|---|
| **1** | **`DirectorAgent`** | [`autodj_planner.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_planner.py) | Multi-objective compatibility graph $S(A,B)$, Beam Search (`K=8`) setlist planner, & `T1`–`T6` archetype selection |
| **2** | **`ProducerAgent`** | [`autodj_producer.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_producer.py) | `ACE-Step 1.5` (1D Overlap-Add Chunked `AutoencoderOobleck`), `MusicGen-Stereo-Melody` (4D stems), & `Suno` bridge/track generation |
| **3** | **`ListenerAgent`** | [`autodj_indexer.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_indexer.py) | 4-Stem separator, sub-ms beat/downbeat tracker, CQT Camelot key detector, & 4-Panel `1024x1024` Spectrogram renderer |
| **4** | **`MixerAgent`** | [`autodj_mixer.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_mixer.py) | `librubberband.so.2` R3 sub-ms beatgrid warping, `T1`–`T6` 4-stem automation, Bass-Swap Law, & `-14 LUFS` / `-1.0 dBTP` mastering |
| **5** | **`CriticAgent`** | [`autodj_critic.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_critic.py) | Stage-1 DSP + Stage-2 Perceptual Audio/Vision QA gate with self-healing `LangGraph` re-mix loop |
| **6** | **`HostAgent`** | [`autodj_crew.py`](file:///home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj_crew.py) | Cached `Kokoro-82M` neural TTS DJ MC synthesis + Broadcast EQ + `-10 dB` mid-band sidechain ducking |
