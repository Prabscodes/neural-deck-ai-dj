# Neural Deck

An AI DJ agent for listeners, not DJs. You describe what you want to hear ("dubstep with a piano hook"),
press **Start listening**, and the music keeps playing. You can send another request at any time and the
next tracks follow it.

Built for CS4680 Prompt Engineering, AI Agent Project.

## How it works

1. **Plan (LLM).** A Gemini "Hydrogen" model turns your request into a small JSON plan: title, style
   description, BPM, key, length, transition type, and overlap.
2. **Validate.** The plan is checked against a strict schema before anything runs
   (`validate_live_plan` in `autodj_live_agent.py`): exact field set, BPM 70-180, key `1A`-`12B`,
   allowed transitions and overlaps, text length limits. Invalid plans are rejected, never executed.
   The model's text is never run as a command. It can only choose among known music operations.
3. **Generate.** Stable Audio 3 Small-Music, running locally through ComfyUI, renders up to three
   candidate takes per slot.
4. **Listen and select.** The Gemini model listens to the outgoing phrase and the candidate handoffs
   as actually rendered, and scores genre, groove, instrument, and continuity. If listening fails, a
   bounded signal-based fallback picks a take.
5. **Mix.** `autodj_beatmatch.py` estimates beat grids and time-stretches within a conservative ratio.
   When the grid is weak it uses a short fade instead of forcing a beat match.
6. **Play.** The browser decodes and schedules the audio chunks on the Web Audio clock, acknowledges
   played chunks, and the server keeps a bounded lookahead queue so work does not run away.
7. **Audit.** Every plan, generation, listening result, steer request, retry, stop, and failure is
   appended to `dj_agent_runs/actions.jsonl`.

The compact **Mix monitor** (bass, mids, highs) and the animated pixel DJ are presentation only. They
do not control the mix.

## Requirements

- Linux with an NVIDIA GPU (developed on an RTX 5090).
- Python 3.11. The app environment needs the packages in `requirements.txt`.
- FFmpeg built with Rubber Band, used for pitch-preserving tempo changes.
- A local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install with its own Python environment,
  plus these model files in its base directory:
  - `models/checkpoints/stable_audio_3_small_music.safetensors`
  - `models/text_encoders/t5gemma_b_b_ul2.safetensors`
- A Gemini planner, either:
  - **CLI (default):** the Antigravity CLI (`agy`) signed in locally, or
  - **API:** `AUTODJ_GEMINI_TRANSPORT=api`, a model ID you have access to in `AUTODJ_GEMINI_MODEL`, and
    `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) set in the server's environment.

The model weights and any keys are not part of this repository.

## Run

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # edit it; .env is git-ignored

set -a; source .env; set +a   # the app does not read .env itself
./start_live_stable.sh
```

Open http://127.0.0.1:8300/. The launcher starts the local renderer on port 8296 if it is not already
running. The first two tracks buffer during the intro, so the first playback can take a minute or more.

Note: `start_live_stable.sh` still has the original author's paths as defaults for `AUTODJ_COMFY_DIR`
and `AUTODJ_COMFY_BASE`. Always set both on another machine.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AUTODJ_GEMINI_TRANSPORT` | `cli` or `api` | `cli` |
| `AUTODJ_GEMINI_MODEL` | model ID for API mode | see `autodj_gemini.py` |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | API key, server-side only | unset |
| `AUTODJ_COMFY_DIR` | ComfyUI install | launcher default |
| `AUTODJ_COMFY_BASE` | ComfyUI base directory holding `models/` | launcher default |
| `AUTODJ_COMFY_PORT` | renderer port | `8296` |
| `AUTODJ_WEB_PORT` | web port | `8298` |
| `AUTODJ_CACHE_DIR` | generated WAV cache | `/dev/shm/autodj_stable_live` |
| `AUTODJ_AGENT_DIR` | audit log and run output | `dj_agent_runs` |
| `AUTODJ_ENABLE_LEGACY_STUDIO` | `1` enables the older studio at `/studio` and its `/api/*` routes | off |
| `AUTODJ_MAX_STEERS` | max follow-up requests per session | `25` |
| `AUTODJ_MAX_TRACKS` | max tracks per session before it stops | `60` |

## Public mode (no GPU)

`autodj_public.py` is a slim server for a shared host. It serves only the listener page and
`/api/dj-live/*`, never imports torch, and needs no GPU or ComfyUI. Instead of rendering, it plays a
bank of tracks pre-rendered offline with Stable Audio. The Gemini planner still runs live for every
track and its plan is validated as usual.

```bash
# once, on a machine with the renderer running (a GPU):
.venv/bin/python tools/build_bank.py --out bank      # 48 tracks, about 52 MB

# on the public host:
AUTODJ_BANK_DIR=bank AUTODJ_GEMINI_TRANSPORT=api AUTODJ_GEMINI_MODEL=<model> GEMINI_API_KEY=<secret> \
  python autodj_public.py --host 0.0.0.0 --port 8300
```

| Variable | Purpose | Default |
|---|---|---|
| `AUTODJ_BANK_DIR` | folder with `bank.json` and the audio | `bank` |
| `AUTODJ_MAX_SESSIONS` | concurrent listeners | `3` |
| `AUTODJ_MAX_STARTS_PER_HOUR` | session starts per client | `6` |
| `AUTODJ_LISTENING` | `off` skips the per-track audio review to save API calls | `on` |
| `AUTODJ_TRUST_PROXY` | `1` reads the client address from `X-Forwarded-For` (only behind a proxy) | off |

The bank is a limited set of styles, so requests outside it get the closest match. Tracks are labeled
by the prompts used, not by listening review.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_demo_route.py tests/test_live_dj.py tests/test_beatmatch.py tests/test_stable_live_cues.py
```

These check routing, plan validation, queueing, beat matching, and cue detection. They do not judge
how the music sounds.

## What is and is not verified

- **Playback continuity:** earlier live runs scheduled consecutive tracks without gaps and
  replenished the queue. This is measurable from the playback timing.
- **Music quality:** subjective and still mixed. A beat-aligned transition can still sound wrong, and
  listening reviews of full sets have scored it well below perfect.
- **Startup:** the first audio took roughly 45 to 106 seconds in earlier runs.

## Repository layout

| Path | Role |
|---|---|
| `autodj_server.py` | FastAPI server, listener page, live API (`/api/dj-live/...`) |
| `autodj_live_agent.py` | live orchestration, plan validation, selection, queue |
| `autodj_gemini.py` | Gemini transport (CLI or API) and error handling |
| `autodj_stable_audio.py` | Stable Audio generation through ComfyUI |
| `autodj_beatmatch.py` | beat grid estimation and tempo matching |
| `autodj_frontend.html`, `autodj_web/` | listener page, scripts, styles, artwork |
| `tests/` | automated tests |

The remaining `autodj_*.py` modules and the older `PROJECT.md`, `AGENT_DEMO.md`, and `TEST_*.md` belong
to the earlier studio workflow and are kept for history.

## Safety

- LLM output is schema-validated and reduced to a fixed set of music operations.
- No shell commands, file paths, or network targets come from model output.
- Only `/api/dj-live/*` and `/api/dj-agent/*` are served by default. The older studio routes (library scan, host drop, and others, which take server paths) return 404 unless `AUTODJ_ENABLE_LEGACY_STUDIO=1`.
- Each session is capped on follow-up requests and total tracks. API calls retry 429 and transient 5xx errors up to three attempts with capped backoff.
- Confirmation prompts are not used: the live path performs no destructive operations. It only deletes its own temporary WAV files.
- The planner key is read from the server's environment only. The listener page and its scripts do not reference it.
- Actions are logged for audit. The log can contain listener requests, so it is git-ignored.

## License

Source code: [PolyForm Noncommercial License 1.0.0](LICENSE). Free for noncommercial use, including
study and coursework. This license covers this repository's code only.

Music is generated with Stable Audio 3 under the Stability AI Community License (research and
non-commercial use are free; the license says you own the outputs, to the extent the law allows, and
outputs are not derivative works of the model). Its Gemma-licensed text encoder is covered by the Gemma
Terms of Use. Model weights are not distributed here. See [NOTICE](NOTICE). Powered by Stability AI.
Track planning uses Google's Gemini API.
