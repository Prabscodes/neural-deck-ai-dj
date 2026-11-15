# LLM-directed DJ demo

The prompt-first home page supports **Start live DJ** and **Short mix**. In live mode, Gemini 4 Hydrogen chooses each next track and either a cut, linear crossfade, or equal-power crossfade. The app validates each plan, generates completed audio segments, and schedules them in the browser. You can update the direction during playback; a track being planned or already queued may still play before the change takes effect. Stop ends playback immediately, while any in-progress model call finishes cooperatively before another run can start.

The queue is bounded and stays at most three segments ahead. When generation is slower than playback, the player displays buffering and resumes once audio is available. It delivers completed WAV segments, not incremental low-latency token streaming. If the tab disconnects, a lease eventually stops background generation. The actual transition and backend for each track are shown in the page and saved in the action log.

With `AUTODJ_MUSIC_BACKEND=ace_step`, the live route generates audio locally with cached ACE-Step. The optional dubstep arranger layers sampled piano, drums and synthesized bass with ACE-Step texture. Otherwise the `neural` path uses the existing MusicGen GPU pipeline and pretrained local weights. Its implementation creates a short MusicGen passage, repeats it to fit the target duration, and blends it with procedural audio and an Oobleck VAE-derived signal. It is not full ACE-Step song generation. The `synth` path is a procedural electronic synthesizer: it uses tempo and key, but does not reproduce an arbitrary genre or interpret the text style. Neither path guarantees musical quality or prompt fidelity.

## Local use

Log into Antigravity CLI and make sure `agy` is on the DJ server’s PATH. The planner sends up to 16 seconds of previously generated audio to Gemini for the next track. From this project directory run:

```sh
export AUTODJ_GEMINI_TRANSPORT=cli
export AUTODJ_GEMINI_MODEL='gemini-4-hydrogen-low'
export AUTODJ_MUSIC_BACKEND=ace_step
export AUTODJ_DUBSTEP_ARRANGER=1
export AUTODJ_AGENT_DIR="$PWD/dj_agent_runs"
.venv/bin/python autodj.py --no-browser
```

Open http://127.0.0.1:8088/, enter a request, and click **Start live DJ**. Press **Update direction** after changing the prompt, or **Stop live DJ** to end playback. Use **Advanced studio** to open the original deck controls. If the planner is unavailable or its plan fails validation, the panel reports an error. The neural path requires working CUDA and the project's cached model dependencies. The plan and generation requests can take time; the button stays busy while they run.

For an API-only submission, set `AUTODJ_GEMINI_TRANSPORT=api`, `AUTODJ_GEMINI_MODEL=gemini-4-hydrogen`, and provide `GEMINI_API_KEY` as a server environment secret.

`dj_agent_runs/actions.jsonl` contains the request, plan, actions, outcomes and errors. It and the generated WAV files contain user data; protect them accordingly. The prompt-first live and short-mix routes allow one job at a time and accepts no LLM-supplied file paths or shell commands. The existing studio was built for localhost and allows local file access; do not expose it publicly without authentication, isolation, resource limits and a deployment review.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_dj_agent.py
```

The automated test uses a fixed plan and exercises procedural audio, validation and the action log. A live Gemini API and GPU run must also be checked on the target machine. For the assignment, a public GitHub repository, a deployment that the instructor can access, and a 1–2 minute video are still required.
