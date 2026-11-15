# Project: NEURAL DECK Web Studio Progressive Disclosure Redesign (`ai-auto-dj`)

## Architecture
- **Single Page Studio Bundle**: `WEB_STUDIO_HTML` embedded in `autodj_server.py:285-2035` served at `GET /` with zero external frontend build step.
- **Backend Service & State**: FastAPI application factory `create_app()` managing `studio_state` (decks A/B, stem gains, archetypes, profiles, telemetry events) protected by `render_lock`.
- **Audio & DSP Core**: WebAudio API frontend client receiving audio streams from `/api/audio/*`, waveform peak data from `/api/envelope/*`, and telemetry over `/ws/telemetry`.
- **E2E Testing Track**: Automated Playwright test harness in `tests/test_ui_playwright.py` verifying viewports, zero overflow, bounding-box collision detection, progressive disclosure defaults, and 4 interactive workflows in headless Chromium.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Single-Row Command Bar | Hero 1-Click Quick Mix CTA, streamlined transport controls, collapsible tech specs pill, master audio player & LUFS readout in a single <=58px row | M1 | explorer_survey_1 |
| 2 | Dual CDJ-3000 Deck Booth | Deck A and Deck B with motorized jog wheels, OLED pitch/key readouts, 8 RGB hot-cue pads, and track selectors | M1 | explorer_survey_1 |
| 3 | Collapsible Stem Isolator Drawers | `#stemDrawerA` and `#stemDrawerB` housing 4-stem sliders (Drums, Bass, Vocals, Other) with mute/solo buttons, collapsible to let deck platters breathe | M1 | explorer_survey_1 |
| 4 | Central Mixer & Transition Radar | Active archetype badge (T1–T6), Camelot wheel SVG radar, 64-point bass-swap envelope canvas, and master crossfader | M1 | explorer_survey_1 |
| 5 | 5-Tab Workstation Dock | Switchable tabs for Crate/Library, AI Producer, 6-Agent Matrix & MC, Spectrogram & Critic QA, and Telemetry/Logs bounding dock height to ~240px | M1 | explorer_survey_1 |
| 6 | Preservation of 93 DOM IDs | Strict retention of all 93 existing DOM IDs for complete backward compatibility with client scripts | M1 | explorer_survey_1 |
| 7 | Preservation of 7 Test String Markers | Exact preservation of `"waveCanvasA"`, `"waveCanvasB"`, `"camelotSvg"`, `"envCanvas"`, `"fftCanvas"`, `"runQuickStart"`, `"soloStemAudio"`, and `"AI-Auto-DJ: 6-Agent Virtual DJ"` | M1 | explorer_survey_2, tests/test_autodj.py |
| 8 | Hardware-Inspired Visual Polish | Modern dark-matte DJ hardware aesthetic (Pioneer CDJ-3000 / DJM-900NXS2), neon accents, generous whitespace, legible typography, reduced badge density | M1 | explorer_survey_1, ORIGINAL_REQUEST §R2 |
| 9 | Zero Viewport Overflow (1920x1080 & 1440x900) | Elimination of the 257px vertical overflow at 1440x900 and 1px at 1920x1080 by bounding total page height to ~820px, guaranteeing zero horizontal overflow | M1 | explorer_survey_1, explorer_survey_3 |
| 10 | Non-Overlapping Interactive Elements | Programmatic collision-free layout ensuring no two interactive buttons, inputs, or sliders overlap | M1 | explorer_survey_3, ORIGINAL_REQUEST §Acceptance |
| 11 | Streamlined Interaction Feedback | Minimal clicks for Quick Mix, clear busy/ready status badges, smooth drawer transitions, and responsive tab switching | M2 | explorer_survey_1, ORIGINAL_REQUEST §R2 |
| 12 | State & WebAudio Synchronization | Seamless real-time state updates via `/api/state` and `/ws/telemetry`, waveform scrubbing, jog wheel animation, and stem gain sync | M2 | explorer_survey_2 |
| 13 | High-Resolution Spectrogram Lightbox | Full 1024x1024 4-panel visual-acoustic spectrogram inspection dialog with backdrop dismiss | M2 | explorer_survey_1, explorer_survey_3 |
| 14 | E2E Test Suite (Tiers 1-4) | Automated headless Playwright suite covering viewports, overflow, bounding-box collision detection, progressive disclosure defaults, and 4 interactive workflows | E2E | explorer_survey_3, ORIGINAL_REQUEST §Acceptance |
| 15 | Adversarial Coverage Hardening (Tier 5) | White-box edge-case testing: rapid tab toggles, empty crate states, concurrent requests, audio autoplay fallbacks | M3 | PROJECT Pattern Phase 2 |
| 16 | External GPU & Resource Safety Guard | Verification that no ComfyUI or other GPU processes are stopped, and model weight caches are untouched | M3 | explorer_survey_2, ORIGINAL_REQUEST §R3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite Track | Design and implement `tests/test_ui_playwright.py` with 4-tier coverage, publish `TEST_READY.md` | none | DONE |
| M1 | Progressive Disclosure Architecture & Hardware Visual Polish | Restructure `WEB_STUDIO_HTML` with single-row Command Bar, CDJ-3000 deck booth, collapsible stem drawers, 5-tab Workstation Dock, dark-matte hardware styling, <=820px height bounding, zero overflow at 1920x1080 & 1440x900, zero overlaps, preserving all 93 DOM IDs and 7 test markers | none | DONE |
| M2 | Interaction Flow & State Synchronization | Wire up smooth drawer/tab transitions, status feedback, WebAudio waveform updates, WebSocket telemetry, and spectrogram modal | M1 | DONE |
| M3 | Final E2E Certification & Adversarial Hardening | Pass 100% of E2E Playwright suite (`TEST_READY.md`) + full pytest backend suite (`tests/test_autodj.py`), execute Tier 5 adversarial testing | M2, E2E | DONE |

## Interface Contracts
### `WEB_STUDIO_HTML` ↔ `autodj_server.py`
- `index_page() -> str`: Returns `WEB_STUDIO_HTML`.
- Required string markers:
  - `"AI-Auto-DJ: 6-Agent Virtual DJ"`
  - `"waveCanvasA"`
  - `"waveCanvasB"`
  - `"camelotSvg"`
  - `"envCanvas"`
  - `"fftCanvas"`
  - `"runQuickStart"`
  - `"soloStemAudio"`

### Frontend DOM ↔ Client JavaScript
- All 93 DOM IDs must remain accessible in `document.getElementById(...)`.
- Tab navigation triggers:
  - Tab buttons: `#tabBtnCrate`, `#tabBtnProducer`, `#tabBtnAgents`, `#tabBtnCritic`, `#tabBtnTelemetry`, `#btnToggleDock`
  - Tab panes: `#tabPaneCrate`, `#tabPaneProducer`, `#tabPaneAgents`, `#tabPaneCritic`, `#tabPaneTelemetry`
- Stem drawer triggers:
  - `#btnToggleStemsA`, `#stemDrawerA`
  - `#btnToggleStemsB`, `#stemDrawerB`

### E2E Test Suite ↔ Test Runner
- Command: `/home/prab/.gemini/antigravity/scratch/ai-auto-dj/.venv/bin/pytest tests/test_ui_playwright.py`
- Viewports tested: `1920x1080` and `1440x900`
- Pass criteria:
  - `scrollWidth <= innerWidth`
  - Overlapping interactive bounding boxes == 0
  - Progressive disclosure default: advanced dock tabs not simultaneously expanded
  - Zero uncaught `pageerror` and zero `console.error`
  - Backend regression pass: `/home/prab/.gemini/antigravity/scratch/ai-auto-dj/.venv/bin/pytest tests/test_autodj.py`

## Code Layout
- `autodj_server.py`: Contains `WEB_STUDIO_HTML` (lines 285–2035) and FastAPI routes. Owned by Implementation Track Workers.
- `tests/test_ui_playwright.py`: Headless Playwright test suite. Owned exclusively by E2E Testing Track Workers.
- `tests/test_autodj.py`: Existing backend unit and integration test suite. Read-only verification target.
