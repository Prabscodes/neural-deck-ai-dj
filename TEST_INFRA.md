# E2E Test Infra: NEURAL DECK Web Studio (`ai-auto-dj`)

## Test Philosophy
- Opaque-box, requirement-driven, derived strictly from `ORIGINAL_REQUEST.md`.
- Headless Playwright browser automation in Python using Chromium.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory & Test Mapping
| # | Feature | Source (Requirement) | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Pairwise) | Tier 4 (Workflows) |
|---|---------|----------------------|:-----------------:|:-----------------:|:-----------------:|:------------------:|
| 1 | 1-Click Auto-DJ / Quick Mix | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Deck A/B Waveforms & Playheads | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | Transport Controls (Jog Play/Pause, Cues) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | Central Mixer & Crossfader | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 5 | Transition Archetypes (T1–T6) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 6 | 4-Stem Isolator Drawers (Drums, Bass, Vocals, Other) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 7 | Crate & Library Drawer/Tab | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 8 | AI Track Producer Studio Tab | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 9 | 6-Agent Matrix & Host MC Tab | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 10 | Spectrogram & Critic QA Tab / Lightbox | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 11 | Telemetry & Certified Transitions Tab | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 12 | Viewport Responsiveness (1920x1080 & 1440x900) | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ | ✓ |
| 13 | Zero Overlapping Interactive Elements | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ | ✓ |
| 14 | Zero Console / Page Errors | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test runner: `/home/prab/.gemini/antigravity/scratch/ai-auto-dj/.venv/bin/pytest tests/test_ui_playwright.py`
- Test file: `/home/prab/.gemini/antigravity/scratch/ai-auto-dj/tests/test_ui_playwright.py`
- Browser: Headless Chromium via `playwright.sync_api`
- Ephemeral server fixture: background thread on free dynamic port with isolated `/dev/shm` cache.

## Coverage Goals
- Tier 1: Viewport & Layout checks (scrollWidth <= innerWidth at 1920x1080 and 1440x900).
- Tier 2: Progressive disclosure default checks (stem drawers closed, dock tabs switchable, spec modal closed).
- Tier 3: Zero bounding-box overlaps across all visible interactive elements.
- Tier 4: Real-world user flows (1-Click QuickStart mix, deck track loading & waveform scrub, crossfader & stem EQ adjust, AI producer generation & spectrogram modal open/close) with 0 console errors.
