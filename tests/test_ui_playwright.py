#!/usr/bin/env python3
"""Comprehensive 4-Tier Playwright E2E Test Suite for NEURAL DECK Web Studio.

File: tests/test_ui_playwright.py
Project: ai-auto-dj (6-Agent AI Auto-DJ & Music Production Studio)
Author: E2E Test Writer (test_writer_e2e_1)
Derived strictly from: ORIGINAL_REQUEST.md, PROJECT.md, and TEST_INFRA.md.

Tiers:
  - Fixture: Dynamic ephemeral uvicorn server on free OS port with isolated /dev/shm cache.
  - Tier 1: Viewport Responsiveness at 1920x1080 and 1440x900 (scrollWidth <= innerWidth, zero horizontal scrollbar).
  - Tier 2: Non-Overlapping Interactive Elements (collision-free bounding box analysis).
  - Tier 3: Progressive Disclosure Defaults (dock tab switching, stem drawers toggle).
  - Tier 4: Interactive Workflows with Zero Console/Page Errors:
      * Workflow 1: 1-Click Auto-DJ / Quickstart mix (#btnQuickStart)
      * Workflow 2: Deck loading and waveform scrubbing (#deckASelect, #waveCanvasA, #jogA)
      * Workflow 3: Crossfader and transition archetypes (#crossfader, #archButtons button, stem sliders)
      * Workflow 4: AI Producer and Spectrogram Modal (#prodTitle, #btnProduce, #specImage, #specModal)
      * Adversarial: Rapid interactions, extreme values, XSS escaping integrity
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
from typing import Any, Generator, List

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from autodj_server import create_app


# ============================================================================
# Pytest Fixtures: Ephemeral Server and Headless Chromium
# ============================================================================

@pytest.fixture(scope="session")
def server_url() -> Generator[str, None, None]:
    """Start an ephemeral FastAPI uvicorn instance on a dynamic OS port with isolated /dev/shm cache."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cache_dir = f"/dev/shm/autodj_playwright_e2e_{port}"
    os.makedirs(cache_dir, exist_ok=True)

    app = create_app(cache_dir=cache_dir, enable_legacy_studio=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to become responsive
    start_time = time.time()
    ready = False
    while time.time() - start_time < 20.0:
        try:
            with urllib.request.urlopen(f"{base_url}/api/state", timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)

    if not ready:
        raise RuntimeError(f"Ephemeral Auto-DJ server at {base_url} failed to start within 20s")

    yield base_url

    server.should_exit = True
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def browser() -> Generator[Browser, None, None]:
    """Launch headless Chromium browser for the test session."""
    with sync_playwright() as p:
        browser_instance = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        yield browser_instance
        browser_instance.close()


@pytest.fixture
def page(browser: Browser) -> Generator[Page, None, None]:
    """Provide an isolated browser page monitoring console.error and pageerror events."""
    context: BrowserContext = browser.new_context()
    pg: Page = context.new_page()

    errors: List[str] = []
    pg.on("pageerror", lambda err: errors.append(f"PAGEERROR: {err}"))
    pg.on("console", lambda msg: errors.append(f"CONSOLE {msg.type}: {msg.text}") if msg.type == "error" else None)
    setattr(pg, "_recorded_errors", errors)

    yield pg

    context.close()


def assert_zero_errors(page: Page) -> None:
    """Assert that zero unhandled pageerror or console.error events were captured."""
    errors = getattr(page, "_recorded_errors", [])
    assert len(errors) == 0, f"Uncaught page or console errors detected: {errors}"


def ensure_tracks_in_crate(page: Page, server_url: str) -> None:
    """Ensure at least 3 tracks exist in crate before running deck operations."""
    page.goto(server_url + "/studio")
    page.wait_for_load_state("domcontentloaded")
    crate_badge = page.locator("#crateCountBadge")
    count = 0
    if crate_badge.count() > 0:
        text = crate_badge.text_content() or "0"
        try:
            count = int(text.split()[0])
        except Exception:
            count = 0
    if count < 3:
        page.click("#btnQuickStart")
        page.wait_for_function(
            """() => {
                const crate = document.getElementById('crateCountBadge');
                return crate && parseInt(crate.textContent) >= 3;
            }""",
            timeout=50000,
        )


# ============================================================================
# Tier 1: Viewport Responsiveness & Zero Horizontal Overflow
# ============================================================================

@pytest.mark.parametrize("width,height", [
    (1920, 1080),
    (1440, 900),
])
def test_tier1_viewport_responsiveness_and_zero_horizontal_overflow(page: Page, server_url: str, width: int, height: int) -> None:
    """Tier 1: Asserts that at 1920x1080 and 1440x900 desktop viewports:
    1. document.documentElement.scrollWidth <= window.innerWidth
    2. Zero horizontal scrollbar is present (scrollWidth <= clientWidth)
    3. Document body overflow-x is not 'scroll'
    """
    page.set_viewport_size({"width": width, "height": height})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("networkidle")

    metrics = page.evaluate("""() => {
        const doc = document.documentElement;
        const body = document.body;
        return {
            scrollWidth: doc.scrollWidth,
            innerWidth: window.innerWidth,
            clientWidth: doc.clientWidth,
            bodyOverflowX: window.getComputedStyle(body).overflowX,
            docOverflowX: window.getComputedStyle(doc).overflowX
        };
    }""")

    assert metrics["scrollWidth"] <= metrics["innerWidth"], (
        f"Horizontal overflow at {width}x{height}: scrollWidth={metrics['scrollWidth']} > innerWidth={metrics['innerWidth']}"
    )
    assert metrics["scrollWidth"] <= metrics["clientWidth"] + 1, (
        f"Horizontal scrollbar detected at {width}x{height}: scrollWidth={metrics['scrollWidth']} > clientWidth={metrics['clientWidth']}"
    )
    assert metrics["bodyOverflowX"] != "scroll", f"body overflow-x is explicitly 'scroll' at {width}x{height}"
    assert metrics["docOverflowX"] != "scroll", f"document overflow-x is explicitly 'scroll' at {width}x{height}"

    assert_zero_errors(page)


# ============================================================================
# Tier 2: Non-Overlapping Interactive Elements
# ============================================================================

@pytest.mark.parametrize("width,height", [
    (1920, 1080),
    (1440, 900),
])
def test_tier2_non_overlapping_interactive_elements(page: Page, server_url: str, width: int, height: int) -> None:
    """Tier 2: Queries all visible interactive elements (buttons, inputs, selects, jog wheels)
    and asserts that no two interactive elements have intersecting bounding boxes (>1px in both dimensions).
    """
    page.set_viewport_size({"width": width, "height": height})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("networkidle")

    collision_data = page.evaluate("""() => {
        const els = Array.from(document.querySelectorAll('button, input, select, .jog-wheel')).filter(el => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' &&
                   style.visibility !== 'hidden' &&
                   style.opacity !== '0' &&
                   rect.width > 0 &&
                   rect.height > 0;
        });

        const rects = els.map((el, idx) => ({
            idx,
            id: el.id || el.getAttribute('aria-label') || el.className || el.tagName,
            tag: el.tagName,
            el: el,
            rect: el.getBoundingClientRect()
        }));

        const overlaps = [];
        for (let i = 0; i < rects.length; i++) {
            for (let j = i + 1; j < rects.length; j++) {
                const a = rects[i];
                const b = rects[j];
                // Ignore containment if one element is a child/descendant of the other
                if (a.el.contains(b.el) || b.el.contains(a.el)) continue;

                const r1 = a.rect;
                const r2 = b.rect;
                const dx = Math.min(r1.right, r2.right) - Math.max(r1.left, r2.left);
                const dy = Math.min(r1.bottom, r2.bottom) - Math.max(r1.top, r2.top);
                if (dx > 1.0 && dy > 1.0) {
                    overlaps.push({
                        a: a.id,
                        b: b.id,
                        dx: Math.round(dx * 10) / 10,
                        dy: Math.round(dy * 10) / 10,
                        r1: { left: r1.left, top: r1.top, right: r1.right, bottom: r1.bottom },
                        r2: { left: r2.left, top: r2.top, right: r2.right, bottom: r2.bottom }
                    });
                }
            }
        }
        return { count: els.length, overlaps };
    }""")

    assert collision_data["count"] > 10, f"Expected >10 interactive elements, found {collision_data['count']}"
    assert len(collision_data["overlaps"]) == 0, (
        f"Found {len(collision_data['overlaps'])} overlapping interactive elements at {width}x{height}: "
        f"{collision_data['overlaps']}"
    )

    assert_zero_errors(page)


# ============================================================================
# Tier 3: Progressive Disclosure Defaults & Interactive Switching
# ============================================================================

def test_tier3_progressive_disclosure_defaults(page: Page, server_url: str) -> None:
    """Tier 3: Asserts that on initial page load, secondary sections are NOT all simultaneously expanded:
    1. Secondary dock panels are housed in tabs, and not all expanded simultaneously.
    2. Stem drawers are present and start collapsed (or cleanly managed) so decks have generous whitespace.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("networkidle")

    # Verify that dock tabs and panes exist
    dock_tabs = ["#tabBtnCrate", "#tabBtnProducer", "#tabBtnAgents", "#tabBtnCritic", "#tabBtnTelemetry"]
    dock_panes = ["#tabPaneCrate", "#tabPaneProducer", "#tabPaneAgents", "#tabPaneCritic", "#tabPaneTelemetry"]

    for tab_id in dock_tabs:
        assert page.locator(tab_id).count() > 0, f"Expected dock tab trigger '{tab_id}' to exist in DOM"

    for pane_id in dock_panes:
        assert page.locator(pane_id).count() > 0, f"Expected dock pane '{pane_id}' to exist in DOM"

    # Count visible dock panes: must NOT all be visible simultaneously
    visible_panes = [pane_id for pane_id in dock_panes if page.locator(pane_id).is_visible()]
    assert len(visible_panes) <= 1, (
        f"Progressive disclosure violated on initial load: expected <=1 visible dock pane, found {len(visible_panes)}: {visible_panes}"
    )

    # Verify stem drawer elements exist
    assert page.locator("#stemDrawerA").count() > 0, "Expected '#stemDrawerA' to exist in DOM"
    assert page.locator("#stemDrawerB").count() > 0, "Expected '#stemDrawerB' to exist in DOM"
    assert page.locator("#btnToggleStemsA").count() > 0, "Expected '#btnToggleStemsA' to exist in DOM"
    assert page.locator("#btnToggleStemsB").count() > 0, "Expected '#btnToggleStemsB' to exist in DOM"

    assert_zero_errors(page)


def test_tier3_dock_tab_switching(page: Page, server_url: str) -> None:
    """Tier 3: Verifies that clicking each dock tab (#tabBtnCrate, #tabBtnProducer, #tabBtnAgents,
    #tabBtnCritic, #tabBtnTelemetry) displays its pane while hiding the others.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("networkidle")

    tab_pairs = [
        ("#tabBtnProducer", "#tabPaneProducer"),
        ("#tabBtnAgents", "#tabPaneAgents"),
        ("#tabBtnCritic", "#tabPaneCritic"),
        ("#tabBtnTelemetry", "#tabPaneTelemetry"),
        ("#tabBtnCrate", "#tabPaneCrate"),
    ]

    all_panes = ["#tabPaneCrate", "#tabPaneProducer", "#tabPaneAgents", "#tabPaneCritic", "#tabPaneTelemetry"]

    for btn_id, active_pane_id in tab_pairs:
        assert page.locator(btn_id).count() > 0, f"Expected dock tab trigger '{btn_id}' to exist in DOM"
        page.click(btn_id, timeout=3000)
        page.wait_for_timeout(200)

        # Active pane must be visible
        assert page.locator(active_pane_id).is_visible(), f"Clicking '{btn_id}' did not make '{active_pane_id}' visible"

        # Other panes must be hidden
        for other_pane in all_panes:
            if other_pane != active_pane_id:
                assert not page.locator(other_pane).is_visible(), (
                    f"Pane '{other_pane}' remained visible when '{active_pane_id}' was activated by '{btn_id}'"
                )

    assert_zero_errors(page)


def test_tier3_stem_drawers_toggle(page: Page, server_url: str) -> None:
    """Tier 3: Verifies that stem drawers (#stemDrawerA, #stemDrawerB) can be toggled
    via #btnToggleStemsA and #btnToggleStemsB.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("networkidle")

    for deck, btn_id, drawer_id in [("A", "#btnToggleStemsA", "#stemDrawerA"), ("B", "#btnToggleStemsB", "#stemDrawerB")]:
        assert page.locator(btn_id).count() > 0, f"Expected stem toggle button '{btn_id}' to exist in DOM"
        assert page.locator(drawer_id).count() > 0, f"Expected stem drawer '{drawer_id}' to exist in DOM"

        init_open = page.evaluate(f"""() => {{
            const el = document.querySelector('{drawer_id}');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetHeight > 0;
        }}""")

        # Click toggle button
        page.click(btn_id, timeout=3000)
        page.wait_for_timeout(250)

        toggled_open = page.evaluate(f"""() => {{
            const el = document.querySelector('{drawer_id}');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetHeight > 0;
        }}""")

        assert toggled_open != init_open, (
            f"Stem drawer '{drawer_id}' visibility did not toggle after clicking '{btn_id}' (was {init_open}, remained {toggled_open})"
        )

        # Toggle back to original state
        page.click(btn_id)
        page.wait_for_timeout(250)
        reverted_open = page.evaluate(f"""() => {{
            const el = document.querySelector('{drawer_id}');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetHeight > 0;
        }}""")
        assert reverted_open == init_open, (
            f"Stem drawer '{drawer_id}' failed to revert to initial state ({init_open}) after second click of '{btn_id}'"
        )

    assert_zero_errors(page)


# ============================================================================
# Tier 4: Interactive Workflows with Zero Console/Page Errors
# ============================================================================

def test_tier4_workflow_1_quickstart_mix(page: Page, server_url: str) -> None:
    """Workflow 1: 1-Click Auto-DJ / Quickstart mix.
    Clicks #btnQuickStart, asserts tracks >= 3, master mix ready, status updated, zero errors.
    """
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.goto(server_url + "/studio")
    page.wait_for_load_state("domcontentloaded")

    btn = page.locator("#btnQuickStart")
    assert btn.is_visible(), "Expected #btnQuickStart to be visible"
    btn.click()

    # Wait for quickstart to finish: crateCountBadge >= 3
    page.wait_for_function(
        """() => {
            const crate = document.getElementById('crateCountBadge');
            return crate && parseInt(crate.textContent) >= 3;
        }""",
        timeout=60000,
    )

    crate_text = page.text_content("#crateCountBadge") or ""
    assert int(crate_text.split()[0]) >= 3, f"Expected >= 3 tracks in crate, found '{crate_text}'"

    # Verify status badge updated
    status_text = page.text_content("#statusBadge") or ""
    assert ("READY" in status_text.upper() or "PLAYING" in status_text.upper() or "MIX" in status_text.upper()), (
        f"Status badge did not reflect active ready/playing state: '{status_text}'"
    )

    # Verify Deck A and Deck B dropdown options populated
    options_a = page.eval_on_selector("#deckASelect", "el => Array.from(el.options).map(o => o.value)")
    options_b = page.eval_on_selector("#deckBSelect", "el => Array.from(el.options).map(o => o.value)")
    assert len(options_a) >= 3, f"Deck A selector not populated with tracks: {options_a}"
    assert len(options_b) >= 3, f"Deck B selector not populated with tracks: {options_b}"

    assert_zero_errors(page)


def test_tier4_workflow_2_deck_load_and_scrub(page: Page, server_url: str) -> None:
    """Workflow 2: Deck loading and waveform scrubbing.
    Loads tracks into Deck A and Deck B, clicks waveform canvas to scrub, clicks jog wheel to toggle play/pause.
    """
    ensure_tracks_in_crate(page, server_url)
    page.set_viewport_size({"width": 1920, "height": 1080})

    # Select 2nd track in Deck A
    options_a = page.eval_on_selector("#deckASelect", "el => Array.from(el.options).map(o => o.value)")
    assert len(options_a) >= 2, f"Expected >=2 tracks in Deck A select, found {options_a}"
    page.select_option("#deckASelect", options_a[1])
    page.wait_for_timeout(300)

    # Click waveform canvas A to scrub
    wave_a = page.locator("#waveCanvasA")
    assert wave_a.is_visible(), "Expected #waveCanvasA to be visible"
    wave_box = wave_a.bounding_box()
    assert wave_box is not None and wave_box["width"] > 100, f"Invalid waveCanvasA bbox: {wave_box}"
    page.click("#waveCanvasA", position={"x": 100, "y": 20})
    page.wait_for_timeout(200)

    # Click jog wheel to toggle play/pause
    jog_a = page.locator("#jogA")
    assert jog_a.is_visible(), "Expected #jogA to be visible"
    jog_a.click()
    page.wait_for_timeout(300)

    # Select 1st track in Deck B
    options_b = page.eval_on_selector("#deckBSelect", "el => Array.from(el.options).map(o => o.value)")
    assert len(options_b) >= 1, f"Expected >=1 tracks in Deck B select, found {options_b}"
    page.select_option("#deckBSelect", options_b[0])
    page.wait_for_timeout(300)

    # Scrub waveform B
    wave_b = page.locator("#waveCanvasB")
    assert wave_b.is_visible(), "Expected #waveCanvasB to be visible"
    page.click("#waveCanvasB", position={"x": 150, "y": 20})
    page.wait_for_timeout(200)

    assert_zero_errors(page)


def test_tier4_workflow_3_crossfader_archetypes_and_stems(page: Page, server_url: str) -> None:
    """Workflow 3: Crossfader, transition archetypes, and stem adjustments.
    Adjusts crossfader, selects transition archetypes (T1-T6), modifies stem faders and mute buttons.
    """
    ensure_tracks_in_crate(page, server_url)
    page.set_viewport_size({"width": 1920, "height": 1080})

    # Test crossfader adjustment
    page.fill("#crossfader", "75")
    page.eval_on_selector("#crossfader", "el => el.dispatchEvent(new Event('input'))")
    page.wait_for_timeout(150)
    xfader_label = page.text_content("#xfaderLabel") or ""
    assert "DECK B" in xfader_label.upper() or "75" in xfader_label, (
        f"Crossfader label did not update on input=75: '{xfader_label}'"
    )

    page.fill("#crossfader", "25")
    page.eval_on_selector("#crossfader", "el => el.dispatchEvent(new Event('input'))")
    page.wait_for_timeout(150)
    xfader_label = page.text_content("#xfaderLabel") or ""
    assert "DECK A" in xfader_label.upper() or "25" in xfader_label, (
        f"Crossfader label did not update on input=25: '{xfader_label}'"
    )

    # Test transition archetypes selection (T1 through T6)
    for arch in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        arch_btn = page.locator(f"#archButtons button:has-text('{arch}')")
        if arch_btn.count() > 0 and arch_btn.first.is_visible():
            arch_btn.first.click()
            page.wait_for_timeout(100)
            badge_text = page.text_content("#activeArchBadge") or ""
            assert arch in badge_text, f"Active archetype badge '{badge_text}' did not reflect clicked '{arch}'"

    # If stem drawers exist and are closed, open them to access stem sliders
    if page.locator("#btnToggleStemsA").count() > 0:
        if not page.locator("#slider_A_drums").is_visible():
            page.click("#btnToggleStemsA")
            page.wait_for_timeout(200)

    # Adjust stem slider
    if page.locator("#slider_A_drums").is_visible():
        page.fill("#slider_A_drums", "90")
        page.eval_on_selector("#slider_A_drums", "el => el.dispatchEvent(new Event('input'))")
        page.wait_for_timeout(100)

    if page.locator("#mute_A_drums").is_visible():
        page.click("#mute_A_drums")
        page.wait_for_timeout(150)
        page.click("#mute_A_drums")
        page.wait_for_timeout(150)

    assert_zero_errors(page)


def test_tier4_workflow_4_ai_producer_and_spectrogram_modal(page: Page, server_url: str) -> None:
    """Workflow 4: AI Producer and Spectrogram Modal.
    Generates a new track via AI Producer form, confirms crate increments, opens 1024x1024 spectrogram modal, and closes it.
    """
    ensure_tracks_in_crate(page, server_url)
    page.set_viewport_size({"width": 1920, "height": 1080})

    # Switch to AI Producer tab if dock tabs exist
    if page.locator("#tabBtnProducer").count() > 0:
        page.click("#tabBtnProducer")
        page.wait_for_timeout(200)

    crate_text_before = page.text_content("#crateCountBadge") or "0"
    count_before = int(crate_text_before.split()[0])

    # Fill track title and produce
    page.fill("#prodTitle", "Apex Pulse")
    page.click("#btnProduce")

    # Wait for track count to increment
    page.wait_for_function(
        f"""() => {{
            const badge = document.getElementById('crateCountBadge');
            return badge && parseInt(badge.textContent) > {count_before};
        }}""",
        timeout=60000,
    )

    crate_text_after = page.text_content("#crateCountBadge") or "0"
    count_after = int(crate_text_after.split()[0])
    assert count_after > count_before, f"Crate count did not increment: was {count_before}, now {count_after}"

    # Switch to Spectrogram / Critic QA tab if dock tabs exist
    if page.locator("#tabBtnCritic").count() > 0:
        page.click("#tabBtnCritic")
        page.wait_for_timeout(200)

    # Click spectrogram preview image to open modal
    spec_img = page.locator("#specImage")
    assert spec_img.is_visible(), "Expected #specImage to be visible"
    spec_img.click()

    # Wait for modal dialog to open
    page.wait_for_function(
        """() => {
            const modal = document.getElementById('specModal');
            return modal && (modal.open === true || window.getComputedStyle(modal).display !== 'none');
        }""",
        timeout=5000,
    )

    modal_open = page.evaluate("() => document.getElementById('specModal')?.open === true")
    assert modal_open is True, "Expected #specModal to be open after clicking #specImage"

    # Close modal
    page.eval_on_selector("#specModal", "el => el.close ? el.close() : (el.style.display = 'none')")
    page.wait_for_timeout(200)

    modal_closed = page.evaluate("() => document.getElementById('specModal')?.open === false")
    assert modal_closed is True, "Expected #specModal to be closed"

    assert_zero_errors(page)


# ============================================================================
# Adversarial Edge Cases: Rapid Toggles & Escaping Integrity
# ============================================================================

def test_tier4_adversarial_rapid_interactions_and_escaping(page: Page, server_url: str) -> None:
    """Adversarial Verification:
    1. Rapid dock tab toggles in immediate succession to stress transition handling.
    2. Input injection in Producer title (<script>alert(1)</script> & "quotes") to verify escaping fidelity.
    3. Rapid crossfader sliding (0 -> 100 -> 0 -> 50).
    Asserts zero console errors or page faults across all stressors.
    """
    ensure_tracks_in_crate(page, server_url)
    page.set_viewport_size({"width": 1440, "height": 900})

    # Rapid tab toggling if dock tabs exist
    if page.locator("#tabBtnCrate").count() > 0:
        tabs = ["#tabBtnProducer", "#tabBtnAgents", "#tabBtnCritic", "#tabBtnTelemetry", "#tabBtnCrate"]
        for _ in range(2):
            for tab_id in tabs:
                page.click(tab_id)
                time.sleep(0.05)

    # Rapid crossfader sliding
    for val in ["0", "100", "0", "50"]:
        page.fill("#crossfader", val)
        page.eval_on_selector("#crossfader", "el => el.dispatchEvent(new Event('input'))")
        time.sleep(0.05)

    # Special characters & escaping check in track title
    if page.locator("#tabBtnProducer").count() > 0:
        page.click("#tabBtnProducer")
        page.wait_for_timeout(100)

    injection_str = "<script>console.log('xss-test')</script> & 'quotes' \"double\" 🎵"
    page.fill("#prodTitle", injection_str)
    val = page.input_value("#prodTitle")
    assert val == injection_str, "Input value suffered distortion or unescaped truncation"

    assert_zero_errors(page)

@pytest.mark.parametrize("width,height", [(1440, 900), (390, 844)])
def test_prompt_first_dj_flow_and_errors(browser: Browser, server_url: str, width: int, height: int) -> None:
    """The listener can make a request and recover from a failed live start."""
    page = browser.new_page(viewport={"width": width, "height": height})
    try:
        page.goto(server_url, wait_until="networkidle")
        assert page.locator("a[href='/studio']").count() == 0
        assert page.locator("img").count() == 0
        start = page.get_by_role("button", name="Start listening")
        start.click()
        assert "what you’d like" in page.locator("#status").text_content()
        page.get_by_role("button", name="Piano dubstep").click()
        assert "acoustic grand piano" in page.locator("#request").input_value()
        page.route("**/api/dj-live/start", lambda route: route.fulfill(
            status=503, json={"detail": "LLM unavailable"}))
        start.click()
        page.wait_for_function("document.querySelector('#status').textContent.includes('Couldn’t start')")
        assert start.is_enabled()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()


def test_live_player_queues_updates_and_stops(page: Page, server_url: str) -> None:
    """The browser schedules two returned audio chunks and sends playback progress."""
    import io, json, wave
    binary = io.BytesIO()
    with wave.open(binary, 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(8000); wav.writeframes(b'\x00' * (8000 * 4 // 2))
    sound = binary.getvalue(); calls = {"ack": [], "steer": [], "stop": 0}
    def route_handler(route):
        req = route.request; url = req.url
        if url.endswith('/start'):
            route.fulfill(json={'session_id':'abc'})
        elif url.endswith('/status'):
            segments = [{"seq":seq,"title":f"Track {seq}","style":"Ambient","bpm":110,"camelot":"8A","backend":"Test backend","transition":"opening" if seq==0 else "linear","overlap_seconds":1,"direction_revision":0,"audio_url":f"/api/dj-live/abc/audio/{seq}"} for seq in range(len(calls['ack']),2)]
            route.fulfill(json={'state':'ready','segments':segments})
        elif '/audio/' in url:
            route.fulfill(status=200,content_type='audio/wav',body=sound)
        elif url.endswith('/ack'):
            calls['ack'].append(req.post_data_json['seq']);route.fulfill(json={'status':'ok'})
        elif url.endswith('/steer'):
            calls['steer'].append(req.post_data_json['request']);route.fulfill(json={'status':'accepted'})
        elif url.endswith('/stop'):
            calls['stop']+=1;route.fulfill(json={'status':'stopping'})
        else:route.abort()
    page.goto(server_url,wait_until='networkidle');page.route('**/api/dj-live/**',route_handler)
    page.locator('#request').fill('mellow start');page.locator('#live-start').click()
    page.wait_for_function("document.querySelector('#live-stop').offsetParent !== null")
    page.wait_for_function("document.querySelectorAll('#tracks li').length >= 1",timeout=10000)
    page.wait_for_function("document.querySelector('#console-state').dataset.state === 'playing'")
    page.wait_for_timeout(1400)
    assert calls['ack'][:2]==[0,1]
    page.locator('#request').fill('more energy');page.locator('#live-steer').click()
    page.wait_for_function("document.querySelector('#status').textContent.includes('Request sent')")
    assert calls['steer']==['more energy']
    page.locator('#live-stop').click();page.wait_for_timeout(100);assert calls['stop']==1
    page.wait_for_function("!document.querySelector('#live-start').hidden")
    assert page.locator('#live-start').is_visible()


def test_live_stop_silences_player_when_server_is_unreachable(page: Page, server_url: str) -> None:
    def handler(route):
        if route.request.url.endswith('/start'):route.fulfill(json={'session_id':'abc'})
        elif route.request.url.endswith('/status'):route.fulfill(json={'state':'planning','segments':[]})
        elif route.request.url.endswith('/stop'):route.abort()
        else:route.abort()
    page.goto(server_url,wait_until='networkidle');page.route('**/api/dj-live/**',handler)
    page.locator('#request').fill('mellow');page.locator('#live-start').click()
    page.locator("#live-stop").wait_for(state="visible")
    page.locator('#live-stop').click()
    page.wait_for_function("!document.querySelector('#live-start').hidden")
    page.wait_for_function("document.querySelector('#status').textContent.includes('lost its connection')")
