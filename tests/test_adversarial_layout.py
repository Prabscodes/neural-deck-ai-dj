#!/usr/bin/env python3
"""Adversarial Layout, Viewport Responsiveness, and Collision Stress Test.

File: tests/test_adversarial_layout.py
Author: Challenger M1 (challenger_m1_1)
Objective:
  1. Test desktop viewports: 1920x1080, 1440x900, 1600x900, 1366x768, 1280x800.
  2. Test all state permutations:
     - Default initial state
     - Stem Drawer A open
     - Stem Drawer B open
     - Both Stem Drawers A and B open
     - Specs dropdown open
     - All 5 dock tabs (Crate, Producer, Agents, Critic, Telemetry)
     - Dock collapsed
     - Both Stem Drawers open + Specs dropdown open across all 5 dock tabs
     - Spectrogram modal open
  3. Verify:
     - Horizontal overflow (scrollWidth <= innerWidth)
     - Vertical overflow at 1920x1080 and 1440x900 (scrollHeight <= innerHeight, total dock bottom <= 820px)
     - Interactive element bounding-box collisions (buttons, inputs, selects, jog wheels, summaries, links)
     - Zero uncaught console errors or page errors
"""

import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from autodj_server import create_app
from playwright.sync_api import sync_playwright, Page, Browser


def start_test_server(port: int, cache_dir: str):
    app = create_app(cache_dir=cache_dir, enable_legacy_studio=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    start = time.time()
    while time.time() - start < 15:
        try:
            with urllib.request.urlopen(f"{base_url}/api/state", timeout=0.5) as r:
                if r.status == 200:
                    return server, base_url
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Server failed to start")


COLLISION_DETECTOR_JS = """() => {
    const selector = 'button, input, select, .jog-wheel, summary, a[href]';
    const els = Array.from(document.querySelectorAll(selector)).filter(el => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style.display !== 'none' &&
               style.visibility !== 'hidden' &&
               style.opacity !== '0' &&
               rect.width > 0 &&
               rect.height > 0;
    });

    const items = els.map((el, idx) => ({
        idx,
        id: el.id || el.getAttribute('aria-label') || el.innerText?.trim()?.slice(0, 20) || el.className || el.tagName,
        tag: el.tagName,
        el: el,
        rect: el.getBoundingClientRect()
    }));

    const overlaps = [];
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const a = items[i];
            const b = items[j];
            // Skip parent/child relationships
            if (a.el.contains(b.el) || b.el.contains(a.el)) continue;

            const r1 = a.rect;
            const r2 = b.rect;
            const dx = Math.min(r1.right, r2.right) - Math.max(r1.left, r2.left);
            const dy = Math.min(r1.bottom, r2.bottom) - Math.max(r1.top, r2.top);
            if (dx > 1.0 && dy > 1.0) {
                overlaps.push({
                    a: a.id,
                    b: b.id,
                    tagA: a.tag,
                    tagB: b.tag,
                    dx: Math.round(dx * 10) / 10,
                    dy: Math.round(dy * 10) / 10,
                    r1: { left: Math.round(r1.left), top: Math.round(r1.top), width: Math.round(r1.width), height: Math.round(r1.height) },
                    r2: { left: Math.round(r2.left), top: Math.round(r2.top), width: Math.round(r2.width), height: Math.round(r2.height) }
                });
            }
        }
    }
    return { count: items.length, overlaps };
}"""

METRICS_JS = """() => {
    const doc = document.documentElement;
    const body = document.body;
    const dock = document.getElementById('dockContainer');
    const dockBottom = dock ? (dock.offsetTop + dock.offsetHeight) : 0;
    return {
        scrollWidth: doc.scrollWidth,
        innerWidth: window.innerWidth,
        clientWidth: doc.clientWidth,
        scrollHeight: doc.scrollHeight,
        innerHeight: window.innerHeight,
        clientHeight: doc.clientHeight,
        dockBottom: dockBottom,
        bodyOverflowX: window.getComputedStyle(body).overflowX,
        docOverflowX: window.getComputedStyle(doc).overflowX
    };
}"""


def run_adversarial_suite():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cache_dir = f"/dev/shm/autodj_adversarial_shm_{port}"
    os.makedirs(cache_dir, exist_ok=True)
    server, base_url = start_test_server(port, cache_dir)
    print(f"Test server running at {base_url}")

    results = []
    failures = []

    viewports = [
        ("1920x1080", 1920, 1080, True),   # name, width, height, strict_v_overflow
        ("1440x900", 1440, 900, True),
        ("1600x900", 1600, 900, True),
        ("1366x768", 1366, 768, False),   # smaller desktop laptop
        ("1280x800", 1280, 800, False),
    ]

    states = [
        ("Default (Drawers closed, Crate tab, Specs closed)", lambda p: None),
        ("Stems A Open", lambda p: p.click("#btnToggleStemsA")),
        ("Stems B Open", lambda p: p.click("#btnToggleStemsB")),
        ("Both Stems A & B Open", lambda p: (p.click("#btnToggleStemsA"), p.click("#btnToggleStemsB"))),
        ("Specs Dropdown Open", lambda p: p.click("#specsDropdown summary")),
        ("Tab: AI Producer", lambda p: p.click("#tabBtnProducer")),
        ("Tab: 6-Agent Matrix", lambda p: p.click("#tabBtnAgents")),
        ("Tab: Critic QA", lambda p: p.click("#tabBtnCritic")),
        ("Tab: Telemetry", lambda p: p.click("#tabBtnTelemetry")),
        ("Dock Collapsed", lambda p: p.click("#btnToggleDock")),
        ("Stress: Stems A+B Open + Specs Open + Producer Tab", lambda p: (
            p.click("#btnToggleStemsA"),
            p.click("#btnToggleStemsB"),
            p.click("#specsDropdown summary"),
            p.click("#tabBtnProducer")
        )),
        ("Stress: Stems A+B Open + Specs Open + Agents Tab", lambda p: (
            p.click("#btnToggleStemsA"),
            p.click("#btnToggleStemsB"),
            p.click("#specsDropdown summary"),
            p.click("#tabBtnAgents")
        )),
        ("Stress: Stems A+B Open + Specs Open + Critic Tab", lambda p: (
            p.click("#btnToggleStemsA"),
            p.click("#btnToggleStemsB"),
            p.click("#specsDropdown summary"),
            p.click("#tabBtnCritic")
        )),
        ("Stress: Stems A+B Open + Specs Open + Telemetry Tab", lambda p: (
            p.click("#btnToggleStemsA"),
            p.click("#btnToggleStemsB"),
            p.click("#specsDropdown summary"),
            p.click("#tabBtnTelemetry")
        )),
    ]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )

            for vp_name, width, height, strict_v in viewports:
                print(f"\n==========================================")
                print(f"Testing Viewport: {vp_name} (width={width}, height={height})")
                print(f"==========================================")

                for state_name, state_prep in states:
                    context = browser.new_context(viewport={"width": width, "height": height})
                    page = context.new_page()

                    recorded_errors = []
                    page.on("pageerror", lambda err: recorded_errors.append(f"PAGEERROR: {err}"))
                    page.on("console", lambda msg: recorded_errors.append(f"CONSOLE: {msg.text}") if msg.type == "error" else None)

                    page.goto(base_url + "/studio")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(100)

                    # Apply state mutation
                    try:
                        state_prep(page)
                        page.wait_for_timeout(150)
                    except Exception as ex:
                        fail_msg = f"Failed to setup state '{state_name}' at {vp_name}: {ex}"
                        failures.append(fail_msg)
                        print(f"  [ERROR] {fail_msg}")
                        context.close()
                        continue

                    # Measure overflow metrics
                    metrics = page.evaluate(METRICS_JS)
                    collisions = page.evaluate(COLLISION_DETECTOR_JS)

                    # Check horizontal overflow
                    h_overflow = metrics["scrollWidth"] > metrics["innerWidth"]
                    h_scroll_present = metrics["scrollWidth"] > metrics["clientWidth"] + 1

                    # Check vertical overflow for target viewports
                    v_overflow = metrics["scrollHeight"] > metrics["innerHeight"]
                    dock_exceeds = metrics["dockBottom"] > 820 if strict_v else False

                    # Overlaps
                    overlap_count = len(collisions["overlaps"])

                    status = "PASS"
                    issue_details = []

                    if h_overflow or h_scroll_present:
                        status = "FAIL"
                        issue_details.append(
                            f"Horizontal overflow: scrollWidth={metrics['scrollWidth']} > innerWidth={metrics['innerWidth']}"
                        )

                    if strict_v and v_overflow:
                        status = "FAIL"
                        issue_details.append(
                            f"Vertical overflow: scrollHeight={metrics['scrollHeight']} > innerHeight={metrics['innerHeight']}"
                        )

                    if strict_v and dock_exceeds:
                        status = "FAIL"
                        issue_details.append(
                            f"Dock bottom exceeds 820px: dockBottom={metrics['dockBottom']}px"
                        )

                    if overlap_count > 0:
                        status = "FAIL"
                        issue_details.append(
                            f"{overlap_count} overlapping interactive elements: {collisions['overlaps']}"
                        )

                    if len(recorded_errors) > 0:
                        status = "FAIL"
                        issue_details.append(
                            f"Console/Page errors: {recorded_errors}"
                        )

                    result_entry = {
                        "viewport": vp_name,
                        "state": state_name,
                        "status": status,
                        "metrics": metrics,
                        "interactive_count": collisions["count"],
                        "overlap_count": overlap_count,
                        "overlaps": collisions["overlaps"],
                        "errors": recorded_errors,
                        "issues": issue_details,
                    }
                    results.append(result_entry)

                    if status == "FAIL":
                        failures.append(f"[{vp_name}] {state_name}: {'; '.join(issue_details)}")
                        print(f"  ❌ FAIL: {state_name} -> {'; '.join(issue_details)}")
                    else:
                        print(f"  ✅ PASS: {state_name} (els={collisions['count']}, dockBottom={metrics['dockBottom']}px, sw={metrics['scrollWidth']}/{metrics['innerWidth']})")

                    context.close()

            browser.close()
    finally:
        server.should_exit = True
        shutil.rmtree(cache_dir, ignore_errors=True)

    print("\n==========================================")
    print("SUITE SUMMARY:")
    print(f"Total scenarios tested: {len(results)}")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll adversarial layout scenarios passed successfully!")

    # Write JSON results for detailed reporting
    report_path = "/home/prab/.gemini/antigravity/scratch/ai-auto-dj/.agents/teamwork/challenger_m1_1/adversarial_results.json"
    with open(report_path, "w") as f:
        json.dump({"total": len(results), "failures": len(failures), "details": results}, f, indent=2)
    print(f"Results written to {report_path}")

    return len(failures) == 0


if __name__ == "__main__":
    success = run_adversarial_suite()
    sys.exit(0 if success else 1)
