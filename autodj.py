#!/home/prab/.gemini/antigravity/scratch/ai-auto-dj/.venv/bin/python
"""Unified Single Entry Point & GUI Studio Launcher (`autodj.py`) for `ai-auto-dj`.

Launch the entire 6-Agent AI DJ & Music Production Studio GUI with a single command:
    /home/prab/.gemini/antigravity/scratch/ai-auto-dj/autodj.py

From the single NEURAL DECK Web Studio GUI (`http://127.0.0.1:8088`), you can:
1. Click `⚡ 1-CLICK QUICK-START DEMO MIX` to synthesize club tracks, run all 6 agents, and play the mix.
2. Scan & index any local folder/track (`ListenerAgent`: 4-stem HTDemucs, BPM/downbeats, CQT Camelot key).
3. Produce new club tracks or harmonic bridges (`ProducerAgent`: ACE-Step 1.5, MusicGen-Melody, Suno v5).
4. Load tracks onto Dual Spinning CDJ Decks (Deck A & Deck B) with 4-stem faders, mutes, solo stem audition,
   and 8 phrase hot-cue pads (`INTRO`, `BUILD`, `DROP`, `OUTRO`).
5. Steer the set energy, select Transition Archetypes (`AUTO`, `T1`–`T6`), inspect the Camelot Harmonic Wheel
   and 4-Panel `1024x1024` Spectrograms (`CriticAgent`), trigger `Kokoro-82M` DJ MC drops (`HostAgent`),
   listen directly in the browser (`<audio>` + WebAudio FFT analyzer) or stream to PipeWire (`pw-play`).
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import threading
import time
import webbrowser

from autodj_crew import AutoDJCrew
from autodj_server import (
    PipeWireLookaheadStreamer,
    QuickStartRequest,
    LibraryScanRequest,
    app,
    create_app,
    export_master_mix_and_cue,
)

__all__ = [
    "AutoDJCrew",
    "PipeWireLookaheadStreamer",
    "app",
    "create_app",
    "export_master_mix_and_cue",
    "main",
]


def _free_stale_autodj_port(host: str, port: int) -> None:
    """If an older autodj.py instance is holding `port`, terminate it cleanly before binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        if sock.connect_ex((host, port)) != 0:
            return
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(["pgrep", "-f", "autodj\\.py"], text=True)
        for line in out.splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex((host, port)) != 0:
                    return
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autodj",
        description="NEURAL DECK // AI-Auto-DJ: Single Unified 6-Agent DJ & Music Production Studio GUI",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088, help="Bind port (default: 8088)")
    parser.add_argument("--cache-dir", default="/dev/shm/autodj_studio", help="Studio RAM disk cache directory")
    parser.add_argument("--scan-dir", default=None, help="Optional local music directory to pre-index on startup")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Pre-generate 3 club tracks and run an initial 6-agent mix before opening the GUI",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab on startup")
    parser.add_argument("--check-only", action="store_true", help="Verify unified studio initialization and exit")
    args = parser.parse_args(argv)

    studio_app = create_app(cache_dir=args.cache_dir)

    if args.scan_dir or args.quickstart:
        from fastapi.testclient import TestClient

        client = TestClient(studio_app)
        if args.scan_dir and os.path.exists(args.scan_dir):
            client.post("/api/library/scan", json={"path": args.scan_dir, "max_files": 12})
        if args.quickstart:
            client.post("/api/quickstart", json={"bars": 8, "enable_host_mc": True})

    url = f"http://{args.host}:{args.port}"
    print("=" * 78)
    print("  NEURAL DECK // AI-AUTO-DJ — 6-Agent Virtual DJ & Production Studio GUI")
    print(f"  Studio URL : {url}")
    print(f"  RAM Cache  : {args.cache_dir}")
    print("=" * 78)

    if args.check_only:
        return 0

    _free_stale_autodj_port(args.host, args.port)

    if not args.no_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        def _open() -> None:
            time.sleep(0.9)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(studio_app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
