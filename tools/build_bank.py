"""Render the pre-generated track bank on a machine with the local Stable Audio renderer.

Usage (renderer must be running, see start_live_stable.sh):
    .venv/bin/python tools/build_bank.py --out bank [--seeds 2] [--limit 5]

Resumable: tracks already listed in bank.json are skipped. Each take is checked with the live
app's own deck analysis and rejected if it is not playable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodj_stable_audio import StableAudio, StableAudioError, prepare_deck  # noqa: E402

SR = 44100
# (slug, production brief, tags matched against the planner's style text, BPM, camelot keys)
SPECS = [
    ("dubstep-piano", "Heavy melodic dubstep with a grand piano hook and growl bass",
     ["dubstep", "piano", "growl", "bass"], 140, ["8A", "10A", "5A"]),
    ("deep-house", "Deep house with warm Rhodes chords, round sub bass and shuffled hi-hats",
     ["house", "deep house", "rhodes", "chords"], 124, ["8A", "9A", "6A"]),
    ("melodic-techno", "Melodic techno with a hypnotic arpeggio, driving kick and rumbling bass",
     ["techno", "melodic", "arpeggio", "driving"], 126, ["7A", "9A", "11A"]),
    ("drum-and-bass", "Liquid drum and bass with atmospheric pads, fast breakbeat and rolling sub",
     ["drum and bass", "dnb", "breakbeat", "pads"], 174, ["8A", "10A", "6A"]),
    ("future-bass", "Future bass with bright detuned supersaw chords, snappy snare and wide bass",
     ["future bass", "supersaw", "bright", "chords"], 150, ["8B", "10B", "11B"]),
    ("trap", "Hard trap beat with deep 808 bass, rolling hi-hats and a dark bell melody",
     ["trap", "808", "hi-hats", "bell"], 140, ["5A", "7A", "8A"]),
    ("trance", "Uplifting trance with a plucked lead, rolling bassline and big breakdown",
     ["trance", "uplifting", "pluck", "lead"], 138, ["8A", "10A", "9B"]),
    ("lofi-piano", "Mellow lo-fi hip hop with soft piano chords, dusty drums and warm bass",
     ["lo-fi", "lofi", "chill", "mellow", "piano"], 85, ["8A", "5A", "10B"]),
]


def load(out: Path) -> dict:
    path = out / "bank.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"tracks": []}


def save(out: Path, bank: dict) -> None:
    tmp = out / "bank.json.tmp"
    tmp.write_text(json.dumps(bank, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out / "bank.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="bank")
    parser.add_argument("--seeds", type=int, default=2, help="variants per style and key")
    parser.add_argument("--limit", type=int, default=0, help="stop after rendering this many (0 = all)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = StableAudio(SR)
    renderer.check()
    bank = load(out)
    have = {t["id"] for t in bank["tracks"]}
    rendered = 0
    for slug, brief, tags, bpm, keys in SPECS:
        for key in keys:
            for variant in range(args.seeds):
                track_id = f"{slug}-{key.lower()}-{variant + 1}"
                if track_id in have:
                    continue
                if args.limit and rendered >= args.limit:
                    save(out, bank)
                    print(f"limit reached; {len(bank['tracks'])} tracks in bank")
                    return
                plan = {"style": brief, "bpm": bpm, "camelot": key}
                for attempt in range(3):
                    seed = 5000 + 97 * len(bank["tracks"]) + 1009 * attempt
                    try:
                        wave = renderer.render(plan, seed, lambda: False)
                        prepare_deck(wave, SR, bpm)  # raises if the take is not playable
                        break
                    except (StableAudioError, ValueError, KeyError, IndexError) as exc:
                        print(f"  {track_id}: attempt {attempt + 1} rejected: {exc}")
                else:
                    print(f"{track_id}: skipped")
                    continue
                filename = f"{track_id}.ogg"
                sf.write(str(out / filename), wave, SR, format="OGG", subtype="VORBIS")
                bank["tracks"].append({"id": track_id, "file": filename, "style": brief, "tags": tags,
                                       "bpm": bpm, "camelot": key, "seed": seed,
                                       "seconds": round(len(wave) / SR, 1)})
                save(out, bank)
                rendered += 1
                print(f"{track_id}: ok ({(out / filename).stat().st_size / 1e6:.1f} MB)")
    save(out, bank)
    total = sum((out / t["file"]).stat().st_size for t in bank["tracks"]) / 1e6
    print(f"done: {len(bank['tracks'])} tracks, {total:.0f} MB")


if __name__ == "__main__":
    main()
