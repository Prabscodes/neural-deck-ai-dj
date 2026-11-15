import json

import numpy as np
import pytest
import soundfile as sf

from autodj_bank import BankAudio, _key_distance
from autodj_stable_audio import StableAudioError

TRACKS = [
    ("dub-8a", ["dubstep", "piano"], 140, "8A"),
    ("dub-10a", ["dubstep", "piano"], 140, "10A"),
    ("dub-fast", ["dubstep", "piano"], 175, "8A"),
    ("house", ["house", "rhodes"], 124, "8A"),
    ("trap", ["trap", "808"], 140, "5A"),
]


def make_bank(path, tracks=TRACKS, sr=8000, channels=2, fmt="FLAC"):
    entries = []
    for i, (track_id, tags, bpm, key) in enumerate(tracks):
        name = f"{track_id}.flac"
        t = np.arange(sr) / sr
        wave = (np.sin(2 * np.pi * (200 + 30 * i) * t) * .2).astype("float32")
        sf.write(str(path / name), np.tile(wave[:, None], (1, channels)), sr, format=fmt)
        entries.append({"id": track_id, "file": name, "tags": tags, "bpm": bpm, "camelot": key})
    (path / "bank.json").write_text(json.dumps({"tracks": entries}))
    return path


def plan(style="heavy dubstep with grand piano", bpm=140, camelot="8A"):
    return {"style": style, "bpm": bpm, "camelot": camelot}


def never():
    return False


def test_best_match_by_style_tempo_and_key(tmp_path):
    bank = BankAudio(8000, str(make_bank(tmp_path)))
    ranked = [e["id"] for e in bank.rank(plan())]
    assert ranked[0] == "dub-8a"
    assert ranked.index("dub-10a") < ranked.index("house")
    assert ranked.index("dub-8a") < ranked.index("dub-fast")  # tempo closeness matters
    assert [e["id"] for e in bank.rank(plan("warm deep house with rhodes chords", 124))][0] == "house"


def test_three_takes_in_one_slot_differ(tmp_path):
    bank = BankAudio(8000, str(make_bank(tmp_path)))
    picks = []
    for attempt in range(3):
        bank._recent.clear()  # isolate the seed rule from the recency rule
        bank.render(plan(), 84 + 1009 * attempt, never)
        picks.append(bank._recent[-1])
    assert len(set(picks)) == 3


def test_recent_tracks_are_penalized(tmp_path):
    bank = BankAudio(8000, str(make_bank(tmp_path)))
    first = bank.rank(plan())[0]["id"]
    bank.render(plan(), 0, never)
    assert bank.rank(plan())[0]["id"] != first


def test_output_is_stereo_float_at_requested_rate(tmp_path):
    for channels in (1, 2):
        d = tmp_path / str(channels); d.mkdir()
        wave = BankAudio(16000, str(make_bank(d, channels=channels))).render(plan(), 0, never)
        assert wave.dtype == np.float32 and wave.shape == (16000, 2)


def test_stopped_check_and_empty_bank(tmp_path):
    bank = BankAudio(8000, str(make_bank(tmp_path)))
    with pytest.raises(StableAudioError, match="stopped"):
        bank.render(plan(), 0, lambda: True)
    empty = tmp_path / "empty"; empty.mkdir()
    (empty / "bank.json").write_text(json.dumps({"tracks": []}))
    with pytest.raises(StableAudioError, match="no playable"):
        BankAudio(8000, str(empty)).check()
    with pytest.raises(StableAudioError, match="missing"):
        BankAudio(8000, str(tmp_path / "nope"))


def test_malformed_and_path_escaping_entries_are_skipped(tmp_path):
    make_bank(tmp_path)
    data = json.loads((tmp_path / "bank.json").read_text())
    good = data["tracks"][0]
    data["tracks"] += [
        {**good, "id": "evil", "file": "../secret.flac"},
        {**good, "id": "abs", "file": "/etc/passwd"},
        {**good, "id": "badkey", "camelot": "13A"},
        {**good, "id": "nofile", "file": "missing.flac"},
        {**good},  # duplicate id
        {"id": "short"},
    ]
    (tmp_path / "bank.json").write_text(json.dumps(data))
    ids = [e["id"] for e in BankAudio(8000, str(tmp_path)).entries]
    assert ids == [t[0] for t in TRACKS]


def test_camelot_distance():
    assert _key_distance("8A", "8A") == 0
    assert _key_distance("8A", "9A") == 1 and _key_distance("12A", "1A") == 1
    assert _key_distance("8A", "8B") == 1
    assert _key_distance("8A", "10A") is None and _key_distance("8A", "9B") is None
