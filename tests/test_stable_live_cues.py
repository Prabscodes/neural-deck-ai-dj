import numpy as np
import pytest

from autodj_stable_audio import StableAudioError, prepare_deck


def test_live_cue_preserves_opening_but_skips_beatless_mix_in():
    rate = 8_000
    audio = np.zeros(75 * rate, dtype=np.float32)
    time = np.arange(69 * rate) / rate
    audio[:len(time)] = .04 * np.sin(2 * np.pi * 440 * time)
    pulse_time = np.arange(rate // 5) / rate
    kick = .65 * np.sin(2 * np.pi * 65 * pulse_time) * np.exp(-16 * pulse_time)
    for onset in np.arange(10, 68, 60/140):
        start = round(onset * rate)
        audio[start:start + len(kick)] += kick
    stereo = np.column_stack((audio, audio))
    opening = prepare_deck(stereo, rate, 140)
    assert opening.cue_seconds == 0
    incoming = prepare_deck(stereo, rate, 140, incoming=True)
    assert 6 <= incoming.cue_seconds <= 12
    assert incoming.entry_has_kick
    assert incoming.opening and abs(incoming.opening.bpm - 140) < 1
    assert incoming.closing and abs(incoming.closing.bpm - 140) < 1
    assert np.abs(incoming.wave).max() <= .761
    assert incoming.trimmed > 5
    with pytest.raises(StableAudioError):
        prepare_deck(np.zeros_like(stereo), rate, 140)


def test_listening_prefers_half_time_fit_to_a_better_signal_grid(monkeypatch):
    import io
    import json
    from types import SimpleNamespace

    import soundfile as sf
    from autodj_live_agent import choose_listening_take

    sr = 8000
    decks = [SimpleNamespace(wave=np.full((52 * sr, 2), level, dtype=np.float32), score=score)
             for level, score in ((.08, 4), (.16, 2))]

    def listen(model, system, context, *, audio_wav, schema, timeout):
        assert model == "gemini-4-hydrogen-low"
        audio, rate = sf.read(io.BytesIO(audio_wav))
        assert rate == 24000 and audio.shape == (20 * rate, 2)
        assert np.mean(audio[rate:3 * rate]) < np.mean(audio[11 * rate:13 * rate])
        return json.dumps({"takes": [
            {"genre": 3, "groove": 3, "instrument": 10, "reason": "piano house"},
            {"genre": 7, "groove": 8, "instrument": 6, "reason": "half-time drop with piano"},
        ]})

    monkeypatch.setattr("autodj_live_agent.generate_text", listen)
    chosen, review = choose_listening_take(
        decks, {"style": "dubstep and piano"}, "heavy dubstep", "gemini-4-hydrogen-low", sr)
    assert chosen == 1
    assert review["state"] == "selected"


def test_outgoing_phrase_and_rendered_handoff_drive_selection(monkeypatch):
    import io
    import json
    from types import SimpleNamespace

    import soundfile as sf
    from autodj_live_agent import choose_listening_take, prepare_handoff

    sr = 8000
    outgoing = np.full((8 * sr, 2), .11, dtype=np.float32)
    tail = outgoing[-3 * sr:]
    decks = [SimpleNamespace(wave=np.full((52 * sr, 2), level, dtype=np.float32),
                             score=score, opening=None, entry_has_kick=False)
             for level, score in ((.03, 3), (.17, 1))]
    plan = {"style": "dubstep and piano", "transition": "equal_power", "overlap_seconds": 2}
    handoffs = [prepare_handoff(deck, tail, None, 0, plan, sr) for deck in decks]

    def listen(model, system, context, *, audio_wav, schema, timeout):
        audio, rate = sf.read(io.BytesIO(audio_wav))
        assert rate == 22050 and len(audio) == round(32.5 * rate)
        assert np.allclose(audio[rate:7*rate], .11, atol=.0002)
        for start, deck, handoff in zip((8.5, 20.5), decks, handoffs):
            live_join = handoff.join(tail, deck.wave)
            payload = audio[round((start+3.2)*rate):round((start+3.6)*rate)]
            assert np.allclose(payload, live_join[round(3.4*sr)], atol=.0002)
        assert "continuity" in schema["properties"]["takes"]["items"]["required"]
        return json.dumps({"takes": [
            {"genre": 9, "groove": 9, "instrument": 9, "continuity": 2, "reason": "momentum drops"},
            {"genre": 7, "groove": 7, "instrument": 7, "continuity": 9, "reason": "kick and bass carry on"},
        ]})

    monkeypatch.setattr("autodj_live_agent.generate_text", listen)
    chosen, review = choose_listening_take(
        decks, plan, "rising dubstep with piano", "gemini-4-hydrogen-low", sr,
        outgoing=outgoing, handoffs=handoffs)
    assert chosen == 1
    assert review["heard_outgoing_phrase"]
