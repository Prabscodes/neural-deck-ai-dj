"""Sampled acoustic piano and controlled dubstep stems over an ACE-Step bed."""
from __future__ import annotations

from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy import signal
import soundfile as sf

SOUNDFONT = Path('/usr/share/sounds/sf2/FluidR3_GM.sf2')


def _vlq(number: int) -> bytes:
    out = [number & 127]
    while number > 127:
        number >>= 7
        out.insert(0, (number & 127) | 128)
    return bytes(out)


def _piano(notes, bpm, sr):
    events = [(0, bytes([0xC0, 0]))]
    for beat, length, pitch, velocity in notes:
        start = round(480 * beat)
        end = round(480 * (beat + length))
        events.extend([(start, bytes([0x90, pitch, velocity])), (end, bytes([0x80, pitch, 0]))])
    events.sort(key=lambda item: item[0])
    tempo = round(60_000_000 / bpm).to_bytes(3, 'big')
    track = bytearray(b'\x00\xff\x51\x03' + tempo)
    last = 0
    for when, event in events:
        track.extend(_vlq(when - last) + event)
        last = when
    track.extend(b'\x00\xff\x2f\x00')
    with TemporaryDirectory(prefix='autodj-piano-') as tmp:
        midi, wav = Path(tmp)/'piano.mid', Path(tmp)/'piano.wav'
        midi.write_bytes(b'MThd' + struct.pack('>IHHH', 6, 0, 1, 480) + b'MTrk' + struct.pack('>I', len(track)) + track)
        subprocess.run(['fluidsynth','-ni','-R','0','-C','0','-g','0.9','-r',str(sr),'-F',str(wav),str(SOUNDFONT),str(midi)],
                       check=True, capture_output=True, timeout=30)
        audio, measured_rate = sf.read(wav, dtype='float32', always_2d=True)
    if measured_rate != sr or audio.shape[1] != 2:
        raise RuntimeError('Unexpected piano renderer audio format')
    return audio


PROFILES = (
    'Deep dub: sparse piano answer, resonant wobble, syncopated two-step',
    'Melodic half-time: rolled piano chords, detuned Reese, layered backbeat',
    'Peak-time: octave piano hook, gated FM bass, kick-and-snare stop/start',
)


def track_profile(seed: int) -> str:
    return PROFILES[(seed-1) % len(PROFILES)]


def _envelope(time, attack, release):
    return np.minimum(time/attack, 1) * np.minimum((time[-1]-time)/release, 1)


def _bass_patch(role, time, hz, cycles):
    """Render three pitched voices, with analytic harmonics or band-limited FM."""
    phase = 2*np.pi*hz*time
    sweep = (.5 + .5*np.cos(2*np.pi*cycles*time))**.9
    if role == 0:
        # Resonant low-pass-style sweep over harmonics of a single root.
        out = np.zeros_like(time)
        center = 2 + 5*sweep
        for harmonic in range(2, 10):
            resonant = np.exp(-.5*((harmonic-center)/1.5)**2)
            out += np.sin(harmonic*phase)*(.48/harmonic+.38*resonant)
        return np.column_stack((out, out))
    if role == 1:
        # Two band-limited saw stacks, detuned to create a moving Reese tone.
        channels = []
        for cents in (-10, 10):
            detuned = phase * 2**(cents/1200)
            out = sum(np.sin(h*detuned)/h for h in range(2, 12))
            channels.append(out*(.78+.22*sweep))
        return np.column_stack(channels)
    # Integer carrier/modulator ratios retain pitch as the modulation changes.
    amount = .5 + 2.5*sweep
    left = np.sin(3*phase + amount*np.sin(2*phase))
    right = np.sin(3*phase + (.8*amount)*np.sin(2*phase+.3))
    return np.column_stack((left, right))*(.3+.7*sweep[:,None])


def _hook(role, bar, chord, root, drop):
    """Four-bar phrases with distinct register, note contour, and phrasing."""
    a,b,c = chord
    if role == 0:
        phrases = [[(.18,12+c,.7,95),(1.5,12+a,.48,84),(3.25,12+b,.37,78)],
                   [(.75,12+b,.8,91),(2.25,c,.62,86)],
                   [(0,12+c,.63,94),(1.25,12+b,.42,80),(3.5,12+a,.32,81)],
                   [(.5,c,.44,88),(1.4,12+a,.62,93),(2.7,b,.44,77)]]
    elif role == 1:
        voicing = (a-12,c,b+12)
        phrases = [[(.02*i,p,2.55-.08*i,79+i*4) for i,p in enumerate(voicing)],
                   [(1+.028*i,p,2.3-.07*i,75+i*3) for i,p in enumerate(voicing[::-1])],
                   [(0,voicing[0],2.2,87),(.42,c,1.5,82),(.47,a+12,1.64,93),(3.2,b,.4,76)],
                   [(.8+.024*i,p,1.86-.1*i,80+i*4) for i,p in enumerate(voicing)]]
    else:
        phrases = [[(0,a,.31,90),(0,a+12,.31,105),(.46,c+12,.23,95),(1.25,b+12,.45,99)],
                   [(.13,b+12,.26,98),(.67,c+12,.28,104),(1.36,a+12,.6,92),(3.5,b,.23,75)],
                   [(0,c,.31,92),(0,c+12,.31,104),(.5,a+12,.4,93),(1.2,b+12,.48,101)],
                   [(.06,b,.28,88),(.59,c+12,.23,105),(1.02,b+12,.26,90),(1.56,a+12,.46,97)]]
    notes = phrases[bar%4]
    if drop and role == 0:
        notes = notes[:1] if bar%2 == 0 else []
    elif drop and role == 1:
        notes = notes if bar%2 else notes[:2]
    elif drop and role == 2 and bar%2:
        notes = notes[:2]
    return [(root+24+pitch, step, length, velocity) for step,pitch,length,velocity in notes]


def arrange(bed: np.ndarray, sr: int, bpm: float, bars: int, root_midi: int,
            minor: bool, seed: int) -> np.ndarray:
    """Use sampled piano, composed note patterns and a restrained ACE-Step texture."""
    rng = np.random.default_rng(seed)
    role = (seed-1)%len(PROFILES)
    seconds_per_beat = 60/bpm
    samples = round(bars*4*seconds_per_beat*sr)
    percussion = np.zeros(samples,dtype=np.float32)
    ducking = np.zeros(samples,dtype=np.float32)
    sub = np.zeros(samples,dtype=np.float32)
    bass = np.zeros((samples,2),dtype=np.float32)
    atmosphere = np.zeros_like(bass)
    # Leave the model's texture audible in the breaks. Keep its noise and
    # unsynchronized transients out of the drop and mix-out bar.
    texture_filter = signal.butter(2,[280,1200],fs=sr,btype='bandpass',output='sos')
    texture = signal.sosfilt(texture_filter,bed[:samples],axis=0)
    notes = []
    scale = (0,2,3,5,7,8,10) if minor else (0,2,4,5,7,9,11)
    progression = ((0,5,0,6),(5,2,6,0),(0,3,5,4))[role]

    def add(target, beat, values):
        offset = round(beat*seconds_per_beat*sr)
        count = min(len(values),len(target)-offset)
        if count > 0:
            target[offset:offset+count] += values[:count]

    def hit(beat, kind, volume):
        duration = {'kick':.18,'snare':.17,'rim':.09,'hat':.055}[kind]
        time = np.arange(round(duration*sr))/sr
        attack = np.minimum(time/.002,1)
        if kind == 'kick':
            hz = (48,54,49)[role]
            pitch = 2*np.pi*(hz*time + 34*(1-np.exp(-time*94))/94)
            sample = .74*np.sin(pitch)*np.exp(-time*(26+role*2))
            sample += .11*np.sin(2*np.pi*1550*time)*np.exp(-time*220)
            add(ducking,beat,.6*volume*np.exp(-time/.068))
        elif kind == 'snare':
            hz = (162,202,186)[role]
            noise = signal.sosfilt(signal.butter(2,[1400,5200],fs=sr,btype='bandpass',output='sos'),
                                   rng.standard_normal(len(time)))
            sample = .41*np.sin(2*np.pi*hz*time)*np.exp(-time*37) + .22*noise*np.exp(-time*44)
            add(ducking,beat,.22*volume*np.exp(-time/.039))
        elif kind == 'rim':
            sample = (.5*np.sin(2*np.pi*540*time)+.19*np.sin(2*np.pi*1320*time))*np.exp(-time*75)
        else:
            noise = signal.sosfilt(signal.butter(2,[6000,8500],fs=sr,btype='bandpass',output='sos'),
                                   rng.standard_normal(len(time)))
            sample = .25*noise*np.exp(-time*103)
        add(percussion,beat,volume*attack*sample)

    patch_filter = signal.butter(2,[125,(1080,850,1750)[role]],fs=sr,btype='bandpass',output='sos')
    for bar in range(bars):
        beat = 4*bar
        position = bar/bars
        mix_out = bars > 1 and bar == bars-1
        intro = position < (.20,.25,.12)[role]
        interlude = .50 <= position < (.63,.69,.61)[role]
        drop = not (intro or interlude or mix_out)
        degree = progression[(bar//2)%4]
        chord = tuple(scale[d%7]+12*(d//7) for d in (degree,degree+2,degree+4))
        if not mix_out:
            for pitch,step,length,velocity in _hook(role,bar,chord,root_midi,drop):
                notes.append((beat+step,length,pitch,velocity+int(rng.integers(-4,5))))
        if intro or interlude:
            start = round(beat*seconds_per_beat*sr)
            end = min(len(texture),round((beat+4)*seconds_per_beat*sr))
            if end > start:
                envelope = np.sin(np.linspace(0,np.pi,end-start))**2
                atmosphere[start:end] = .18*texture[start:end]*envelope[:,None]
        if mix_out:
            # A restrained rhythmic cue rather than a broadband noise sweep.
            for step in ((1.5,3.5),(2.5,3.25,3.75),(2.0,3.0,3.5))[role]:
                hit(beat+step,'rim',.22)
        elif drop:
            kicks = (((0,1.65),(0,3.25)),((0,1.5,3.5),(0,2.75)),((0,.75,2.75),(0,1.25,3.5)))[role]
            for i,step in enumerate(kicks[bar%2]):
                hit(beat+step,'kick',1 if i==0 else .65)
            hit(beat+2,'snare',(.86,.95,1)[role])
            hat_steps = ((.66,1.33,2.67,3.33),(.5,1.5,2.5,3.5),(.5,1.25,1.75,2.5,3.25,3.73))[role]
            for i,step in enumerate(hat_steps):
                hit(beat+step,'hat',.68 if i%2==0 else .36)
            if role == 0 and bar%2:
                hit(beat+3.72,'rim',.24)
        elif bar%2:
            hit(beat+2,'rim',.42)
            hit(beat+3.5,'hat',.72)
        if not drop:
            continue
        rhythms = (
            (((1.1,.82,3),(2.58,1.05,4)),((.6,.91,3),(2.38,1.21,3))),
            (((.2,1.56,1),(2.52,1.06,2)),((.72,1.15,2),(2.45,1.39,1))),
            (((.21,.34,6),(.89,.42,5),(2.43,.62,6),(3.39,.35,9)),
             ((.44,.50,7),(1.18,.58,5),(2.63,.44,8),(3.34,.43,6))),
        )
        hz = 440*2**((root_midi+chord[0]-69)/12)
        while hz > 67:
            hz /= 2
        while hz < 33:
            hz *= 2
        for step,length,cycles in rhythms[role][bar%2]:
            time = np.arange(round(length*seconds_per_beat*sr))/sr
            envelope = _envelope(time,.014,.044)
            patch = _bass_patch(role,time,hz,cycles/seconds_per_beat)
            pitched = signal.sosfilt(patch_filter,patch,axis=0)
            add(bass,beat+step,((.30,.35,.38)[role]*envelope)[:,None]*pitched)
            phase = 2*np.pi*hz*time
            add(sub,beat+step,envelope*(.38*np.sin(phase)+.05*np.sin(2*phase)))
    piano = _piano(notes,bpm,sr)[:samples]
    piano = signal.sosfilt(signal.butter(2,125,fs=sr,btype='highpass',output='sos'),piano,axis=0)
    piano *= (.46,.50,.47)[role]/max(.01,float(np.max(np.abs(piano))))
    piano_layer = np.zeros_like(bass)
    add(piano_layer,0,piano)
    reflection = round(.028*sr)
    reflected = piano[:samples-reflection,::-1]
    piano_layer[reflection:reflection+len(reflected)] += .14*reflected
    piano_env = signal.lfilter([.002],[1,-.998],np.max(np.abs(piano_layer),axis=1))
    bass *= (1-np.clip(piano_env*2.8,0,.42))[:,None]
    gain = (1-np.clip(ducking,0,.72))[:,None]
    mix = atmosphere + piano_layer + gain*(bass+sub[:,None]) + percussion[:,None]*.79
    peak = float(np.max(np.abs(mix)))
    if peak > .94:
        mix *= .94/peak
    fade = round(.024*sr)
    mix[-fade:] *= np.linspace(1,0,fade)[:,None]
    if not np.isfinite(mix).all():
        raise RuntimeError('Arranged audio contains non-finite samples')
    return np.asarray(mix,dtype=np.float32)
