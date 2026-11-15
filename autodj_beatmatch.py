"""Conservative audio-measured beat matching for the live DJ queue.

The detector estimates a periodic beat grid from onset strength; a requested BPM
is only an octave hint, never the measured result. It abstains when confidence,
tempo ratio, or length is inadequate. Downbeat/phrase detection is not claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import subprocess
import numpy as np
from scipy import signal

@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    phase: float
    confidence: float
    period: float


def estimate_grid(wave: np.ndarray, sr: int, bpm_hint: float | None = None, *, tempo_window: float | None = None) -> BeatGrid | None:
    if wave.ndim != 2 or wave.shape[1] != 2 or len(wave) < 4*sr or not np.isfinite(wave).all():
        return None
    # Work at 11.025 kHz: separate bass/percussion changes from broad piano sustain.
    mono=wave.mean(axis=1).astype(np.float32)
    g=math.gcd(sr,11025); mono=signal.resample_poly(mono,11025//g,sr//g)
    rate=11025; hop=110; size=512
    _,_,z=signal.stft(mono,fs=rate,nperseg=size,noverlap=size-hop,boundary=None,padded=False)
    mag=np.log1p(30*np.abs(z)); df=np.maximum(np.diff(mag,axis=1,prepend=mag[:,:1]),0)
    freqs=np.fft.rfftfreq(size,1/rate)
    flux=(df[(freqs>=35)&(freqs<220)].mean(axis=0)*1.4 + df[(freqs>=220)&(freqs<7000)].mean(axis=0))
    flux=np.maximum(flux-signal.medfilt(flux, kernel_size=21),0)
    if not np.isfinite(flux).all() or float(np.std(flux))<1e-5:return None
    flux=(flux-float(np.mean(flux)))/(float(np.std(flux))+1e-9)
    corr=signal.correlate(flux,flux,mode='full',method='fft')[len(flux)-1:]
    maxlag=min(len(corr)-1,int(round(60*rate/(65*hop))))
    minlag=max(1,int(round(60*rate/(190*hop))))
    candidates=[]
    for lag in range(minlag+1,maxlag):
        if corr[lag]>=corr[lag-1] and corr[lag]>=corr[lag+1]:
            bpm=60*rate/(hop*lag)
            if tempo_window is not None and bpm_hint:
                if not any(65 <= target <= 190 and abs(target / bpm_hint - 1) <= tempo_window
                           for target in (bpm / 2, bpm, bpm * 2)):
                    continue
            # Restrict to plausible tempo from the audio, not arbitrary prompt tempo.
            prior=1.0
            if bpm_hint and 60<=bpm_hint<=190:
                distance=min(abs(math.log2(bpm/bpm_hint)),abs(math.log2(2*bpm/bpm_hint)),abs(math.log2(bpm/(2*bpm_hint))))
                prior=math.exp(-0.30*distance)
            candidates.append((float(corr[lag]/(corr[0]+1e-9))*prior,lag,bpm))
    if not candidates:return None
    strength,lag,bpm=max(candidates)
    if strength<0.08:return None
    # Sub-frame interpolation reduces phase drift across longer clips.
    y0,y1,y2=map(float,(corr[lag-1],corr[lag],corr[lag+1]))
    den=y0-2*y1+y2
    refined=lag + (0.5*(y0-y2)/den if abs(den)>1e-9 else 0.0)
    bpm=60*rate/(hop*refined)
    # Resolve common 1/2x tempo ambiguity using the requested tempo, while retaining measured period.
    if bpm_hint:
        possible=[x for x in (bpm/2,bpm,bpm*2) if 65<=x<=190]
        bpm=min(possible,key=lambda x:abs(math.log(x/bpm_hint)))
    period=60/bpm
    hopsec=hop/rate
    lagf=period/hopsec
    phases=np.arange(max(1,round(lagf)))
    positive=np.maximum(flux,0)
    scores=[]
    for p in phases:
        idx=np.rint(np.arange(p,len(positive),lagf)).astype(int)
        idx=idx[idx<len(positive)]
        scores.append(float(np.mean(positive[idx])) if len(idx) else 0.0)
    phase_idx=int(np.argmax(scores)); peak=float(scores[phase_idx]); median=float(np.median(scores))
    prominence=(peak-median)/(peak+1e-9)
    confidence=float(np.clip(min(strength/0.25,1)*min(prominence/0.32,1),0,1))
    if confidence<0.30 or duration_beats(len(wave),sr,period)<6:return None
    # scipy STFT times refer to window centres, not starts.
    phase=((phase_idx*hop + size/2)/rate)%period
    return BeatGrid(float(bpm),phase,confidence,period)


def duration_beats(samples:int,sr:int,period:float)->float:
    return samples/(sr*period)


def stretch_audio(wave:np.ndarray,sr:int,tempo_ratio:float)->np.ndarray:
    """Change tempo while preserving pitch; fail rather than silently resample pitch."""
    if abs(tempo_ratio-1)<0.003:return wave.copy()
    if not 0.92<=tempo_ratio<=1.08:raise ValueError('Tempo difference too large')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','f32le','-ar',str(sr),'-ac','2','-i','pipe:0',
         '-af',f'rubberband=tempo={tempo_ratio:.8f}','-f','f32le','-acodec','pcm_f32le','pipe:1']
    p=subprocess.run(cmd,input=np.ascontiguousarray(wave,dtype='<f4').tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,check=True)
    out=np.frombuffer(p.stdout,dtype='<f4').reshape(-1,2).copy()
    if len(out)<sr or not np.isfinite(out).all():raise ValueError('Invalid pitch-preserving stretch output')
    return out


def align_incoming(tail:np.ndarray, incoming:np.ndarray, sr:int, outgoing:BeatGrid | None,
                   outgoing_tail_origin:int, incoming_grid:BeatGrid | None, overlap:int, *, analysis_seconds:int | None = None):
    """Return adjusted incoming wave and truthful measured transition diagnostics."""
    info={'beat_matched':False,'reason':'beat confidence too low','outgoing_bpm':None,'incoming_bpm':None,'phase_error_ms':None}
    if outgoing is None or incoming_grid is None or overlap<=0:return incoming,info
    info.update(outgoing_bpm=round(outgoing.bpm,2),incoming_bpm=round(incoming_grid.bpm,2))
    if min(outgoing.confidence,incoming_grid.confidence)<0.3:return incoming,info
    ratio=outgoing.bpm/incoming_grid.bpm
    if not 0.92<=ratio<=1.08:
        info['reason']='measured tempo difference exceeds safe stretch range';return incoming,info
    try:
        adjusted=stretch_audio(incoming,sr,ratio)
        # Re-measure after stretching instead of assuming DSP has no delay.
        analysis = adjusted[:analysis_seconds * sr] if analysis_seconds is not None else adjusted
        current=estimate_grid(analysis,sr,outgoing.bpm,
                              tempo_window=.06 if analysis_seconds is not None else None)
        if current is None or abs(current.bpm/outgoing.bpm-1)>0.035:
            info['reason']='could not verify stretched beat grid';return incoming,info
        start=(outgoing_tail_origin+len(tail)-overlap)/sr
        target_phase=(outgoing.phase-start)%outgoing.period
        trim=((current.phase-target_phase)%outgoing.period)
        trim_samples=round(trim*sr)
        if trim_samples>=len(adjusted)-sr:
            info['reason']='insufficient incoming audio after alignment';return incoming,info
        adjusted=adjusted[trim_samples:]
        actual=(current.phase-trim_samples/sr-target_phase)%outgoing.period
        err=min(actual,outgoing.period-actual)
        info.update(beat_matched=True,reason='estimated grids aligned',phase_error_ms=round(err*1000,2),
                    outgoing_confidence=round(outgoing.confidence,3),incoming_confidence=round(current.confidence,3),
                    tempo_ratio=round(ratio,5),trim_seconds=round(trim_samples/sr,4),stretched_bpm=round(current.bpm,2))
        return adjusted,info
    except (OSError,ValueError,subprocess.SubprocessError) as e:
        info['reason']='pitch-preserving stretch unavailable: '+str(e)[:120]
        return incoming,info


def trim_trailing_silence(wave:np.ndarray,sr:int)->tuple[np.ndarray,float]:
    """Drop long digital-silence tails, preserving a short natural decay."""
    if len(wave)<2*sr:return wave,0.0
    block=max(1,round(sr*.05));usable=len(wave)//block
    if not usable:return wave,0.0
    blocks=wave[:usable*block].reshape(usable,block,2)
    levels=np.sqrt(np.mean(np.square(blocks,dtype=np.float64),axis=(1,2)))
    active=np.flatnonzero(levels>0.001)
    if not len(active):return wave,0.0
    end=min(len(wave),int((active[-1]+1)*block+0.15*sr))
    if len(wave)-end<sr:return wave,0.0
    return wave[:end],(len(wave)-end)/sr


def bass_swap_crossfade(tail: np.ndarray, head: np.ndarray, overlap: int, sr: int, kind: str) -> np.ndarray:
    """Crossfade the high bands while passing the low end from one deck to the other."""
    t = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
    lowpass = signal.butter(2, 150, fs=sr, output='sos')
    old = tail[-overlap:]
    new = head[:overlap]
    old_low = signal.sosfiltfilt(lowpass, tail, axis=0)[-overlap:]
    new_low = signal.sosfiltfilt(lowpass, head, axis=0)[:overlap]
    bass_in = np.clip((t - .43) / .14, 0, 1)
    bass_in = bass_in * bass_in * (3 - 2 * bass_in)
    if kind == 'equal_power':
        old_gain, new_gain = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    else:
        old_gain, new_gain = 1 - t, t
    cross = (old - old_low) * old_gain + (new - new_low) * new_gain
    cross += old_low * (1 - bass_in) + new_low * bass_in
    # Attenuate only overload, smoothly returning to the original gain at either edge.
    peak = np.max(np.abs(cross), axis=1, keepdims=True)
    gain = np.maximum(peak / .995, 1)
    return np.asarray(cross / gain, dtype=np.float32)
