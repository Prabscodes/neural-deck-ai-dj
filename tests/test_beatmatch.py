import numpy as np
from scipy.signal import find_peaks
from autodj_beatmatch import estimate_grid, align_incoming, trim_trailing_silence

SR=22050
def clicks(bpm,phase,seconds=16):
    out=np.zeros((SR*seconds,2),dtype=np.float32)
    for t in np.arange(phase,seconds-0.01,60/bpm):
        i=int(t*SR); x=np.arange(200)/SR
        out[i:i+200]=(.7*np.sin(2*np.pi*100*x)*np.exp(-x*100))[:,None]
    return out

def test_measured_grid_and_silence_abstention():
    a=clicks(140,.12);g=estimate_grid(a,SR,140)
    assert g and abs(g.bpm-140)<1 and abs(g.phase-.12)<.03
    assert estimate_grid(np.zeros_like(a),SR,140) is None
    assert estimate_grid(a[:SR],SR,140) is None

def test_measured_alignment_of_actual_clicks():
    a=clicks(140,.12); b=clicks(136,.27);ga=estimate_grid(a,SR,140);gb=estimate_grid(b,SR,140)
    tail=a[-3*SR:];adjusted,meta=align_incoming(tail,b,SR,ga,len(a)-len(tail),gb,2*SR)
    assert meta['beat_matched'] and 1.01<meta['tempo_ratio']<1.04
    def peaks(x):return find_peaks(np.abs(x[:,0]),height=.1,distance=int(SR*.3))[0]/SR
    old=peaks(tail[-2*SR:]);new=peaks(adjusted[:2*SR])
    assert len(old)>=4 and len(new)>=4
    assert max(min(abs(new-t)) for t in old)*1000<20

def test_distant_tempos_fall_back_without_falsely_claiming_match():
    a=clicks(140,.12);b=clicks(120,.08);ga=estimate_grid(a,SR,140);gb=estimate_grid(b,SR,140)
    adjusted,meta=align_incoming(a[-3*SR:],b,SR,ga,len(a)-3*SR,gb,2*SR)
    assert not meta['beat_matched'] and 'difference' in meta['reason']
    assert np.array_equal(adjusted,b)

def test_only_long_silent_tails_trimmed():
    a=clicks(140,.12);a[-2*SR:]=0
    adjusted,trim=trim_trailing_silence(a,SR)
    assert trim>1 and len(adjusted)>10*SR
    short=np.concatenate([clicks(140,.12),np.zeros((SR//2,2),dtype=np.float32)])
    assert trim_trailing_silence(short,SR)[1]==0


def test_tempo_policy_respects_the_listener():
    from autodj_live_agent import constrain_live_tempo
    g=estimate_grid(clicks(140,.12),SR,140)
    plan={'bpm':128}
    assert constrain_live_tempo(plan,'play dubstep with piano',None,None)['bpm']==140
    assert constrain_live_tempo(plan,'play dubstep at 128 bpm',None,None)['bpm']==128
    assert constrain_live_tempo(plan,'continue this vibe',{'bpm':140},g)['bpm']==140
    assert constrain_live_tempo(plan,'make it slower',{'bpm':140},g)['bpm']==128
