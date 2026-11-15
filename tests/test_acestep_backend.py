import threading, time
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from autodj_acestep_inference import _caption
from autodj_live_agent import LiveDJ


def test_prompt_keeps_request_and_adds_specific_bass():
    assert 'cello' in _caption('quiet cello ambient')
    result = _caption('dubstep with piano')
    assert 'piano' in result.lower() and 'wobble' in result.lower()


def test_explicit_backend_never_uses_synth(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTODJ_MUSIC_BACKEND', 'ace_step')
    calls=[]; paused=threading.Event(); release=threading.Event()
    class Producer:
        def produce_track(self, **kw):
            calls.append(kw)
            sf.write(kw['output_path'],np.zeros((44100*4,2),dtype='float32'),44100)
            return SimpleNamespace(metadata={'neural_gpu_active':True},backend_used='fake ACE')
    def planner(*args):
        if calls: paused.set();release.wait(5)
        return {'title':'test','style':'dubstep with piano','bpm':140,'camelot':'8A','bars':8,'backend':'synth','transition':'cut','overlap_seconds':1}
    dj=LiveDJ(str(tmp_path),Producer(),lambda *x:None,threading.Lock(),planner)
    sid=dj.start('dubstep with piano','test',True)['session_id']
    assert paused.wait(5)
    assert calls[0]['use_neural_gpu'] is True
    assert dj.status(sid)['segments'][0]['backend']=='fake ACE'
    dj.stop(sid);release.set()
    deadline=time.monotonic()+5
    while dj.status(sid)['state']!='stopped' and time.monotonic()<deadline:time.sleep(.01)
    assert dj.status(sid)['state']=='stopped'
