import json, threading, time
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from autodj_agent import AgentError
from autodj_live_agent import LiveDJ, mix_transition, validate_live_plan

BASE={'title':'Opening','style':'soft instrumental','bpm':100,'camelot':'8A','bars':8,'backend':'synth','transition':'linear','overlap_seconds':1}

class Producer:
    def __init__(self,sr=8000): self.sr=sr
    def produce_track(self,**kw):
        t=np.arange(self.sr*4)/self.sr
        sig=np.sin(2*np.pi*220*t).astype('float32')*.15
        sf.write(kw['output_path'],np.column_stack((sig,sig)),self.sr)
        return SimpleNamespace(metadata={}, backend_used='Procedural test synth')

def wait_for(fn, timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=fn()
        if value:return value
        time.sleep(.01)
    raise AssertionError('Timed out')

def test_transition_shapes_and_validation():
    a=np.ones((20,2),dtype='float32')*.25;b=np.ones((20,2),dtype='float32')*.5
    cut,n=mix_transition(a,b,'cut',5);assert len(cut)==40 and n==0
    mix,n=mix_transition(a,b,'linear',5);assert len(mix)==35 and n==5
    assert np.allclose(mix[15],a[0]) and np.allclose(mix[19],b[0])
    equal,n=mix_transition(a,b,'equal_power',5);assert len(equal)==35 and n==5
    with pytest.raises(AgentError):validate_live_plan({**BASE,'transition':'shell'},False)
    with pytest.raises(AgentError):validate_live_plan({**BASE,'backend':'neural'},False)
    with pytest.raises(AgentError):validate_live_plan({**BASE,'bars':400},False)

def test_stream_queue_steering_backpressure_stop_and_audit(tmp_path):
    received=[]; records=[]; second_started=threading.Event();continue_second=threading.Event()
    def planner(direction,previous,neural,model):
        received.append((direction,previous))
        if len(received)==2:second_started.set();assert continue_second.wait(3)
        return {**BASE,'title':f'Track {len(received)}','transition':'equal_power'}
    gate=threading.Lock(); dj=LiveDJ(str(tmp_path),Producer(),lambda *e:records.append(e),gate,planner=planner,sample_rate=8000)
    sid=dj.start('soft start','test',False)['session_id']
    assert second_started.wait(3)
    initial=wait_for(lambda: dj.status(sid)['segments'])
    assert initial[0]['seq']==0 and initial[0]['transition']=='opening'
    dj.steer(sid,'more energetic');continue_second.set()
    segments=wait_for(lambda: (x if len(x:=dj.status(sid)['segments'])==3 else None))
    assert [s['seq'] for s in segments]==[0,1,2]
    assert segments[1]['transition']=='equal_power' and segments[1]['overlap_seconds']==1
    assert segments[2]['direction_revision']==1
    assert received[2][0]=='more energetic' and received[2][1]['title']=='Track 2'
    with pytest.raises(AgentError):dj.ack(sid,1)
    assert dj.segment_path(sid,0).is_file()
    dj.ack(sid,0)
    assert not (tmp_path/sid/'segment_0.wav').exists()
    wait_for(lambda:len(dj.status(sid)['segments'])==3)
    dj.stop(sid);wait_for(lambda:dj.status(sid)['state']=='stopped')
    assert not gate.locked()
    assert not list((tmp_path/sid).glob('segment_*.wav'))
    assert any(x[1:3]==('steer','accepted') for x in records)

def test_stop_during_planning_prevents_generation(tmp_path):
    started=threading.Event();release=threading.Event();gate=threading.Lock()
    def planner(*args):started.set();assert release.wait(3);return BASE
    class NoProduce:
        def produce_track(self,**kw):raise AssertionError('should not generate')
    dj=LiveDJ(str(tmp_path),NoProduce(),lambda *x:None,gate,planner=planner,sample_rate=8000)
    sid=dj.start('music','model',False)['session_id'];assert started.wait(3)
    dj.stop(sid)
    with pytest.raises(AgentError):dj.start('another','model',False)
    release.set();wait_for(lambda:dj.status(sid)['state']=='stopped');assert not gate.locked()

def test_http_stream_routes_and_shared_exclusion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import torch
    monkeypatch.setenv('AUTODJ_AGENT_DIR',str(tmp_path/'live-runs'))
    monkeypatch.setenv('AUTODJ_GEMINI_TRANSPORT','api')
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    from autodj_server import create_app
    app=create_app(cache_dir=str(tmp_path/'studio'),enable_legacy_studio=True)
    dj=app.state.llm_live_dj;dj.producer=Producer(sr=44100);dj.planner=lambda *args:BASE
    client=TestClient(app)
    response=client.post('/api/dj-live/start',json={'request':'mellow music'})
    assert response.status_code==200
    sid=response.json()['session_id']
    segment=wait_for(lambda: (v[0] if (v:=client.get(f'/api/dj-live/{sid}/status').json()['segments']) else None))
    audio=client.get(segment['audio_url']);assert audio.status_code==200 and audio.headers['content-type'].startswith('audio/wav')
    assert client.post('/api/produce',json={'title':'conflict'}).status_code==409
    assert client.post(f'/api/dj-live/{sid}/steer',json={'request':'higher energy'}).status_code==200
    assert client.post(f'/api/dj-live/{sid}/ack',json={'seq':segment['seq']}).status_code==200
    assert client.get(segment['audio_url']).status_code==404
    assert client.post(f'/api/dj-live/{sid}/stop',json={}).status_code==200
    wait_for(lambda: client.get(f'/api/dj-live/{sid}/status').json()['state']=='stopped')

def test_gemini_planner_retries_invalid_response_and_sends_wav(monkeypatch):
    import base64
    import io
    from autodj_gemini import DEFAULT_GEMINI_MODEL
    from autodj_live_agent import plan_next
    sent = []
    replies = iter([{**BASE, "camelot": "15Z"}, BASE])
    audio = io.BytesIO()
    sf.write(audio, np.zeros((8000, 2)), 8000, format="WAV", subtype="PCM_16")
    wav = audio.getvalue()
    monkeypatch.setenv("AUTODJ_GEMINI_TRANSPORT", "api")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(req, timeout):
        assert req.full_url.endswith(f"/{DEFAULT_GEMINI_MODEL}:generateContent")
        assert req.get_header("X-goog-api-key") == "test-key"
        sent.append(json.loads(req.data))
        response = {"candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"text": json.dumps(next(replies))}]},
        }]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("autodj_gemini.urlopen", fake_urlopen)
    assert plan_next("soft", BASE, False, DEFAULT_GEMINI_MODEL, audio_wav=wav) == BASE
    assert len(sent) == 2
    schema = sent[0]["generationConfig"]["responseJsonSchema"]
    assert schema["properties"]["camelot"]["enum"][-1] == "12B"
    parts = sent[1]["contents"][0]["parts"]
    assert "plan_validation_error" in json.loads(parts[0]["text"])
    assert parts[1]["inlineData"]["mimeType"] == "audio/wav"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == wav


def test_live_planner_sends_previous_generated_track(tmp_path, monkeypatch):
    from autodj_gemini import DEFAULT_GEMINI_MODEL

    seen = []
    records = []
    third_plan = threading.Event()
    finish = threading.Event()

    def fake_api(model, system, message, *, audio_wav, schema):
        assert model == DEFAULT_GEMINI_MODEL
        seen.append(audio_wav)
        if len(seen) == 3:
            third_plan.set()
            assert finish.wait(3)
        return json.dumps(BASE)

    monkeypatch.delenv("AUTODJ_MUSIC_BACKEND", raising=False)
    monkeypatch.setattr("autodj_live_agent.generate_text", fake_api)
    dj = LiveDJ(str(tmp_path), Producer(), lambda *args: records.append(args),
                threading.Lock(), sample_rate=8000)
    sid = dj.start("mellow", DEFAULT_GEMINI_MODEL, False)["session_id"]
    try:
        assert third_plan.wait(3)
        assert seen[0] is None
        import io
        waveform, sr = sf.read(io.BytesIO(seen[1]), always_2d=True)
        assert sr == 8000 and waveform.shape[1] == 2
        assert np.any(waveform != 0)
        assert [s["seq"] for s in dj.status(sid)["segments"]] == [0, 1]
        assert any(r[1] == "live_plan" and r[2] == "started"
                   and r[3]["has_audio_context"] for r in records)
    finally:
        dj.stop(sid)
        finish.set()
        wait_for(lambda: dj.status(sid)["state"] in ("stopped","failed"))


def test_cli_hydrogen_reads_only_previous_wav(monkeypatch):
    import io
    import subprocess
    from pathlib import Path
    from autodj_gemini import GeminiError, configured_model, generate_text, require_planner

    monkeypatch.delenv("AUTODJ_GEMINI_TRANSPORT", raising=False)
    monkeypatch.delenv("AUTODJ_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("autodj_gemini.shutil.which", lambda _: "/test/agy")
    audio = io.BytesIO()
    sf.write(audio, np.zeros((8000, 2)), 8000, format="WAV", subtype="PCM_16")
    wav = audio.getvalue()
    sandbox_paths = []
    tool_filename = "previous.wav"

    def fake_cli(args, *, cwd, stdin, capture_output, text, timeout):
        assert args[:3] == ["/test/agy", "--model", "gemini-4-hydrogen-low"]
        assert "--sandbox" in args and "--mode" in args
        folder = Path(cwd)
        assert list(folder.iterdir()) == [folder/"previous.wav"]
        assert (folder/"previous.wav").read_bytes() == wav
        sandbox_paths.append(folder)
        events = [
            {"event": "step_update", "step_update": {
                "state": "DONE", "tool_name": "read_file",
                "tool_info": {"parameters": {"Path": str(folder/tool_filename)}},
            }},
            {"event": "result", "result": {"status": "SUCCESS", "response": json.dumps(BASE)}},
        ]
        return subprocess.CompletedProcess(args, 0, "\n".join(map(json.dumps, events)), "")

    monkeypatch.setattr("autodj_gemini.subprocess.run", fake_cli)
    require_planner()
    model = configured_model()
    assert json.loads(generate_text(model, "Plan a DJ track.", "dubstep", audio_wav=wav)) == BASE
    assert not sandbox_paths[-1].exists()

    tool_filename = "../unexpected.wav"
    with pytest.raises(GeminiError, match="unexpected file"):
        generate_text(model, "Plan a DJ track.", "dubstep", audio_wav=wav)
