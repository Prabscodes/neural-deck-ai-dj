import json
import pytest
from autodj_agent import AgentError, AgentRunner, validate_plan

PLAN = {'tracks': [
    {'title':'A','style':'soft instrumental','bpm':100,'camelot':'8A','bars':2,'backend':'synth'},
    {'title':'B','style':'upbeat instrumental','bpm':120,'camelot':'9A','bars':2,'backend':'synth'},
]}

def test_rejects_unsupported_actions_and_unavailable_neural():
    with pytest.raises(AgentError): validate_plan({'tracks': PLAN['tracks'], 'command':'rm -rf /'}, False)
    with pytest.raises(AgentError): validate_plan({'tracks':[dict(PLAN['tracks'][0],backend='neural'), PLAN['tracks'][1]]}, False)
    with pytest.raises(AgentError): validate_plan({'tracks':[dict(PLAN['tracks'][0],bars=500), PLAN['tracks'][1]]}, False)

def test_real_procedural_audio_and_audit(tmp_path):
    from autodj_producer import ProducerAgent
    import soundfile as sf
    runner=AgentRunner(str(tmp_path), ProducerAgent(cache_dir=str(tmp_path)))
    result=runner.run('soft then energetic','test',False,planner=lambda *args: PLAN)
    mix=tmp_path / (result['run_id']+'_mix.wav')
    audio=sf.info(str(mix))
    assert audio.channels == 2 and audio.samplerate == 44100 and audio.duration > 0
    assert all('Procedural' in x['backend'] for x in result['outputs'])
    events=[json.loads(line) for line in (tmp_path/'actions.jsonl').read_text().splitlines()]
    assert [(e['action'],e['status']) for e in events] == [('plan','started'),('plan','completed'),('generate','started'),('generate','completed'),('generate','started'),('generate','completed'),('mix','completed')]

def test_failed_plan_is_logged(tmp_path):
    runner=AgentRunner(str(tmp_path),None)
    def fail(*args): raise AgentError('model offline')
    with pytest.raises(AgentError): runner.run('music','test',False,planner=fail)
    events=[json.loads(line) for line in (tmp_path/'actions.jsonl').read_text().splitlines()]
    assert events[-1]['status']=='failed' and 'model offline' in events[-1]['detail']['error']
