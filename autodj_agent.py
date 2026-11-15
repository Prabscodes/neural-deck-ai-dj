"""Validated LLM-directed DJ planning and auditable execution."""
from __future__ import annotations
import json, re, time, uuid
from pathlib import Path
from typing import Any, Callable
from autodj_gemini import GeminiError, generate_text

class AgentError(Exception):
    pass

SYSTEM = '''You are a DJ planning an instrumental set. Return only JSON with a "tracks" array of exactly two objects. Each object must have title, style, bpm, camelot, bars, backend. title and style are short strings. bpm is 70-180, camelot is 1A-12A or 1B-12B, bars is 2, 4 or 8, backend is "synth" or "neural". The neural backend requires an available GPU and MusicGen; synth is a basic procedural electronic synthesizer and does not follow the requested genre. Choose neural when it is available, otherwise synth. Never invent tools or paths. Describe the requested progression in the two styles.'''

def validate_plan(value: Any, neural_available: bool) -> dict:
    if not isinstance(value, dict) or set(value) != {'tracks'} or not isinstance(value['tracks'], list) or len(value['tracks']) != 2:
        raise AgentError('Plan must contain exactly two tracks.')
    tracks=[]
    for item in value['tracks']:
        if not isinstance(item, dict) or set(item) != {'title','style','bpm','camelot','bars','backend'}:
            raise AgentError('Invalid track fields.')
        title, style = item['title'], item['style']
        if not isinstance(title,str) or not 1 <= len(title.strip()) <= 80 or not isinstance(style,str) or not 1 <= len(style.strip()) <= 300:
            raise AgentError('Invalid track title or style.')
        bpm, bars, cam, backend = item['bpm'], item['bars'], item['camelot'], item['backend']
        if isinstance(bpm,bool) or not isinstance(bpm,(int,float)) or not 70 <= bpm <= 180 or type(bars) is not int or bars not in (2,4,8):
            raise AgentError('Invalid BPM or bar count.')
        if not isinstance(cam,str) or not re.fullmatch(r'(?:[1-9]|1[0-2])[AB]',cam):
            raise AgentError('Invalid Camelot key.')
        if backend not in ('synth','neural') or (backend == 'neural' and not neural_available):
            raise AgentError('Requested backend is unavailable.')
        tracks.append(dict(title=title.strip(),style=style.strip(),bpm=bpm,camelot=cam,bars=bars,backend=backend))
    return {'tracks': tracks}

def request_plan(prompt: str, neural_available: bool, model: str) -> dict:
    if not prompt.strip() or len(prompt) > 1000:
        raise AgentError('Enter a request of at most 1000 characters.')
    schema = {"type": "object", "required": ["tracks"], "properties": {
        "tracks": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
            "type": "object", "required": ["title", "style", "bpm", "camelot", "bars", "backend"],
            "properties": {name: {"type": value} for name, value in {
                "title": "string", "style": "string", "bpm": "number",
                "camelot": "string", "bars": "integer", "backend": "string",
            }.items()},
        }},
    }}
    try:
        answer = generate_text(model, SYSTEM + f' Neural available: {neural_available}.',
                               prompt, schema=schema)
        return validate_plan(json.loads(answer), neural_available)
    except (GeminiError, ValueError) as exc:
        raise AgentError(str(exc)) from exc


class AgentRunner:
    def __init__(self, directory: str, producer: Any):
        self.root=Path(directory).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.producer=producer
    def log(self, run_id: str, action: str, status: str, detail: Any) -> None:
        record={'time':time.time(),'run_id':run_id,'action':action,'status':status,'detail':detail}
        with (self.root/'actions.jsonl').open('a',encoding='utf8') as out:
            out.write(json.dumps(record,ensure_ascii=False)+'\n')
    def run(self,prompt: str,model: str,neural_available: bool, planner: Callable = request_plan) -> dict:
        import numpy as np
        import soundfile as sf
        from autodj_indexer import SAMPLE_RATE
        run_id=uuid.uuid4().hex
        self.log(run_id,'plan','started',{'request':prompt,'model':model})
        try:
            plan=validate_plan(planner(prompt,neural_available,model),neural_available)
            self.log(run_id,'plan','completed',plan)
            audios=[]; outputs=[]
            for i,track in enumerate(plan['tracks']):
                self.log(run_id,'generate','started',{'track':i+1, **track})
                path=self.root / f'{run_id}_{i}.wav'
                art=self.producer.produce_track(title=track['title'],style=track['style'],target_bpm=track['bpm'],target_camelot=track['camelot'],bars=track['bars'],output_path=str(path),use_neural_gpu=track['backend']=='neural',seed=i+1)
                is_neural=bool(art.metadata.get('neural_gpu_active'))
                if track['backend']=='neural' and not is_neural:
                    raise AgentError('Neural generation failed or was unavailable; procedural audio was not presented as neural output.')
                audio,sr=sf.read(art.output_path,dtype='float32',always_2d=True)
                if sr != SAMPLE_RATE or not len(audio): raise AgentError('Producer returned invalid audio.')
                audios.append(audio); outputs.append({'backend':art.backend_used,'duration_sec':art.duration_sec})
                self.log(run_id,'generate','completed',outputs[-1])
            overlap=min(SAMPLE_RATE*2,len(audios[0])//3,len(audios[1])//3)
            fade=np.linspace(0,1,overlap,dtype=np.float32)[:,None]
            mix=np.concatenate([audios[0][:-overlap],audios[0][-overlap:]*(1-fade)+audios[1][:overlap]*fade,audios[1][overlap:]])
            sf.write(str(self.root/f'{run_id}_mix.wav'),mix,SAMPLE_RATE)
            self.log(run_id,'mix','completed',{'crossfade_seconds':overlap/SAMPLE_RATE})
            return {'run_id':run_id,'plan':plan,'outputs':outputs,'audio_url':f'/api/dj-agent/audio/{run_id}'}
        except Exception as exc:
            self.log(run_id,'run','failed',{'error':str(exc)})
            raise
