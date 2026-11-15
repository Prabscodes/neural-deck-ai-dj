"""Optional, local ACE-Step 1.5 turbo text-to-music backend for live DJ tracks.

This adapter never downloads weights or unloads other applications' models. The
separate 5 Hz language model is not used. Failures are reported to the caller;
no procedural audio is substituted for neural generation.
"""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

_LOCK = threading.Lock()


def _restore_rotary(model: Any, device: str) -> int:
    """Restore nonpersistent rotary buffers left uninitialized by some HF loaders."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3RotaryEmbedding):
            fresh = Qwen3RotaryEmbedding(module.config, device=device)
            module.inv_freq = fresh.inv_freq
            module.original_inv_freq = fresh.original_inv_freq
            module.attention_scaling = fresh.attention_scaling
            if not torch.isfinite(module.inv_freq).all():
                raise RuntimeError('ACE-Step rotary positions are invalid')
            count += 1
    if not count:
        raise RuntimeError('ACE-Step rotary position layers were not found')
    return count


def _caption(style: str) -> str:
    """Use the listener's requested style; strengthen the evaluated piano-dubstep case."""
    original = style.strip()
    low = original.lower()
    if 'dubstep' in low and ('piano' in low or 'keyboard' in low) and not any(x in low for x in ('chill','mellow','gentle','soft','ambient')):
        return ("Instrumental festival dubstep, half-time. A close-miked acoustic grand piano plays "
                "a distinct short hook in the breakdown. The drop alternates piano answers with "
                "syncopated talking FM growls, rhythmic wobble fills and clean sub. Heavy kick, "
                "cracking snare on beat three; sparse hats. Phrase changes, drum fill into the drop, "
                "short drum-and-bass outro for mixing. No vocals. " + original)
    return original


def generate(*, driver: Any, prompt: str, output_path: str, target_bpm: float,
             target_camelot: str, bars: int, seed: int) -> Any:
    """Create one full stereo clip, with conditioning hints rather than guaranteed BPM/key."""
    from autodj_producer import ProducedAudio, chunked_oobleck_decode, _ensure_vector_quantize_pytorch_shim
    from autodj_indexer import CAMELOT_MAJOR, CAMELOT_MINOR
    from diffusers import AutoencoderOobleck
    from transformers import AutoModel, AutoTokenizer

    if not math.isfinite(target_bpm) or not 40 <= target_bpm <= 220 or not 1 <= bars <= 64:
        raise ValueError('Invalid BPM or bar count for ACE-Step generation')
    root = Path(driver.acestep_snap or '')
    for name in ('Qwen3-Embedding-0.6B', 'acestep-v15-turbo', 'vae'):
        if not (root / name).is_dir():
            raise FileNotFoundError('ACE-Step cached model components are unavailable')
    if not torch.cuda.is_available():
        raise RuntimeError('ACE-Step live generation requires an available CUDA GPU')
    duration = bars * 4 * 60.0 / target_bpm
    frames = int(round(duration * 25))
    if not 25 <= frames <= 3000:
        raise ValueError('ACE-Step track must be between 1 and 120 seconds')
    keys = {**{v: f'{k} major' for k, v in CAMELOT_MAJOR.items()},
            **{v: f'{k} minor' for k, v in CAMELOT_MINOR.items()}}
    key = keys.get(target_camelot, '')
    words = prompt.lower()
    use_arrangement = (
        os.environ.get('AUTODJ_DUBSTEP_ARRANGER') == '1'
        and 'dubstep' in words
        and 'piano' in words
        and not any(term in words for term in ('no piano', 'without piano', 'chill', 'ambient', 'mellow'))
    )
    caption = _caption(prompt)
    if use_arrangement:
        caption = ("Quiet tonal background texture for an instrumental dubstep arrangement. Soft sustained "
                   "harmonic atmosphere, no drums, no lead, no noise sweeps.")
    meta = f'- bpm: {round(target_bpm)}\n- timesignature: 4\n- keyscale: {key}\n- duration: {round(duration, 2)} seconds\n'
    text_prompt = ('# Instruction\nFill the audio semantic mask based on the given conditions:'
                   f'\n\n# Caption\n{caption}\n\n# Metas\n{meta}<|endoftext|>')
    lyrics = '# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>'
    device, dtype = 'cuda', torch.bfloat16

    with _LOCK:
        free, _ = torch.cuda.mem_get_info()
        if free < 10 * 1024**3:
            raise RuntimeError(f'ACE-Step needs at least 10 GiB free GPU memory; {free / 1024**3:.1f} GiB available')
        _ensure_vector_quantize_pytorch_shim()
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(str(root / 'Qwen3-Embedding-0.6B'), local_files_only=True)
            text = AutoModel.from_pretrained(str(root / 'Qwen3-Embedding-0.6B'), local_files_only=True, dtype=dtype).to(device).eval()
            _restore_rotary(text, device)
            tokens = tokenizer([text_prompt], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            lyric_tokens = tokenizer([lyrics], padding=True, truncation=True, max_length=2048, return_tensors='pt').to(device)
            text_hidden = text(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
            lyric_hidden = text.get_input_embeddings()(lyric_tokens.input_ids)
            text_mask, lyric_mask = tokens.attention_mask.bool(), lyric_tokens.attention_mask.bool()
            if not torch.isfinite(text_hidden).all() or not torch.isfinite(lyric_hidden).all():
                raise RuntimeError('ACE-Step text conditioning is invalid')
            del text, tokens, lyric_tokens
            gc.collect()
            torch.cuda.empty_cache()
            dit = AutoModel.from_pretrained(str(root / 'acestep-v15-turbo'), trust_remote_code=True, local_files_only=True, dtype=dtype).to(device).eval()
            _restore_rotary(dit, device)
            silence = torch.load(root / 'acestep-v15-turbo' / 'silence_latent.pt', map_location='cpu', weights_only=True).transpose(1, 2).to(device=device, dtype=dtype)
            if silence.shape[1] < max(frames, 750) or not torch.isfinite(silence).all():
                raise RuntimeError('ACE-Step silence conditioning is invalid')
            source = silence[:, :frames, :].contiguous()
            reference = silence[:, :750, :].contiguous()
            result = dit.generate_audio(text_hidden_states=text_hidden, text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden, lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=reference,
                refer_audio_order_mask=torch.tensor([0], device=device, dtype=torch.long),
                src_latents=source, chunk_masks=torch.ones_like(source),
                is_covers=torch.zeros(1, device=device, dtype=torch.long), silence_latent=silence,
                precomputed_lm_hints_25Hz=torch.zeros_like(source), seed=int(seed), shift=3.0)
            latents = result['target_latents'].transpose(1, 2).contiguous()
            if not torch.isfinite(latents).all():
                raise RuntimeError('ACE-Step diffusion produced invalid audio latents')
            del dit, text_hidden, lyric_hidden, source, reference, silence, result
            gc.collect()
            torch.cuda.empty_cache()
            vae = AutoencoderOobleck.from_pretrained(str(root / 'vae'), local_files_only=True, torch_dtype=dtype).to(device).eval()
            audio = chunked_oobleck_decode(vae, latents, chunk_frames=250, overlap_frames=25)[0].numpy()
            del vae, latents
            gc.collect()
            torch.cuda.empty_cache()
    if audio.ndim != 2 or audio.shape[0] != 2 or not np.isfinite(audio).all():
        raise RuntimeError('ACE-Step decoder produced invalid stereo audio')
    audio = resample_poly(audio, 147, 160, axis=1).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio *= 0.98 / peak
    backend = 'ACE-Step 1.5 turbo (local)'
    profile = None
    if use_arrangement:
        from autodj_dubstep_arranger import arrange, track_profile
        profile = track_profile(int(seed))
        key_name, _, key_mode = key.partition(' ')
        pitches = ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')
        audio = arrange(audio.T, 44100, target_bpm, bars,
                        36 + pitches.index(key_name), key_mode == 'minor', int(seed)).T
        backend = 'ACE-Step + sampled piano and arranged dubstep stems'
    output = Path(output_path).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio.T, 44100, subtype='PCM_16')
    return ProducedAudio(output_path=str(output), backend_used=backend, mode='local',
        task_type='full_song', sample_rate=44100, duration_sec=round(audio.shape[1]/44100, 4),
        target_bpm=target_bpm, target_camelot=target_camelot,
        metadata={'neural_gpu_active': True, 'model': 'ACE-Step 1.5 turbo', 'generation_mode': 'text2music',
                  'lm_used': False, 'seed': int(seed), 'bars': bars, 'caption': caption,
                  'timing_key_are_requests': not use_arrangement, 'arranged_downbeat': use_arrangement, 'pre_peak_limit': peak, 'arrangement_profile': profile})
