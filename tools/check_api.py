"""One-call check that the Gemini API path works with your key, model, JSON schema and audio input.

    read -rs GEMINI_API_KEY; export GEMINI_API_KEY
    AUTODJ_GEMINI_TRANSPORT=api AUTODJ_GEMINI_MODEL=gemini-4-hydrogen .venv/bin/python tools/check_api.py

Prints only status information, never the key. Makes two small requests (text, then text plus 1 s of audio).
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodj_gemini import GeminiError, configured_model, configured_transport, generate_text  # noqa: E402

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ok", "heard"],
          "properties": {"ok": {"type": "boolean"}, "heard": {"type": "string", "maxLength": 80}}}


def main() -> int:
    os.environ["AUTODJ_GEMINI_TRANSPORT"] = "api"
    model = configured_model()
    print(f"transport={configured_transport()} model={model}")
    wave = (np.sin(2 * np.pi * 440 * np.arange(44100) / 44100) * .2).astype("float32")
    wav = io.BytesIO()
    sf.write(wav, np.column_stack((wave, wave)), 44100, format="WAV", subtype="PCM_16")
    ok = True
    for label, audio in (("text + JSON schema", None), ("text + audio + JSON schema", wav.getvalue())):
        try:
            reply = generate_text(model, "Answer in JSON. In 'heard' say what the input is, in a few words.",
                                  "Say ok=true.", audio_wav=audio, schema=SCHEMA, timeout=60)
            parsed = json.loads(reply)
            print(f"PASS  {label}: {parsed}")
        except (GeminiError, ValueError) as exc:
            ok = False
            print(f"FAIL  {label}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
