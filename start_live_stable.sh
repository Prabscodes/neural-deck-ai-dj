#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
comfy="${AUTODJ_COMFY_DIR:-/mnt/storage/projects/ComfyUI}"
models="${AUTODJ_COMFY_BASE:-/home/prab/Documents/Codex/2026-11-12/the/work/dj-stable-audio}"
comfy_port="${AUTODJ_COMFY_PORT:-8296}"
dj_port="${AUTODJ_WEB_PORT:-8298}"

export AUTODJ_MUSIC_BACKEND=stable_audio
export AUTODJ_COMFY_URL="http://127.0.0.1:$comfy_port"
export AUTODJ_GEMINI_TRANSPORT="${AUTODJ_GEMINI_TRANSPORT:-cli}"
export AUTODJ_CACHE_DIR="${AUTODJ_CACHE_DIR:-/dev/shm/autodj_stable_live}"

renderer=""
trap 'if [[ -n "$renderer" ]]; then kill "$renderer" 2>/dev/null || true; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
health="$AUTODJ_COMFY_URL/models/checkpoints"
if ! curl -fs --max-time 2 "$health" >/dev/null; then
    (
        cd "$comfy"
        exec .venv/bin/python main.py --listen 127.0.0.1 --port "$comfy_port" \
            --base-directory "$models" --disable-all-custom-nodes --disable-api-nodes --disable-auto-launch
    ) >> "$models/comfy-live.log" 2>&1 &
    renderer=$!
    for _ in {1..45}; do
        if curl -fs --max-time 2 "$health" >/dev/null; then break; fi
        if ! kill -0 "$renderer" 2>/dev/null; then break; fi
        sleep 1
    done
fi
if ! curl -fs --max-time 2 "$health" >/dev/null; then
    echo "Stable Audio renderer unavailable. Check $models/comfy-live.log." >&2
    exit 1
fi

cd "$project"
.venv/bin/python autodj_server.py --host 127.0.0.1 --port "$dj_port"
