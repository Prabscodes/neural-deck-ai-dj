# Public Neural Deck: pre-rendered bank + live Gemini planner. No GPU, no torch.
# Build context needs the bank/ folder (committed; rebuild with python tools/build_bank.py).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-public.txt .
RUN pip install --no-cache-dir -r requirements-public.txt
COPY autodj_public.py autodj_live_agent.py autodj_agent.py autodj_gemini.py autodj_bank.py \
     autodj_stable_audio.py autodj_beatmatch.py autodj_frontend.html ./
COPY autodj_web ./autodj_web
COPY bank ./bank
RUN useradd -m app && mkdir dj_agent_runs && chown -R app /app
USER app
ENV AUTODJ_BANK_DIR=/app/bank PORT=8300 PYTHONUNBUFFERED=1
# Set GEMINI_API_KEY (secret), AUTODJ_GEMINI_TRANSPORT=api and AUTODJ_GEMINI_MODEL at run time.
# Set AUTODJ_TRUST_PROXY=1 only when a proxy in front sets X-Forwarded-For.
CMD ["sh", "-c", "python autodj_public.py --host 0.0.0.0 --port ${PORT}"]
