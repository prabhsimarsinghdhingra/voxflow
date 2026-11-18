# VoxFlow

Low-latency agentic voice AI. Browser mic → WebSocket → faster-whisper (streaming STT) → LangGraph agent (intent, tools, memory) → vLLM (streaming LLM) → sentence-level edge-tts → browser speaker. Timeouts, retries with backoff and text fallbacks at every stage; Redis session memory; PostgreSQL turn log.

## Quick start (Windows PowerShell / VS Code terminal)
1. Install Python 3.10+, then start an LLM server (below).
2. `Copy-Item .env.example .env`
3. `.\run.ps1 -Bench` to benchmark, then `.\run.ps1 -Serve` and open http://localhost:8000/

### LLM server (pick one)
- **Ollama, native Windows:** install from ollama.com, `ollama pull qwen2.5:1.5b`, then in `.env` set `VLLM_URL=http://localhost:11434/v1` and `VLLM_MODEL=qwen2.5:1.5b`.
- **vLLM (NVIDIA GPU):** vLLM does not run natively on Windows. In WSL2 Ubuntu: `pip install vllm; vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001`. Or use `docker compose up --build` (starts vLLM, app, Redis, Postgres).

GPU speech recognition: `WHISPER_DEVICE=cuda` (needs CUDA 12 + cuDNN 9). CPU `base.en` works everywhere.

## Protocol (`/ws?session=ID`)
Client: `{"type":"audio","data":"<b64 PCM16 16kHz>"}`, `{"type":"end"}`, `{"type":"interrupt"}` (barge-in), or `{"type":"text","text":"..."}` for testing.
Server: `transcript` (partial/final), `token`, `audio` (MP3, base64), `speak_text` (browser TTS fallback), `metrics`.

## Metrics
`ttft_ms` = end of speech → first LLM token. `e2e_ms` = end of speech → first audio. `python -m voxflow.benchmark --levels 1 2 4 8 16` writes `dashboard/benchmark.json` (p50/p95/p99, turns/s, fallback rate).

## Tests
`python -m pytest -q` runs 13 unit tests against in-process fakes (`voxflow/testing.py`), so no models are needed. Production code never imports them.

## Dashboard on Vercel
Static site in `dashboard/`. Run the benchmark, commit `dashboard/benchmark.json`, push. In Vercel: Add New Project → import the repo → Root Directory `dashboard` → Framework "Other" → Deploy. The voice backend itself needs a GPU/long-lived server and cannot run on Vercel.
