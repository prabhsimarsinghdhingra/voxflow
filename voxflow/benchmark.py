"""Real-stack benchmark: spoken utterances (edge-tts) -> Whisper -> LangGraph -> LLM server -> TTS.
Reports TTFT, end-to-end latency (end of speech -> first audio), throughput and concurrency scaling."""
import argparse
import asyncio
import datetime as dt
import json
import pathlib
import statistics
import time

from .config import Settings
from .pipeline import VoicePipeline
from .store import Store

UTTERANCES = ["Hello there, how are you today?", "What is 12 times 7?", "What time is it right now?",
              "What is the weather in Bengaluru?", "Where is my order 48213?",
              "Tell me one interesting fact about the moon."]
CHUNK = 3200  # 100 ms of 16 kHz PCM16


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p / 100))], 1)


def dist(xs):
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "mean": round(statistics.mean(xs), 1)}


async def make_audio(cfg, cache):
    """Synthesize each utterance once, decode to 16 kHz mono PCM16 bytes."""
    import edge_tts
    import numpy as np
    from faster_whisper import decode_audio
    cache.mkdir(exist_ok=True)
    out = []
    for i, text in enumerate(UTTERANCES):
        f = cache / f"u{i}.mp3"
        if not f.exists():
            await edge_tts.Communicate(text, cfg.tts_voice).save(str(f))
        pcm = (decode_audio(str(f), sampling_rate=16000) * 32767).astype(np.int16).tobytes()
        out.append(pcm)
    return out


async def stream_pcm(pcm):
    for i in range(0, len(pcm), CHUNK):
        yield pcm[i:i + CHUNK]


async def session(pipe, k, turns, clips):
    out = []
    for i in range(turns):
        async for ev in pipe.handle_turn(f"s{k}", stream_pcm(clips[(k + i) % len(clips)])):
            if ev["type"] == "metrics":
                out.append(ev)
    return out


async def run_level(pipe, n, turns, clips):
    pipe.store = Store()
    t0 = time.perf_counter()
    res = [m for r in await asyncio.gather(*(session(pipe, i, turns, clips) for i in range(n))) for m in r]
    wall = time.perf_counter() - t0
    avg = lambda k: round(statistics.mean(m[k] for m in res), 1)
    return {"concurrency": n, "turns": len(res), "ttft": dist([m["ttft_ms"] for m in res]),
            "e2e": dist([m["e2e_ms"] for m in res]), "throughput_tps": round(len(res) / wall, 2),
            "fallback_rate": round(sum(m["fallback"] for m in res) / len(res), 4),
            "retries": sum(m["retries"] for m in res),
            "stages": {"stt": avg("stt_ms"), "plan": avg("plan_ms"), "llm": avg("llm_ttft_ms"),
                       "tts": round(max(0.0, avg("e2e_ms") - avg("stt_ms") - avg("plan_ms") - avg("llm_ttft_ms")), 1)}}


async def amain(a):
    cfg = Settings()
    print("Loading models and generating test audio...")
    pipe = VoicePipeline(cfg)
    await pipe.warmup()
    clips = await make_audio(cfg, pathlib.Path(a.audio_cache))
    levels = []
    for n in a.levels:
        r = await run_level(pipe, n, a.turns, clips)
        levels.append(r)
        print(f"{n:>3} sessions | TTFT p50 {r['ttft']['p50']:>7} ms | E2E p50 {r['e2e']['p50']:>7} "
              f"p95 {r['e2e']['p95']:>7} ms | {r['throughput_tps']:>5} turns/s | fallback {r['fallback_rate']:.1%}")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "stt": f"faster-whisper {cfg.whisper_model} ({cfg.whisper_device})",
                               "llm": cfg.llm_model, "tts": f"edge-tts {cfg.tts_voice}", "levels": levels}, indent=2))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--out", default="dashboard/benchmark.json")
    ap.add_argument("--audio-cache", default="bench_audio")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
