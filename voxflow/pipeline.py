"""Streaming turn pipeline: audio -> STT -> graph plan -> LLM tokens -> sentence TTS."""
import asyncio
import base64
import re
import time

from .agent import build_graph
from .config import Settings
from .providers import build
from .store import Store

FALLBACK_REPLY = "Sorry, I'm having trouble right now. Could you say that again?"
SENT_END = re.compile(r"[.!?]\s*$")


class LLMUnavailable(Exception):
    pass


async def _tap(chunks, mark):
    async for c in chunks:
        yield c
    mark["end"] = time.perf_counter()  # moment the speaker stopped


class VoicePipeline:
    def __init__(self, settings=None, stt=None, llm=None, tts=None, store=None, graph=None):
        self.cfg = settings or Settings()
        d_stt, d_llm, d_tts = build(self.cfg) if None in (stt, llm, tts) else (None, None, None)
        self.stt, self.llm, self.tts = stt or d_stt, llm or d_llm, tts or d_tts
        self.store = store or Store(self.cfg.redis_url, self.cfg.database_url)
        self.graph = graph or build_graph()

    async def _llm_tokens(self, messages, counters):
        """Retry with exponential backoff until the first token arrives, then stream."""
        for attempt in range(self.cfg.llm_retries + 1):
            agen = self.llm.stream(messages)
            try:
                first = await asyncio.wait_for(agen.__anext__(), self.cfg.llm_first_token_timeout)
            except StopAsyncIteration:
                return
            except Exception:
                counters["retries"] += 1
                await agen.aclose()
                await asyncio.sleep(0.05 * 2 ** attempt)
                continue
            yield first
            async for tok in agen:
                yield tok
            return
        raise LLMUnavailable()

    async def _speak(self, sentence, counters):
        """Synthesize one sentence; degrade to text-only (client-side TTS) on failure."""
        try:
            async def collect():
                return [c async for c in self.tts.synthesize(sentence)]
            audio = await asyncio.wait_for(collect(), self.cfg.tts_timeout)
            return {"type": "audio", "mime": getattr(self.tts, "mime", "audio/pcm"), "text": sentence, "data": base64.b64encode(b"".join(audio)).decode()}
        except Exception:
            counters["tts_fallbacks"] += 1
            return {"type": "speak_text", "text": sentence}

    async def handle_turn(self, sid, chunks):
        mark, c = {}, {"retries": 0, "tts_fallbacks": 0}
        t0 = time.perf_counter()
        rel = lambda: (time.perf_counter() - mark.get("end", t0)) * 1000  # ms since speech ended
        text = ""
        async for text, final in self.stt.stream(_tap(chunks, mark)):
            yield {"type": "transcript", "text": text, "final": final}
        mark.setdefault("end", time.perf_counter())
        stt_ms = rel()

        state = {"text": text, "history": await self.store.history(sid)}
        try:
            plan = await asyncio.wait_for(self.graph.ainvoke(state), self.cfg.plan_timeout)
        except Exception:  # a stalled tool/graph degrades to plain chat
            plan = {"intent": "chat", "messages": [{"role": "system", "content": "You are VoxFlow."},
                                                   {"role": "user", "content": text}]}
        plan_ms = rel()

        reply, buf, first_tok, first_out, fallback, n_tok, llm_first = "", "", None, None, False, 0, None
        llm_start = time.perf_counter()
        try:
            async for tok in self._llm_tokens(plan["messages"], c):
                if first_tok is None:
                    first_tok, llm_first = rel(), (time.perf_counter() - llm_start) * 1000
                n_tok += 1
                reply, buf = reply + tok, buf + tok
                yield {"type": "token", "text": tok}
                if (SENT_END.search(buf) and len(buf) >= 15) or len(buf) > 120:
                    ev = await self._speak(buf.strip(), c)
                    first_out = rel() if first_out is None else first_out
                    buf = ""
                    yield ev
            if not reply:
                raise LLMUnavailable()
        except Exception:
            fallback = True
            if not n_tok:
                reply = buf = FALLBACK_REPLY
                yield {"type": "token", "text": buf}
        if buf.strip():
            ev = await self._speak(buf.strip(), c)
            first_out = rel() if first_out is None else first_out
            yield ev

        await self.store.append(sid, "user", text)
        await self.store.append(sid, "assistant", reply.strip())
        e2e = first_out if first_out is not None else rel()
        m = {"type": "metrics", "session": sid, "intent": plan.get("intent", "chat"), "stt_ms": stt_ms,
             "plan_ms": plan_ms - stt_ms, "llm_ttft_ms": llm_first or 0.0, "ttft_ms": first_tok or e2e,
             "e2e_ms": e2e, "total_ms": rel(), "tokens": n_tok, "retries": c["retries"],
             "tts_fallbacks": c["tts_fallbacks"], "fallback": fallback}
        await self.store.log_turn(m)
        yield m

    async def warmup(self):
        """Load models and open connections before the first caller arrives."""
        async def silence():
            yield b"\x00" * 32000
        try:
            async for _ in self.stt.stream(silence()):
                pass
            async for _ in self.llm.stream([{"role": "user", "content": "hi"}]):
                break
            async for _ in self.tts.synthesize("Ready."):
                break
        except Exception as e:
            print("warmup warning:", e)

    def summary(self):
        t = self.store.turns
        if not t:
            return {"turns": 0}
        e = sorted(x["e2e_ms"] for x in t)
        return {"turns": len(t), "e2e_p50_ms": round(e[len(e) // 2], 1),
                "e2e_p95_ms": round(e[min(len(e) - 1, int(len(e) * 0.95))], 1),
                "fallback_rate": sum(x["fallback"] for x in t) / len(t)}
