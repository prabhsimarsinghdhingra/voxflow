import asyncio
import unittest

from voxflow.config import Settings
from voxflow.pipeline import VoicePipeline
from voxflow.testing import MockLLM, MockSTT, MockTTS
from voxflow.store import Store


async def audio(text):
    b = text.encode()
    for i in range(0, len(b), 5):
        yield b[i:i + 5]


async def turn(pipe, text, sid="t"):
    return [e async for e in pipe.handle_turn(sid, audio(text))]


def last(evs):
    return evs[-1]


class FlakyLLM(MockLLM):
    def __init__(self, fail_first=1, **kw):
        super().__init__(**kw)
        self.calls, self.fail_first = 0, fail_first

    async def stream(self, messages):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("boom")
        async for t in super().stream(messages):
            yield t


class SlowLLM(MockLLM):
    async def stream(self, messages):
        await asyncio.sleep(5)
        yield "never"


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def pipe(self, **kw):
        llm = kw.pop("llm", None) or MockLLM()
        return VoicePipeline(Settings(redis_url="", database_url="", **kw), stt=MockSTT(), llm=llm,
                             tts=MockTTS(), store=Store())

    async def test_full_turn_streams_everything(self):
        evs = await turn(self.pipe(), "hello there")
        kinds = {e["type"] for e in evs}
        self.assertTrue({"transcript", "token", "audio", "metrics"} <= kinds)
        m = last(evs)
        self.assertFalse(m["fallback"])
        self.assertLess(m["e2e_ms"], 1000)
        self.assertLessEqual(m["ttft_ms"], m["e2e_ms"])

    async def test_tool_calling_calculator(self):
        evs = await turn(self.pipe(), "what is 12 * 7")
        said = " ".join(e["text"] for e in evs if e["type"] == "token")
        self.assertIn("84", said)
        self.assertEqual(last(evs)["intent"], "calc")

    async def test_intents(self):
        p = self.pipe()
        self.assertEqual(last(await turn(p, "weather in Pune"))["intent"], "weather")
        self.assertEqual(last(await turn(p, "where is order 48213"))["intent"], "order")

    async def test_retry_then_success(self):
        evs = await turn(self.pipe(llm=FlakyLLM(fail_first=1)), "hello")
        m = last(evs)
        self.assertEqual(m["retries"], 1)
        self.assertFalse(m["fallback"])

    async def test_fallback_when_llm_down(self):
        p = self.pipe(llm=FlakyLLM(fail_first=99), llm_retries=2)
        evs = await turn(p, "hello")
        m = last(evs)
        self.assertTrue(m["fallback"])
        self.assertEqual(m["retries"], 3)
        self.assertTrue(any(e["type"] in ("audio", "speak_text") for e in evs))

    async def test_first_token_timeout_triggers_fallback(self):
        p = self.pipe(llm=SlowLLM(), llm_first_token_timeout=0.05, llm_retries=1)
        self.assertTrue(last(await turn(p, "hello"))["fallback"])

    async def test_contextual_memory(self):
        p = self.pipe()
        await turn(p, "hello", "s1")
        await turn(p, "what is 2 + 2", "s1")
        h = await p.store.history("s1")
        self.assertEqual([x["role"] for x in h], ["user", "assistant"] * 2)

    async def test_sessions_are_isolated(self):
        p = self.pipe()
        await turn(p, "hello", "a")
        self.assertEqual(await p.store.history("b"), [])

    async def test_concurrent_sessions(self):
        p = self.pipe()
        res = await asyncio.gather(*(turn(p, "what is 3 * 3", f"c{i}") for i in range(30)))
        self.assertEqual(len(res), 30)
        self.assertTrue(all(not last(r)["fallback"] for r in res))

    async def test_summary(self):
        p = self.pipe()
        await turn(p, "hello")
        self.assertEqual(p.summary()["turns"], 1)


if __name__ == "__main__":
    unittest.main()
