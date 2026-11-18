"""Deterministic fakes used ONLY by the unit tests. Production never imports this module."""
import asyncio
import codecs
import random


class MockSTT:
    async def stream(self, chunks):
        dec, text = codecs.getincrementaldecoder("utf-8")("ignore"), ""
        async for c in chunks:
            await asyncio.sleep(0.002)
            text += dec.decode(c)
            yield text.strip(), False
        yield text.strip(), True


class MockLLM:
    def __init__(self, slots=8, seed=7):
        self.slots, self.rng, self._sem = slots, random.Random(seed), None

    async def stream(self, messages):
        self._sem = self._sem or asyncio.Semaphore(self.slots)
        async with self._sem:
            await asyncio.sleep(0.05)
            system, user = messages[0]["content"], messages[-1]["content"]
            tool = next((l[13:] for l in system.splitlines() if l.startswith("Tool result: ")), "none")
            if tool != "none":
                text = f"Sure. {tool}. Anything else?"
            elif "Intent: greeting" in system:
                text = "Hello! I'm VoxFlow. How can I help you today?"
            else:
                text = f"Got it. You said {user}. Tell me more."
            for w in text.split(" "):
                await asyncio.sleep(0.005)
                yield w + " "


class MockTTS:
    mime = "audio/pcm"

    async def synthesize(self, text):
        await asyncio.sleep(0.02)
        yield b"\x00" * (len(text) * 160)
