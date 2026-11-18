"""Real providers: faster-whisper (STT), vLLM/OpenAI-compatible server (LLM), edge-tts (TTS)."""
import asyncio
import json


class WhisperSTT:
    """Streaming-ish STT on 16 kHz mono PCM16: a partial transcript every ~2 s, final on utterance end."""
    PARTIAL_BYTES = 64000  # 2 s of 16 kHz PCM16

    def __init__(self, model="base.en", device="cpu"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")

    def _run(self, pcm: bytes) -> str:
        import numpy as np
        if len(pcm) < 3200:
            return ""
        audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()

    async def stream(self, chunks):
        loop, buf, done, last = asyncio.get_running_loop(), bytearray(), asyncio.Event(), 0

        async def reader():  # keeps consuming audio while a partial decode runs
            async for c in chunks:
                buf.extend(c)
            done.set()

        task = asyncio.create_task(reader())
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), 0.5)
            except asyncio.TimeoutError:
                pass
            if not done.is_set() and len(buf) - last >= self.PARTIAL_BYTES:
                last = len(buf)
                yield await loop.run_in_executor(None, self._run, bytes(buf)), False
        await task
        yield await loop.run_in_executor(None, self._run, bytes(buf)), True


class VLLMClient:
    """Streams from any OpenAI-compatible server: vLLM (`:8001/v1`) or Ollama (`:11434/v1`)."""

    def __init__(self, base_url, model, api_key="EMPTY"):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key

    async def stream(self, messages):
        import httpx
        body = {"model": self.model, "messages": messages, "stream": True, "max_tokens": 160, "temperature": 0.4}
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=3.0)) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self.api_key}"}) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        ch = json.loads(line[6:])["choices"]
                        delta = ch[0]["delta"].get("content") if ch else None
                        if delta:
                            yield delta


class EdgeTTS:
    """Neural TTS via edge-tts (needs internet). Emits MP3 chunks."""
    mime = "audio/mpeg"

    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice

    async def synthesize(self, text):
        import edge_tts
        async for ch in edge_tts.Communicate(text, self.voice).stream():
            if ch["type"] == "audio":
                yield ch["data"]


def build(s):
    return (WhisperSTT(s.whisper_model, s.whisper_device), VLLMClient(s.vllm_url, s.llm_model, s.llm_api_key),
            EdgeTTS(s.tts_voice))
