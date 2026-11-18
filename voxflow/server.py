import asyncio
import base64
import json
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pipeline import VoicePipeline

pipeline = None
ROOT = pathlib.Path(__file__).resolve().parent.parent


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = VoicePipeline()  # loads Whisper, connects to the LLM server
    return pipeline


@asynccontextmanager
async def lifespan(_):
    if os.getenv("VOX_WARMUP", "1") == "1":
        await get_pipeline().warmup()
    yield


app = FastAPI(title="VoxFlow", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DASH = ROOT / "dashboard"


async def _drain(q):
    while True:
        c = await q.get()
        if c is None:
            return
        yield c


@app.get("/health")
async def health():
    import httpx
    cfg = get_pipeline().cfg
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            llm_up = (await c.get(cfg.vllm_url.rstrip("/") + "/models")).status_code == 200
    except Exception:
        llm_up = False
    return {"ok": True, "llm_up": llm_up, "llm": cfg.llm_model, "stt": cfg.whisper_model}


@app.get("/")
async def root():
    return RedirectResponse("/app/")


@app.get("/api/metrics")
async def metrics():
    return get_pipeline().summary()


@app.get("/api/benchmark")
async def benchmark():
    f = DASH / "benchmark.json"
    return json.loads(f.read_text()) if f.exists() else {"levels": []}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, session: str = "anon"):
    """Client sends {"type":"audio","data":b64} | {"type":"text","text":..} | {"type":"end"} | {"type":"interrupt"}"""
    await ws.accept()
    task, q = None, None

    async def run(queue):
        async for ev in get_pipeline().handle_turn(session, _drain(queue)):
            await ws.send_json(ev)

    try:
        while True:
            m = await ws.receive_json()
            t = m.get("type")
            if t in ("audio", "text"):
                if task is None or task.done():
                    q = asyncio.Queue()
                    task = asyncio.create_task(run(q))
                q.put_nowait(base64.b64decode(m["data"]) if t == "audio" else m["text"].encode())
                if t == "text":
                    q.put_nowait(None)
            elif t == "end" and q is not None:
                q.put_nowait(None)
            elif t == "interrupt" and task and not task.done():  # barge-in
                task.cancel()
                await ws.send_json({"type": "interrupted"})
    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()


if (ROOT / "client").exists():
    app.mount("/app", StaticFiles(directory=str(ROOT / "client"), html=True), name="client")
if DASH.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASH), html=True), name="dashboard")
