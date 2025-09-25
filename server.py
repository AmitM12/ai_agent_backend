# server.py (hardened + ASR stats/finals + connect logs + auto-greeting + strong health + explicit model)
import os
import json
import asyncio
import logging
from typing import Optional, Deque, List, Tuple
from collections import deque
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
from aiohttp import ClientSession, WSMsgType, ClientResponseError, ClientTimeout
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

PORT = int(os.getenv("PORT", "3001"))
DG_API_KEY = os.getenv("DG_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Explicit default; override via env with a model your account supports
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

ASR_LANGUAGE_HINT = (os.getenv("ASR_LANGUAGE_HINT") or "").strip()  # "", "multi", "hi", "en"
ASR_ENDPOINTING_MS = int(os.getenv("ASR_ENDPOINTING_MS", "500"))
TTS_QUEUE_MAX = int(os.getenv("TTS_QUEUE_MAX", "12"))                # sentences
TTS_SENTENCE_MAX_CHARS = int(os.getenv("TTS_SENTENCE_MAX_CHARS", "280"))

# Auto-greeting toggle (bot speaks first on connect)
AUTO_GREETING = os.getenv("AUTO_GREETING", "true").lower() in ("1", "true", "yes")

REQUIRED_ENV = {
    "DG_API_KEY": DG_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Track missing env for clearer /health and WS error
app.state.missing_env = []

@app.on_event("startup")
async def _validate_env():
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    app.state.missing_env = missing
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        app.state.env_ok = False
        return
    # quick ping check to catch obvious key issues (best-effort; don't block server if net hiccups)
    try:
        timeout = ClientTimeout(total=6)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get("https://api.elevenlabs.io/v1/models", headers={"xi-api-key": ELEVENLABS_API_KEY}) as r:
                app.state.env_ok = r.status < 400
                if app.state.env_ok:
                    logger.info("[Health] ElevenLabs key OK")
                else:
                    logger.warning(f"[Health] ElevenLabs key check status {r.status}")
    except Exception as e:
        logger.warning(f"[Health] Startup key check failed (continuing): {e}")
        app.state.env_ok = True  # keep green to avoid blocking local dev

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": bool(getattr(app.state, "env_ok", False)),
        "missing": getattr(app.state, "missing_env", []),
    })

# ----- Utilities -----
END_PUNCT = ("!", "?", ".", "।")
def jd(x) -> str: return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

async def ws_send_json(ws: WebSocket, obj: dict):
    try:
        await ws.send_text(jd(obj))
    except Exception as e:
        logger.warning(f"client_ws send_text failed: {e}")

def _segment_sentences(buf: str) -> Tuple[List[str], str]:
    out: List[str] = []
    i = 0
    while i < len(buf):
        if buf[i] in END_PUNCT:
            j = i + 1
            while j < len(buf) and buf[j].isspace():
                j += 1
            seg = buf[:j].strip()
            if seg:
                out.append(seg)
            buf = buf[j:]
            i = 0
        else:
            i += 1
    return out, buf

def _cap_sentence(s: str) -> str:
    s = s.strip()
    return s if len(s) <= TTS_SENTENCE_MAX_CHARS else s[:TTS_SENTENCE_MAX_CHARS] + "…"

# ----- WebSocket endpoint -----
@app.websocket("/ws")
async def socket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logger.info("Client connected")

    if not getattr(app.state, "env_ok", False):
        await ws_send_json(client_ws, {
            "type": "error",
            "source": "server",
            "code": "CONFIG_MISSING",
            "message": "Server missing required configuration.",
            "details": {"missing": getattr(app.state, "missing_env", [])}
        })
        with suppress(Exception):
            await client_ws.close()
        return

    session: Optional[ClientSession] = None
    dg_ws = None
    oai_ws = None

    # state
    speaking = False
    llm_buf = ""
    tts_queue: Deque[str] = deque()
    tts_task: Optional[asyncio.Task] = None
    gen_id = 0  # increments on each barge-in to invalidate old TTS

    # ASR debug counters/tasks
    bytes_up = 0
    stats_task: Optional[asyncio.Task] = None

    # --- helpers
    async def cleanup():
        nonlocal session, dg_ws, oai_ws, tts_task, stats_task
        logger.info("Cleaning up connection")
        with suppress(Exception):
            if stats_task and not stats_task.done():
                stats_task.cancel()
                await asyncio.gather(stats_task, return_exceptions=True)
        with suppress(Exception):
            if tts_task and not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
        with suppress(Exception):
            if dg_ws and not dg_ws.closed:
                await dg_ws.close(code=1000)
        with suppress(Exception):
            if oai_ws and not oai_ws.closed:
                await oai_ws.close(code=1000)
        with suppress(Exception):
            if session:
                await session.close()
        with suppress(Exception):
            await client_ws.close()

    async def publish_stats():
        try:
            while True:
                await asyncio.sleep(1.0)
                await ws_send_json(client_ws, {
                    "type": "asr.stats",
                    "bytesUp": bytes_up,
                    "connected": True
                })
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"stats publisher stopped: {e}")

    # async def connect_deepgram(session: ClientSession):
    #     params_primary = {
    #         "model": "nova-3",
    #         "encoding": "opus",
    #         "sample_rate": "48000",
    #         "channels": "1",
    #         "smart_format": "true",
    #         "interim_results": "true",
    #         "endpointing": str(ASR_ENDPOINTING_MS),
    #     }
    #     if ASR_LANGUAGE_HINT:
    #         params_primary["language"] = ASR_LANGUAGE_HINT  # e.g., "multi"|"en"|"hi"

    #     params_minimal = {"model": "nova-3", "encoding": "opus", "sample_rate": "48000", "channels": "1"}

    #     async def _connect(params: dict):
    #         qs = "&".join(f"{k}={v}" for k, v in params.items())
    #         url = f"wss://api.deepgram.com/v1/listen?{qs}"
    #         return await session.ws_connect(
    #             url,
    #             headers={"Authorization": f"Token {DG_API_KEY}"},
    #             heartbeat=30,
    #             autoping=True,
    #         )

    #     try:
    #         return await _connect(params_primary)
    #     except ClientResponseError as cre:
    #         logger.warning(f"[Deepgram] handshake {cre.status}: {cre.message}")
    #         if cre.status == 400:
    #             try:
    #                 logger.info("[Deepgram] retry minimal params")
    #                 return await _connect(params_minimal)
    #             except Exception as e2:
    #                 logger.error(f"[Deepgram] minimal connect failed: {e2}")
    #                 raise
    #         raise
    #     except Exception as e:
    #         logger.error(f"[Deepgram] connect error: {e}")
    #         raise

    async def connect_deepgram(session: ClientSession):
        # A) Let Deepgram auto-detect container/codec (often best for MediaRecorder webm/opus)
        params_auto = {
            "model": "nova-3",
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": str(ASR_ENDPOINTING_MS),
        }
        if ASR_LANGUAGE_HINT:
            params_auto["language"] = ASR_LANGUAGE_HINT

        # B) Explicit opus
        params_opus = {
            "model": "nova-3",
            "encoding": "opus",
            "sample_rate": "48000",
            "channels": "1",
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": str(ASR_ENDPOINTING_MS),
        }
        if ASR_LANGUAGE_HINT:
            params_opus["language"] = ASR_LANGUAGE_HINT

        # C) Minimal
        params_min = {"model": "nova-3"}

        async def _connect(params: dict, label: str):
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"wss://api.deepgram.com/v1/listen?{qs}"
            logger.info(f"[Deepgram] connecting with profile {label}: {qs}")
            return await session.ws_connect(
                url,
                headers={"Authorization": f"Token {DG_API_KEY}"},
                heartbeat=30,
                autoping=True,
            )

        # Try profiles in order
        for label, params in (("AUTO", params_auto), ("OPUS", params_opus), ("MIN", params_min)):
            try:
                ws = await _connect(params, label)
                logger.info(f"[Deepgram] connected using profile {label}")
                return ws
            except ClientResponseError as cre:
                logger.warning(f"[Deepgram] handshake ({label}) {cre.status}: {cre.message}")
                # continue to next profile
            except Exception as e:
                logger.warning(f"[Deepgram] connect error ({label}): {e}")
                # continue to next profile

        # If all failed, raise last error
        raise RuntimeError("Failed to connect to Deepgram with all profiles")


    async def connect_openai(session: ClientSession):
        url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
        ws = await session.ws_connect(
            url,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            heartbeat=30,
            autoping=True,
        )
        await ws.send_str(jd({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": (
                    "You are a friendly, empathetic voice assistant for India. "
                    "Reply in the user’s language (Hindi/English/Hinglish), 1–2 sentences, TTS-friendly."
                )
            }
        }))
        return ws

    async def handle_barge_in():
        """Cancel current TTS & response; clear queue; notify client."""
        nonlocal tts_task, speaking, llm_buf, tts_queue, oai_ws, gen_id
        gen_id += 1  # invalidate in-flight TTS
        if tts_task and not tts_task.done():
            tts_task.cancel()
            with suppress(Exception):
                await asyncio.gather(tts_task, return_exceptions=True)
        speaking = False
        tts_queue.clear()
        llm_buf = ""
        await ws_send_json(client_ws, {"type": "bot.audio.clear"})
        with suppress(Exception):
            if oai_ws and not oai_ws.closed:
                await oai_ws.send_str(jd({"type": "response.cancel"}))

    async def tts_chunk(text: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        params = {"optimize_streaming_latency": "3", "output_format": "mp3_22050_32"}
        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.35, "similarity_boost": 0.9, "style": 0.65, "use_speaker_boost": True}
        }
        # bounded retries for transient errors
        attempts = 0
        while True:
            attempts += 1
            try:
                async with session.post(
                    url, params=params, json=body,
                    headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
                ) as resp:
                    resp.raise_for_status()
                    bufs = [chunk async for chunk in resp.content.iter_chunked(4096)]
                    return b"".join(bufs)
            except ClientResponseError as cre:
                # 4xx: don't retry except maybe 429
                if cre.status == 429 and attempts < 3:
                    await asyncio.sleep(0.4 * attempts)
                    continue
                raise
            except Exception:
                if attempts < 3:
                    await asyncio.sleep(0.4 * attempts)
                    continue
                raise

    async def drain_tts(current_gen: int):
        nonlocal speaking, tts_task
        if speaking or not tts_queue:
            return

        async def _run(my_gen: int):
            nonlocal speaking
            speaking = True
            try:
                while tts_queue:
                    # If barge-in happened, abort
                    if my_gen != gen_id:
                        return
                    text = _cap_sentence(tts_queue.popleft())
                    try:
                        audio = await tts_chunk(text)
                    except Exception as e:
                        await ws_send_json(client_ws, {
                            "type": "error", "source": "tts", "code": "ELEVENLABS_FAILED",
                            "message": "TTS request failed", "details": str(e)[:200]
                        })
                        return
                    # If client closed during TTS
                    try:
                        await client_ws.send_bytes(audio)
                        await ws_send_json(client_ws, {"type": "bot.audio.chunk", "bytes": len(audio)})
                    except Exception as e:
                        logger.info(f"Client send_bytes failed (likely closed): {e}")
                        return
            finally:
                speaking = False

        tts_task = asyncio.create_task(_run(current_gen))

    async def read_client():
        nonlocal bytes_up
        try:
            while True:
                data = await client_ws.receive()
                t = data.get("type")
                if t == "websocket.disconnect":
                    raise WebSocketDisconnect()

                if t == "websocket.receive":
                    if (b := data.get("bytes")):
                        if dg_ws and not dg_ws.closed:
                            with suppress(Exception):
                                await dg_ws.send_bytes(b)
                                bytes_up += len(b)  # count upstream bytes to Deepgram
                    elif (txt := data.get("text")):
                        with suppress(Exception):
                            obj = json.loads(txt)
                            if obj.get("type") == "barge-in":
                                await handle_barge_in()
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.warning(f"Client reader error: {e}")

    async def read_deepgram():
        nonlocal llm_buf
        try:
            async for m in dg_ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        evt = json.loads(m.data)
                    except Exception as e:
                        logger.warning(f"[Deepgram] JSON parse error: {e}")
                        continue

                    et = evt.get("type")  # e.g., "Results", "Error", "Warning", "Metadata"
                    logger.info(f"[Deepgram] event type: {et}")

                    # Handle explicit DG errors/warnings
                    if et in ("Error", "error"):
                        msg = evt.get("message") or evt
                        logger.error(f"[Deepgram] ERROR: {msg}")
                        await ws_send_json(client_ws, {"type": "error", "source": "asr",
                                                    "code": "DEEPGRAM_ERROR", "message": str(msg)[:300]})
                        continue
                    if et in ("Warning", "warning"):
                        msg = evt.get("message") or evt
                        logger.warning(f"[Deepgram] WARNING: {msg}")
                        await ws_send_json(client_ws, {"type": "warn", "code": "DEEPGRAM_WARNING", "message": str(msg)[:300]})
                        # keep going

                    # Normal transcript path (Deepgram "Results" shape)
                    ch = (evt.get("channel") or {})
                    alt0 = (ch.get("alternatives") or [{}])[0]
                    transcript = alt0.get("transcript") or ""
                    is_final = bool(evt.get("is_final") or evt.get("speech_final") or False)

                    # Show unknown-but-informative events in console (helps confirm DG is alive)
                    if not transcript and et not in ("Results", None):
                        await ws_send_json(client_ws, {"type": "asr.event", "eventType": et})
                        continue

                    if transcript:
                        logger.info(f"[ASR]{' (final)' if is_final else ''}: {transcript}")
                        await ws_send_json(client_ws, {"type": "asr.partial", "text": transcript, "final": is_final})

                    if transcript and is_final:
                        conf = (alt0.get("confidence") or None)
                        await ws_send_json(client_ws, {"type": "asr.final", "text": transcript, "confidence": conf})
                        if speaking:
                            await handle_barge_in()
                        llm_buf = ""
                        
                        # TEMP ECHO MODE: speak back what user said
                        tts_queue.append(transcript)
                        await drain_tts(gen_id)
                        
                        # Normal mode: send to OpenAI for response
                        # with suppress(Exception):
                        #     await oai_ws.send_str(jd({
                        #         "type": "response.create",
                        #         "response": {"instructions": transcript}
                        #     }))

                elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.warning(f"Deepgram reader error: {e}")
            await ws_send_json(client_ws, {"type": "error", "source": "asr", "code": "ASR_STREAM_FAILED",
                                        "message": "ASR stream ended unexpectedly"})


    async def read_openai():
        nonlocal llm_buf
        try:
            async for m in oai_ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        evt = json.loads(m.data)
                    except Exception as e:
                        logger.warning(f"OpenAI parse error: {e}")
                        continue

                    et = evt.get("type")
                    if et in ("response.text.delta", "response.output_text.delta"):
                        delta = evt.get("delta") or ""
                        await ws_send_json(client_ws, {"type": "bot.text.delta", "delta": delta})
                        # segment & enqueue into TTS
                        segs, llm_buf = _segment_sentences(llm_buf + delta)
                        for s in segs:
                            if len(tts_queue) >= TTS_QUEUE_MAX:
                                tts_queue.popleft()
                                await ws_send_json(client_ws, {"type": "warn", "code": "TTS_QUEUE_OVERFLOW"})
                            tts_queue.append(s)
                        await drain_tts(gen_id)
                        continue

                    if et in ("response.text.done", "response.done", "response.completed"):
                        tail = llm_buf.strip()
                        if tail:
                            if len(tts_queue) >= TTS_QUEUE_MAX:
                                tts_queue.popleft()
                                await ws_send_json(client_ws, {"type": "warn", "code": "TTS_QUEUE_OVERFLOW"})
                            tts_queue.append(tail)
                            llm_buf = ""
                        await drain_tts(gen_id)
                        continue

                    if et == "error":
                        await ws_send_json(client_ws, {"type": "error", "source": "openai", "code": "OPENAI_ERROR",
                                                        "message": evt.get("error") or "OpenAI error"})
                    else:
                        logger.debug(f"OpenAI event passthrough: {et}")
                elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.warning(f"OpenAI reader error: {e}")
            await ws_send_json(client_ws, {"type": "error", "source": "openai", "code": "OPENAI_STREAM_FAILED",
                                            "message": "OpenAI stream ended unexpectedly"})

    # --- Connection orchestration
    try:
        timeout = ClientTimeout(total=0, sock_connect=20, sock_read=120)
        session = aiohttp.ClientSession(timeout=timeout)

        # Retry upstream connects a few times if transient
        async def _retry(coro_factory, name: str, attempts: int = 3):
            delay = 0.5
            for i in range(1, attempts + 1):
                try:
                    return await coro_factory()
                except Exception as e:
                    if i == attempts:
                        raise
                    logger.info(f"{name} connect failed (attempt {i}): {e}; retrying…")
                    await asyncio.sleep(delay)
                    delay *= 1.6

        dg_ws = await _retry(lambda: connect_deepgram(session), "Deepgram")
        logger.info("[Deepgram] connected")

        oai_ws = await _retry(lambda: connect_openai(session), "OpenAI")
        logger.info("[OpenAI] connected")

        # Start ASR stats publisher
        stats_task = asyncio.create_task(publish_stats())

        # OPTIONAL auto-greeting so the bot speaks first
        if AUTO_GREETING:
            try:
                await oai_ws.send_str(jd({
                    "type": "response.create",
                    "response": {
                        "instructions": "Namaste! Main madad kar sakti hoon—kaunsi gaadi ya variant dekh rahe ho? (Short answer please.)"
                    }
                }))
            except Exception as e:
                logger.warning(f"OpenAI greeting failed: {e}")

        tasks = [
            asyncio.create_task(read_client()),
            asyncio.create_task(read_deepgram()),
            asyncio.create_task(read_openai()),
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        with suppress(Exception):
            await asyncio.gather(*pending, return_exceptions=True)

    except Exception as e:
        logger.error(f"Upstream orchestration error: {e}")
        await ws_send_json(client_ws, {"type": "error", "source": "server", "code": "UPSTREAM_CONNECT_FAILED",
                                        "message": "Failed to connect to upstream services", "details": str(e)[:200]})
    finally:
        await cleanup()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")
