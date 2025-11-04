import os
import json
import asyncio
import logging
import base64
import uuid
from typing import Dict, Optional, Deque, List, Tuple
from collections import deque
from contextlib import suppress

from fastapi import FastAPI, WebSocket, Request, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from aiohttp import ClientResponseError, WSMsgType
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Environment & Config
# -----------------------------
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("server")

PORT = int(os.getenv("PORT", "3001"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g., https://your-app.example.com

# Echo mode (bypass LLM and speak back user's words)
ECHO_MODE = (os.getenv("ECHO_MODE", "false").lower() in ("1","true","yes"))
ECHO_STRICT_NO_SPLIT = (os.getenv("ECHO_STRICT_NO_SPLIT", "true").lower() in ("1","true","yes"))

# OpenAI (optional if not using LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a concise, friendly sales assistant for an Indian auto dealership. "
    "Reply in the caller's language (Hindi/English/Hinglish), 1–2 short sentences, TTS-friendly."
)

# Deepgram (Telephony ASR)
DG_API_KEY = os.getenv("DG_API_KEY")
ASR_ENDPOINTING_MS = int(os.getenv("ASR_ENDPOINTING_MS", "800"))
ASR_LANGUAGE_HINT = (os.getenv("ASR_LANGUAGE_HINT") or "").strip()

# EnableX Voice API
ENABLEX_APP_ID = os.getenv("ENABLEX_APP_ID")
ENABLEX_APP_KEY = os.getenv("ENABLEX_APP_KEY")
ENABLEX_BASE = os.getenv("ENABLEX_BASE", "https://api.enablex.io")

# ElevenLabs (TTS) — now MANDATORY
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
AUTO_GREETING = (os.getenv("AUTO_GREETING", "true").lower() in ("1","true","yes"))
TTS_SENTENCE_MAX_CHARS = int(os.getenv("TTS_SENTENCE_MAX_CHARS", "280"))

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------
# In-memory store for generated audio
# -----------------------------
AUDIO_STORE: Dict[str, bytes] = {}  # id -> audio bytes

class CallSession:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.playing: bool = False
        self.gen_id: int = 0
        self.llm_buf: str = ""
        self.tts_queue: Deque[str] = deque()
        self.speaking_task: Optional[asyncio.Task] = None
        self.oai_session: Optional[ClientSession] = None
        self.extra: Dict[str, str] = {}

SESSIONS: Dict[str, CallSession] = {}

# -----------------------------
# Utilities
# -----------------------------
END_PUNCT = ("!", "?", ".", "।")

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

async def ensure_session(voice_id: str) -> CallSession:
    if voice_id not in SESSIONS:
        SESSIONS[voice_id] = CallSession(voice_id)
    return SESSIONS[voice_id]

# -----------------------------
# Health (now requires ElevenLabs keys)
# -----------------------------
@app.get("/health")
async def health():
    missing = []
    for k in ["PUBLIC_BASE_URL", "DG_API_KEY", "ENABLEX_APP_ID", "ENABLEX_APP_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]:
        if not globals().get(k):
            missing.append(k)
    # OpenAI is optional unless ECHO_MODE is false and you want LLM replies
    if not ECHO_MODE and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY (optional unless using LLM)")
    return JSONResponse({"ok": len(missing) == 0, "echo_mode": ECHO_MODE, "missing": missing})

# -----------------------------
# Serve synthesized audio (µ-law 8k) as audio/basic
# -----------------------------
from fastapi.responses import Response

@app.get("/tts/{clip_id}.ulaw")
async def serve_tts_ulaw(clip_id: str):
    data = AUDIO_STORE.get(clip_id)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")

    # Most telephony stacks expect 'audio/basic' for G.711 μ-law
    # Some use 'audio/ulaw' — if ENX complains, try that instead.
    return Response(
        content=data,
        media_type="audio/basic",
        headers={
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store"
        }
    )

# -----------------------------
# EnableX Webhook
# -----------------------------
@app.post("/enablex/webhook")
async def enablex_webhook(req: Request):
    try:
        payload = await req.json()
    except Exception:
        payload = {"raw": await req.body()}

    logger.info(f"[ENX EVT] {payload}")

    voice_id = payload.get("voice_id") or payload.get("callId") or payload.get("id")
    state = (payload.get("state") or payload.get("call_state") or "").lower()
    if not voice_id:
        return {"ok": True}

    session = await ensure_session(voice_id)

    if state in ("answered", "connected", "live"):
        # Start media stream
        wss_url = f"{PUBLIC_BASE_URL.replace('http','ws').replace('https','wss')}/enablex/stream?voice_id={voice_id}"
        await enablex_start_media_stream(voice_id, wss_url)

        # Optional EL greeting (no built-in TTS)
        if AUTO_GREETING:
            with suppress(Exception):
                await play_sentence(session, "Namaste! Kaise madad kar sakti hoon? Ek chhota sa sawaal puchiye.")
        return {"ok": True}

    # If tenant posts recognized text here, handle it
    if (payload.get("state") or "").lower() == "recognized" and payload.get("text"):
        try:
            await handle_final_text(session, payload["text"])
        except Exception:
            logger.exception("[WEBHOOK] recognized->handle_final_text failed")
        return {"ok": True}

    rec_text = payload.get("recognized_text") or payload.get("asr_text")
    if rec_text:
        try:
            await handle_final_text(session, rec_text)
        except Exception:
            logger.exception("[WEBHOOK] handle_final_text failed")

    return {"ok": True}

# --- Deepgram keepalive ping ---
async def _dg_keepalive(ws, interval: int = 5):
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_str('{"type":"KeepAlive"}')
    except Exception:
        pass

# -----------------------------
# ENX media stream WS → Deepgram
# -----------------------------
from aiohttp import WSServerHandshakeError

@app.websocket("/enablex/stream")
async def enablex_stream(ws: WebSocket):
    await ws.accept()
    params = ws.query_params
    voice_id: Optional[str] = params.get("voice_id")
    session: Optional[CallSession] = None
    if voice_id:
        session = await ensure_session(voice_id)

    logger.info(f"[ENX WS] connected path={ws.url.path} voice_id={voice_id or 'unknown'}")

    dg_params = {
        "model": "nova-3",
        "encoding": "mulaw",
        "sample_rate": "8000",
        "channels": "1",
        "interim_results": "true",
        "smart_format": "true",
        "endpointing": str(ASR_ENDPOINTING_MS),
    }
    qs = "&".join(f"{k}={v}" for k, v in dg_params.items())
    dg_url = f"wss://api.deepgram.com/v1/listen?{qs}"

    timeout = ClientTimeout(total=0, sock_connect=20, sock_read=120)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers={"Authorization": f"Token {DG_API_KEY}"}) as http:
            try:
                dg = await http.ws_connect(dg_url, heartbeat=30, autoping=True, compress=0)
                logger.info(f"[DG] connected: {dg_url}")
                ka_task = asyncio.create_task(_dg_keepalive(dg))
            except WSServerHandshakeError as e:
                logger.error(f"[DG] handshake failed {e.status} {e.message}")
                await ws.close(code=1011)
                return
            except Exception:
                logger.exception("[DG] connect failed")
                await ws.close(code=1011)
                return

            dg_closing = False

            async def pump_enx_to_dg():
                nonlocal session, voice_id, dg_closing
                total_bytes = 0
                try:
                    while True:
                        ev = await ws.receive()

                        if ev.get("type") == "websocket.disconnect":
                            logger.info("[ENX WS] disconnect")
                            break

                        if ev.get("text") is not None:
                            try:
                                obj = json.loads(ev["text"])
                            except Exception:
                                logger.warning("[ENX->DG] non-JSON text ignored")
                                continue

                            call_id = obj.get("voice_id") or obj.get("callId") or obj.get("id")
                            if call_id and not session:
                                voice_id = call_id
                                session = await ensure_session(voice_id)
                                logger.info(f"[ENX WS] bound voice_id from JSON: {voice_id}")

                            st = (obj.get("state") or "").lower()
                            if st == "start_media":
                                logger.info("[ENX] start_media received")
                                continue

                            if st == "recognized" and obj.get("text"):
                                if session and session.playing:
                                    with suppress(Exception):
                                        await enablex_stop_play(session.voice_id)
                                    session.playing = False
                                if session:
                                    await handle_final_text(session, obj["text"])
                                else:
                                    logger.warning("[ENX] recognized but no session; dropping")
                                continue

                            payload_b64 = (
                                obj.get("payload")
                                or (obj.get("media") or {}).get("payload")
                                or (obj.get("audio") or {}).get("data")
                                or obj.get("data")
                            )
                            if payload_b64:
                                try:
                                    if dg_closing or dg.closed:
                                        return
                                    audio_bytes = base64.b64decode(payload_b64)
                                    if not audio_bytes:
                                        continue
                                    total_bytes += len(audio_bytes)
                                    if total_bytes % 32768 < 1600:
                                        logger.info(f"[ENX->DG] +{len(audio_bytes)}B (total {total_bytes}B)")
                                    await dg.send_bytes(audio_bytes)
                                except Exception as e:
                                    logger.warning(f"[ENX->DG] JSON payload decode failed: {e}")
                            continue

                        if ev.get("bytes") is not None:
                            if dg_closing or dg.closed:
                                return
                            buf = ev["bytes"]
                            ulaw_bytes = buf[12:] if len(buf) > 12 else buf
                            if not ulaw_bytes:
                                continue
                            total_bytes += len(ulaw_bytes)
                            if total_bytes % 32768 < 1600:
                                logger.info(f"[ENX->DG] +{len(ulaw_bytes)}B (total {total_bytes}B)")
                            try:
                                await dg.send_bytes(ulaw_bytes)
                            except Exception as e:
                                logger.warning(f"[ENX->DG] send to Deepgram failed: {e}")
                            continue

                finally:
                    with suppress(Exception):
                        await dg.close()

            async def pump_dg_to_bot():
                nonlocal dg_closing
                try:
                    async for m in dg:
                        if m.type == WSMsgType.TEXT:
                            try:
                                evt = json.loads(m.data)
                            except Exception:
                                continue

                            t = evt.get("type")
                            if t in ("Warning", "Error"):
                                logger.warning(f"[DG] {t}: {evt}")
                            if t == "Metadata":
                                logger.info(f"[DG] metadata: {evt.get('request_id')}")
                            if t == "Results":
                                alt0 = (evt.get("channel", {}).get("alternatives") or [{}])[0]
                                transcript = alt0.get("transcript") or ""
                                is_final = bool(evt.get("is_final") or evt.get("speech_final"))
                                if transcript:
                                    logger.info(f"[ASR]{' (final)' if is_final else ''}: {transcript}")
                                if transcript and is_final and session:
                                    if session.playing:
                                        with suppress(Exception): await enablex_stop_play(session.voice_id)
                                        session.playing = False
                                    await handle_final_text(session, transcript)

                        elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                            dg_closing = True
                            try:
                                logger.warning(f"[DG] closing: code={dg.close_code} reason={getattr(dg, 'close_message', None)}")
                            except Exception:
                                logger.warning("[DG] closing")
                            break
                except Exception as e:
                    dg_closing = True
                    logger.warning(f"[DG] reader error: {e}")

            await asyncio.gather(pump_enx_to_dg(), pump_dg_to_bot())
            with suppress(Exception):
                ka_task.cancel()

    except Exception:
        logger.exception("[ENX WS] handler crashed")
        with suppress(Exception):
            await ws.close(code=1011)

@app.websocket("/ws")
async def enablex_stream_alias(ws: WebSocket):
    await enablex_stream(ws)

# -----------------------------
# Core bot logic: Echo/LLM → ElevenLabs → ENX play URL
# -----------------------------
async def handle_final_text(session: CallSession, user_text: str):
    logger.info(f"[BOT] handle_final_text (echo_mode={ECHO_MODE}): {user_text}")

    if ECHO_MODE:
        reply_text = user_text
    else:
        reply_text = await llm_reply(user_text)

    # No EnableX built-in TTS anywhere — always ElevenLabs
    if ECHO_MODE and ECHO_STRICT_NO_SPLIT:
        segs = [reply_text]
    else:
        segs, tail = _segment_sentences(reply_text)
        if tail.strip():
            segs.append(tail.strip())

    for s in segs:
        if len(s) > TTS_SENTENCE_MAX_CHARS:
            s = s[:TTS_SENTENCE_MAX_CHARS] + "…"
        await play_sentence(session, s)

async def _openai_with_retries(url: str, body: dict, headers: dict, attempts: int = 3) -> dict:
    delay = 0.5
    for i in range(1, attempts + 1):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=body, headers=headers) as r:
                    if r.status == 429:
                        ra = r.headers.get("Retry-After")
                        wait = float(ra) if ra else delay
                        if i == attempts:
                            return {"_error": True, "status": r.status, "text": await r.text()}
                        await asyncio.sleep(wait); delay *= 1.7
                        continue
                    r.raise_for_status()
                    return await r.json()
        except Exception:
            if i == attempts: raise
            await asyncio.sleep(delay + (0.1 * i)); delay *= 1.7
    return {"_error": True, "status": 500, "text": "exhausted"}

async def llm_reply(user_text: str) -> str:
    if not OPENAI_API_KEY:
        return "Sorry, I’m having trouble right now—please try again."
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.4,
        "max_tokens": 96,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        data = await _openai_with_retries(url, body, headers, attempts=3)
        if data.get("_error"):
            logger.error(f"[LLM] OpenAI error after retries: {data}")
            return "Sorry, I’m having trouble right now—please try again."
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        logger.exception("[LLM] OpenAI chat completion failed")
        return "Sorry, I’m having trouble right now—please try again."

async def play_sentence(session: CallSession, text: str):
    voice_id = session.voice_id
    session.gen_id += 1
    my_gen = session.gen_id

    clip_id, url = await elevenlabs_synth_to_url(text)
    if my_gen != session.gen_id:
        return  # invalidated by barge-in
    await enablex_play_url(voice_id, url)
    session.playing = True

# -----------------------------
# EnableX helpers (URL playback only — no built-in TTS)
# -----------------------------
def enx_auth() -> BasicAuth:
    if not (ENABLEX_APP_ID and ENABLEX_APP_KEY):
        raise RuntimeError("Missing EnableX credentials")
    return BasicAuth(ENABLEX_APP_ID, ENABLEX_APP_KEY)

async def enablex_start_media_stream(voice_id: str, stream_wss_url: str):
    url = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/stream"
    body = {"stream_dest": stream_wss_url}
    async with aiohttp.ClientSession(auth=enx_auth()) as s:
        r = await s.put(url, json=body)
        txt = await r.text()
        if r.status >= 400:
            logger.error(f"[ENX START STREAM] {r.status} {txt}")
        else:
            logger.info(f"[ENX START STREAM] {r.status} {txt}")

async def enablex_play_url(voice_id: str, media_url: str):
    url = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
    body_candidates = [
        {"url": media_url},
        {"prompt_url": media_url},
    ]
    async with aiohttp.ClientSession(auth=enx_auth()) as s:
        for body in body_candidates:
            r = await s.put(url, json=body)
            txt = await r.text()
            if r.status < 400:
                logger.info(f"[ENX PLAY URL] {r.status} {txt}")
                return
            logger.warning(f"[ENX PLAY URL attempt] {r.status} {txt} with body={body}")
        logger.error("[ENX PLAY URL] All body variants failed; check tenant OpenAPI for the correct field.")

async def enablex_stop_play(voice_id: str):
    async with aiohttp.ClientSession(auth=enx_auth()) as s:
        url1 = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play/stop"
        r1 = await s.put(url1)
        if r1.status < 400:
            logger.info(f"[ENX STOP PLAY] {r1.status} {await r1.text()}")
            return
        url2 = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
        r2 = await s.put(url2, json={"command": "stop"})
        logger.info(f"[ENX STOP PLAY fallback] {r2.status} {await r2.text()}")

# -----------------------------
# Local mock endpoints (optional)
# -----------------------------
@app.put("/__mock/voice/v1/call/{voice_id}/stream")
async def _mock_stream(voice_id: str, body: dict = Body(...)):
    logger.info(f"[MOCK ENX STREAM] voice_id={voice_id} body={body}")
    return {"status": "ok", "type": "stream.started", "voice_id": voice_id, "body": body}

@app.put("/__mock/voice/v1/call/{voice_id}/play")
async def _mock_play(voice_id: str, body: dict = Body(...)):
    logger.info(f"[MOCK ENX PLAY] voice_id={voice_id} body={body}")
    return {"status": "ok", "type": "play.accepted", "voice_id": voice_id, "body": body}

@app.put("/__mock/voice/v1/call/{voice_id}/play/stop")
async def _mock_play_stop(voice_id: str):
    logger.info(f"[MOCK ENX PLAY STOP] voice_id={voice_id}")
    return {"status": "ok", "type": "play.stopped", "voice_id": voice_id}

# -----------------------------
# ElevenLabs synth → µ-law URL
# -----------------------------
async def elevenlabs_synth_to_url(text: str) -> Tuple[str, str]:
    if not (ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        raise RuntimeError("Missing ElevenLabs config")
    api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

    # IMPORTANT: telephony-friendly format
    params = {"optimize_streaming_latency": "3", "output_format": "ulaw_8000"}

    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": 
        {
            "stability": 0.35, 
            "similarity_boost": 0.9, 
            "style": 0.65, 
            "use_speaker_boost": True
        }
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}

    attempts = 0
    while True:
        attempts += 1
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(api_url, params=params, json=body, headers=headers) as resp:
                    resp.raise_for_status()
                    bufs = [chunk async for chunk in resp.content.iter_chunked(4096)]
                    ulaw = b"".join(bufs)
                    clip_id = uuid.uuid4().hex
                    AUDIO_STORE[clip_id] = ulaw
                    base = PUBLIC_BASE_URL or f"http://localhost:{PORT}"
                    url  = f"{base}/tts/{clip_id}.ulaw"   # <- return a .ulaw URL
                    logger.info(f"[TTS] EL clip={clip_id} bytes={len(ulaw)} url=/tts/{clip_id}.ulaw")
                    return clip_id, url
        except ClientResponseError as cre:
            if cre.status == 429 and attempts < 3:
                await asyncio.sleep(0.4 * attempts)
                continue
            raise
        except Exception:
            if attempts < 3:
                await asyncio.sleep(0.4 * attempts)
                continue
            raise

# -----------------------------
# Root
# -----------------------------
@app.get("/")
async def root():
    return PlainTextResponse("EnableX Voice Bot server is running.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL is not set; remote playback via URL will fail.")
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
