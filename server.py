import os
import json
import asyncio
import logging
import base64
import uuid
from typing import Dict, Optional, Deque, List, Tuple
from collections import deque
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from aiohttp import ClientResponseError
from aiohttp import WSMsgType
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Environment & Config
# -----------------------------
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("server")

PORT = int(os.getenv("PORT", "3001"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g., https://your-app.onrender.com

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a concise, friendly sales assistant for an Indian auto dealership. "
    "Reply in the caller's language (Hindi/English/Hinglish), 1–2 short sentences, TTS-friendly."
)

# Deepgram (Telephony ASR)
DG_API_KEY = os.getenv("DG_API_KEY")
ASR_ENDPOINTING_MS = int(os.getenv("ASR_ENDPOINTING_MS", "800"))  # silence timeout for finals
ASR_LANGUAGE_HINT = (os.getenv("ASR_LANGUAGE_HINT") or "").strip()  # e.g., "en", "hi", "multi"

# EnableX Voice API
ENABLEX_APP_ID = os.getenv("ENABLEX_APP_ID")
ENABLEX_APP_KEY = os.getenv("ENABLEX_APP_KEY")
ENABLEX_BASE = os.getenv("ENABLEX_BASE", "https://api.enablex.io")

# TTS
USE_ELEVENLABS_TTS = (os.getenv("USE_ELEVENLABS_TTS", "false").lower() in ("1","true","yes"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
TTS_SENTENCE_MAX_CHARS = int(os.getenv("TTS_SENTENCE_MAX_CHARS", "280"))
TTS_QUEUE_MAX = int(os.getenv("TTS_QUEUE_MAX", "12"))
AUTO_GREETING = (os.getenv("AUTO_GREETING", "true").lower() in ("1","true","yes"))

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
# Simple in-memory stores
# -----------------------------
AUDIO_STORE: Dict[str, bytes] = {}  # id -> MP3 bytes for ElevenLabs playback via URL

class CallSession:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.playing: bool = False
        self.gen_id: int = 0              # increment to invalidate in-flight playback
        self.llm_buf: str = ""
        self.tts_queue: Deque[str] = deque()
        self.speaking_task: Optional[asyncio.Task] = None
        self.oai_session: Optional[ClientSession] = None
        self.extra: Dict[str, str] = {}   # free-form

SESSIONS: Dict[str, CallSession] = {}

# -----------------------------
# Utilities
# -----------------------------
END_PUNCT = ("!", "?", ".", "।")

def jd(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

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
# Health & Config check
# -----------------------------
@app.get("/health")
async def health():
    missing = []
    for k in ["PUBLIC_BASE_URL", "OPENAI_API_KEY", "DG_API_KEY", "ENABLEX_APP_ID", "ENABLEX_APP_KEY"]:
        if not globals().get(k):
            missing.append(k)
    if USE_ELEVENLABS_TTS:
        for k in ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]:
            if not globals().get(k):
                missing.append(k)
    return JSONResponse({"ok": len(missing) == 0, "missing": missing})

# -----------------------------
# Serve synthesized MP3s from memory via a stable URL
# -----------------------------
@app.get("/tts/{clip_id}.mp3")
async def serve_tts(clip_id: str):
    data = AUDIO_STORE.get(clip_id)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return StreamingResponse(iter([data]), media_type="audio/mpeg")

# -----------------------------
# EnableX Webhook: call lifecycle & control
# -----------------------------
@app.post("/enablex/webhook")
async def enablex_webhook(req: Request):
    try:
        payload = await req.json()
    except Exception:
        payload = {"raw": await req.body()}

    logger.info(f"[ENX EVT] {payload}")

    # Common fields seen in EnableX voice webhook payloads
    voice_id = payload.get("voice_id") or payload.get("callId") or payload.get("id")
    state = (payload.get("state") or payload.get("call_state") or "").lower()

    if not voice_id:
        return {"ok": True}

    session = await ensure_session(voice_id)

    # When call is answered/connected: start media stream to our WS endpoint
    if state in ("answered", "connected", "live"):
        wss_url = f"{PUBLIC_BASE_URL.replace('http','ws').replace('https','wss')}/enablex/stream?voice_id={voice_id}"
        await enablex_start_media_stream(voice_id, wss_url)
        if AUTO_GREETING:
            # Short greeting using built-in TTS (fast path)
            with suppress(Exception):
                await enablex_play_text(voice_id, "Namaste! Kaise madad kar sakti hoon? Ek chhota sa sawaal puchiye.")
        return {"ok": True}
    
    # EnableX-recognized webhook shape
    if (payload.get("state") or "").lower() == "recognized" and payload.get("text"):
        logger.info(f"[WEBHOOK] recognized: {payload['text']}")
        try:
            await handle_final_text(session, payload["text"])
        except Exception as e:
            logger.exception("[WEBHOOK] recognized->handle_final_text failed")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
        return {"ok": True}

    # You may receive playstate / ASR results here depending on project config.
    # If your tenant posts recognized text, handle it here similarly to Deepgram finals.
    rec_text = payload.get("recognized_text") or payload.get("asr_text")
    if rec_text:
        try:
            await handle_final_text(session, rec_text)
        except Exception as e:
            logger.exception("[WEBHOOK] handle_final_text failed")
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    return {"ok": True}

# -----------------------------
# EnableX media stream WebSocket (caller audio -> Deepgram telephony)
# -----------------------------
@app.websocket("/enablex/stream")
async def enablex_stream(ws: WebSocket):
    await ws.accept()
    params = ws.query_params
    voice_id = params.get("voice_id")
    if not voice_id:
        await ws.close(code=4000)
        return
    session = await ensure_session(voice_id)
    logger.info(f"[ENX WS] connected for voice_id={voice_id}")

    # Connect Deepgram with 8kHz telephony settings
    dg_params = {
        "model": "nova-3",
        "encoding": "mulaw",      # telephony μ-law
        "sample_rate": "8000",
        "channels": "1",
        "smart_format": "true",
        "interim_results": "true",
        "endpointing": str(ASR_ENDPOINTING_MS),
    }
    if ASR_LANGUAGE_HINT:
        dg_params["language"] = ASR_LANGUAGE_HINT

    qs = "&".join(f"{k}={v}" for k, v in dg_params.items())
    dg_url = f"wss://api.deepgram.com/v1/listen?{qs}"

    timeout = ClientTimeout(total=0, sock_connect=20, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout, headers={"Authorization": f"Token {DG_API_KEY}"}) as http:
        dg = await http.ws_connect(dg_url, heartbeat=30, autoping=True)

        async def pump_enx_to_dg():
            try:
                while True:
                    ev = await ws.receive()
                    t = ev.get("type")
                    if t == "websocket.disconnect":
                        break

                    # A) Text frames: JSON control (e.g., {"state":"start_media"}) or recognized text
                    if ev.get("text") is not None:
                        try:
                            obj = json.loads(ev["text"])
                        except Exception:
                            continue

                        # control events from EnableX, e.g. {"state":"start_media"}
                        st = (obj.get("state") or "").lower()
                        if st == "start_media":
                            logger.info("[ENX] start_media received")
                            continue

                        # some tenants post recognized text via WS JSON
                        if st == "recognized" and obj.get("text"):
                            # treat as a final utterance
                            if session.playing:
                                with suppress(Exception):
                                    await enablex_stop_play(session.voice_id)
                                session.playing = False
                            await handle_final_text(session, obj["text"])
                            continue

                        # optional: other JSON formats carrying base64 audio
                        payload_b64 = (
                            obj.get("payload")
                            or (obj.get("media") or {}).get("payload")
                            or (obj.get("audio") or {}).get("data")
                            or obj.get("data")
                        )
                        if payload_b64:
                            try:
                                audio_bytes = base64.b64decode(payload_b64)
                                await dg.send_bytes(audio_bytes)
                            except Exception as e:
                                logger.warning(f"[ENX->DG] JSON payload decode failed: {e}")
                        continue

                    # B) Binary frames: 12-byte header + base64 μ-law (per EnableX example)
                    if ev.get("bytes") is not None:
                        buf = ev["bytes"]
                        payload = buf[12:] if len(buf) > 12 else buf
                        try:
                            # payload is base64 ascii; python can b64-decode bytes directly
                            ulaw_bytes = base64.b64decode(payload, validate=False)
                        except Exception:
                            # if not base64, try as raw μ-law
                            ulaw_bytes = payload
                        try:
                            await dg.send_bytes(ulaw_bytes)
                        except Exception as e:
                            logger.warning(f"[ENX->DG] send to Deepgram failed: {e}")
                        continue
            except WebSocketDisconnect:
                pass
            finally:
                with suppress(Exception):
                    await dg.close()


        async def pump_dg_to_bot():
            try:
                async for m in dg:
                    if m.type == WSMsgType.TEXT:
                        try:
                            evt = json.loads(m.data)
                        except Exception:
                            continue
                        et = evt.get("type")
                        if et == "Results":
                            alt0 = (evt.get("channel", {}).get("alternatives") or [{}])[0]
                            transcript = alt0.get("transcript") or ""
                            is_final = bool(evt.get("is_final") or evt.get("speech_final"))
                            if transcript:
                                logger.info(f"[ASR]{' (final)' if is_final else ''}: {transcript}")
                            if transcript and is_final:
                                # Barge-in: stop any current play
                                if session.playing:
                                    with suppress(Exception):
                                        await enablex_stop_play(session.voice_id)
                                    session.playing = False
                                await handle_final_text(session, transcript)
                    elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
            except Exception as e:
                logger.warning(f"[DG] reader error: {e}")

        await asyncio.gather(pump_enx_to_dg(), pump_dg_to_bot())

# -----------------------------
# Core bot logic: LLM → TTS → EnableX play
# -----------------------------
async def handle_final_text(session: CallSession, user_text: str):
    """Handle a finalized user utterance: call LLM, speak back via EnableX."""
    logger.info(f"[BOT] handle_final_text: {user_text}")

    # 1) LLM reply (short, sentence-level)
    reply_text = await llm_reply(user_text)

    # 2) Sentence chunking for early-start playback
    segs, tail = _segment_sentences(reply_text)
    if tail.strip():
        segs.append(tail.strip())

    # 3) Enqueue & start playback
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
                        # Respect Retry-After if present, else backoff
                        ra = r.headers.get("Retry-After")
                        wait = float(ra) if ra else delay
                        if i == attempts:
                            # surface the last response as error json for logging context
                            return {"_error": True, "status": r.status, "text": await r.text()}
                        await asyncio.sleep(wait)
                        delay *= 1.7  # exponential-ish
                        continue
                    r.raise_for_status()
                    return await r.json()
        except Exception as e:
            if i == attempts:
                raise
            await asyncio.sleep(delay + (0.1 * i))  # small jitter
            delay *= 1.7
    # Shouldn’t get here
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
        "temperature": 0.4,   # slightly lower to be more deterministic
        "max_tokens": 96,     # smaller responses start faster
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        data = await _openai_with_retries(url, body, headers, attempts=3)
        if data.get("_error"):
            logger.error(f"[LLM] OpenAI error after retries: {data}")
            return "Sorry, I’m having trouble right now—please try again."  # single sentence (no split)
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        logger.exception("[LLM] OpenAI chat completion failed")
        # Keep it to a **single sentence** so segmentation doesn’t create two plays
        return "Sorry, I’m having trouble right now—please try again."

async def play_sentence(session: CallSession, text: str):
    voice_id = session.voice_id
    session.gen_id += 1
    my_gen = session.gen_id

    if USE_ELEVENLABS_TTS:
        # Path B: synth with ElevenLabs -> serve via /tts/<id>.mp3 -> EnableX /play with URL
        clip_id, url = await elevenlabs_synth_to_url(text)
        if my_gen != session.gen_id:
            return  # invalidated by barge-in
        await enablex_play_url(voice_id, url)
        session.playing = True
    else:
        # Path A: fast path (EnableX built-in TTS)
        await enablex_play_text(voice_id, text)
        session.playing = True

# -----------------------------
# EnableX helpers
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

async def enablex_play_text(voice_id: str, text: str, language: str = "en-IN", voice: str = "female"):
    url = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
    body = {"text": text, "language": language, "voice": voice}
    async with aiohttp.ClientSession(auth=enx_auth()) as s:
        r = await s.put(url, json=body)
        txt = await r.text()
        if r.status >= 400:
            logger.error(f"[ENX PLAY TEXT] {r.status} {txt}")
        else:
            logger.info(f"[ENX PLAY TEXT] {r.status} {txt}")

async def enablex_play_url(voice_id: str, media_url: str):
    """Play a remote audio URL. Some tenants use `url`, others require a specific key.
    If your tenant only supports stored prompts, upload first and use prompt_name instead.
    """
    url = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
    # Try common shapes: {"url": ...} then fallback to {"prompt_url": ...}
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
        logger.error("[ENX PLAY URL] All body variants failed; check your tenant's OpenAPI for the correct field name.")

async def enablex_stop_play(voice_id: str):
    # Attempt common stop controls: direct stop endpoint, or command body
    async with aiohttp.ClientSession(auth=enx_auth()) as s:
        # Variant 1: dedicated stop endpoint
        url1 = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play/stop"
        r1 = await s.put(url1)
        if r1.status < 400:
            logger.info(f"[ENX STOP PLAY] {r1.status} {await r1.text()}")
            return
        # Variant 2: command in body
        url2 = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
        r2 = await s.put(url2, json={"command": "stop"})
        logger.info(f"[ENX STOP PLAY fallback] {r2.status} {await r2.text()}")

# -----------------------------
# Local fake EnableX endpoints (self-mock for testing)
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
# ElevenLabs synth → URL
# -----------------------------
async def elevenlabs_synth_to_url(text: str) -> Tuple[str, str]:
    if not (ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        raise RuntimeError("Missing ElevenLabs config")

    api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    params = {"optimize_streaming_latency": "3", "output_format": "mp3_22050_32"}
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.35, "similarity_boost": 0.9, "style": 0.65, "use_speaker_boost": True}
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
                    mp3 = b"".join(bufs)
                    clip_id = uuid.uuid4().hex
                    AUDIO_STORE[clip_id] = mp3
                    base = PUBLIC_BASE_URL or f"http://localhost:{PORT}"
                    url = f"{base}/tts/{clip_id}.mp3"
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

# https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream?output_format=ulaw_8000`;