
import os
import json
import asyncio
import logging
import base64
import uuid
from typing import Dict, Optional, Deque, List, Tuple
from collections import deque
from contextlib import suppress

# Latency Calculation
import time

from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiohttp
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from aiohttp import ClientResponseError
from aiohttp import WSMsgType
from system_prompt import system_prompt

from dotenv import load_dotenv

load_dotenv()

from downlink_ulaw import ULAW_STORE, elevenlabs_ulaw_to_url
# -----------------------------
# Environment & Config
# -----------------------------
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("server")

PORT = int(os.getenv("PORT", "3001"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g., https://your-app.onrender.com

# Echo mode (bypass LLM and speak back user's words)
ECHO_MODE = (os.getenv("ECHO_MODE", "false").lower() in ("1","true","yes"))
# If True, don't split the echoed text into sentences—play exactly what user said
ECHO_STRICT_NO_SPLIT = (os.getenv("ECHO_STRICT_NO_SPLIT", "true").lower() in ("1","true","yes"))
# Optional: echo interims too (usually NO, gets choppy)
ECHO_INTERIMS = (os.getenv("ECHO_INTERIMS", "false").lower() in ("1","true","yes"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# SYSTEM_PROMPT = (
#     "You are a concise, friendly sales assistant for an Indian auto dealership. "
#     "Reply in the caller's language (Hindi/English/Hinglish), 1–2 short sentences, TTS-friendly."
# )

SYSTEM_PROMPT = system_prompt

# Deepgram (Telephony ASR)
DG_API_KEY = os.getenv("DG_API_KEY")
ASR_ENDPOINTING_MS = int(os.getenv("ASR_ENDPOINTING_MS", "120"))  # silence timeout for finals
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
TTS_SENTENCE_MAX_CHARS = int(os.getenv("TTS_SENTENCE_MAX_CHARS", "120"))
TTS_QUEUE_MAX = int(os.getenv("TTS_QUEUE_MAX", "12"))
AUTO_GREETING = (os.getenv("AUTO_GREETING", "true").lower() in ("1","true","yes"))

# Debugging
DEBUG_VOICE = (os.getenv("DEBUG_VOICE", "false").lower() in ("1", "true", "yes"))
DEBUG_TTS_DUMP_DIR = os.getenv("DEBUG_TTS_DUMP_DIR", "")

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
PREGEN_ULAW: Dict[str, bytes] = {}

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

        self.ws: Optional[WebSocket] = None   # active EnableX websocket
        self.stream_id: Optional[str] = None # from start_media event
        self.out_seq: int = 0                # sequence for media we send

        # NEW: purely in-memory dialogue
        self.turn: int = 0
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.last_final: str = "" 
        self.send_lock = asyncio.Lock() 

        # --- Latency metrics (all in seconds; we convert to ms when logging) ---
        self.asr_last_audio_ts: float = 0.0   # when we last forwarded caller audio to ASR
        self.asr_last_final_ts: float = 0.0   # when we got a final transcript

        self.llm_start_ts: float = 0.0
        self.llm_end_ts: float = 0.0

        self.tts_start_ts: float = 0.0
        self.tts_end_ts: float = 0.0

        self.downlink_start_ts: float = 0.0
        self.downlink_end_ts: float = 0.0

SESSIONS: Dict[str, CallSession] = {}

# -----------------------------
# Utilities
# -----------------------------
END_PUNCT = ("!", "?", ".", "।")

async def wait_for_stream_id(session: CallSession, timeout: float = 2.0):
    """
    Wait a bit for EnableX to send start_media so stream_id is available.
    Prevents early downlink failures.
    """
    t0 = asyncio.get_event_loop().time()
    while not session.stream_id and (asyncio.get_event_loop().time() - t0) < timeout:
        await asyncio.sleep(0.05)

def _trim_history_for_budget(history: List[Dict[str, str]], max_chars: int = 8000) -> List[Dict[str, str]]:
    if not history:
        return []
    sys = [m for m in history if m["role"] == "system"][:1]
    rest = [m for m in history if m["role"] != "system"]
    total = sum(len(m["content"]) for m in sys)
    kept = []
    for m in reversed(rest):
        c = len(m["content"])
        if kept and total + c > max_chars:
            break
        kept.append(m)
        total += c
    kept.reverse()
    return (sys + kept) if sys else kept

def _is_simple_greeting(text: str) -> bool:
    t = text.strip().lower()
    # tweak this list as you like
    GREETINGS = {
        "hello", "hi", "hii", "hey", "helo",
        "hello.", "hi.", "hey.", "haan", "haan ji", "yes", "ya", "yeah"
    }
    return t in GREETINGS

# async def llm_reply_with_history(session: CallSession) -> str:
#     if not OPENAI_API_KEY:
#         return "Sorry, I’m having trouble right now—please try again."

#     messages = _trim_history_for_budget(session.history, max_chars=8000)
#     body = {
#         "model": OPENAI_CHAT_MODEL,
#         "messages": messages,
#         "temperature": 0.4,
#         "max_tokens": 120,
#     }
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     data = await _openai_with_retries("https://api.openai.com/v1/chat/completions", body, headers, attempts=3)
#     if data.get("_error"):
#         logger.error(f"[LLM] OpenAI error after retries: {data}")
#         return "Sorry, I’m having trouble right now—please try again."
#     return (data["choices"][0]["message"]["content"] or "").strip()

async def llm_reply_with_history(session: CallSession, user_text: str) -> str:
    """
    Streaming LLM call using the shared app.state.http ClientSession.

    - Sends full chat history (trimmed) with `stream: True`.
    - Reads "data: {...}" SSE lines as they arrive.
    - Logs:
        * LLM ttfb_ms  -> time to first token
        * LLM latency_ms -> time until full answer is done
    - Returns the full concatenated reply text (for now we still TTS once at the end).
    """
    if not OPENAI_API_KEY:
        return "Sorry, I’m having trouble right now—please try again."

    # 1) Build messages for this turn (system + prior history)
    messages = _trim_history_for_budget(session.history, max_chars=8000)

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 48,
        "stream": True,  # streaming mode
    }

    # We'll collect all text pieces here
    chunks: List[str] = []

    # Timing
    session.llm_start_ts = time.perf_counter()
    first_chunk_ts: Optional[float] = None

    # 2) Get the shared HTTP session from app.state.http
    http: Optional[ClientSession] = getattr(getattr(app, "state", None), "http", None)

    # Fallback: if for some reason startup hasn't created it (e.g. in unit tests),
    # create a local temporary session and close it at the end.
    created_here = False
    if http is None:
        http = aiohttp.ClientSession(timeout=ClientTimeout(total=0))
        created_here = True

    try:
        # 3) Open the streaming request using the shared session
        async with http.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()

            # Read raw bytes lines as they arrive
            async for raw_line in resp.content:
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line:
                    continue

                # OpenAI SSE format: lines starting with "data: "
                if not line.startswith(b"data:"):
                    # Could be comments/heartbeats, ignore
                    continue

                data_bytes = line[len(b"data:"):].strip()

                # End-of-stream marker
                if data_bytes == b"[DONE]":
                    break

                # Parse JSON chunk
                try:
                    payload = json.loads(data_bytes.decode("utf-8"))
                except Exception:
                    continue

                choices = payload.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if not piece:
                    continue

                # Append piece to our growing text
                chunks.append(piece)

                # First token -> log TTFB
                if first_chunk_ts is None:
                    first_chunk_ts = time.perf_counter()
                    ttfb_ms = (first_chunk_ts - session.llm_start_ts) * 1000.0
                    logger.info(
                        "[METRIC] LLM ttfb_ms=%.1f turn=%d user_text=%r",
                        ttfb_ms,
                        session.turn,
                        user_text,
                    )

        # 4) Stream finished: compute total LLM latency
        session.llm_end_ts = time.perf_counter()
        llm_ms = (session.llm_end_ts - session.llm_start_ts) * 1000.0

        reply_text = "".join(chunks).strip()
        if not reply_text:
            reply_text = "Sorry, I’m having trouble right now—please try again."

        logger.info(
            "[METRIC] LLM latency_ms=%.1f turn=%d user_text=%r",
            llm_ms,
            session.turn,
            user_text,
        )

        return reply_text

    except Exception:
        # If streaming fails, fall back to the old non-streaming helper
        logger.exception("[LLM] streaming failed, falling back to non-streaming call")

        body_fallback = {
            "model": OPENAI_CHAT_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 48,
        }

        session.llm_start_ts = time.perf_counter()
        data = await _openai_with_retries(url, body_fallback, headers, attempts=3)
        session.llm_end_ts = time.perf_counter()
        llm_ms = (session.llm_end_ts - session.llm_start_ts) * 1000.0

        if data.get("_error"):
            logger.error(f"[LLM] OpenAI error after retries (fallback): {data}")
            return "Sorry, I’m having trouble right now—please try again."

        logger.info(
            "[METRIC] LLM latency_ms=%.1f turn=%d user_text=%r",
            llm_ms,
            session.turn,
            user_text,
        )

        return (data["choices"][0]["message"]["content"] or "").strip()

    finally:
        # Only close the HTTP session if we created a temporary one here
        if created_here:
            with suppress(Exception):
                await http.close()


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

def _print_session_history(session: CallSession):
    # Skip the system message; print only user/assistant turns
    lines = []
    for m in session.history:
        if m["role"] == "system":
            continue
        who = "USER" if m["role"] == "user" else "BOT "
        lines.append(f"{who}: {m['content']}")
    log = "\n".join(lines) if lines else "(no dialogue)"
    logger.info("\n====== DIALOGUE (voice_id=%s, turns=%d) ======\n%s\n================== END ==================\n",
                session.voice_id, session.turn, log)

# -----------------------------
# Health & Config check
# -----------------------------
@app.get("/health")
async def health():
    missing = []
    # Always require these:
    for k in ["PUBLIC_BASE_URL", "DG_API_KEY", "ENABLEX_APP_ID", "ENABLEX_APP_KEY"]:
        if not globals().get(k):
            missing.append(k)

    # Only require OpenAI when NOT in echo mode
    if not ECHO_MODE and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    # ElevenLabs only if you chose that path
    if USE_ELEVENLABS_TTS:
        for k in ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]:
            if not globals().get(k):
                missing.append(k)

    return JSONResponse({"ok": len(missing) == 0, "echo_mode": ECHO_MODE, "missing": missing})

# On startup
@app.on_event("startup")
async def _http_pool():
    app.state.http = aiohttp.ClientSession(timeout=ClientTimeout(total=0))

    # Optional: pre-generate ULaw for very common phrases
    greeting_text = (
        "Hello, mai Raj bol raha hu, ABC Motors se. "
        "Kya mai John se baat kar raha hu?"
    )
    if ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        try:
            logger.info("[BOOT] pre-generating ULaw for greeting...")
            clip_id, _ = await elevenlabs_ulaw_to_url(greeting_text, PUBLIC_BASE_URL or "", PORT)
            b = ULAW_STORE.get(clip_id, b"")
            b = strip_wav_header_if_present(b)
            if b:
                PREGEN_ULAW[greeting_text] = b
                logger.info("[BOOT] greeting ULaw cached, %d bytes", len(b))
        except Exception:
            logger.exception("[BOOT] failed to pre-generate greeting ULaw")


@app.on_event("shutdown")
async def _http_close():
    with suppress(Exception): await app.state.http.close()

# Then use app.state.http instead of creating a new ClientSession each call

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
# DOWNLINK SETUP
# -----------------------------
@app.get("/tts_ulaw/{clip_id}.ulaw")
async def serve_tts_ulaw(clip_id: str):
    data = ULAW_STORE.get(clip_id)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    # 'audio/basic' is the canonical MIME for μ-law
    return StreamingResponse(iter([data]), media_type="audio/basic")

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
    
    end_states = {"completed", "disconnected", "hangup", "ended", "failed", "terminated"}
    if state in end_states:
        _print_session_history(session)
        # Optional: free memory after printing
        with suppress(Exception):
            SESSIONS.pop(session.voice_id, None)
        return {"ok": True}
    
    # EnableX-recognized webhook shape
    if (payload.get("state") or "").lower() == "recognized" and payload.get("text"):
        t = payload["text"].strip()
        if t == session.last_final:
            logger.info("[ENX] duplicate recognized ignored")
            return {"ok": True}
        session.last_final = t
        logger.info(f"[WEBHOOK] recognized: {payload['text']}")
        try:
            await handle_final_text(session, t)
            return {"ok": True}
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

# --- Deepgram keepalive ping (prevents idle timeouts) ---
async def _dg_keepalive(ws, interval: int = 5):
    try:
        while True:
            await asyncio.sleep(interval)
            # Deepgram Listen WS understands {"type":"KeepAlive"}
            await ws.send_str('{"type":"KeepAlive"}')
    except Exception:
        # socket closed or task canceled — safe to ignore
        pass

# -----------------------------
# EnableX media stream WebSocket (caller audio -> Deepgram telephony)
# -----------------------------
from aiohttp import WSServerHandshakeError

@app.websocket("/enablex/stream")
async def enablex_stream(ws: WebSocket):
    await ws.accept()

    # 1) Try query param OR common headers first
    params = ws.query_params
    hdrs = {k.lower(): v for k, v in ws.headers.items()}
    voice_id: Optional[str] = (
        params.get("voice_id")
        or hdrs.get("x-voice-id")
        or hdrs.get("x-voiceid")
        or hdrs.get("x-call-id")
        or hdrs.get("x-enx-callid")
    )

    # 2) Fallback to an ephemeral id so you at least have a session for logs
    if not voice_id:
        voice_id = f"ws-{uuid.uuid4().hex[:8]}-ephemeral"

    session = await ensure_session(voice_id)
    session.ws = ws # bind for downlink use
    logger.info(f"[ENX WS] connected path={ws.url.path} voice_id={voice_id}")

    from urllib.parse import quote_plus

    # --- Deepgram connect (8k μ-law telephony) ---
    dg_params = {
        "model": "nova-3",
        "encoding": "mulaw",
        "sample_rate": "8000",
        "channels": "1",
        "interim_results": "true",
        "smart_format": "true",
        "endpointing": str(ASR_ENDPOINTING_MS),
        # Optionally add "language": "en" or "hi" if you really need a fixed language.
        # Avoid "multi" unless your DG account/docs confirm it for nova-3 Listen.
    }

    if ASR_LANGUAGE_HINT:
        dg_params["language"] = ASR_LANGUAGE_HINT  # e.g. "multi" or "hi"

    qs = "&".join(f"{k}={quote_plus(str(v))}" for k, v in dg_params.items())
    dg_url = f"wss://api.deepgram.com/v1/listen?{qs}"
    logger.info(f"[DG] url -> {dg_url}")

    timeout = ClientTimeout(total=0, sock_connect=20, sock_read=120)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers={"Authorization": f"Token {DG_API_KEY}"}
        ) as http:
            try:
                dg = await http.ws_connect(dg_url, heartbeat=30, autoping=True, compress=0)
                logger.info(f"[DG] connected: {dg_url}")
                ka_task = asyncio.create_task(_dg_keepalive(dg))
            except WSServerHandshakeError as e:
                logger.error(f"[DG] handshake failed {e.status} {e.message}")
                with suppress(Exception): await ws.close(code=1011)
                return
            except Exception:
                logger.exception("[DG] connect failed")
                with suppress(Exception): await ws.close(code=1011)
                return

            # --- Send explicit Settings FIRST (prevents early close) ---
            # cfg = {
            #     "type": "Configure",
            #     "audio": { "encoding": "mulaw", "sample_rate": 8000, "channels": 1 },
            #     "model": "nova-3",
            #     "interim_results": True,
            #     "smart_format": True,
            #     "endpointing": ASR_ENDPOINTING_MS,
            # }
            # if ASR_LANGUAGE_HINT:
            #     cfg["language"] = ASR_LANGUAGE_HINT
            # await dg.send_str(json.dumps(cfg))

            dg_closing = False  # flip to True when DG begins closing

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
                            # control/recognized/optional b64 audio
                            try:
                                obj = json.loads(ev["text"])
                                # always try to learn stream_id 
                                sid = (
                                    obj.get("stream_id")
                                    or (obj.get("start") or {}).get("stream_id")
                                    or (obj.get("media") or {}).get("stream_id")
                                    or obj.get("streamId")  # just in case ENX uses camelCase
                                )
                                if sid and not session.stream_id:
                                    session.stream_id = sid
                                    logger.info(f"[ENX] learned stream_id={sid} from event={obj.get('event')}")

                            except Exception:
                                logger.warning("[ENX->DG] non-JSON text ignored")
                                continue

                            call_id = obj.get("voice_id") or obj.get("callId") or obj.get("id")
                            if call_id and not session:
                                voice_id = call_id
                                session = await ensure_session(voice_id)
                                logger.info(f"[ENX WS] bound voice_id from JSON: {voice_id}")
                            if call_id and (call_id != voice_id):
                                # 3) Lazy re-bind once we learn the real id
                                old = voice_id
                                voice_id = call_id
                                session = await ensure_session(voice_id)
                                session.ws = ws
                                logger.info(f"[ENX WS] re-bound voice_id {old} -> {voice_id}")

                            event = (obj.get("event") or "").lower()
                            if event in ("start_media", "start", "media_started"):
                                session.stream_id = obj.get("stream_id")
                                real_vid = (obj.get("start") or {}).get("voice_id")
                                if real_vid and real_vid != voice_id:
                                    old = voice_id
                                    voice_id = real_vid
                                    session = await ensure_session(voice_id)
                                    session.ws = ws
                                    session.stream_id = obj.get("stream_id")
                                    logger.info(f"[ENX WS] re-bound voice_id {old} -> {voice_id}")
                                logger.info(f"[ENX] start_media stream_id={session.stream_id}")
                                continue

                            if event == "media":
                                payload_b64 = ((obj.get("media") or {}).get("payload"))  # per guide
                                if payload_b64:
                                    audio_bytes = base64.b64decode(payload_b64)
                                    if audio_bytes and not dg_closing and not dg.closed:
                                        # --- LATENCY: mark when we last sent user audio to ASR ---
                                        session.asr_last_audio_ts = time.perf_counter()
                                        await dg.send_bytes(audio_bytes)
                                continue

                            if event == "stop_media":
                                if DEBUG_VOICE:
                                    logger.info(
                                        "[DEBUG VOICE] stop_media received stream_id=%s playing=%s gen_id=%d",
                                        session.stream_id,
                                        session.playing,
                                        session.gen_id,
                                    )
                                logger.info("[ENX] stop_media received")
                                break

                            logger.info(f"[ENX WS] unhandled event={event!r} obj={obj}")

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
                                        continue  # don't forward empties
                                    total_bytes += len(audio_bytes)
                                    if total_bytes % 32768 < 1600:
                                        logger.info(f"[ENX->DG] +{len(audio_bytes)}B (total {total_bytes}B)")
                                    
                                    # --- LATENCY: mark when we last sent user audio to ASR ---
                                    session.asr_last_audio_ts = time.perf_counter()
                                    await dg.send_bytes(audio_bytes)
                                except Exception as e:
                                    logger.warning(f"[ENX->DG] JSON payload decode failed: {e}")
                            continue

                        if ev.get("bytes") is not None:
                            if dg_closing or dg.closed:
                                return
                            buf = ev["bytes"]

                            # If your ENX binary framing has a 12-byte header, strip it; else just use buf as-is.
                            ulaw_bytes = buf[12:] if len(buf) > 12 else buf

                            # Skip empty frames
                            if not ulaw_bytes:
                                continue

                            total_bytes += len(ulaw_bytes)
                            if total_bytes % 32768 < 1600:
                                logger.info(f"[ENX->DG] +{len(ulaw_bytes)}B (total {total_bytes}B)")

                            try:
                                # --- LATENCY: mark when we last sent user audio to ASR ---
                                session.asr_last_audio_ts = time.perf_counter()
                                await dg.send_bytes(ulaw_bytes)
                            except Exception as e:
                                logger.warning(f"[ENX->DG] send to Deepgram failed: {e}")
                            continue

                finally:
                    try:
                        _print_session_history(session)
                    finally:
                        with suppress(Exception):
                            SESSIONS.pop(session.voice_id, None)
                    if session:
                        session.ws = None
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

                            # Log server-side issues early
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
                                    t = transcript.strip()
                                    if t == session.last_final:
                                        logger.info("[ASR] duplicate final ignored")
                                        continue
                                    session.last_final = t

                                    # --- LATENCY: ASR endpointing + decode ---
                                    now = time.perf_counter()
                                    if session.asr_last_audio_ts:
                                        asr_latency_ms = (now - session.asr_last_audio_ts) * 1000.0
                                        logger.info(
                                            "[METRIC] ASR endpoint+decode latency_ms=%.1f text=%r",
                                            asr_latency_ms,
                                            t,
                                        )
                                    session.asr_last_final_ts = now

                                    if session.playing:
                                        with suppress(Exception): await enablex_stop_play(session)
                                        session.playing = False
                                    asyncio.create_task(handle_final_text(session, t))

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
    # Reuse the same handler
    await enablex_stream(ws)

# -----------------------------
# Core bot logic: Echo or LLM → TTS → EnableX play
# -----------------------------
async def handle_final_text(session: CallSession, user_text: str):
    logger.info(f"[BOT] handle_final_text (echo_mode={ECHO_MODE}): {user_text}")
    session.turn += 1

    # add user turn in memory
    session.history.append({"role": "user", "content": user_text})
    # cap size: keep system + latest ~49 msgs
    if len(session.history) > 50:
        system = session.history[0]
        session.history = [system] + session.history[-49:]

    # choose reply
    if ECHO_MODE:
        reply_text = user_text
    # Fast path: first real user turn and simple greeting → no LLM
    elif session.turn == 1 and _is_simple_greeting(user_text):
        reply_text = (
            "Hello, mai Raj bol raha hu, ABC Motors se. "
            "Kya mai John se baat kar raha hu?"
        )

    else:
        # streaming llm call with history and latency metrics
        reply_text = await llm_reply_with_history(session, user_text)
    
    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] LLM reply turn=%d raw=%r",
            session.turn,
            reply_text,
        )

    # add assistant turn in memory
    session.history.append({"role": "assistant", "content": reply_text})

    # IMPORTANT: bump gen ONCE for this turn
    session.gen_id += 1
    my_gen = session.gen_id

    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] handle_final_text gen=%d user=%r",
            my_gen,
            user_text,
        )

    # enqueue instead of speaking inline
    enqueue_reply(session, reply_text, my_gen)

    # for s in segs:
    #     if not (ECHO_MODE and ECHO_STRICT_NO_SPLIT):
    #         if len(s) > TTS_SENTENCE_MAX_CHARS:
    #             s = s[:TTS_SENTENCE_MAX_CHARS] + "…"
    #     await play_sentence(session, s)


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
    
# -----------------------------
# TTS queue and speaker loop
# -----------------------------
async def speaker_loop(session: CallSession):
    """
    Consume session.tts_queue and play sentences sequentially.
    Runs in background so ASR keeps listening.
    """
    while True:
        # If no more TTS queued, shut down the loop
        if not session.tts_queue:
            if DEBUG_VOICE:
                logger.info("[DEBUG VOICE] speaker_loop idle exit (queue empty)")
            return

        text, my_gen = session.tts_queue.popleft()

        if DEBUG_VOICE:
            logger.info(
                "[DEBUG VOICE] speaker_loop pop gen=%d len=%d text=%r",
                my_gen,
                len(text),
                text,
            )

        # If a newer user turn arrived, skip this (stale) audio
        if my_gen != session.gen_id:
            continue

        # Play this sentence. After it finishes, loop back and
        # either grab the next one or exit if queue is now empty.
        await play_sentence(session, text, my_gen)


def enqueue_reply(
        session: CallSession, 
        reply_text: str, 
        my_gen: int):
    """
    Enqueue ONE TTS clip per reply.
    This avoids sending multiple 'media' events for a single bot turn,
    which is what seems to make EnableX drop everything after the first.
    """
    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] enqueue_reply (no-split) gen=%d text=%r",
            my_gen,
            reply_text,
        )

    text = (reply_text or "").strip()
    if not text:
        return

    # Still keep a safety cap on very long replies
    if len(text) > TTS_SENTENCE_MAX_CHARS:
        text = text[:TTS_SENTENCE_MAX_CHARS] + "…"

    session.tts_queue.append((text, my_gen))

    if not session.speaking_task or session.speaking_task.done():
        session.speaking_task = asyncio.create_task(speaker_loop(session))



def _ulaw_duration_sec(clip_id: str) -> float:
    """Compute duration from μ-law bytes @8kHz."""
    b = ULAW_STORE.get(clip_id) or b""
    return len(b) / 8000.0

async def _wait_until_done(session: CallSession, est_sec: float, my_gen: int):
    """
    Wait until playback should be finished, unless a new gen invalidates us (barge-in/new turn).
    Poll in small steps so we yield to the event loop.
    """
    # A small safety buffer to account for network jitter / player start-up
    target = asyncio.get_event_loop().time() + est_sec + 0.10
    while asyncio.get_event_loop().time() < target:
        # If a newer utterance started, abort waiting
        if my_gen != session.gen_id:
            return
        await asyncio.sleep(0.05)
    session.playing = False


# -----------------------------
# TTS + EnableX play
# DOWNLINK SETUP
# -----------------------------

def strip_wav_header_if_present(b: bytes) -> bytes:
    # WAV files start with "RIFF....WAVE"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        idx = b.find(b"data")
        if idx != -1 and idx + 8 <= len(b):
            data_len = int.from_bytes(b[idx+4:idx+8], "little", signed=False)
            start = idx + 8
            logger.warning("[ULAW] WAV header detected, stripping to raw ulaw")
            return b[start:start + data_len]
        logger.warning("[ULAW] WAV header detected but data chunk not found; stripping first 44 bytes")
        return b[44:]
    return b

async def play_sentence(session: CallSession, text: str, my_gen: int):
    voice_id = session.voice_id

    if my_gen != session.gen_id:
        return

    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] play_sentence gen=%d voice=%s len=%d text=%r",
            my_gen,
            voice_id,
            len(text),
            text,
        )

    await wait_for_stream_id(session, timeout=2.0)



    # If we have a pre-generated ULaw clip for this exact text, use it
    ulaw_bytes: bytes = b""
    if text in PREGEN_ULAW:
        ulaw_bytes = PREGEN_ULAW[text]
    else:
        # 1) Synth μ-law
        try:
            # --- LATENCY: TTS synth start ---
            session.tts_start_ts = time.perf_counter()

            clip_id, _ = await elevenlabs_ulaw_to_url(text, PUBLIC_BASE_URL or "", PORT)
            ulaw_bytes = ULAW_STORE.get(clip_id, b"")
            ulaw_bytes = strip_wav_header_if_present(ulaw_bytes)

            session.tts_end_ts = time.perf_counter()
            tts_ms = (session.tts_end_ts - session.tts_start_ts) * 1000.0

            logger.info(
                "[METRIC] TTS synth latency_ms=%.1f chars=%d bytes=%d",
                tts_ms,
                len(text),
                len(ulaw_bytes),
            )

            if DEBUG_VOICE:
                logger.info(
                    "[DEBUG VOICE] elevenlabs_ulaw clip_id=%s bytes=%d",
                    clip_id,
                    len(ulaw_bytes),
                )
        except Exception:
            logger.exception("[ULAW] synth failed")
            await enablex_play_text(voice_id, text)
            session.playing = True
            await asyncio.sleep(min(len(text) / 12.0, 3.0))
            session.playing = False
            return

    # 2) Padding (tiny bit of trailing silence)
    if ulaw_bytes:
        ulaw_bytes += bytes([0xFF]) * 240

    est_sec = len(ulaw_bytes) / 8000.0 if ulaw_bytes else max(0.6, min(len(text) / 12.0, 4.0))
    logger.info(
        "[ENX DOWNLINK SIMPLE] sending %d bytes (~%.2fs) gen=%d",
        len(ulaw_bytes),
        est_sec,
        my_gen,
    )

    if my_gen != session.gen_id:
        return

    session.playing = True

    # --- LATENCY: Telephony send + jitter buffer wait ---

    # 4a) measure WS send to EnableX
    session.downlink_start_ts = time.perf_counter()
    ok = await enablex_send_ulaw_simple(session, ulaw_bytes)
    send_done_ts = time.perf_counter()
    send_ms = (send_done_ts - session.downlink_start_ts) * 1000.0

    logger.info(
        "[METRIC] DOWNLINK send_ms=%.1f bytes=%d",
        send_ms,
        len(ulaw_bytes),
    )

    if ok:
        # 4b) measure our wait time for playout (approx jitter + playback)
        wait_start = time.perf_counter()
        await _wait_until_done(session, est_sec, my_gen)
        wait_end = time.perf_counter()
        session.downlink_end_ts = wait_end

        wait_ms = (wait_end - wait_start) * 1000.0
        end_to_end_ms = 0.0
        if session.asr_last_audio_ts:
            end_to_end_ms = (wait_end - session.asr_last_audio_ts) * 1000.0

        logger.info(
            "[METRIC] DOWNLINK wait_ms=%.1f est_audio_ms=%.0f end_to_end_turn_ms=%.1f",
            wait_ms,
            est_sec * 1000.0,
            end_to_end_ms,
        )
    else:
        session.playing = False
        # Fallback: URL-based play (no detailed metrics here)
        clip_id = uuid.uuid4().hex
        ULAW_STORE[clip_id] = ulaw_bytes
        url = f"{PUBLIC_BASE_URL}/tts_ulaw/{clip_id}.ulaw"
        await enablex_play_url(voice_id, url)
        session.playing = True
        await _wait_until_done(session, est_sec, my_gen)


# async def play_sentence(session: CallSession, text: str, my_gen: int):
#     """
#     Downlink: ElevenLabs -> μ-law 8k bytes -> URL -> EnableX /play with that URL.
    
#     Steps to send downlink
#     1. synth ulaw
#     2. send ulaw bytes via enablex_send_ulaw
#     """
#     voice_id = session.voice_id

#     # if canceled mid-way, don’t play
#     if my_gen != session.gen_id:
#         return
#     if DEBUG_VOICE:
#         logger.info(
#             "[DEBUG VOICE] play_sentence gen=%d voice=%s len=%d text=%r",
#             my_gen,
#             voice_id,
#             len(text),
#             text,
#         )
    
#     # LATENCYYYYYYY: wait for stream_id to be ready
#     await wait_for_stream_id(session, timeout=2.0)

#     # 1) Synthesize μ-law and host it at /tts_ulaw/<id>.ulaw
#     try:
#         clip_id, _ = await elevenlabs_ulaw_to_url(text, PUBLIC_BASE_URL or "", PORT)
#         ulaw_bytes = ULAW_STORE.get(clip_id, b"")
#         ulaw_bytes = strip_wav_header_if_present(ulaw_bytes)
#         if DEBUG_VOICE:
#             logger.info(
#                 "[DEBUG VOICE] elevenlabs_ulaw clip_id=%s bytes_before_pad=%d",
#                 clip_id,
#                 len(ulaw_bytes),
#             )
#     except Exception:
#         logger.exception("[ULAW] synth failed")
#         # fallback: ENX built-in TTS (keeps the call alive)
#         await enablex_play_text(voice_id, text)
#         session.playing = True
#         # Conservative wait so we don't immediately stack another /play
#         await asyncio.sleep(min(len(text) / 12.0, 3.0))
#         session.playing = False
#         return

#     # 2) Padding
#     if ulaw_bytes:
#         ulaw_bytes += bytes([0xFF]) * 240

#     if DEBUG_VOICE:
#         logger.info(
#             "[DEBUG VOICE] elevenlabs_ulaw bytes_after_pad=%d (~%.2fs)",
#             len(ulaw_bytes),
#             len(ulaw_bytes) / 8000.0 if ulaw_bytes else 0.0,
#         )

#     # 3) If we got pre-empted (barge-in / newer gen), don't play
#     if my_gen != session.gen_id:
#         return

#     # 4) Compute duration from μ-law length and play
#     est_sec = len(ulaw_bytes)/8000.0
#     if est_sec <= 0:
#         est_sec = max(0.6, min(len(text)/12.0, 4.0))  # crude fallback estimate
#     logger.info(f"[ENX DOWNLINK] sending {len(ulaw_bytes)} bytes, est={est_sec:.2f}s gen={my_gen}")

#     # 5) Send downlink via WS
#     # start playing flag BEFORE streaming so barge-in can stop mid-stream
#     session.playing = True

#     ok = await enablex_send_ulaw_streamed(session, ulaw_bytes, my_gen, chunk_ms=20)

#     if ok:
#         logger.info("[ENX DOWNLINK] WS send OK")
#         await _wait_until_done(session, est_sec, my_gen)
#     else:
#         logger.warning("[ENX DOWNLINK] WS send FAILED -> falling back to REST play_url")
#         session.playing = False  # reset before fallback

#         clip_id = uuid.uuid4().hex
#         ULAW_STORE[clip_id] = ulaw_bytes
#         url = f"{PUBLIC_BASE_URL}/tts_ulaw/{clip_id}.ulaw"
#         await enablex_play_url(voice_id, url)

#         session.playing = True
#         await _wait_until_done(session, est_sec, my_gen)



# -----------------------------
# EnableX helpers
# -----------------------------

def enx_auth() -> BasicAuth:
    if not (ENABLEX_APP_ID and ENABLEX_APP_KEY):
        raise RuntimeError("Missing EnableX credentials")
    return BasicAuth(ENABLEX_APP_ID, ENABLEX_APP_KEY)

async def enablex_start_media_stream(voice_id: str, stream_wss_url: str):
    url = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/stream"
    body = {"wss_host": stream_wss_url}
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

async def enablex_stop_play(session: CallSession):
    if session.ws and session.stream_id:
        msg = {"event": "clear_media", "stream_id": session.stream_id, "voice_id": session.voice_id}
        if DEBUG_VOICE:
            logger.info(
                "[DEBUG VOICE] sending clear_media stream_id=%s voice_id=%s",
                session.stream_id,
                session.voice_id,
            )
        with suppress(Exception):
            async with session.send_lock:
                await session.ws.send_text(json.dumps(msg))
        logger.info("[ENX] clear_media sent")
        return

        # # Variant 2: command in body
        # url2 = f"{ENABLEX_BASE}/voice/v1/call/{voice_id}/play"
        # r2 = await s.put(url2, json={"command": "stop"})
        # logger.info(f"[ENX STOP PLAY fallback] {r2.status} {await r2.text()}")

# -----------------------------
# EnableX downlink: send μ-law audio via WS
# -----------------------------

# async def enablex_send_ulaw(session: CallSession, ulaw_bytes: bytes):
#     """
#     Send μ-law audio back to EnableX on the SAME WS.
#     """
#     if not session.ws or not session.stream_id:
#         logger.warning("[ENX DOWNLINK] ws/stream_id missing, can't send media")
#         return False

#     session.out_seq += 1
#     msg = {
#         "event": "media",
#         "voice_id": session.voice_id,
#         "stream_id": session.stream_id,
#         "media": {
#             "seq": session.out_seq,
#             "timestamp": int(time.time() * 1000),
#             "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
#             "payload": base64.b64encode(ulaw_bytes).decode("utf-8"),
#         },
#     }
#     await session.ws.send_text(json.dumps(msg))
#     return True

# -----------------------------
# EnableX downlink: send μ-law audio via WS in ONE BIG CHUNK (spec clone)
# -----------------------------
async def enablex_send_ulaw_simple(
        session: CallSession, 
        ulaw_bytes: bytes) -> bool:
    """
    Minimal, spec-clone send:
    - One big 'media' event
    - Exactly the JSON shape from the EnableX guide
    """
    if not session.ws or not session.stream_id:
        logger.warning("[ENX DOWNLINK] ws/stream_id missing, can't send media (simple)")
        return False

    session.out_seq += 1
    msg = {
        "event": "media",
        "voice_id": session.voice_id,
        "stream_id": session.stream_id,
        "media": {
            "seq": session.out_seq,
            "timestamp": int(time.time() * 1000),
            "format": {
                "encoding": "ulaw",
                "sample_rate": 8000,
                "channels": 1,
            },
            "payload": base64.b64encode(ulaw_bytes).decode("utf-8"),
        },
    }

    # msg = {
    #     "event": "media",
    #     "voice_id": session.voice_id,
    #     "stream_id": session.stream_id,
    #     "media": {
    #         "payload": base64.b64encode(ulaw_bytes).decode("utf-8"),
    #     },
    # }

    msg_for_log = {**msg, "media": {**msg["media"], "payload": f"<{len(ulaw_bytes)} bytes b64>"}}
    logger.info("[ENX DOWNLINK SIMPLE] JSON: %s", jd(msg_for_log))

    try:
        async with session.send_lock:
            await session.ws.send_text(json.dumps(msg))
        logger.info(
            "[ENX DOWNLINK SIMPLE] sent %d bytes in one media event seq=%d",
            len(ulaw_bytes),
            session.out_seq,
        )
        return True
    except Exception as e:
        logger.warning(f"[ENX DOWNLINK SIMPLE] send failed: {e}")
        session.ws = None
        session.stream_id = None
        return False


# -----------------------------
# EnableX downlink: send μ-law audio via WS in REALTIME chunks
# -----------------------------
async def enablex_send_ulaw_streamed(
    session: CallSession,
    ulaw_bytes: bytes,
    my_gen: int,
    chunk_ms: int = 20,         # standard 20ms telephony frame
    prebuffer_chunks: int = 5   # send first 5 chunks fast to fill jitter buffer
    ):
    if not session.ws or not session.stream_id:
        logger.warning("[ENX DOWNLINK] ws/stream_id missing, can't send media")
        return False

    bytes_per_ms = 8  # 8000 samples/sec = 8 bytes/ms in ulaw
    chunk_size = bytes_per_ms * chunk_ms
    total_bytes = len(ulaw_bytes)

    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] enablex_send_ulaw_streamed gen=%d total_bytes=%d chunk_ms=%d",
            my_gen,
            total_bytes,
            chunk_ms,
        )

    loop = asyncio.get_event_loop()
    start_time = loop.time()

    # Split into fixed frames
    chunks = [
        ulaw_bytes[i:i + chunk_size]
        for i in range(0, len(ulaw_bytes), chunk_size)
        if ulaw_bytes[i:i + chunk_size]
    ]

    for n, chunk in enumerate(chunks):
        # stop sending stale audio instantly
        if my_gen != session.gen_id:
            logger.info("[ENX DOWNLINK] canceled mid-stream due to new gen")
            return True

        session.out_seq += 1

        if DEBUG_VOICE and n < 3:  # log first few chunks for detail
            logger.info(
                "[DEBUG VOICE] downlink chunk idx=%d size=%d seq=%d",
                n,
                len(chunk),
                session.out_seq,
            )

        msg = {
            "event": "media",
            "voice_id": session.voice_id,
            "stream_id": session.stream_id,
            "media": {
                "seq": session.out_seq,
                "timestamp": int(time.time() * 1000),
                "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
                "payload": base64.b64encode(chunk).decode("utf-8"),
            },
        }

        try:
            async with session.send_lock:
                await session.ws.send_text(json.dumps(msg))
        except Exception as e:
            logger.warning(f"[ENX DOWNLINK] send failed: {e}")
            session.ws = None
            session.stream_id = None
            return False

        # ---- pacing ----
        # Prebuffer: first few chunks are sent immediately
        if n >= prebuffer_chunks:
            # target send moment for this chunk
            target = start_time + (n - prebuffer_chunks + 1) * (chunk_ms / 1000.0)
            now = loop.time()
            if target > now:
                await asyncio.sleep(target - now)

    if DEBUG_VOICE:
        logger.info(
            "[DEBUG VOICE] enablex_send_ulaw_streamed gen=%d DONE chunks=%d total_bytes=%d",
            my_gen,
            len(chunks),
            total_bytes,
        )

    return True

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
        "voice_settings": {"stability": 0.8, "similarity_boost": 0.7, "style": 0.65, "use_speaker_boost": True}
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


# change 2 - working for first sentence only.

