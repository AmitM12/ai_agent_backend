# downlink_elevenlabs.py
import os
import json
import asyncio
import logging
from typing import Dict, Optional, Callable, List

import aiohttp
from aiohttp import ClientSession, ClientTimeout, WSMsgType, WSServerHandshakeError

logger = logging.getLogger("downlink")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ------- ENV (re-used from your .env) -------
ENABLEX_WSS        = os.getenv("ENABLEX_AUDIOSOCKET_WSS", "wss://api.enablex.io:9090/audiosocket/socket.io/")
ENABLEX_APP_ID     = os.getenv("ENABLEX_APP_ID") or ""
ENABLEX_APP_KEY    = os.getenv("ENABLEX_APP_KEY") or ""

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY") or ""
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or ""
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
EL_OPT_LATENCY      = os.getenv("EL_OPT_LATENCY", "3")  # 0..4 (3 is a good balance)
OUTPUT_FORMAT       = "ulaw_8000"                      # telephony

# ------- Connection model -------
class _Conn:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.ready: bool = False                     # set True after 'start_media'
        self.queue: List[Callable[[], None]] = []    # actions to run once ready
        self.lock = asyncio.Lock()                   # serialize TTS per call
        self._http_session: Optional[ClientSession] = None

CONNS: Dict[str, _Conn] = {}

def _enx_headers(voice_id: str) -> Dict[str, str]:
    return {
        "X-App-ID": ENABLEX_APP_ID,
        "X-App-Key": ENABLEX_APP_KEY,
        "X-Voice-Id": voice_id,
    }

async def ensure(voice_id: str) -> _Conn:
    """
    Ensure an AudioSocket is open for this voice_id.
    Returns when socket is open; 'ready' flips True upon 'start_media'.
    """
    if voice_id in CONNS and CONNS[voice_id].ws and not CONNS[voice_id].ws.closed:
        return CONNS[voice_id]

    conn = _Conn(voice_id)
    CONNS[voice_id] = conn

    timeout = ClientTimeout(total=0, sock_connect=20, sock_read=0)
    http = ClientSession(timeout=timeout)
    conn._http_session = http

    try:
        ws = await http.ws_connect(
            ENABLEX_WSS,
            headers=_enx_headers(voice_id),
            autoping=True,
            heartbeat=30,
        )
        conn.ws = ws
        logger.info(f"[DL] AudioSocket connected ({voice_id})")

        async def reader():
            try:
                async for m in ws:
                    if m.type == WSMsgType.TEXT:
                        try:
                            obj = json.loads(m.data)
                            st = (obj.get("state") or "").lower()
                            if st == "start_media":
                                conn.ready = True
                                logger.info(f"[DL] start_media received ({voice_id}) — ready")
                                # flush queued actions
                                queued = conn.queue[:]
                                conn.queue.clear()
                                for fn in queued:
                                    try:
                                        fn()
                                    except Exception:
                                        logger.exception("[DL] queued action failed")
                        except Exception:
                            pass  # ignore any non-JSON text
                    elif m.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                        logger.warning(f"[DL] WS closing ({voice_id})")
                        break
            except Exception:
                logger.exception(f"[DL] reader crashed ({voice_id})")
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass
                try:
                    await http.close()
                except Exception:
                    pass
                CONNS.pop(voice_id, None)
                logger.info(f"[DL] AudioSocket cleaned up ({voice_id})")

        asyncio.create_task(reader())
        return conn

    except WSServerHandshakeError as e:
        await http.close()
        logger.error(f"[DL] handshake failed: {e.status} {e.message}")
        raise
    except Exception:
        await http.close()
        logger.exception("[DL] ws_connect failed")
        raise

async def say(voice_id: str, text: str):
    """
    Stream ElevenLabs (μ-law 8k) into the AudioSocket for this voice_id.
    Queues playback until 'start_media' if needed.
    """
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("Downlink requires ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID")

    conn = await ensure(voice_id)
    ws = conn.ws
    if ws is None or ws.closed:
        raise RuntimeError("AudioSocket not open")

    async def _do_stream():
        async with conn.lock:
            if ws.closed:
                return
            logger.info(f"[DL] TTS→WS start ({voice_id}) text={text!r}")

            path = f"/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
            params = {"output_format": OUTPUT_FORMAT, "optimize_streaming_latency": EL_OPT_LATENCY}
            body = {
                "text": text,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {
                    "stability": 0.35, "similarity_boost": 0.9, "style": 0.65, "use_speaker_boost": True
                },
            }
            timeout = ClientTimeout(total=0, sock_connect=20, sock_read=0)
            async with ClientSession(timeout=timeout) as s:
                async with s.post(
                    f"https://api.elevenlabs.io{path}",
                    params=params, json=body,
                    headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                ) as r:
                    if r.status != 200:
                        txt = await r.text()
                        raise RuntimeError(f"ElevenLabs {r.status}: {txt}")

                    async for chunk in r.content.iter_chunked(4096):
                        if not chunk:
                            continue
                        if ws.closed:
                            break
                        await ws.send_bytes(chunk)

            logger.info(f"[DL] TTS→WS done ({voice_id})")

    # If not yet 'ready', defer until 'start_media' arrives
    if not conn.ready:
        logger.info(f"[DL] not ready — queueing TTS until start_media ({voice_id})")
        conn.queue.append(lambda: asyncio.create_task(_do_stream()))
    else:
        await _do_stream()

async def close(voice_id: str):
    """Optional: close the AudioSocket manually."""
    conn = CONNS.pop(voice_id, None)
    if not conn:
        return
    try:
        if conn.ws and not conn.ws.closed:
            await conn.ws.close()
    finally:
        if conn._http_session:
            await conn._http_session.close()
