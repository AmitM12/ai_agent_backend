# downlink_ulaw.py
import os
import uuid
import json
import asyncio
import logging
from typing import Tuple, Dict

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientResponseError

logger = logging.getLogger("downlink.ulaw")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

from dotenv import load_dotenv

load_dotenv()

# ---- ElevenLabs env ----
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY") or ""
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") or ""
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
EL_OPT_LATENCY      = os.getenv("EL_OPT_LATENCY", "3")  # 0..4 (3 = good balance)

# In-memory store for raw ULAW bytes (served via /tts_ulaw/<id>.ulaw)
ULAW_STORE: Dict[str, bytes] = {}

async def elevenlabs_ulaw_to_url(text: str, public_base_url: str, fallback_local_port: int) -> Tuple[str, str]:
    """
    Synthesize `text` with ElevenLabs as 8 kHz μ-law.
    Store bytes in-memory and return (clip_id, url) where url points to /tts_ulaw/<clip_id>.ulaw
    """
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("Missing ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID")

    # ElevenLabs streaming endpoint (ULAW 8000 Hz).
    api_path = f"/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    params = {"output_format": "ulaw_8000", "optimize_streaming_latency": EL_OPT_LATENCY}
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.9,
            "style": 0.65,
            "use_speaker_boost": True
        }
    }
    timeout = ClientTimeout(total=0, sock_connect=20, sock_read=0)

    # bounded retries for transient failures
    attempts = 0
    while True:
        attempts += 1
        try:
            async with ClientSession(timeout=timeout) as s:
                async with s.post(
                    f"https://api.elevenlabs.io{api_path}",
                    params=params, json=body,
                    headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
                ) as r:
                    if r.status != 200:
                        txt = await r.text()
                        raise RuntimeError(f"ElevenLabs {r.status}: {txt}")

                    # Collect raw μ-law bytes
                    bufs = [chunk async for chunk in r.content.iter_chunked(4096)]
                    ulaw = b"".join(bufs)

        except ClientResponseError as cre:
            # retry 429 a couple of times
            if cre.status == 429 and attempts < 3:
                await asyncio.sleep(0.4 * attempts)
                continue
            raise
        except Exception:
            if attempts < 3:
                await asyncio.sleep(0.4 * attempts)
                continue
            raise
        break

    clip_id = uuid.uuid4().hex
    ULAW_STORE[clip_id] = ulaw

    base = (public_base_url or "").strip() or f"http://localhost:{fallback_local_port}"
    url = f"{base}/tts_ulaw/{clip_id}.ulaw"

    logger.info(f"[ULAW] clip ready id={clip_id} bytes={len(ulaw)} url={url}")
    return clip_id, url
