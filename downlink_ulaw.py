# downlink_ulaw.py
import os
import uuid
import json
import asyncio
import logging
from typing import Tuple, Dict

import aiohttp
from aiohttp import ClientResponseError

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
EL_OPT_LATENCY      = os.getenv("EL_OPT_LATENCY", "4")

ELEVENLABS_OPTIMIZE_STREAMING_LATENCY = os.getenv("ELEVENLABS_OPTIMIZE_STREAMING_LATENCY", "0")
ELEVENLABS_ADD_SILENCE_MS = int(os.getenv("ELEVENLABS_ADD_SILENCE_MS", "0"))

ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", "0.45"))
ELEVENLABS_SIMILARITY = float(os.getenv("ELEVENLABS_SIMILARITY", "0.80"))
ELEVENLABS_STYLE = float(os.getenv("ELEVENLABS_STYLE", "0.00"))
ELEVENLABS_SPEAKER_BOOST = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() in ("1","true","yes")

ULAW_STORE: Dict[str, bytes] = {}

async def elevenlabs_ulaw_to_url(text: str, public_base_url: str, port: int) -> Tuple[str, str]:
    """
    Generate a single, complete μ-law (8k) clip from ElevenLabs for the given sentence
    and host it at /tts_ulaw/<clip_id>.ulaw. Interface unchanged.
    """
    if not (ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        raise RuntimeError("Missing ElevenLabs config")

    api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

    # Keep telephony format; quality-first latency to reduce artifacts
    params = {
        "output_format": "ulaw_8000",
        "optimize_streaming_latency": ELEVENLABS_OPTIMIZE_STREAMING_LATENCY,  # "0" recommended here
    }

    body = {
        "text": text.strip(),
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": ELEVENLABS_STABILITY,
            "similarity_boost": ELEVENLABS_SIMILARITY,
            "style": ELEVENLABS_STYLE,
            "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST,
        },
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}

    attempts = 0
    while True:
        attempts += 1
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(api_url, params=params, json=body, headers=headers) as resp:
                    resp.raise_for_status()
                    # Buffer the FULL sentence clip (no partials)
                    chunks = [chunk async for chunk in resp.content.iter_chunked(4096)]
                    ulaw = b"".join(chunks)
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
        else:
            break

    # Optional tiny de-click: μ-law silence (0xFF) pre-roll—doesn't change format
    if ELEVENLABS_ADD_SILENCE_MS > 0:
        ulaw = (b"\xff" * int(8000 * ELEVENLABS_ADD_SILENCE_MS / 1000)) + ulaw

    clip_id = uuid.uuid4().hex
    ULAW_STORE[clip_id] = ulaw

    base = public_base_url or f"http://localhost:{port}"
    return clip_id, f"{base}/tts_ulaw/{clip_id}.ulaw"
