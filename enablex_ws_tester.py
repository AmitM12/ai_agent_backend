# enablex_ws_tester.py
import asyncio, json, base64, uuid, argparse, urllib.parse, math, struct, wave, audioop
from typing import Iterator, Tuple, Optional
import websockets

# -------- Helpers: URL/IDs --------
def get_qs_param(url: str, key: str) -> Optional[str]:
    q = urllib.parse.urlparse(url).query
    params = urllib.parse.parse_qs(q)
    v = params.get(key)
    return v[0] if v else None

def ensure_voice_id(ws_url: str, cli_voice_id: Optional[str]) -> str:
    v = cli_voice_id or get_qs_param(ws_url, "voice_id")
    if not v:
        raise SystemExit("voice_id missing (use --voice-id or include ?voice_id=... in --ws URL)")
    return v

# -------- μ-law framing --------
def pcm16_to_mulaw(pcm16: bytes) -> bytes:
    # PCM16 -> μ-law (1 byte per sample)
    return audioop.lin2ulaw(pcm16, 2)

def to_mono_pcm16(pcm: bytes, sampwidth: int, nchannels: int) -> bytes:
    out = pcm
    if nchannels == 2:
        out = audioop.tomono(out, sampwidth, 1.0, 0.0)  # take left
    if sampwidth != 2:
        out = audioop.lin2lin(out, sampwidth, 2)
    return out

def resample_pcm16(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm16
    # stateful rate conversion; we don't need state across calls for offline buffer
    converted, _ = audioop.ratecv(pcm16, 2, 1, src_rate, dst_rate, None)
    return converted

def chunk_bytes(data: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i:i+size]

# -------- WAV loader (robust) --------
def load_wav_as_mulaw_frames(path: str, frame_ms: int = 20) -> Tuple[Iterator[bytes], int]:
    """
    Loads any WAV, converts to mono 16-bit 8kHz PCM, then μ-law,
    and yields frames of N bytes where N = 8kHz * 1ch * (frame_ms/1000).
    """
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    # to mono/16-bit
    pcm16 = to_mono_pcm16(raw, sampwidth, channels)
    # to 8kHz
    pcm16_8k = resample_pcm16(pcm16, rate, 8000)
    # to μ-law
    ulaw = pcm16_to_mulaw(pcm16_8k)

    frame_bytes = int(8000 * (frame_ms / 1000.0))  # e.g., 20ms -> 160 bytes μ-law
    return chunk_bytes(ulaw, frame_bytes), frame_bytes

# -------- Tone generator (fallback) --------
def generate_tone_mulaw(duration_sec: float = 3.0, freq: float = 440.0, frame_ms: int = 20) -> Tuple[Iterator[bytes], int]:
    """
    Generates a simple sine tone (PCM16 8kHz), converts to μ-law, chunks by frame_ms.
    """
    sr = 8000
    total = int(sr * duration_sec)
    # build PCM16
    amp = 8000
    pcm16 = bytearray()
    for n in range(total):
        s = int(amp * math.sin(2 * math.pi * freq * (n / sr)))
        pcm16 += struct.pack("<h", s)
    ulaw = pcm16_to_mulaw(bytes(pcm16))
    frame_bytes = int(sr * (frame_ms / 1000.0))  # 160 at 20ms
    return chunk_bytes(ulaw, frame_bytes), frame_bytes

# -------- Main streamer --------
async def stream_media(ws_url: str, voice_id: str, frames: Iterator[bytes], frame_bytes: int, frame_ms: int):
    # Connect; disable pings so we don’t interfere with timing
    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
        stream_id = str(uuid.uuid4())

        # per-EnableX shape
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({"event": "start_media", "start": {"voice_id": voice_id, "stream_id": stream_id}}))

        seq = 0
        for chunk in frames:
            if not chunk:
                continue
            b64 = base64.b64encode(chunk).decode("ascii")
            msg = {
                "event": "media",
                "media": {
                    "payload": b64,
                    "seq": seq,
                    "format": {"encoding": "mulaw", "sample_rate": 8000, "channels": 1},
                },
            }
            await ws.send(json.dumps(msg))
            seq += 1
            # Real-time pacing: 20ms per 160 μ-law bytes
            await asyncio.sleep(frame_ms / 1000.0)

        await ws.send(json.dumps({"event": "stop_media"}))
        # give the server a beat to log/close
        await asyncio.sleep(0.2)

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="EnableX WS media test client")
    ap.add_argument("--ws", required=True, help="wss://<host>/enablex/stream?voice_id=CALLID")
    ap.add_argument("--voice-id", help="Override voice_id (else read from ?voice_id=)")
    ap.add_argument("--wav", help="Path to WAV file to stream (any format; auto-converts)")
    ap.add_argument("--frame-ms", type=int, default=20, help="Frame size in ms (default 20)")
    args = ap.parse_args()

    voice_id = ensure_voice_id(args.ws, args.voice_id)

    if args.wav:
        frames_iter, frame_bytes = load_wav_as_mulaw_frames(args.wav, args.frame_ms)
    else:
        frames_iter, frame_bytes = generate_tone_mulaw(3.0, 440.0, args.frame_ms)

    asyncio.run(stream_media(args.ws, voice_id, frames_iter, frame_bytes, args.frame_ms))

if __name__ == "__main__":
    main()
