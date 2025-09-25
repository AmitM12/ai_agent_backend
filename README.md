# ai_agent_backend
Python backend for the AI sales agent

## Backend Flow

- Each client WS connection initialises shared state—queues, buffers, and tasks—and refuses the session early if any critical env configuration is missing.
- `connect_deepgram` and `connect_openai` retry with multiple Deepgram parameter profiles and send an immediate `session.update` to shape OpenAI’s persona/instructions.
- Telemetry on bytes streamed to Deepgram is published every second to help the UI confirm audio actually reached the ASR service.
- `read_client` forwards binary mic frames upstream and watches for client-issued barge-in commands to cancel both TTS and LLM work-in-progress.
- `read_deepgram` handles Deepgram warnings/errors, streams interim transcripts, and on finals currently enqueues an echo reply while resetting the LLM buffer.
- `read_openai` consumes text deltas/final markers, relays them to the client, segments sentences, and pushes them into the TTS queue with overflow protection.

## ASR / LLM / TTS Orchestration

- User audio → server WS → Deepgram: each final transcript notifies the UI (`asr.final`) and, in normal mode, would trigger `response.create` to OpenAI (that call is presently commented out, replaced by a temporary echo).
- OpenAI’s realtime WS streams `response.text.delta` events; each delta both updates the UI and accumulates in `llm_buf` for sentence detection.
- As soon as a sentence boundary is detected, the text is queued for TTS, keeping only the most recent `TTS_QUEUE_MAX` items to avoid backlog under long turns.
- ElevenLabs requests are serialized via `drain_tts`, which launches a background task that fetches MP3 audio for each sentence and streams it back to the browser.
- Barge-in increments `gen_id`, cancels outstanding TTS tasks, clears buffers, tells the UI to drop queued audio, and sends `response.cancel` upstream so the LLM stops generating.

## Chunking & Buffers

- `_segment_sentences` walks the LLM buffer looking for punctuation (including the Hindi danda) to release clean, TTS-friendly chunks while retaining any trailing fragment for the next delta.
  - `TTS_SENTENCE_MAX_CHARS` caps very long sentences, adding an ellipsis before request submission to ElevenLabs.
- `tts_queue` (deque) feeds `drain_tts`; overflow ejects the oldest entry and notifies the client so the UI can warn about dropped speech.
- The front end mirrors this with `playQueueRef`, scheduling decoded buffers back-to-back and dropping the oldest clip if the client-side queue grows past 16 items.
- The ASR stats publisher maintains a simple byte counter so the UI can display whether audio is still flowing even before Deepgram returns text.
