"""
stt_service.py

Wraps OpenAI's Whisper transcription API. Used by routes_voice.py's
/voice/transcribe endpoint.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL = "whisper-1"  # cheapest, non-streaming — fine for turn-based answers, not live back-and-forth

_client: OpenAI | None = None


class TranscriptionEmptyError(RuntimeError):
    """Raised when Whisper returns empty text — the audio was likely
    silent or unreadable. Kept distinct from a plain RuntimeError
    (e.g. a missing API key) so the route can map each to the
    correct HTTP status: this one is the client's problem (422,
    try recording again), a missing key is a server problem (500)."""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — check that your .env file is loaded.")
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe(audio_file) -> str:
    """
    Transcribes an audio file to text.

    audio_file must be a file-like object with a `.name` attribute
    carrying a recognizable extension (Whisper uses it to infer the
    audio format) — e.g. open(path, "rb"), or an io.BytesIO with
    .name manually set, as routes_voice.py does for uploaded bytes.
    """
    client = _get_client()
    response = client.audio.transcriptions.create(
        model=MODEL,
        file=audio_file,
    )
    text = response.text.strip()
    if not text:
        raise TranscriptionEmptyError(
            "Transcription returned empty text — the audio may be silent or unreadable."
        )
    return text
