"""
tts_service.py

Wraps OpenAI's text-to-speech API. Used by routes_voice.py's
/voice/synthesize endpoint.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL = "tts-1"  # cheap, per-minute billing — fine for short interview questions
VOICE = "ash"            # default OpenAI voice; swap for whichever fits your product's tone

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — check that your .env file is loaded.")
        _client = OpenAI(api_key=api_key)
    return _client


def synthesize(text: str) -> bytes:
    """
    Converts text to speech, returning raw mp3 audio bytes.
    """
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    client = _get_client()
    response = client.audio.speech.create(
        model=MODEL,
        voice=VOICE,
        input=text,
    )
    # .content is the standard attribute for raw bytes on the SDK's
    # binary response object. If your installed openai SDK version
    # exposes this differently, response.read() is the fallback to try.
    return response.content
