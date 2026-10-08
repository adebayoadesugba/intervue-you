"""
routes_voice.py

Voice endpoints — pure utilities, deliberately decoupled from
InterviewSession entirely:

  POST /voice/transcribe   -> audio in, transcribed text out
  POST /voice/synthesize   -> text in, audio out

Neither endpoint touches session state. The frontend is expected to:
  1. Record the candidate's spoken answer, POST it to
     /voice/transcribe, get back plain text, then POST that text to
     the EXISTING /interview/{session_id}/answer endpoint (unchanged)
     — the same path a typed answer already takes.
  2. Take the `question` / `next_question` text already returned by
     /interview/start or /interview/{id}/answer, POST it to
     /voice/synthesize, and play the returned audio.

Keeping voice as a pure text<->audio utility — rather than a
duplicate "answer-with-audio" endpoint — means the actual interview
turn logic (ask_next/record_answer) exists in exactly one place,
used identically whether the candidate typed or spoke.
"""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.responses import Response

from app.services import stt_service, tts_service

router = APIRouter(prefix="/voice", tags=["voice"])


class TranscribeResponse(BaseModel):
    text: str


class SynthesizeRequest(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (webm, wav, mp3, m4a — whatever the
    browser's MediaRecorder produced) and returns the transcribed text.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Whisper's API needs a file-like object with a .name attribute
    # (it reads the extension to figure out the audio format) — a
    # raw UploadFile doesn't expose that the way open() does, so wrap
    # the bytes and attach the original filename manually.
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = file.filename or "audio.webm"

    try:
        text = stt_service.transcribe(audio_file)
    except stt_service.TranscriptionEmptyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # e.g. missing API key — a server config problem, not the
        # candidate's fault, so this is 500 not 422.
        raise HTTPException(status_code=500, detail=str(e))

    return TranscribeResponse(text=text)


@router.post("/synthesize")
def synthesize_speech(req: SynthesizeRequest):
    """
    Converts text to speech and returns mp3 audio.
    """
    try:
        audio_bytes = tts_service.synthesize(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return a standard Response instead of a StreamingResponse
    return Response(content=audio_bytes, media_type="audio/mpeg")