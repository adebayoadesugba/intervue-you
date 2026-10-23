"""
routes_interview.py

FastAPI endpoints for running an interview session over HTTP.
Provides strict input validation, concurrency locks against double-submits,
abandoned session handling, background TTL memory pruning, and real-time Server-Sent Events (SSE) streaming.
"""

import time
import json
import asyncio
import logging
from typing import Literal
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents import report_agent, resume_agent
from app.interview_loop import ask_next, record_answer
from app.models.session import InterviewSession
from app.rag import retriever
from app.agents.interviewer_agent import QuestionBankExhausted, get_next_question_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/interview", tags=["interview"])

# Session TTL configuration (2 hours in seconds)
SESSION_TTL_SECONDS = 7200


class SessionStore:
    """
    Thread-safe session store with per-session async locks to prevent double-submit race conditions.
    Can be adapted to Redis or SQLite for multi-worker production deployments.
    """
    def __init__(self):
        self._sessions: dict[str, InterviewSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_accessed: dict[str, float] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def get(self, session_id: str) -> InterviewSession | None:
        session = self._sessions.get(session_id)
        if session:
            self._last_accessed[session_id] = time.monotonic()
        return session

    def set(self, session: InterviewSession) -> None:
        self._sessions[session.session_id] = session
        self._last_accessed[session.session_id] = time.monotonic()

    def prune_stale(self, max_age_seconds: int = SESSION_TTL_SECONDS) -> int:
        now = time.monotonic()
        stale_ids = [
            sid for sid, last_active in self._last_accessed.items()
            if (now - last_active) > max_age_seconds
        ]
        for sid in stale_ids:
            self._sessions.pop(sid, None)
            self._locks.pop(sid, None)
            self._last_accessed.pop(sid, None)
        return len(stale_ids)


store = SessionStore()


# --- Pydantic Input Schemas ---

class StartRequest(BaseModel):
    category: str = Field(..., min_length=2, max_length=50, description="Interview category/role")
    difficulty: str = Field(..., min_length=2, max_length=20, description="Initial difficulty level")
    stack: list[str] = Field(default_factory=lambda: ["core"], max_items=5, description="Technology stacks")
    mode: Literal["text", "voice", "video"] = Field("text", description="Delivery mode")
    time_limit_minutes: int = Field(10, ge=1, le=60, description="Session duration in minutes (1-60)")
    cv_text: str | None = Field(None, max_length=12000, description="Optional raw candidate CV text (capped at 12k chars)")


class StartResponse(BaseModel):
    session_id: str
    question: str
    is_follow_up: bool = False


class AnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=4000, description="Candidate answer (1-4000 chars)")


class AnswerResponse(BaseModel):
    ended: bool
    next_question: str | None = None
    reason: str | None = None  # "time_limit" | "agenda_completed" | "bank_exhausted" | "already_ended"
    is_follow_up: bool = False


# --- Helper Functions ---

def _get_session_or_404(session_id: str) -> InterviewSession:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found or expired. Please start a new session."
        )
    return session


def _trigger_background_cleanup(background_tasks: BackgroundTasks) -> None:
    """Schedules session cleanup in the background without blocking API responses."""
    background_tasks.add_task(store.prune_stale)


# --- Standard Non-Streaming Routes ---

@router.post("/start", response_model=StartResponse)
async def start_interview(req: StartRequest, background_tasks: BackgroundTasks):
    _trigger_background_cleanup(background_tasks)

    # 1. Instantiate Session with Domain Error Protection
    try:
        session = InterviewSession(
            category=req.category,
            difficulty=req.difficulty,
            stack=req.stack,
            mode=req.mode,
            duration_minutes=req.time_limit_minutes,
            cv_text=req.cv_text
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 2. Pre-Flight Vector DB Check
    is_valid, err_msg = retriever.verify_session_agenda_fillable(session)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question Bank Configuration Error: {err_msg}"
        )

    # 3. Process Optional CV Extraction
    if req.cv_text and req.cv_text.strip():
        profile = resume_agent.extract_resume_profile(req.cv_text)
        if profile and profile.suggested_questions:
            session.resume_questions = profile.suggested_questions
            logger.info(f"CV extracted {len(profile.suggested_questions)} project deep-dive questions.")

    store.set(session)

    # 4. Generate First Question
    try:
        next_q = ask_next(session)
        if not next_q:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize session question."
            )
    except QuestionBankExhausted as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return StartResponse(
        session_id=session.session_id,
        question=next_q.phrased,
        is_follow_up=getattr(next_q, "is_follow_up", False),
    )


@router.post("/{session_id}/answer", response_model=AnswerResponse)
async def submit_answer(session_id: str, req: AnswerRequest, background_tasks: BackgroundTasks):
    _trigger_background_cleanup(background_tasks)
    session = _get_session_or_404(session_id)
    lock = store.get_lock(session_id)

    # Concurrency Lock (Double-Submit Protection)
    async with lock:
        # Reject if session is already ended or no question is awaiting an answer
        has_pending_question = len(session.questions_asked) > len(session.scores)
        if session.ended or not has_pending_question:
            return AnswerResponse(ended=True, reason="already_ended")

        # 1. Record and Evaluate Answer
        record_answer(session, req.answer)

        # 2. Check Expiration & Agenda Completion
        if session.should_end():
            session.end()
            elapsed = time.monotonic() - session.start_timestamp
            reason = "time_limit" if elapsed >= session.duration_seconds else "agenda_completed"
            return AnswerResponse(ended=True, reason=reason)

        # 3. Fetch Next Question
        try:
            next_q = ask_next(session)
        except QuestionBankExhausted:
            session.end()
            return AnswerResponse(ended=True, reason="bank_exhausted")

        if not next_q:
            session.end()
            return AnswerResponse(ended=True, reason="agenda_completed")

        return AnswerResponse(
            ended=False,
            next_question=next_q.phrased,
            is_follow_up=getattr(next_q, "is_follow_up", False),
        )


# --- New Real-Time SSE Streaming Routes ---

@router.post("/start/stream")
async def start_interview_stream(req: StartRequest, background_tasks: BackgroundTasks):
    _trigger_background_cleanup(background_tasks)

    try:
        session = InterviewSession(
            category=req.category,
            difficulty=req.difficulty,
            stack=req.stack,
            mode=req.mode,
            duration_minutes=req.time_limit_minutes,
            cv_text=req.cv_text
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    is_valid, err_msg = retriever.verify_session_agenda_fillable(session)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question Bank Configuration Error: {err_msg}"
        )

    if req.cv_text and req.cv_text.strip():
        profile = resume_agent.extract_resume_profile(req.cv_text)
        if profile and profile.suggested_questions:
            session.resume_questions = profile.suggested_questions
            logger.info(f"CV extracted {len(profile.suggested_questions)} project deep-dive questions.")

    store.set(session)

    async def event_generator():
        try:
            # Emit initial session metadata event
            yield f"data: {json.dumps({'type': 'session_created', 'session_id': session.session_id})}\n\n"

            stream_gen = get_next_question_stream(session)
            final_nq = None
            for event_type, data in stream_gen:
                if event_type == "meta":
                    yield f"data: {json.dumps({'type': 'meta', 'is_follow_up': data.is_follow_up})}\n\n"
                elif event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
                elif event_type == "done":
                    final_nq = data

            if not final_nq:
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Failed to initialize session question.'})}\n\n"
                return

            # Commit question atomically to session state
            session.add_question(
                question_text=final_nq.phrased,
                raw_text=final_nq.raw,
                reference_points=final_nq.reference_points,
                is_follow_up=final_nq.is_follow_up,
                slot_id=final_nq.slot_id,
                question_type=final_nq.question_type,
                competency=final_nq.competency,
                question_id=final_nq.question_id,
                stack=final_nq.stack
            )

            yield f"data: {json.dumps({'type': 'done', 'session_id': session.session_id, 'question': final_nq.phrased, 'is_follow_up': final_nq.is_follow_up})}\n\n"

        except Exception as e:
            logger.exception("start/stream failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{session_id}/answer/stream")
async def submit_answer_stream(session_id: str, req: AnswerRequest, background_tasks: BackgroundTasks):
    _trigger_background_cleanup(background_tasks)
    session = _get_session_or_404(session_id)
    lock = store.get_lock(session_id)

    async def event_generator():
        try:
            async with lock:
                has_pending_question = len(session.questions_asked) > len(session.scores)
                if session.ended or not has_pending_question:
                    yield f"data: {json.dumps({'type': 'ended', 'ended': True, 'reason': 'already_ended'})}\n\n"
                    return

                # 1. Record and Evaluate Answer
                record_answer(session, req.answer)

                # 2. Check Expiration & Agenda Completion
                if session.should_end():
                    session.end()
                    elapsed = time.monotonic() - session.start_timestamp
                    reason = "time_limit" if elapsed >= session.duration_seconds else "agenda_completed"
                    yield f"data: {json.dumps({'type': 'ended', 'ended': True, 'reason': reason})}\n\n"
                    return

                # 3. Stream Next Question Tokens
                try:
                    stream_gen = get_next_question_stream(session)
                    final_nq = None
                    for event_type, data in stream_gen:
                        if event_type == "meta":
                            yield f"data: {json.dumps({'type': 'meta', 'is_follow_up': data.is_follow_up})}\n\n"
                        elif event_type == "token":
                            yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
                        elif event_type == "done":
                            final_nq = data

                    if not final_nq:
                        session.end()
                        yield f"data: {json.dumps({'type': 'ended', 'ended': True, 'reason': 'agenda_completed'})}\n\n"
                        return

                    # Commit question atomically to session state
                    session.add_question(
                        question_text=final_nq.phrased,
                        raw_text=final_nq.raw,
                        reference_points=final_nq.reference_points,
                        is_follow_up=final_nq.is_follow_up,
                        slot_id=final_nq.slot_id,
                        question_type=final_nq.question_type,
                        competency=final_nq.competency,
                        question_id=final_nq.question_id,
                        stack=final_nq.stack
                    )

                    yield f"data: {json.dumps({'type': 'done', 'ended': False, 'next_question': final_nq.phrased, 'is_follow_up': final_nq.is_follow_up})}\n\n"

                except QuestionBankExhausted:
                    session.end()
                    yield f"data: {json.dumps({'type': 'ended', 'ended': True, 'reason': 'bank_exhausted'})}\n\n"
        except Exception:
            logger.exception("answer/stream failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Internal error, please retry.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    session = _get_session_or_404(session_id)

    if not session.ended:
        session.end()

    # Generate 4D Scorecard Report
    report = report_agent.generate_report(session)

    return {
        "session_id": session.session_id,
        "category": session.category,
        "metrics": {
            "questions_attempted": len(session.scores),
            "follow_ups_needed": report.get("follow_up_count", 0),
            "incomplete_technical_slots": report.get("incomplete_technical_slots", 0),
            "overall_percentage": f"{report.get('overall_score') or 0}%",
            "scores_by_difficulty": session.scores_by_difficulty(),
        },
        "report": report,
    }