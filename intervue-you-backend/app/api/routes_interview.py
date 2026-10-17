"""
routes_interview.py

FastAPI endpoints for running an interview session over HTTP.
Provides strict input validation, concurrency locks against double-submits,
abandoned session handling, and background TTL memory pruning.
"""

import time
import asyncio
import logging
from typing import Literal
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.agents import report_agent, resume_agent
from app.interview_loop import ask_next, record_answer
from app.models.session import InterviewSession
from app.rag import retriever
from app.agents.interviewer_agent import QuestionBankExhausted

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


# --- API Routes ---

@router.post("/start", response_model=StartResponse)
async def start_interview(req: StartRequest, background_tasks: BackgroundTasks):
    _trigger_background_cleanup(background_tasks)

    # 1. Instantiate Session with Domain Error Protection (Fix Bug 5)
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

    # Concurrency Lock (Fix Bug 4: Double-Submit Protection)
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


@router.get("/{session_id}/report")
async def get_report(session_id: str):
    session = _get_session_or_404(session_id)

    # Handle Abandoned Sessions (Fix Bug 3)
    # If the user closed the tab, check if duration expired and auto-end session on demand
    elapsed = time.monotonic() - session.start_timestamp
    if elapsed >= session.duration_seconds and not session.pending_follow_up():
        session.end()

    if not session.should_end():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is still in progress. Complete all questions or wait for time to expire."
        )

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