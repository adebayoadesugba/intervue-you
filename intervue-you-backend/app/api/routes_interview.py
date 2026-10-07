"""
routes_interview.py

FastAPI endpoints for running an interview session over HTTP.

Each HTTP request is stateless, so interview_loop.run_session()'s
single blocking while loop (built for the CLI) doesn't fit directly.
Instead, this calls the same underlying primitives —
interview_loop.ask_next() and interview_loop.record_answer() — one
per request, so the actual turn logic lives in exactly one place and
can't drift out of sync between the CLI and the API.

  POST /interview/start          -> creates a session, returns first question
  POST /interview/{id}/answer    -> records an answer, evaluates it, returns
                                     the next question (or marks the session ended)
  GET  /interview/{id}/report    -> returns the final report once ended

Sessions are held in memory, keyed by session_id. Fine for local
development and a single-server deployment; does NOT survive a
server restart or scale across multiple processes — swap SESSIONS
for a real store (Redis, a DB table) before a multi-instance
production deployment.
"""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents import interviewer_agent
from app.interview_loop import ask_next, record_answer
from app.models.session import InterviewSession
from app.agents import report_agent  


router = APIRouter(prefix="/interview", tags=["interview"])

SESSIONS: dict[str, InterviewSession] = {}
SESSION_START_TIMES: dict[str, float] = {}
SESSION_TIME_LIMITS: dict[str, int | None] = {}


class StartRequest(BaseModel):
    category: str
    difficulty: str
    mode: str = "text"
    max_questions: int = 8
    time_limit_seconds: int | None = None  # None = no time limit, only max_questions applies


class StartResponse(BaseModel):
    session_id: str
    question: str


class AnswerRequest(BaseModel):
    answer: str


class AnswerResponse(BaseModel):
    ended: bool
    next_question: str | None = None
    reason: str | None = None  # "time_limit" | "max_questions" | "bank_exhausted" | "already_ended"


def _time_is_up(session_id: str) -> bool:
    limit = SESSION_TIME_LIMITS.get(session_id)
    if limit is None:
        return False
    return (time.monotonic() - SESSION_START_TIMES[session_id]) >= limit


def _get_session(session_id: str) -> InterviewSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/start", response_model=StartResponse)
def start_interview(req: StartRequest):
    session = InterviewSession(
        category=req.category,
        difficulty=req.difficulty,
        mode=req.mode,
        max_questions=req.max_questions,
    )

    SESSIONS[session.session_id] = session
    SESSION_START_TIMES[session.session_id] = time.monotonic()
    SESSION_TIME_LIMITS[session.session_id] = req.time_limit_seconds

    try:
        next_q = ask_next(session)
    except interviewer_agent.QuestionBankExhausted as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StartResponse(session_id=session.session_id, question=next_q.phrased)


@router.post("/{session_id}/answer", response_model=AnswerResponse)
def submit_answer(session_id: str, req: AnswerRequest):
    session = _get_session(session_id)

    # Reject only if nothing is actually waiting for an answer — NOT
    # just because should_end() is true. The question that pushes the
    # session over its limit is still unanswered at the moment it's
    # asked, and deserves to be recorded before the session actually
    # ends. Checking should_end() alone here would silently drop that
    # final answer, exactly as it did before this fix.
    has_pending_question = len(session.questions_asked) > len(session.scores)

    # Time check happens here — between turns, before whatever would
    # come next (follow-up or fresh) — same principle as the CLI loop.

    if not has_pending_question:
        return AnswerResponse(ended=True, reason="already_ended")

    record_answer(session, req.answer)

    if _time_is_up(session_id):
        session.end()
        return AnswerResponse(ended=True, reason="time_limit")
   

    if session.should_end():
        return AnswerResponse(ended=True, reason="max_questions")

    try:
        next_q = ask_next(session)
    except interviewer_agent.QuestionBankExhausted:
        session.end()
        return AnswerResponse(ended=True, reason="bank_exhausted")

    return AnswerResponse(ended=False, next_question=next_q.phrased)


@router.get("/{session_id}/report")
def get_report(session_id: str):
    session = _get_session(session_id)
    
    if not session.should_end():
        raise HTTPException(status_code=400, detail="Session is still in progress")

    # Generate the AI written report
    ai_feedback = report_agent.generate_report(session)

    # Combine the raw metrics with the AI feedback
    return {
        "session_id": session.session_id,
        "category": session.category,
        "metrics": {
            "questions_asked": session.original_question_count,
            "max_questions": session.max_questions,
            "total_turns_incl_follow_ups": len(session.questions_asked),
            "average_score": round(session.average_score(), 1),
            "overall_percentage": f"{session.percentage_score()}%", # <--- New strict percentage
            "scores_by_difficulty": session.scores_by_difficulty(),
        },
        "feedback": ai_feedback,
    }