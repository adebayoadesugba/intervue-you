"""
interview_loop.py

Orchestrates one full interview session: repeatedly calls the
interviewer agent to get the next question, waits for the candidate's
answer, calls the evaluator agent to score it, and mutates the session
with the results — until session.should_end() is true.

This is the one place that owns session mutation end to end. Both
agents stay read-only / return-value-only, exactly as designed.

This module does NOT handle text vs. voice I/O or FastAPI routing —
get_answer_fn is injected by the caller, so this same loop works
whether answers come from stdin (this file's own CLI runner), a web
request, or a transcribed voice clip.
"""

import time
from collections.abc import Callable

from app.agents import evaluator_agent, interviewer_agent
from app.models.session import InterviewSession


def run_turn(session: InterviewSession, get_answer_fn: Callable[[str], str]) -> None:
    """
    Runs exactly one question-and-answer turn:
      ask -> get_answer_fn(question) -> evaluate -> record.

    get_answer_fn receives the phrased question and must return the
    candidate's answer as plain text (already transcribed, if voice).

    Raises whatever interviewer_agent.get_next_question() raises if the
    question bank is exhausted — callers should catch RuntimeError and
    end the session gracefully rather than let it propagate unhandled.
    """
    next_q = interviewer_agent.get_next_question(session)
    session.add_question(
        next_q.phrased,
        raw_text=next_q.raw,
        reference_points=next_q.reference_points,
        is_follow_up=next_q.is_follow_up,
    )

    answer = get_answer_fn(next_q.phrased)
    session.add_answer(answer)

    evaluation = evaluator_agent.evaluate_answer(
        session=session,
        question_asked=session.questions_asked[-1].text,
        candidate_answer=answer,
        reference_points=session.questions_asked[-1].reference_points,
    )
    session.add_evaluation(evaluation)


def run_session(
    session: InterviewSession,
    get_answer_fn: Callable[[str], str],
    time_limit_seconds: int | None = None,
) -> None:
    """
    Runs turns until the session ends — max_questions is hit, the time
    limit is reached, session.end() was called externally, or the
    question bank runs dry.

    The time check happens BETWEEN turns, not mid-turn: a turn already
    in progress is always allowed to finish. This means actual elapsed
    time can run slightly past time_limit_seconds (by however long one
    question-answer-evaluate cycle takes) — a deliberate choice, not
    an oversight; see the design discussion this was built from.
    """
    start_time = time.monotonic() if time_limit_seconds else None

    while not session.should_end():
        if start_time is not None and (time.monotonic() - start_time) >= time_limit_seconds:
            print(f"\n[time limit of {time_limit_seconds}s reached — ending session]")
            session.end()
            break

        try:
            run_turn(session, get_answer_fn)
        except interviewer_agent.QuestionBankExhausted as e:
            # Genuinely unrecoverable without more data — end gracefully.
            # Any OTHER RuntimeError (e.g. the evaluator failing to parse
            # a response) is deliberately NOT caught here — it propagates
            # as a real error rather than being mislabeled as "interview
            # complete."
            print(f"\n[ending session early: {e}]")
            session.end()
            break


def _cli_get_answer(question: str) -> str:
    """Simple text-mode answer source for local testing — swap this out
    for a FastAPI request handler or the voice pipeline later."""
    print(f"\nInterviewer: {question}")
    return input("You: ")


if __name__ == "__main__":
    # Manual end-to-end smoke test — requires OPENAI_API_KEY set and
    # vector_db/ already built. Answer in the terminal to try it live.
    test_session = InterviewSession(category="frontend", difficulty="junior", max_questions=5)
    run_session(test_session, _cli_get_answer, time_limit_seconds=600)  # Fixed to 600 seconds

    print("\n--- Session summary ---")
    print(f"Questions asked: {len(test_session.questions_asked)}")
    print(f"Average score: {test_session.average_score():.1f}")
    print(f"Final difficulty: {test_session.difficulty}")
    print(f"By difficulty: {test_session.scores_by_difficulty()}")