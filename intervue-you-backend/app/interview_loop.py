"""
interview_loop.py

Core turn logic for running an interview session.

ask_next() and record_answer() are the two primitives everything else
is built from. run_session() below is a blocking CLI/local-test loop
built on top of them; app/api/routes_interview.py calls the same two
primitives directly, one per HTTP request, instead of blocking in a
loop. The actual business logic lives in exactly one place either way.
"""

import time
from collections.abc import Callable

from app.agents import evaluator_agent, interviewer_agent
from app.agents.interviewer_agent import NextQuestion
from app.models.session import AnswerEvaluation, InterviewSession


def ask_next(session: InterviewSession) -> NextQuestion:
    """Gets the next question (follow-up or fresh) and records it on
    the session. Raises interviewer_agent.QuestionBankExhausted if
    nothing is left to ask — callers should catch this and end the
    session gracefully rather than let it propagate as a crash."""
    next_q = interviewer_agent.get_next_question(session)
    session.add_question(
        next_q.phrased,
        raw_text=next_q.raw,
        reference_points=next_q.reference_points,
        is_follow_up=next_q.is_follow_up,
    )
    return next_q


def record_answer(session: InterviewSession, answer: str) -> AnswerEvaluation:
    """Records the candidate's answer to the most recently asked
    question, evaluates it, and applies the evaluation — which may
    shift session.difficulty and/or set up a pending follow-up for
    the next call to ask_next()."""
    session.add_answer(answer)
    evaluation = evaluator_agent.evaluate_answer(
        session=session,
        question_asked=session.questions_asked[-1].text,
        candidate_answer=answer,
        reference_points=session.questions_asked[-1].reference_points,
    )
    session.add_evaluation(evaluation)
    return evaluation


def run_turn(session: InterviewSession, get_answer_fn: Callable[[str], str]) -> None:
    """One full ask -> answer -> evaluate cycle, blocking on
    get_answer_fn for the candidate's response. Used by run_session()
    for local/CLI testing only — the API uses ask_next()/record_answer()
    directly instead, since it can't block on a callback per request."""
    next_q = ask_next(session)
    answer = get_answer_fn(next_q.phrased)
    record_answer(session, answer)


def run_session(
    session: InterviewSession,
    get_answer_fn: Callable[[str], str],
    time_limit_seconds: int | None = None,
) -> None:
    """
    Blocking loop for local/CLI testing — runs turns until the session
    ends (max_questions hit, time limit reached, session.end() called
    externally, or the question bank runs dry). The time check happens
    between turns, before whatever would come next, follow-up included
    — a turn already in progress always finishes.
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
            print(f"\n[ending session early: {e}]")
            session.end()
            break


def _cli_get_answer(question: str) -> str:
    print(f"\nInterviewer: {question}")
    return input("You: ")


if __name__ == "__main__":
    test_session = InterviewSession(category="frontend", difficulty="junior", max_questions=5)
    run_session(test_session, _cli_get_answer, time_limit_seconds=300)

    print("\n--- Session summary ---")
    print(f"Questions asked: {test_session.original_question_count}")  # topics only, not follow-ups
    print(f"Total turns (incl. follow-ups): {len(test_session.questions_asked)}")
    print(f"Average score: {test_session.average_score():.1f}")
    print(f"Final difficulty: {test_session.difficulty}")
    print(f"By difficulty: {test_session.scores_by_difficulty()}")