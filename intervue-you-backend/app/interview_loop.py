"""
interview_loop.py

Core turn logic for running an interview session.
Handles atomic state commits, contextual follow-up evaluation, 
and explicit agenda completion signaling.
"""

import logging
from collections.abc import Callable

from app.agents import evaluator_agent, interviewer_agent
from app.agents.interviewer_agent import NextQuestion
from app.models.session import AnswerEvaluation, InterviewSession

logger = logging.getLogger(__name__)


def ask_next(session: InterviewSession) -> NextQuestion | None:
    """
    Retrieves the next question from the Blueprint Engine.
    If no questions remain or the agenda is complete, marks session ended.
    """
    next_q = interviewer_agent.get_next_question(session)
    
    if not next_q:
        logger.info(f"Session {session.session_id}: Agenda completed or time expired.")
        session.end()
        return None

    # Atomic addition: session.add_question handles index advancement safely
    session.add_question(
        question_text=next_q.phrased,
        raw_text=next_q.raw,
        reference_points=next_q.reference_points,
        is_follow_up=next_q.is_follow_up,
        slot_id=next_q.slot_id,
        question_type=next_q.question_type,
        competency=next_q.competency,
        question_id=next_q.question_id,
    )
    return next_q


def record_answer(session: InterviewSession, answer: str) -> AnswerEvaluation:
    """
    Evaluates candidate response and commits answer + score atomically.
    Includes parent turn context when evaluating follow-up exchanges.
    Manages follow-up queue state and main agenda slot advancement.
    """
    if not session.questions_asked:
        raise RuntimeError("Cannot record answer: No question has been asked in this session.")

    last_asked = session.questions_asked[-1]
    
    # Check if an evaluation was already recorded for this exact turn
    if last_asked.evaluation is not None:
        logger.warning(f"Turn already evaluated for question: '{last_asked.text[:30]}...'")
        return last_asked.evaluation

    # Build evaluation context (Include parent Q&A context for follow-ups)
    evaluation_question = last_asked.text
    if last_asked.is_follow_up and len(session.questions_asked) >= 2:
        parent_q = session.questions_asked[-2]
        parent_ans = parent_q.answer_text or "(No initial answer provided)"
        evaluation_question = (
            f"Original Question: {parent_q.text}\n"
            f"Candidate Initial Answer: {parent_ans}\n"
            f"Follow-Up Question: {last_asked.text}"
        )

    # Determine question_type with explicit fallback logging
    q_type = last_asked.question_type
    if not q_type:
        logger.warning("Missing question_type on AskedQuestion. Falling back to 'technical'.")
        q_type = "technical"

    # Evaluate BEFORE mutating session state
    evaluation = evaluator_agent.evaluate_answer(
        session=session,
        question_asked=evaluation_question,
        candidate_answer=answer,
        reference_points=last_asked.reference_points,
        question_type=q_type,
    )

    # Atomic state commit
    session.add_answer(answer)
    session.add_evaluation(evaluation)
    
    # --- Follow-Up & Slot Progression Management ---
    needs_follow_up = (
        getattr(evaluation, "needs_follow_up", False)
        and not last_asked.is_follow_up  # Cap at 1 follow-up per slot
        and bool(getattr(evaluation, "follow_up_question", None))
    )

    if needs_follow_up:
        # Queue follow-up question; DO NOT advance main slot index
        session.set_pending_follow_up(evaluation.follow_up_question)
        logger.info(f"Follow-up queued for main slot {session.current_slot_index}. Holding slot index.")
    else:
        # Clear follow-up state and advance main slot pointer
        session.clear_pending_follow_up()
        session.advance_slot()
        logger.info(f"Main slot finished. Advanced agenda slot index to {session.current_slot_index}.")

    return evaluation


def run_turn(session: InterviewSession, get_answer_fn: Callable[[str], str]) -> None:
    """Runs a single ask -> answer -> evaluate turn for CLI testing."""
    next_q = ask_next(session)
    if not next_q:
        return
        
    answer = get_answer_fn(next_q.phrased)
    record_answer(session, answer)


def run_session(
    session: InterviewSession,
    get_answer_fn: Callable[[str], str],
) -> None:
    """Blocking turn loop for CLI testing."""
    while not session.should_end():
        try:
            run_turn(session, get_answer_fn)
        except interviewer_agent.QuestionBankExhausted as e:
            logger.info(f"Ending session: {e}")
            session.end()
            break


def _cli_get_answer(question: str) -> str:
    print(f"\nInterviewer: {question}")
    return input("You: ")


if __name__ == "__main__":
    test_session = InterviewSession(category="frontend", difficulty="junior", duration_minutes=5)
    run_session(test_session, _cli_get_answer)

    print("\n--- Session Summary ---")
    print(f"Original Topics Asked: {test_session.original_question_count}")
    print(f"Total Turns (incl. follow-ups): {len(test_session.questions_asked)}")
    print(f"Evaluations Recorded: {len(test_session.scores)}")
    print(f"Final Difficulty: {test_session.difficulty}")