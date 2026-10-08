"""
interviewer_agent.py

Decides what to ask next in an interview session:
  1. If the previous evaluation flagged a follow-up, ask that directly
     (no retrieval, no rephrasing — it's already conversational).
  2. Otherwise, retrieve a fresh candidate question from the question
     bank and rephrase it to read naturally in context.

This module does NOT mutate the session — it returns a NextQuestion
with both the phrased text (to show/speak to the candidate) and the
raw text (to pass into session.add_question(..., raw_text=...) so
future retrieval calls can correctly exclude it). The caller owns
all session mutation, same as every other agent in this project.

Run directly as a smoke test:
    python interviewer_agent.py
"""

import os
import random
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from app.models.session import InterviewSession
from app.rag.retriever import get_relevant_questions

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"      # cheap/fast is fine here — this is templating, not deep reasoning
RETRIEVAL_K = 3              # fetch a few candidates, not just the top match, for variety
HISTORY_WINDOW = 6           # how many recent turns to use as context

_client: OpenAI | None = None


class QuestionBankExhausted(RuntimeError):
    """Raised when no unused questions remain for a session's current
    category/difficulty. Deliberately a distinct type from plain
    RuntimeError, so callers can catch this specifically and end the
    session gracefully — without also accidentally swallowing an
    unrelated RuntimeError (e.g. an evaluator API failure) as if it
    meant the same thing."""


@dataclass
class NextQuestion:
    phrased: str   # what the candidate actually sees/hears
    raw: str       # original question-bank wording, for dedup — equals `phrased` for follow-ups
    reference_points: list[str]  # for the evaluator; empty for follow-ups (no bank entry)
    is_follow_up: bool  # explicit, not inferred — lets session.py exclude follow-ups from max_questions


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — check that your .env file is loaded.")
        _client = OpenAI(api_key=api_key)
    return _client


def _build_query_text(session: InterviewSession) -> str:
    """Turns recent conversation history into text used to semantically
    rank candidate questions. Empty on the very first turn — retriever.py
    already handles an empty query_text explicitly (unranked sample)."""
    if not session.history:
        return ""
    recent = session.history[-HISTORY_WINDOW:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)


def _phrase_question(raw_question: str, session: InterviewSession) -> str:
    """Wraps a raw question-bank question so it reads as the next line
    of an ongoing conversation, rather than a flat recital."""
    client = _get_client()

    # Check if this is the very first question of the session
    is_first_question = len(session.history) == 0

    if is_first_question:
        bridge_instruction = (
            "This is the very first question of the interview. Start with a brief, warm "
            "introductory phrase that will set the tone for the conversation. DO NOT compliment or refer to previous answers, because the candidate "
            "has not said anything yet."
        )
    else:
        bridge_instruction = (
            "CRITICAL INSTRUCTION: Since this is an interactive interview, ALWAYS start your response "
            "by briefly and naturally reacting to the candidate's last answer depending on how good their response was "
            "(e.g., 'That's a solid explanation.' if they gave a really good answer and no follow-up is needed, "
            "'Great point.' if they gave a decent answer and a follow-up is needed, "
            "'Makes perfect sense. Moving on...', or 'Good attempt.' if their response was fair but could be improved) before transitioning "
            "to the new question. Do not answer the question for them, just bridge the conversation smoothly."
        )

    system_prompt = f"""
You are conducting a {session.difficulty}-level {session.category} interview.
Rephrase the question below so it reads naturally in conversation.

{bridge_instruction}

Question to ask:
{raw_question}
""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session.history[-HISTORY_WINDOW:])
    messages.append({"role": "user", "content": "Ask the next question now."})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def get_next_question(session: InterviewSession) -> NextQuestion:
    """
    Returns the next question to ask, as a NextQuestion(phrased, raw).

    Raises RuntimeError if there are no unused questions left in the
    bank for this session's category/difficulty — the caller should
    catch this and end the session gracefully (e.g. route to the
    final report) rather than let it crash the interview loop.
    """
    follow_up = session.pending_follow_up()
    if follow_up:
        # Already generated in context by the evaluator agent — no
        # question-bank entry to track, so raw == phrased and there
        # are no reference_points to carry forward.
        return NextQuestion(phrased=follow_up, raw=follow_up, reference_points=[], is_follow_up=True)

    query_text = _build_query_text(session)
    candidates = get_relevant_questions(
        category=session.category,
        difficulty=session.difficulty,
        stack=session.stack,
        query_text=query_text,
        exclude_questions=session.asked_question_texts(),
        k=RETRIEVAL_K,
    )

    if not candidates:
        raise QuestionBankExhausted(
            f"No unused questions left for category='{session.category}', "
            f"difficulty='{session.difficulty}', stack='{session.stack}'. Widen the question bank "
            f"or end the session."
        )

    # Pick among the top few rather than always the single best match —
    # avoids every same-state session feeling identically scripted.
    chosen = random.choice(candidates)
    phrased = _phrase_question(chosen["question"], session)

    return NextQuestion(
        phrased=phrased,
        raw=chosen["question"],
        reference_points=chosen.get("reference_points", []),
        is_follow_up=False,
    )


if __name__ == "__main__":
    # Manual smoke test — requires OPENAI_API_KEY to be set in .env,
    # and vector_db/ to already exist (run embed_questions.py first).
    test_session = InterviewSession(category="frontend", difficulty="junior", stack="core")
    result = get_next_question(test_session)
    print("Phrased (shown to candidate):\n", result.phrased)
    print("\nRaw (used for dedup):\n", result.raw)