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


@dataclass
class NextQuestion:
    phrased: str   # what the candidate actually sees/hears
    raw: str       # original question-bank wording, for dedup — equals `phrased` for follow-ups


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

    system_prompt = f"""
You are conducting a {session.difficulty}-level {session.category} interview.
Rephrase the question below so it reads naturally as the next line in an
ongoing conversation. You may briefly acknowledge the candidate's last
answer first if that fits, but do not change the technical substance of
the question, and do not answer it yourself. Return only the question
you would say next — no preamble, no labels.
Act like a professional interviewer dont answer any question that is not related to the interview, 
Keep it concise and clear and also ask follow-up questions if need else ask the next question.

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
        # question-bank entry to track, so raw == phrased here.
        return NextQuestion(phrased=follow_up, raw=follow_up)

    query_text = _build_query_text(session)
    candidates = get_relevant_questions(
        category=session.category,
        difficulty=session.difficulty,
        query_text=query_text,
        exclude_questions=session.asked_question_texts(),
        k=RETRIEVAL_K,
    )

    if not candidates:
        raise RuntimeError(
            f"No unused questions left for category='{session.category}', "
            f"difficulty='{session.difficulty}'. Widen the question bank "
            f"or end the session."
        )

    # Pick among the top few rather than always the single best match —
    # avoids every same-state session feeling identically scripted.
    chosen_raw = random.choice(candidates)["question"]
    phrased = _phrase_question(chosen_raw, session)

    return NextQuestion(phrased=phrased, raw=chosen_raw)


if __name__ == "__main__":
    # Manual smoke test — requires OPENAI_API_KEY to be set in .env,
    # and vector_db/ to already exist (run embed_questions.py first).
    test_session = InterviewSession(category="frontend", difficulty="junior")
    result = get_next_question(test_session)
    print("Phrased (shown to candidate):\n", result.phrased)
    print("\nRaw (used for dedup):\n", result.raw)
