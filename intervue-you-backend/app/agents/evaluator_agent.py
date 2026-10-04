"""
evaluator_agent.py

Scores the candidate's most recent answer using structured outputs,
returning an AnswerEvaluation — the same shape already defined in
session.py, so the caller can pass the result straight into
session.add_evaluation() with no conversion step.

Does not mutate the session — same convention as interviewer_agent.py.
The caller is responsible for calling session.add_evaluation(result).

Run directly as a smoke test:
    python -m app.agents.evaluator_agent
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

from app.models.session import AnswerEvaluation, InterviewSession

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
HISTORY_WINDOW = 6

_client: OpenAI | None = None


# Structured-output schema. Mirrors AnswerEvaluation in session.py field
# for field — kept as a *separate* Pydantic class here (rather than
# reusing the dataclass) because structured outputs require a Pydantic
# BaseModel, while session.py deliberately avoids importing pydantic to
# stay a dependency-free, easily testable module.
class AnswerEvaluationSchema(BaseModel):
    score: int
    technical_accuracy: Literal["poor", "fair", "good", "excellent"]
    communication_clarity: Literal["poor", "fair", "good", "excellent"]
    feedback: str
    should_follow_up: bool
    follow_up_question: str | None
    next_question_difficulty: Literal["easier", "same", "harder"]


EVALUATION_SYSTEM_PROMPT = """
You are evaluating a candidate's answer in a {difficulty}-level {category} interview.

Score the answer on a scale of 1-10, assess technical accuracy and
communication clarity, and give short, specific feedback (1-2 sentences).

Reference points a strong answer would cover:
{reference_points}

Decide should_follow_up:
- true if the answer was vague, incomplete, or raises something worth
  probing deeper before moving on
- false if the answer was clear enough to move to a new question

If should_follow_up is true, write a natural, specific follow_up_question.
If false, follow_up_question must be null.

Decide next_question_difficulty based on this answer alone:
- "harder" if the answer was confident, accurate, and complete
- "same" if adequate but not exceptional
- "easier" if the answer showed clear gaps or confusion
""".strip()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — check that your .env file is loaded.")
        _client = OpenAI(api_key=api_key)
    return _client


def _to_dataclass(parsed: AnswerEvaluationSchema) -> AnswerEvaluation:
    """Converts the Pydantic structured-output result into the plain
    dataclass session.py expects, so the caller never has to know two
    slightly different shapes exist."""
    return AnswerEvaluation(
        score=parsed.score,
        technical_accuracy=parsed.technical_accuracy,
        communication_clarity=parsed.communication_clarity,
        feedback=parsed.feedback,
        should_follow_up=parsed.should_follow_up,
        follow_up_question=parsed.follow_up_question,
        next_question_difficulty=parsed.next_question_difficulty,
    )


def evaluate_answer(
    session: InterviewSession,
    question_asked: str,
    candidate_answer: str,
    reference_points: list[str],
) -> AnswerEvaluation:
    """
    Evaluates the candidate's most recent answer against the question
    that prompted it. Does not read session.history for the answer
    itself — question_asked and candidate_answer are passed explicitly
    so this function works identically whether the answer came from
    text or voice (already transcribed by this point), and so a caller
    testing this in isolation doesn't need a populated session.history
    to get a meaningful result.
    """
    client = _get_client()

    system_prompt = EVALUATION_SYSTEM_PROMPT.format(
        difficulty=session.difficulty,
        category=session.category,
        reference_points="\n".join(f"- {point}" for point in reference_points) or "(none provided)",
    )

    user_prompt = f"Question: {question_asked}\n\nCandidate's answer: {candidate_answer}"

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=AnswerEvaluationSchema,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        # The model refused or couldn't produce a valid structured
        # response — fail loudly rather than silently returning a
        # fabricated default evaluation the caller can't distinguish
        # from a real one.
        raise RuntimeError("Evaluator returned no parsed result — check the raw response for a refusal.")

    return _to_dataclass(parsed)


if __name__ == "__main__":
    # Manual smoke test — requires OPENAI_API_KEY to be set in .env.
    test_session = InterviewSession(category="frontend", difficulty="junior")

    result = evaluate_answer(
        session=test_session,
        question_asked="What is semantic HTML, and why is it important?",
        candidate_answer="Semantic HTML is the practice of using HTML tags that clearly describe the meaning and role of their content to both the browser and the developer, rather than just how the content looks, Semantic Examples: <header>, <nav>, <main>, <article>, <section>, <aside>, <footer>, <figure>, <time>, and headings (<h1> through <h6>).",
        reference_points=[
            "uses tags with meaningful names (e.g. <header>, <article>, <footer>)",
            "improves accessibility for screen readers",
            "helps search engines understand content (SEO)",
        ],
    )

    print(result)
