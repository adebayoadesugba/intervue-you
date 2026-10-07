"""
report_agent.py

Analyzes an completed InterviewSession and generates a structured,
human-readable final performance report. It looks at the trajectory
of the scores, the difficulty shifts, and the specific feedback given
during the interview to write a cohesive summary.
"""

import os
from pydantic import BaseModel
from openai import OpenAI

from app.models.session import InterviewSession

MODEL = "gpt-4.1-mini"
_client: OpenAI | None = None


class FinalReportSchema(BaseModel):
    overall_summary: str
    strengths: list[str]
    areas_for_improvement: list[str]
    difficulty_progression_note: str


REPORT_SYSTEM_PROMPT = """
You are an expert technical interviewer writing a final performance report for a candidate.
You will be provided with the statistics and turn-by-turn breakdown of their interview.

Your goal is to write a professional and highly specific evaluation.

Guidelines:
1. 'overall_summary': Write a 2-3 sentence overview of their performance. You MUST explicitly state their "Overall Final Score" percentage in this summary.
2. 'strengths': List 2-3 specific technical concepts they handled well.
3. 'areas_for_improvement': List 1-2 specific technical gaps or communication issues.
4. 'difficulty_progression_note': Write 1 sentence explaining how the difficulty shifted.

CRITICAL TIME PENALTY: Compare "Questions Asked" to "Target Max Questions". 
If the candidate answered fewer questions than the target before time ran out, 
you MUST explicitly penalize them in the `overall_summary` and `areas_for_improvement`. 
State that they struggled with time management, lacked conciseness, and failed to complete the technical assessment. 
""".strip()

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def generate_report(session: InterviewSession) -> dict:
    """
    Generates a structured final report from the session state.
    Returns a dictionary matching FinalReportSchema so it can be directly
    returned by FastAPI as JSON.
    """
    # If the user didn't submit a single valid answer, return a fallback.
    if not session.scores:
        return {
            "overall_summary": "The session ended before any technical answers could be fully evaluated.",
            "strengths": [],
            "areas_for_improvement": ["Ensure you manage your time effectively to complete the technical questions."],
            "difficulty_progression_note": "Difficulty remained unchanged as no answers were scored.",
        }

    # Build a text summary of the turns to give the LLM context
    turn_summaries = []
    for q, score in zip(session.questions_asked, session.scores):
        q_type = "Follow-up" if q.is_follow_up else "Main Question"
        turn_summaries.append(
            f"- {q_type} ({q.difficulty} level): Score {score.score}/10. "
            f"Evaluator Feedback: {score.feedback}"
        )
    
    context = "\n".join(turn_summaries)

    user_prompt = f"""
Category: {session.category}
Initial Difficulty: {session.questions_asked[0].difficulty if session.questions_asked else session.difficulty}
Max Questions: {session.max_questions}
Questions Asked (Main Topics Completed): {session.original_question_count}
Final Average Quality Score: {session.average_score():.1f}/10
Overall Final Score (Penalizes unanswered questions): {session.percentage_score()}%

Turn-by-turn breakdown:
{context}
"""

    client = _get_client()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FinalReportSchema,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Report agent failed to parse structured output.")

    return parsed.model_dump()