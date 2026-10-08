"""
session.py

Holds all state for one interview session: the running conversation
history, which questions have been asked (and at what difficulty each
was asked), every evaluation collected so far, and the current
(adaptive) difficulty level.

This object gets passed into the retriever, the interviewer agent, and
the evaluator agent — it's the shared state that lets the loop behave
like one continuous interview rather than isolated Q&A pairs.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# Not every interview category fits a junior/mid/senior progression —
# a graduate-trainee or NYSC screening interview typically has one tier,
# not three. Each category gets its own ladder here; anything not
# listed falls back to DEFAULT_LADDER. Add a category's own entry only
# when it genuinely needs more than one difficulty level.
DIFFICULTY_LADDERS: dict[str, list[str]] = {
    "graduate-trainee": ["general"],
    "nysc-trainee": ["general"],
    "uiux": ["entry", "advanced", "expert"],  # <--- Add your custom progression here
    "copywriting": ["entry", "expert"]        # <--- Or a simple two-tier ladder
}
DEFAULT_LADDER = ["junior", "mid", "senior"]

def get_ladder(category: str) -> list[str]:
    return DIFFICULTY_LADDERS.get(category, DEFAULT_LADDER)


@dataclass
class AnswerEvaluation:
    """Mirrors the structured-output schema the evaluator agent returns.
    Defined here too so session.py has no import dependency on the
    agents package — keeps this module easy to test on its own."""
    score: int
    technical_accuracy: Literal["poor", "fair", "good", "excellent"]
    communication_clarity: Literal["poor", "fair", "good", "excellent"]
    feedback: str
    should_follow_up: bool
    follow_up_question: str | None
    next_question_difficulty: Literal["easier", "same", "harder"]


@dataclass
class AskedQuestion:
    """A question as it was actually asked — difficulty is captured at
    ask-time, not read later from the session's (by-then-changed)
    current difficulty. This is what makes an honest, difficulty-aware
    category_breakdown possible in the final report."""
    text: str
    difficulty: str
    raw_text: str | None = None  # original question-bank wording, before phrasing;
                                  # falls back to `text` when there's no bank entry
                                  # (e.g. a follow-up question has nothing to fall back from)
    reference_points: list[str] = field(default_factory=list)  # empty for follow-ups
    is_follow_up: bool = False
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InterviewSession:
    category: str
    difficulty: str
    stack: list[str] = field(default_factory=lambda: ["core"]) # Stores the specific framework/stack chosen
    mode: Literal["text", "voice", "video"] = "text"

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    history: list[dict] = field(default_factory=list)              # [{"role": ..., "content": ...}]
    questions_asked: list[AskedQuestion] = field(default_factory=list)
    scores: list[AnswerEvaluation] = field(default_factory=list)

    max_questions: int = 8
    ended: bool = False
    original_question_count: int = 0  # fresh topics only — follow-ups don't increment this
    consecutive_follow_ups: int = 0

    def __post_init__(self) -> None:
        # Normalize once, here, so every downstream file (retriever,
        # agents) can trust these values are always clean lowercase
        # and a valid ladder entry — no need to re-check everywhere.
        self.category = self.category.strip().lower()
        self.difficulty = self.difficulty.strip().lower()
        self.stack = [s.strip().lower() for s in self.stack]

        ladder = get_ladder(self.category)
        if self.difficulty not in ladder:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}' for category "
                f"'{self.category}'. Must be one of {ladder}."
            )

    # --- mutators -----------------------------------------------------

    def add_question(
        self,
        question_text: str,
        raw_text: str | None = None,
        reference_points: list[str] | None = None,
        is_follow_up: bool = False,
    ) -> None:
        if is_follow_up:
            self.consecutive_follow_ups += 1
        else:
            self.consecutive_follow_ups = 0
            self.original_question_count += 1

        self.history.append({"role": "assistant", "content": question_text})
        self.questions_asked.append(
            AskedQuestion(
                text=question_text,
                difficulty=self.difficulty,
                raw_text=raw_text,
                reference_points=reference_points or [],
                is_follow_up=is_follow_up,
            )
        )

    def add_answer(self, answer_text: str) -> None:
        self.history.append({"role": "user", "content": answer_text})

    def add_evaluation(self, evaluation: AnswerEvaluation) -> None:
        self.scores.append(evaluation)
        self._apply_difficulty_shift(evaluation.next_question_difficulty)

    def _apply_difficulty_shift(self, direction: str) -> None:
        ladder = get_ladder(self.category)
        idx = ladder.index(self.difficulty)
        if direction == "harder":
            idx = min(idx + 1, len(ladder) - 1)
        elif direction == "easier":
            idx = max(idx - 1, 0)
        self.difficulty = ladder[idx]

    # --- state checks ---------------------------------------------------

    def asked_question_texts(self) -> list[str]:
        """Plain list of RAW question-bank text — what retriever.py's
        exclude_questions argument needs to match against, since Chroma
        stores the original unphrased wording, not whatever the
        interviewer agent rephrased it into for the candidate."""
        return [q.raw_text or q.text for q in self.questions_asked]

    def pending_follow_up(self) -> str | None:
        """Returns the follow-up question to ask next, if the previous
        evaluation flagged one AND we haven't exceeded 2 follow-ups. Otherwise None."""
        if not self.scores or self.consecutive_follow_ups >= 2:
            return None
            
        if self.scores[-1].should_follow_up:
            return self.scores[-1].follow_up_question
        return None

    def should_end(self) -> bool:
        return self.ended or self.original_question_count >= self.max_questions

    def end(self) -> None:
        self.ended = True

    # --- reporting helpers ------------------------------------------------

    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(e.score for e in self.scores) / len(self.scores)

    def percentage_score(self) -> int:
        """Calculates a strict 0-100% score that penalizes unanswered main questions."""
        if not self.scores or self.max_questions <= 0:
            return 0
        
        # 1. How much of the interview did they actually complete? (e.g., 7 / 10 = 0.7)
        completion_ratio = min(self.original_question_count / self.max_questions, 1.0)
        
        # 2. What was the quality of the answers they DID give? (e.g., 8.0 / 10 = 0.8)
        quality_ratio = self.average_score() / 10.0
        
        # 3. Combine for final penalty (0.7 * 0.8 * 100 = 56%)
        return round(completion_ratio * quality_ratio * 100)

    def scores_by_difficulty(self) -> dict[str, list[int]]:
        """Pairs each evaluation with the difficulty its question was
        actually asked at — only possible because AskedQuestion records
        difficulty at ask-time rather than relying on the session's
        current (by-then-shifted) difficulty."""
        breakdown: dict[str, list[int]] = {}
        for question, evaluation in zip(self.questions_asked, self.scores):
            breakdown.setdefault(question.difficulty, []).append(evaluation.score)
        return breakdown