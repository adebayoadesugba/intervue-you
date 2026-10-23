"""
session.py

Holds all state for one interview session. Agenda progress is DERIVED from
what was actually asked and answered, so it cannot drift.
"""

import time
import uuid
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

DIFFICULTY_LADDERS: dict[str, list[str]] = {
    "graduate-trainee": ["general"],
    "nysc-trainee": ["general"],
    "uiux": ["entry", "advanced", "expert"],
    "copywriting": ["entry", "expert"],
}
DEFAULT_LADDER = ["junior", "mid", "senior"]

MAX_FOLLOW_UPS_PER_SLOT = 1


def get_ladder(category: str) -> list[str]:
    return DIFFICULTY_LADDERS.get(category, DEFAULT_LADDER)


@dataclass
class AgendaSlot:
    slot_id: int
    question_type: str
    competency: str
    stack: str
    is_must_have: bool = True


@dataclass
class AnswerEvaluation:
    score: int
    technical_accuracy: Literal["poor", "fair", "good", "excellent"]
    communication_clarity: Literal["poor", "fair", "good", "excellent"]
    feedback: str
    should_follow_up: bool
    follow_up_question: str | None
    next_question_difficulty: Literal["easier", "same", "harder"]
    rubric_criteria_met: list[str] = field(default_factory=list)
    rubric_criteria_missed: list[str] = field(default_factory=list)


@dataclass
class AskedQuestion:
    text: str
    difficulty: str
    raw_text: str | None = None
    reference_points: list[str] = field(default_factory=list)
    is_follow_up: bool = False
    slot_id: int | None = None
    question_type: str | None = None
    competency: str | None = None
    stack: str | None = None
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    answer_text: str | None = None
    evaluation: AnswerEvaluation | None = None


@dataclass
class InterviewSession:
    category: str
    difficulty: str
    duration_minutes: int = 10
    cv_text: str | None = None
    stack: list[str] = field(default_factory=lambda: ["core"])
    mode: Literal["text", "voice", "video"] = "text"

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_timestamp: float = field(init=False)

    prompt_version: str = "v2.1-strict"
    rubric_version: str = "v1.4-multi-type"
    question_bank_version: str = "2026.10-release"

    agenda: list[AgendaSlot] = field(default_factory=list)
    current_slot_index: int = 0
    user_asked_question_ids: list[str] = field(default_factory=list)

    history: list[dict] = field(default_factory=list)
    questions_asked: list[AskedQuestion] = field(default_factory=list)
    scores: list[AnswerEvaluation] = field(default_factory=list)

    ended: bool = False
    original_question_count: int = 0
    consecutive_follow_ups: int = 0
    pending_follow_up_text: str | None = None
    duration_seconds: int = field(init=False)

    def __post_init__(self) -> None:
        if self.duration_minutes < 1:
            raise ValueError("duration_minutes must be at least 1.")

        self.category = self.category.strip().lower()
        self.difficulty = self.difficulty.strip().lower()
        self.stack = [s.strip().lower() for s in self.stack]

        self.start_timestamp = time.monotonic()
        self.duration_seconds = self.duration_minutes * 60

        ladder = get_ladder(self.category)
        if self.difficulty not in ladder:
            raise ValueError(
                f"Invalid difficulty '{self.difficulty}' for category '{self.category}'. Must be one of {ladder}."
            )

        self.agenda = self._build_time_aware_agenda(self.duration_minutes)

    # ---------------- Agenda construction ----------------

    def _build_time_aware_agenda(self, minutes: int) -> list[AgendaSlot]:
        behavioral_competency = random.choice(["teamwork", "problem_solving", "communication", "teamwork"])

        slots = [
            AgendaSlot(1, "introduction", "communication", "core", is_must_have=True),
            AgendaSlot(2, "behavioral", behavioral_competency, "core", is_must_have=True),
        ]

        slot_counter = 3
        if self.cv_text and self.cv_text.strip():
            slots.append(AgendaSlot(slot_counter, "resume_based", "problem_solving", "core", is_must_have=True))
        else:
            slots.append(AgendaSlot(slot_counter, "behavioral", "teamwork", "core", is_must_have=True))
        slot_counter += 1

        if minutes <= 10:
            tech_slot_count = 1   # 4 slots total
        elif minutes <= 15:
            tech_slot_count = 2   # 5
        elif minutes <= 20:
            tech_slot_count = 3   # 6
        elif minutes <= 30:
            tech_slot_count = 4   # 7
        else:
            tech_slot_count = 5   # 8

        requested_stacks = [s for s in self.stack if s and s.lower() not in ("core", "none", "universal")]
        if not requested_stacks:
            requested_stacks = ["core"]

        active_stacks = requested_stacks[:tech_slot_count]
        base_count = tech_slot_count // len(active_stacks)
        remainder = tech_slot_count % len(active_stacks)

        assigned_stacks: list[str] = []
        for idx, st in enumerate(active_stacks):
            count = base_count + (1 if idx < remainder else 0)
            assigned_stacks.extend([st] * count)

        for target_stack in assigned_stacks:
            slots.append(
                AgendaSlot(
                    slot_id=slot_counter,
                    question_type="technical",
                    competency="technical_depth",
                    stack=target_stack,
                    is_must_have=False,
                )
            )
            slot_counter += 1

        return slots

    # ---------------- Derived agenda progress ----------------

    def _done_slot_ids(self) -> set[int]:
        """Slots whose MAIN (non follow-up) question has been asked AND answered."""
        return {
            q.slot_id
            for q in self.questions_asked
            if q.slot_id is not None and not q.is_follow_up and q.answer_text is not None
        }

    def _next_open_slot_index(self) -> int:
        done = self._done_slot_ids()
        for i, slot in enumerate(self.agenda):
            if slot.slot_id not in done:
                return i
        return len(self.agenda)

    def _is_slot_completed(self, slot: AgendaSlot) -> bool:
        return slot.slot_id in self._done_slot_ids()

    def is_agenda_complete(self) -> bool:
        return self._next_open_slot_index() >= len(self.agenda)

    def advance_slot(self) -> None:
        """Idempotent: syncs the index to what has actually been completed."""
        self.current_slot_index = self._next_open_slot_index()

    def peek_next_slot(self) -> Optional[AgendaSlot]:
        elapsed = time.monotonic() - self.start_timestamp
        remaining = self.duration_seconds - elapsed
        done = self._done_slot_ids()
        idx = self._next_open_slot_index()

        if remaining < 150:
            while idx < len(self.agenda):
                slot = self.agenda[idx]
                if slot.is_must_have and slot.slot_id not in done:
                    return slot
                idx += 1
            return None

        return self.agenda[idx] if idx < len(self.agenda) else None

    def max_turns(self) -> int:
        return len(self.agenda) * (1 + MAX_FOLLOW_UPS_PER_SLOT)

    def follow_ups_used_for_slot(self, slot_id: int | None) -> int:
        if slot_id is None:
            return 0
        return sum(1 for q in self.questions_asked if q.slot_id == slot_id and q.is_follow_up)

    def mark_completed(self, reason: str = "agenda_completed") -> None:
        self.ended = True

    # ---------------- Mutators ----------------

    def set_pending_follow_up(self, question_text: str) -> None:
        self.pending_follow_up_text = question_text

    def clear_pending_follow_up(self) -> None:
        self.pending_follow_up_text = None

    def add_question(
        self,
        question_text: str,
        raw_text: str | None = None,
        reference_points: list[str] | None = None,
        is_follow_up: bool = False,
        slot_id: int | None = None,
        question_type: str | None = None,
        competency: str | None = None,
        stack: str | None = None,
        question_id: str | None = None,
    ) -> None:
        if is_follow_up:
            self.consecutive_follow_ups += 1
        else:
            self.consecutive_follow_ups = 0
            self.original_question_count += 1

        # Whatever was just asked, any stored follow-up is now consumed
        self.pending_follow_up_text = None

        if question_id:
            self.user_asked_question_ids.append(question_id)

        self.history.append({"role": "assistant", "content": question_text})
        self.questions_asked.append(
            AskedQuestion(
                text=question_text,
                difficulty=self.difficulty,
                raw_text=raw_text,
                reference_points=reference_points or [],
                is_follow_up=is_follow_up,
                slot_id=slot_id,
                question_type=question_type,
                competency=competency,
                stack=stack,
            )
        )

    def add_answer(self, answer_text: str) -> None:
        self.history.append({"role": "user", "content": answer_text})
        if self.questions_asked:
            self.questions_asked[-1].answer_text = answer_text

    def add_evaluation(self, evaluation: AnswerEvaluation) -> None:
        self.scores.append(evaluation)
        if self.questions_asked:
            last = self.questions_asked[-1]
            last.evaluation = evaluation
            if last.is_follow_up:   # follow-ups shouldn't double-shift difficulty
                return
        self._apply_difficulty_shift(evaluation.next_question_difficulty)

    def _apply_difficulty_shift(self, direction: str) -> None:
        ladder = get_ladder(self.category)
        idx = ladder.index(self.difficulty) if self.difficulty in ladder else 0
        if direction == "harder":
            idx = min(idx + 1, len(ladder) - 1)
        elif direction == "easier":
            idx = max(idx - 1, 0)
        self.difficulty = ladder[idx]

    # ---------------- State checks ----------------

    def pending_follow_up(self) -> str | None:
        if self.consecutive_follow_ups >= MAX_FOLLOW_UPS_PER_SLOT:
            return None
        return self.pending_follow_up_text

    def _turn_cap_reached(self) -> bool:
        if len(self.questions_asked) < self.max_turns():
            return False
        return self.questions_asked[-1].answer_text is not None

    def should_end(self) -> bool:
        if self.ended:
            return True
        if self.pending_follow_up():
            return False
        if self.is_agenda_complete():
            return True
        if self._turn_cap_reached():
            return True
        elapsed = time.monotonic() - self.start_timestamp
        return elapsed >= self.duration_seconds

    def end(self) -> None:
        self.ended = True

    # ---------------- Reporting ----------------

    def scores_by_difficulty(self) -> dict[str, list[int]]:
        breakdown: dict[str, list[int]] = {}
        for question in self.questions_asked:
            if question.evaluation:
                breakdown.setdefault(question.difficulty, []).append(question.evaluation.score)
        return breakdown