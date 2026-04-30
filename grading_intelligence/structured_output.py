"""
Structured output models for grading results.

Replaces regex-based score extraction with typed Pydantic models.
Every grading result is machine-readable JSON with per-question
scores, confidence levels, reasoning chains, and audit metadata.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import math as _math

from pydantic import BaseModel, Field, computed_field, model_validator


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class QuestionResult(BaseModel):
    """Result for a single exam question (instance-based scoring)."""
    question_number: int
    points_earned: float
    points_possible: float
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH

    # --- Vision layer (what was physically seen) ---
    raw_reading: str = ""  # Exact transcription before interpretation/tolerance
    alternative_readings: list[str] = Field(default_factory=list)  # Other plausible readings
    reading_confidence: str = "HIGH"  # Vision-only confidence (HIGH/LOW)

    # --- Scoring layer (interpretation + rubric application) ---
    student_answer: str = ""  # Final interpretation after tolerance rules
    expected_answer: str = ""
    correct_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    wrong_terms: list[str] = Field(default_factory=list)
    reasoning: str = ""
    bloom_level: Optional[BloomLevel] = None
    shift_detected: bool = False
    answer_hunted: bool = False  # Was the answer found on a non-designated page?
    illegible: bool = False
    faint_content: bool = False

    @computed_field
    @property
    def percentage(self) -> float:
        if self.points_possible == 0:
            return 0.0
        return round((self.points_earned / self.points_possible) * 100, 1)


class Deduction(BaseModel):
    """A single point deduction with reason."""
    points: float
    reason: str
    criterion: str = ""
    propagated: bool = False  # True if this is a propagated error (single-penalty rule)


class SectionResult(BaseModel):
    """Result for a rubric section (deduction-based scoring)."""
    section_name: str
    points_earned: float
    points_possible: float
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    deductions: list[Deduction] = Field(default_factory=list)
    reasoning: str = ""
    bloom_level: Optional[BloomLevel] = None

    @computed_field
    @property
    def percentage(self) -> float:
        if self.points_possible == 0:
            return 0.0
        return round((self.points_earned / self.points_possible) * 100, 1)


class FreeResponseResult(BaseModel):
    """Result for a free response question."""
    question_number: int
    points_earned: float
    points_possible: float
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    rubric_level_matched: str = ""
    reasoning: str = ""
    bloom_level: Optional[BloomLevel] = None


class ModelVote(BaseModel):
    """Record of a single model's grading output for audit trail."""
    provider: str  # "claude", "gemini", "openai"
    model_id: str  # e.g. "claude-sonnet-4-20250514"
    score: float
    total: float
    raw_text: str
    confidence: float = 0.0
    tokens_used: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0


class GradingResult(BaseModel):
    """
    Complete grading result for a single student submission.

    This is the canonical output format for all grading operations.
    Replaces raw text parsing with structured, auditable data.
    """
    # Identity
    student_file: str
    student_name: str = ""  # Name as read from the exam (from STUDENT NAME: field)
    name_alternates: list[str] = Field(default_factory=list)  # Alternate handwriting readings
    name_confidence: str = ""  # HIGH or LOW
    name_mismatch: bool = False  # True if name on exam differs from expected
    assignment_title: str = ""
    grading_mode: str = "exam"  # "exam" or "notebook"

    # Scores
    total_score: float
    total_possible: float

    # Detailed breakdown
    question_results: list[QuestionResult] = Field(default_factory=list)
    section_results: list[SectionResult] = Field(default_factory=list)
    free_response_results: list[FreeResponseResult] = Field(default_factory=list)

    # Shift detection
    shift_detected: bool = False
    shift_details: str = ""

    # Comments and notes
    comments: str = ""
    scan_format_notes: str = ""

    # Ensemble metadata
    model_votes: list[ModelVote] = Field(default_factory=list)
    consensus_strategy: str = "single"  # "single", "vote", "debate", "cascade"
    consensus_confidence: float = 1.0

    # Audit
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    grading_duration_ms: int = 0

    # Token usage (aggregated across all models)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_api_calls: int = 0

    @model_validator(mode='after')
    def _clamp_score_bounds(self) -> 'GradingResult':
        """Validate and clamp score at construction time — not at output time."""
        if _math.isnan(self.total_score) or _math.isinf(self.total_score):
            self.total_score = 0
            self.consensus_confidence = 0.0  # Force needs_human_review
            self.comments = "[SCORE ERROR: Invalid score (NaN/Inf) clamped to 0 — MANDATORY human review.]\n" + self.comments
        if self.total_score < 0:
            self.total_score = 0
        if self.total_possible == 0 and self.total_score > 0:
            self.total_score = 0
        if self.total_possible > 0 and self.total_score > self.total_possible:
            self.total_score = self.total_possible
        return self

    @computed_field
    @property
    def percentage(self) -> float:
        if self.total_possible == 0:
            return 0.0
        return round((self.total_score / self.total_possible) * 100, 1)

    @computed_field
    @property
    def needs_human_review(self) -> bool:
        """Flag if any component has LOW confidence or score is anomalous."""
        # Low confidence items
        for q in self.question_results:
            if q.confidence == ConfidenceLevel.LOW:
                return True
        for s in self.section_results:
            if s.confidence == ConfidenceLevel.LOW:
                return True
        for fr in self.free_response_results:
            if fr.confidence == ConfidenceLevel.LOW:
                return True
        # Low ensemble consensus
        if self.consensus_confidence < 0.7:
            return True
        # Perfect zero on a non-trivial exam should be verified
        # (could be blank submission OR complete misread)
        if self.total_possible > 0 and self.total_score == 0 and (self.question_results or self.section_results):
            return True
        # Perfect score should also be verified (rarer but possible LLM over-crediting)
        if self.total_possible > 0 and self.total_score == self.total_possible and self.total_possible >= 10 and (self.question_results or self.section_results):
            return True
        return False

    @computed_field
    @property
    def low_confidence_items(self) -> list[str]:
        """List items that need human review."""
        items = []
        for q in self.question_results:
            if q.confidence == ConfidenceLevel.LOW:
                items.append(f"Q{q.question_number}: {q.reasoning}")
        for s in self.section_results:
            if s.confidence == ConfidenceLevel.LOW:
                items.append(f"{s.section_name}: {s.reasoning}")
        for fr in self.free_response_results:
            if fr.confidence == ConfidenceLevel.LOW:
                items.append(f"FR{fr.question_number}: {fr.reasoning}")
        # Flag anomalous total scores
        if self.total_possible > 0 and self.total_score == 0 and (self.question_results or self.section_results):
            items.append("ZERO SCORE: Student received 0 — verify this is not a misread or blank-page error")
        if self.total_possible > 0 and self.total_score == self.total_possible and self.total_possible >= 10 and (self.question_results or self.section_results):
            items.append("PERFECT SCORE: Student received full marks — verify no over-crediting occurred")
        if self.consensus_confidence < 0.7:
            items.append(f"LOW CONSENSUS: Ensemble confidence {self.consensus_confidence:.2f} — models disagreed")
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def to_legacy_text(self) -> str:
        """Convert back to the legacy text format for backward compatibility."""
        lines = []

        def _fmt(v: float) -> str:
            if _math.isnan(v) or _math.isinf(v):
                return str(v)
            return str(int(v)) if v == int(v) else f"{v:.1f}"

        # Score bounds already enforced by model_validator at construction time

        if self.grading_mode == "exam":
            lines.append(f"SHIFT DETECTED: {'Yes' if self.shift_detected else 'No'}")
            lines.append(f"SHIFT DETAILS: {self.shift_details or 'N/A'}")
            lines.append("")
            lines.append(f"GRADE: {_fmt(self.total_score)} / {_fmt(self.total_possible)}")
            lines.append("")
            lines.append("QUESTION BREAKDOWN:")
            for q in self.question_results:
                terms = []
                if q.correct_terms:
                    terms.append(f"correct: {', '.join(q.correct_terms)}")
                if q.missing_terms:
                    terms.append(f"missing: {', '.join(q.missing_terms)}")
                if q.wrong_terms:
                    terms.append(f"wrong: {', '.join(q.wrong_terms)}")
                detail = "; ".join(terms) if terms else q.reasoning
                lines.append(
                    f"Q{q.question_number} ({q.points_possible:.0f} pts): "
                    f"{q.points_earned:.0f}/{q.points_possible:.0f} "
                    f"[confidence: {q.confidence.value}] — {detail}"
                )

            if self.free_response_results:
                lines.append("")
                lines.append("FREE RESPONSE BREAKDOWN:")
                for fr in self.free_response_results:
                    lines.append(
                        f"FR{fr.question_number} ({fr.points_possible:.0f} pts): "
                        f"{fr.points_earned:.0f}/{fr.points_possible:.0f} "
                        f"[confidence: {fr.confidence.value}] — {fr.reasoning}"
                    )
        else:
            # Notebook mode
            lines.append(f"GRADE: {_fmt(self.total_score)} / {_fmt(self.total_possible)}")
            lines.append("")
            lines.append("BREAKDOWN:")
            for s in self.section_results:
                lines.append(
                    f"- {s.section_name}: {_fmt(s.points_earned)}/{_fmt(s.points_possible)} "
                    f"[confidence: {s.confidence.value}]"
                )
            lines.append("")
            lines.append("DEDUCTIONS:")
            for s in self.section_results:
                for d in s.deductions:
                    lines.append(f"- -{_fmt(d.points)} {d.reason}")

        lines.append("")
        lines.append("COMMENTS:")
        lines.append(self.comments or "[No comments]")
        lines.append("")
        if self.scan_format_notes:
            lines.append("SCAN/FORMAT NOTES:")
            lines.append(self.scan_format_notes)

        return "\n".join(lines)


class LatePenaltyConfig(BaseModel):
    """Late submission penalty configuration."""
    penalty_per_day: float = 0.0  # Points deducted per day late
    max_penalty_percent: float = 100.0  # Cap at this % of total
    grace_period_hours: int = 0  # Hours after deadline before penalties start


class CurveConfig(BaseModel):
    """Grade curve configuration."""
    method: str = "none"  # "none", "linear", "flat_boost", "sqrt", "drop_lowest"
    target_mean: float | None = None  # For linear scaling
    boost_points: float = 0.0  # For flat boost
    drop_count: int = 0  # For drop_lowest


def parse_legacy_grade_text(grade_text: str, mode: str = "exam") -> GradingResult:
    """
    Parse legacy Claude text output into a structured GradingResult.

    This bridges the gap between the current text-based output and the new
    structured format. Handles both exam and notebook modes.
    """
    # Extract score
    score_match = re.search(r"GRADE\s*:\s*([\d.]+)\s*/\s*([\d.]+)", grade_text, re.IGNORECASE)
    total_score = float(score_match.group(1)) if score_match else 0.0
    total_possible = float(score_match.group(2)) if score_match else 0.0

    # Extract shift detection
    shift_match = re.search(r"SHIFT DETECTED:\s*(Yes|No)", grade_text, re.IGNORECASE)
    shift_detected = shift_match.group(1).lower() == "yes" if shift_match else False

    shift_detail_match = re.search(r"SHIFT DETAILS:\s*(.+?)(?:\n|$)", grade_text)
    shift_details = shift_detail_match.group(1).strip() if shift_detail_match else ""
    if shift_details == "N/A":
        shift_details = ""

    # Extract question breakdown (exam mode)
    question_results = []
    q_pattern = re.compile(
        r"Q(\d+)\s*\((\d+)\s*pts?\):\s*([\d.]+)/([\d.]+)\s*\[confidence:\s*(HIGH|LOW)\]\s*[—–-]\s*(.*?)(?=\nQ\d|\nFR\d|\n\n|\nCOMMENTS|\nFREE|\nCONFIDENCE|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in q_pattern.finditer(grade_text):
        q_num = int(m.group(1))
        pts_possible = float(m.group(2))
        pts_earned = float(m.group(3))
        confidence = ConfidenceLevel(m.group(5).upper())
        detail = m.group(6).strip()

        question_results.append(QuestionResult(
            question_number=q_num,
            points_earned=pts_earned,
            points_possible=pts_possible,
            confidence=confidence,
            reasoning=detail,
        ))

    # Extract section breakdown (notebook mode)
    section_results = []
    s_pattern = re.compile(
        r"-\s*(.+?):\s*([\d.]+)/([\d.]+)\s*\[confidence:\s*(HIGH|LOW)\]",
        re.IGNORECASE,
    )
    if mode == "notebook":
        breakdown_match = re.search(r"BREAKDOWN:(.*?)(?=DEDUCTIONS:|COMMENTS:|\Z)", grade_text, re.DOTALL)
        if breakdown_match:
            for m in s_pattern.finditer(breakdown_match.group(1)):
                section_results.append(SectionResult(
                    section_name=m.group(1).strip(),
                    points_earned=float(m.group(2)),
                    points_possible=float(m.group(3)),
                    confidence=ConfidenceLevel(m.group(4).upper()),
                ))

    # Extract comments
    comments_match = re.search(r"COMMENTS:\s*\n(.*?)(?=\nSCAN/FORMAT|\Z)", grade_text, re.DOTALL)
    comments = comments_match.group(1).strip() if comments_match else ""

    # Extract scan/format notes
    scan_match = re.search(r"SCAN/FORMAT NOTES.*?:\s*\n(.*?)$", grade_text, re.DOTALL)
    scan_notes = scan_match.group(1).strip() if scan_match else ""

    # Extract student name (with alternate readings support)
    student_name = ""
    name_alternates = []
    name_confidence = ""
    name_mismatch = False
    name_match = re.search(
        r"STUDENT\s+NAME\s*:\s*(.+?)(?:\s*\[confidence:\s*(HIGH|LOW)\])?$",
        grade_text, re.IGNORECASE | re.MULTILINE,
    )
    if name_match:
        raw_name = name_match.group(1).strip()
        if raw_name.upper() not in ("NOT FOUND", "UNKNOWN", "N/A"):
            # Parse alternate readings: "Best Name (or: Alt1, Alt2)"
            alt_match = re.match(r"^(.+?)\s*\(or:\s*(.+?)\)\s*$", raw_name)
            if alt_match:
                student_name = alt_match.group(1).strip()
                name_alternates = [a.strip() for a in alt_match.group(2).split(",") if a.strip()]
            else:
                student_name = raw_name
            name_confidence = (name_match.group(2) or "").upper()
    mismatch_match = re.search(r"NAME\s+MISMATCH\s*:\s*(Yes)", grade_text, re.IGNORECASE)
    if mismatch_match:
        name_mismatch = True

    return GradingResult(
        student_file="",
        student_name=student_name,
        name_alternates=name_alternates,
        name_confidence=name_confidence,
        name_mismatch=name_mismatch,
        grading_mode=mode,
        total_score=total_score,
        total_possible=total_possible,
        question_results=question_results,
        section_results=section_results,
        shift_detected=shift_detected,
        shift_details=shift_details,
        comments=comments,
        scan_format_notes=scan_notes,
    )
