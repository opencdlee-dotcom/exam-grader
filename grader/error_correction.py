"""
Error Correction Loop — log instructor grade corrections and inject them
as context in future grading sessions to prevent repeat mistakes.

Corrections are persisted to output/corrections_{exam_name}.json and
surfaced back to the grading pipeline as formatted context strings.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Corrections are stored relative to the project root output/ directory.
_OUTPUT_DIR = Path("output")


@dataclass
class GradingCorrection:
    """A single instructor correction to an AI-assigned grade."""

    id: str
    date: str
    exam: str
    question_num: int
    ai_score: float
    instructor_score: float
    student_answer_excerpt: str
    notes: str


def get_corrections_path(exam_name: str) -> str:
    """Return the file path for corrections for the given exam."""
    return str(_OUTPUT_DIR / f"corrections_{exam_name}.json")


def log_correction(
    exam_name: str,
    question_num: int,
    ai_score: float,
    instructor_score: float,
    student_answer_excerpt: str = "",
    notes: str = "",
) -> GradingCorrection:
    """
    Record an instructor grade correction and persist it to disk.

    Creates the corrections file if it does not exist. Appends to the
    existing file if it does.

    Args:
        exam_name: Exam identifier (e.g. "ex2T5").
        question_num: The question number that was mis-graded.
        ai_score: The score the AI originally assigned.
        instructor_score: The score the instructor determined is correct.
        student_answer_excerpt: A brief excerpt of the student's answer.
        notes: Instructor commentary on why the AI was wrong.

    Returns:
        The created GradingCorrection object.
    """
    correction = GradingCorrection(
        id=str(uuid.uuid4()),
        date=date.today().isoformat(),
        exam=exam_name,
        question_num=question_num,
        ai_score=ai_score,
        instructor_score=instructor_score,
        student_answer_excerpt=student_answer_excerpt,
        notes=notes,
    )

    corrections_path = Path(get_corrections_path(exam_name))
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = _load_raw_corrections(corrections_path)
    existing.append(asdict(correction))

    corrections_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info(
        "Correction logged for %s Q%d: AI=%.1f → instructor=%.1f (id=%s)",
        exam_name,
        question_num,
        ai_score,
        instructor_score,
        correction.id,
    )

    return correction


def get_corrections_for_question(
    exam_name: str,
    question_num: int,
) -> list[GradingCorrection]:
    """
    Load and return all corrections for a specific exam and question number.

    Returns an empty list if the corrections file does not exist or has no
    entries for the requested question.
    """
    corrections_path = Path(get_corrections_path(exam_name))
    raw_list = _load_raw_corrections(corrections_path)

    results: list[GradingCorrection] = []
    for item in raw_list:
        try:
            c = GradingCorrection(**item)
            if c.question_num == question_num:
                results.append(c)
        except (TypeError, KeyError) as e:
            logger.debug("Skipping malformed correction record: %s", e)

    return results


def build_correction_context(exam_name: str, question_num: int) -> str:
    """
    Build a formatted context string for a question's past corrections.

    Returns an empty string if there are no corrections for this question.
    The returned string is suitable for injection at the top of a grading prompt.
    """
    corrections = get_corrections_for_question(exam_name, question_num)
    if not corrections:
        return ""

    lines = [
        f"PREVIOUS GRADING CORRECTIONS FOR Q{question_num} (learn from these mistakes):",
    ]

    for c in corrections:
        excerpt = c.student_answer_excerpt or "(no excerpt provided)"
        lines.append(f'- Student wrote: "{excerpt}"')
        lines.append(
            f"  AI gave: {c.ai_score} pts. Instructor corrected to: {c.instructor_score} pts."
        )
        if c.notes:
            lines.append(f"  Note: {c.notes}")

    return "\n".join(lines)


def get_all_correction_context(exam_name: str, question_nums: list[int]) -> str:
    """
    Build concatenated correction context for multiple questions.

    Skips questions that have no corrections. Returns an empty string if
    no corrections exist for any of the requested questions.
    """
    parts: list[str] = []
    for q_num in question_nums:
        ctx = build_correction_context(exam_name, q_num)
        if ctx:
            parts.append(ctx)

    return "\n\n".join(parts)


# ─── Internal helpers ────────────────────────────────────────────────────────


def _load_raw_corrections(path: Path) -> list[dict]:
    """Load the raw corrections JSON list from disk, or return [] if missing/corrupt."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.warning("Corrections file %s does not contain a list; resetting.", path)
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read corrections file %s: %s", path, e)
        return []
