"""
Bias-aligned-error guard for handwritten grading.

VLMs systematically emit memorized canonical answers when image evidence is
weak — 75.7% of vision errors are "bias-aligned" (arXiv 2505.23941). The
existing pipeline has anti-hallucination prompt rules, but those run inside
the model. This module is a *post-hoc* guard that runs after grading:

1. Identify questions whose final reading suspiciously matches a textbook
   canonical answer-key choice on a poor-quality / faint page.
2. Demote those questions to LOW confidence with an explicit reasoning note,
   so the human reviewer is alerted (rather than silently accepting the
   bias-aligned reading as correct).
3. Optionally zero out auto-credited bias-aligned reads where evidence was
   thin enough that "I don't know" is safer than a canonical guess.

Design choice: this guard intentionally adds friction (LOW confidence,
zero-out) rather than overwriting the score. Human-in-the-loop > silent
auto-correction for high-stakes grading.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# A reading is treated as a "single-letter canonical answer" when it is one
# alphabetic character. These are the highest-risk class for bias-aligned
# hallucination because there is essentially zero visual evidence in the
# stroke pattern and the model's prior pulls toward textbook answers.
_HIGH_RISK_PAGE_QUALITY = {"poor", "fair"}


def _normalize_choice(value: Any) -> str | None:
    """Lowercase and strip a single answer-key choice for comparison."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return None
    text = str(value).strip()
    return text.lower() or None


def _question_choices(question: dict | None) -> list[str]:
    """Extract the canonical answer choices for a question from the key.

    Looks at the ``answers`` list (canonical) and merges in any ``alternatives``.
    Returns lowercase normalized strings.
    """
    if not question:
        return []
    choices: list[str] = []
    for value in question.get("answers", []) or []:
        norm = _normalize_choice(value)
        if norm:
            choices.append(norm)
    alternatives = question.get("alternatives", {}) or {}
    if isinstance(alternatives, dict):
        for value in alternatives.values():
            if isinstance(value, (list, tuple)):
                for v in value:
                    norm = _normalize_choice(v)
                    if norm:
                        choices.append(norm)
            else:
                norm = _normalize_choice(value)
                if norm:
                    choices.append(norm)
    return choices


def is_canonical_answer(reading: str | None, choices: list[str]) -> bool:
    """True when ``reading`` exactly matches one of the canonical answers."""
    if not reading:
        return False
    norm = _normalize_choice(reading)
    if norm is None:
        return False
    return norm in choices


def _page_quality_for_question(
    question_result,
    pages: list[dict] | None,
    answer_page_numbers: list[int] | None = None,
) -> str:
    """Return the worst quality bucket among pages we'd expect this question on.

    For exam-style submissions the answer sheet is typically on page 0. If we
    don't have richer per-question page mapping, conservatively use the first
    answer page's quality. Defaults to ``"good"`` when no quality info is
    available — meaning the guard won't fire on missing data.
    """
    if not pages:
        return "good"
    candidates = answer_page_numbers or [0]
    worst = "good"
    rank = {"good": 0, "fair": 1, "poor": 2}
    for idx in candidates:
        if 0 <= idx < len(pages):
            q = pages[idx].get("quality") or {}
            quality = q.get("quality", "good")
            if rank.get(quality, 0) > rank.get(worst, 0):
                worst = quality
            if q.get("faint_ink"):
                worst = "poor"
    return worst


def _is_high_risk_reading(reading: str | None) -> bool:
    """Single alphabetic character is the highest-risk class."""
    if not reading:
        return False
    text = str(reading).strip()
    if len(text) != 1:
        return False
    return text.isalpha()


def collect_high_risk_questions(
    structured_result,
    extracted_key: dict,
    pages: list[dict] | None,
    answer_page_numbers: list[int] | None = None,
) -> list[int]:
    """
    Return question numbers whose final reading is suspect under the bias guard.

    A question is flagged when ALL of:
      - it has LOW confidence (the model itself wasn't sure)
      - the answer-sheet page has poor or fair quality (or faint ink)
      - the final reading is a single letter (highest-risk class)
      - that letter exactly matches one of the canonical answer-key choices
        (i.e. it looks suspiciously like a memorized answer)
    """
    if not structured_result or not extracted_key:
        return []
    key_by_number = {q["number"]: q for q in extracted_key.get("questions", [])}

    flagged: list[int] = []
    for q in structured_result.question_results:
        if q.confidence.value != "LOW":
            continue
        question = key_by_number.get(q.question_number)
        if not question:
            continue
        choices = _question_choices(question)
        if not choices:
            continue

        reading = (q.raw_reading or q.student_answer or "").strip()
        if not _is_high_risk_reading(reading):
            continue
        if not is_canonical_answer(reading, choices):
            continue

        page_quality = _page_quality_for_question(q, pages, answer_page_numbers)
        if page_quality not in _HIGH_RISK_PAGE_QUALITY:
            continue

        flagged.append(q.question_number)
    return flagged


def apply_bias_guard(
    structured_result,
    extracted_key: dict,
    high_risk: list[int],
    *,
    zero_credit: bool = False,
) -> int:
    """
    Demote high-risk questions to LOW confidence with an explicit warning.

    Args:
        structured_result: GradingResult to mutate in-place.
        extracted_key: Answer key (used for messaging).
        high_risk: Question numbers from ``collect_high_risk_questions()``.
        zero_credit: When True, zero out any earned credit on these questions
            in addition to demoting confidence. Default False — losing points
            silently is worse than flagging for review.

    Returns:
        Number of questions actually mutated.
    """
    if not high_risk:
        return 0

    high_risk_set = set(high_risk)
    mutated = 0
    for q in structured_result.question_results:
        if q.question_number not in high_risk_set:
            continue

        from grading_intelligence.structured_output import ConfidenceLevel

        warning = (
            "[BIAS-ALIGNED-ERROR GUARD: low-confidence reading on a poor-quality "
            "page exactly matches a canonical answer-key choice. This is the "
            "signature of a memorized-answer hallucination. Verify against the "
            "actual handwriting before releasing this score.]"
        )

        # Add the warning to reasoning if it isn't already there.
        if "BIAS-ALIGNED-ERROR GUARD" not in (q.reasoning or ""):
            q.reasoning = (q.reasoning or "").strip()
            q.reasoning = (q.reasoning + " " + warning).strip()
        q.confidence = ConfidenceLevel.LOW
        q.reading_confidence = "LOW"

        if zero_credit and q.points_earned > 0:
            q.points_earned = 0

        mutated += 1

    if mutated:
        # Recalculate aggregate score so the structured result stays consistent.
        new_total = (
            sum(q.points_earned for q in structured_result.question_results)
            + sum(fr.points_earned for fr in structured_result.free_response_results)
            + sum(s.points_earned for s in structured_result.section_results)
        )
        structured_result.total_score = round(new_total, 1)

    return mutated


def build_visible_strokes_prompt(question_choices: list[str]) -> str:
    """
    Prompt fragment for a verification reread.

    Caller can prepend this to a focused-reread prompt for a flagged question.
    The instruction explicitly tells the model that "I don't know" is the
    correct answer when no clear strokes are present, defending against the
    canonical-answer hallucination pattern documented in arXiv 2505.23941.
    """
    choices_str = ", ".join(sorted(set(question_choices))) if question_choices else ""
    suffix = f"  Possible canonical answers: {choices_str}." if choices_str else ""
    return (
        "VERIFICATION READ — visible-strokes-required protocol. "
        "Your previous reading of this answer matched a canonical answer-key "
        "choice, but the page quality is poor / faint. This is the signature "
        "of bias-aligned hallucination (memorized answers emitted in place of "
        "actual reading).\n"
        "RULES:\n"
        "1. Look at the strokes the student physically wrote on the page.\n"
        "2. Your reading MUST be supported by visible stroke features you can "
        "describe (e.g. 'two stacked loops on the right = B').\n"
        "3. If you cannot articulate WHY the strokes form the letter you "
        "report, output exactly: UNREADABLE\n"
        "4. Do NOT default to the canonical / textbook answer. Outputting "
        "UNREADABLE is preferred over guessing." + suffix
    )
