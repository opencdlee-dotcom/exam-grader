"""
Post-grading validators — flag suspicious patterns in grading results.

Each validator is a function: (result, config) -> list[str]
  - result: a GradingResult (or any object with question_results, student_name, etc.)
  - config: dict of optional settings (roster, thresholds, etc.)
  - returns: list of flag strings (empty = no issues)

Register new validators with @register. They run automatically via run_all().
This module is course-agnostic and exam-format-agnostic.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grading_intelligence.structured_output import GradingResult

VALIDATORS: list = []


def register(fn):
    """Decorator to register a post-grading validator."""
    VALIDATORS.append(fn)
    return fn


def run_all(result, config: dict | None = None) -> list[str]:
    """Run all registered validators and return combined flags."""
    config = config or {}
    flags = []
    for validator in VALIDATORS:
        try:
            flags.extend(validator(result, config))
        except Exception as e:
            flags.append(f"VALIDATOR ERROR ({validator.__name__}): {e}")
    return flags


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------

@register
def score_consistency(result, config: dict) -> list[str]:
    """
    Flag if objective question subsections diverge suspiciously.

    Groups questions by points_possible (proxy for question type — 1pt questions
    are typically MC/matching, multi-point are fill-in/FR). If one group scores
    >70% and another <30%, the low group likely has misread answers.

    Works for any exam structure — no hardcoded question types.
    """
    questions = getattr(result, "question_results", [])
    if len(questions) < 4:
        return []

    # Group by points_possible
    groups: dict[float, list] = {}
    for q in questions:
        pp = q.points_possible
        if pp <= 0:
            continue
        groups.setdefault(pp, []).append(q)

    flags = []
    threshold_high = config.get("consistency_high", 0.70)
    threshold_low = config.get("consistency_low", 0.30)

    group_pcts = {}
    for pp, qs in groups.items():
        if len(qs) < 3:  # Need at least 3 questions to be meaningful
            continue
        total_earned = sum(q.points_earned for q in qs)
        total_possible = sum(q.points_possible for q in qs)
        if total_possible > 0:
            group_pcts[pp] = total_earned / total_possible

    if len(group_pcts) >= 2:
        max_pct = max(group_pcts.values())
        min_pct = min(group_pcts.values())
        if max_pct >= threshold_high and min_pct <= threshold_low:
            high_group = [pp for pp, pct in group_pcts.items() if pct >= threshold_high]
            low_group = [pp for pp, pct in group_pcts.items() if pct <= threshold_low]
            flags.append(
                f"SCORE INCONSISTENCY: {int(max_pct*100)}% on {high_group[0]}pt questions "
                f"vs {int(min_pct*100)}% on {low_group[0]}pt questions — "
                f"likely misread answers in the low-scoring section"
            )

    return flags


@register
def low_confidence_density(result, config: dict) -> list[str]:
    """Flag if too many answers have LOW confidence."""
    questions = getattr(result, "question_results", [])
    if not questions:
        return []

    threshold = config.get("low_confidence_threshold", 0.30)
    low_count = sum(1 for q in questions if q.confidence.value == "LOW")
    total = len(questions)

    if total > 0 and low_count / total > threshold:
        return [
            f"HIGH UNCERTAINTY: {low_count}/{total} answers have LOW confidence — "
            f"recommend human review of entire exam"
        ]
    return []


@register
def low_reading_confidence_density(result, config: dict) -> list[str]:
    """Flag if too many raw readings have LOW reading_confidence (vision layer)."""
    questions = getattr(result, "question_results", [])
    if not questions:
        return []

    threshold = config.get("low_reading_confidence_threshold", 0.25)
    low_count = sum(
        1 for q in questions
        if getattr(q, "reading_confidence", "HIGH") == "LOW"
    )
    total = len(questions)

    if total > 0 and low_count / total > threshold:
        return [
            f"VISION UNCERTAINTY: {low_count}/{total} answers have LOW reading confidence — "
            f"handwriting may be systematically difficult to read"
        ]
    return []


@register
def name_roster_match(result, config: dict) -> list[str]:
    """
    Fuzzy-match student name against provided roster.
    Flags if name doesn't closely match any roster entry.
    """
    roster = config.get("roster", [])
    if not roster:
        return []

    student_name = getattr(result, "student_name", "")
    if not student_name or student_name.upper() in ("UNKNOWN", "NOT FOUND", "N/A"):
        return ["NAME NOT FOUND: no student name detected on exam"]

    # Fuzzy match against roster
    matches = difflib.get_close_matches(
        student_name.lower(),
        [name.lower() for name in roster],
        n=1,
        cutoff=config.get("name_match_cutoff", 0.6),
    )

    if not matches:
        return [
            f"NAME MISMATCH: '{student_name}' does not match any roster entry — "
            f"verify manually"
        ]

    best_match = matches[0]
    # Find original-case version
    for name in roster:
        if name.lower() == best_match:
            best_match = name
            break

    if best_match.lower() != student_name.lower():
        return [
            f"NAME FUZZY MATCH: '{student_name}' matched to '{best_match}' — verify"
        ]

    return []


@register
def zero_section_check(result, config: dict) -> list[str]:
    """
    Flag if a contiguous block of 3+ questions all score 0.
    This pattern often indicates misread answers or a skipped section,
    not genuine student failure on every question.
    """
    questions = getattr(result, "question_results", [])
    if len(questions) < 3:
        return []

    # Sort by question number
    sorted_qs = sorted(questions, key=lambda q: q.question_number)

    consecutive_zeros = 0
    zero_start = 0
    flags = []

    min_block = config.get("zero_block_min", 3)

    for i, q in enumerate(sorted_qs):
        if q.points_earned == 0 and q.points_possible > 0:
            if consecutive_zeros == 0:
                zero_start = q.question_number
            consecutive_zeros += 1
        else:
            if consecutive_zeros >= min_block:
                zero_end = sorted_qs[i - 1].question_number
                flags.append(
                    f"ZERO BLOCK: Q{zero_start}-Q{zero_end} ({consecutive_zeros} questions) "
                    f"all scored 0 — possible misread or skipped section"
                )
            consecutive_zeros = 0

    # Check trailing block
    if consecutive_zeros >= min_block:
        zero_end = sorted_qs[-1].question_number
        flags.append(
            f"ZERO BLOCK: Q{zero_start}-Q{zero_end} ({consecutive_zeros} questions) "
            f"all scored 0 — possible misread or skipped section"
        )

    return flags


@register
def reading_alternative_audit(result, config: dict) -> list[str]:
    """
    Flag questions where the raw_reading differs from student_answer
    AND an alternative_reading would have changed the score.

    This catches cases where vision misread a letter but an alternate
    reading would have matched the answer key.
    """
    questions = getattr(result, "question_results", [])
    flags = []

    for q in questions:
        raw = getattr(q, "raw_reading", "")
        alts = getattr(q, "alternative_readings", [])
        answer = q.student_answer
        earned = q.points_earned

        if not raw or not alts:
            continue

        # If score is 0 and there are alternative readings, flag for review
        if earned == 0 and alts and q.points_possible > 0:
            flags.append(
                f"Q{q.question_number}: scored 0 but has alternate readings {alts} — "
                f"raw='{raw}', check if an alternate matches the key"
            )

    return flags


@register
def below_threshold_check(result, config: dict) -> list[str]:
    """Flag students scoring below a configurable percentage threshold."""
    threshold_pct = config.get("below_threshold_pct", 50)
    total_score = getattr(result, "total_score", 0)
    total_possible = getattr(result, "total_possible", 0)

    if total_possible <= 0:
        return []

    pct = (total_score / total_possible) * 100
    if pct < threshold_pct:
        return [
            f"BELOW {threshold_pct}%: {total_score}/{total_possible} ({pct:.1f}%) — "
            f"recommend verification pass"
        ]
    return []
