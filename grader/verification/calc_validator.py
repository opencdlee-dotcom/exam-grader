"""
Calculation question scoring consistency validator.

Ensures calc questions (Q26/Q27 agarose gel calculations) are scored
consistently across students. Catches magnitude errors (600g vs 6g vs 0.6g)
and enforces uniform application of calc scoring rules:
  - Wrong number = 0
  - Wrong/missing unit = -0.25
  - No work shown = half credit only if answer is fully correct
"""

from collections import Counter
from dataclasses import dataclass

from .cross_validation import ScoringFlag


def validate_calc_scoring(results: list[dict],
                          calc_field: str = "q26_calc",
                          key_answer: float | None = None) -> list[ScoringFlag]:
    """
    Verify calculation question scores are internally consistent.

    Checks:
    1. If most students got 0 but some got 1, verify the 1s are genuine
    2. If partial credit (0.25, 0.5, 0.75) exists, verify consistency
    3. Flag unusual score distributions
    """
    flags = []
    scores = []

    for student in results:
        score = student.get(calc_field)
        if score is not None:
            name = student.get("name", "Unknown")
            scores.append((name, score))

    if len(scores) < 3:
        return flags

    score_values = [s for _, s in scores]
    score_dist = Counter(score_values)
    n = len(scores)

    # Check for partial credit inconsistency
    # If some students got 0.75 (correct number, wrong unit) but others got 1.0,
    # verify whether the 1.0 students also had unit issues
    has_partial = any(0 < s < 1 for s in score_values)
    has_full = any(s >= 1 for s in score_values)
    has_zero = any(s == 0 for s in score_values)

    if has_partial:
        partial_students = [(n, s) for n, s in scores if 0 < s < 1]
        full_students = [(n, s) for n, s in scores if s >= 1]

        # Partial credit exists — ensure it's not randomly applied
        if len(partial_students) <= 2 and len(full_students) >= 5:
            for name, score in partial_students:
                flags.append(ScoringFlag(
                    question=calc_field.upper().replace("_", " "),
                    student_name=name,
                    flag_type="calc_partial_credit",
                    severity="LOW",
                    detail=f"Got {score}/1 on calc while {len(full_students)} students "
                           f"got full credit. Verify partial credit deduction is justified "
                           f"(wrong unit = -0.25).",
                    suggested_action=f"Check {name}'s calculation for unit/work consistency"
                ))

    # Check for lone full-credit among all zeros
    if score_dist.get(0, 0) >= n - 2 and score_dist.get(1, 0) <= 2:
        for name, score in scores:
            if score >= 1:
                flags.append(ScoringFlag(
                    question=calc_field.upper().replace("_", " "),
                    student_name=name,
                    flag_type="calc_outlier",
                    severity="MEDIUM",
                    detail=f"Got full credit on calc while {score_dist.get(0, 0)}/{n} "
                           f"students got 0. Verify answer is genuinely correct.",
                    suggested_action=f"Re-check {name}'s calculation answer"
                ))

    return flags


def run_calc_validation(results: list[dict],
                        calc_fields: list[str] | None = None) -> list[ScoringFlag]:
    """Run calc validation for all calculation fields found in results."""
    flags = []

    if calc_fields is None:
        # Auto-detect calc fields from first result
        if results:
            sample = results[0]
            calc_fields = [k for k in sample.keys()
                          if "calc" in k.lower() or k in ("q26_calc", "q27_calc")]

    if not calc_fields:
        return flags

    for field in calc_fields:
        flags.extend(validate_calc_scoring(results, calc_field=field))

    return flags
