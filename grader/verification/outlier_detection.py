"""
Statistical outlier detection for grading quality assurance.

Flags students whose scores are statistically unusual, and per-question
anomalies where a student's performance on a single question contradicts
their overall performance level.
"""

import math
from dataclasses import dataclass

from .cross_validation import ScoringFlag


def flag_statistical_outliers(results: list[dict],
                              z_threshold: float = 2.5,
                              total_possible: int = 32) -> list[ScoringFlag]:
    """
    Flag students whose total score is >z_threshold standard deviations
    from the section mean.

    These are candidates for full verification — either the student truly
    performed unusually, or the grading agent made systematic errors.

    Uses population standard deviation (not sample/Bessel) because:
    - We have the full population (all students in the section), not a sample
    - Bessel correction inflates stdev for small classes (n<15), raising the bar
      and causing real outliers to be missed

    Default z_threshold=2.5 aligns with ETS/ACT best practice for exam grading.
    For classes <15 students the effective threshold is capped at 1.75 to
    compensate for reduced statistical power.
    """
    flags = []
    # Filter out sentinel/error records
    valid_results = [r for r in results if not r.get("_batch_load_error")]
    totals = [r.get("score", r.get("total", 0)) or 0 for r in valid_results]

    if len(totals) < 3:
        return flags

    n = len(totals)
    mean = sum(totals) / n
    # Population variance (divide by n, not n-1) — we have the full class
    variance = sum((t - mean) ** 2 for t in totals) / n
    stdev = math.sqrt(variance) if variance > 0 else 0

    if stdev == 0:
        return flags

    # Adaptive threshold: smaller classes need a lower bar
    effective_threshold = z_threshold if n >= 15 else min(z_threshold, 1.75)

    for student in valid_results:
        name = student.get("name", "Unknown")
        total = student.get("score", student.get("total", 0)) or 0
        z = (total - mean) / stdev

        if abs(z) > effective_threshold:
            direction = "below" if z < 0 else "above"
            flags.append(ScoringFlag(
                question="TOTAL",
                student_name=name,
                flag_type="statistical_outlier",
                severity="MEDIUM",
                detail=f"Score {total}/{total_possible} is {abs(z):.1f} SD {direction} "
                       f"mean ({mean:.1f}). Verify grading accuracy.",
                suggested_action=f"Spot-check 5-8 questions for {name}"
            ))

    return flags


def calculate_item_difficulty(results: list[dict]) -> dict[str, float]:
    """
    Calculate p-value (item difficulty) for each question.

    p = proportion of students who got the question correct.
    High p (>0.80) = easy question.
    Low p (<0.30) = hard question.
    """
    q_correct = {}
    q_total = {}

    for student in results:
        q_scores = student.get("q_scores", {})
        for q, score in q_scores.items():
            q_total[q] = q_total.get(q, 0) + 1
            if score >= 1:
                q_correct[q] = q_correct.get(q, 0) + 1

    p_values = {}
    for q in q_total:
        if q_total[q] > 0:
            p_values[q] = q_correct.get(q, 0) / q_total[q]

    return p_values


def irt_residual_check(results: list[dict],
                       total_possible: int = 32) -> list[ScoringFlag]:
    """
    Flag per-question anomalies using item difficulty (p-value).

    - High performer (>70%) got an easy question wrong (p>0.80): verify
    - Low performer (<30%) got a hard question right (p<0.30): verify

    These patterns suggest the grading agent may have misread/misscored
    that specific question.
    """
    flags = []
    p_values = calculate_item_difficulty(results)

    if not p_values:
        return flags

    for student in results:
        name = student.get("name", "Unknown")
        total = student.get("total", 0) or 0
        pct = total / total_possible if total_possible > 0 else 0
        q_scores = student.get("q_scores", {})

        for q, score in q_scores.items():
            p = p_values.get(q)
            if p is None:
                continue

            # High performer missed easy question
            if pct >= 0.70 and score == 0 and p >= 0.80:
                flags.append(ScoringFlag(
                    question=q,
                    student_name=name,
                    flag_type="irt_residual",
                    severity="LOW",
                    detail=f"Strong student ({total}/{total_possible}, {pct*100:.0f}%) "
                           f"missed easy {q} (p={p:.2f}, {p*100:.0f}% of class correct). "
                           f"Possible misread.",
                    suggested_action=f"Verify {name}'s answer for {q}"
                ))

            # Low performer got hard question right
            if pct <= 0.30 and score >= 1 and p <= 0.30:
                flags.append(ScoringFlag(
                    question=q,
                    student_name=name,
                    flag_type="irt_residual",
                    severity="LOW",
                    detail=f"Weak student ({total}/{total_possible}, {pct*100:.0f}%) "
                           f"got hard {q} right (p={p:.2f}, only {p*100:.0f}% correct). "
                           f"Possible overcredit.",
                    suggested_action=f"Verify {name}'s answer for {q}"
                ))

    return flags


def detect_fr_objective_divergence(results: list[dict],
                                   total_possible: int = 32) -> list[ScoringFlag]:
    """
    Flag students where FR score and objective score strongly diverge.

    - FR=0 but objective >70%: agent may have missed FR section entirely
    - FR=full (5/5) but objective <30%: unusual pattern, verify FR scoring
    """
    flags = []

    for student in results:
        name = student.get("name", "Unknown")
        fr_score = student.get("fr_score", None)
        q_scores = student.get("q_scores", {})

        if fr_score is None or not q_scores:
            continue

        objective_total = sum(q_scores.values())
        objective_count = len(q_scores)
        objective_pct = objective_total / objective_count if objective_count > 0 else 0

        # FR = 0 but strong objective performance
        if fr_score == 0 and objective_pct >= 0.70:
            flags.append(ScoringFlag(
                question="FR",
                student_name=name,
                flag_type="fr_objective_divergence",
                severity="MEDIUM",
                detail=f"FR score is 0/5 but objective score is {objective_total}/{objective_count} "
                       f"({objective_pct*100:.0f}%). Agent may have missed FR content "
                       f"or read wrong page.",
                suggested_action=f"Re-read {name}'s FR answer on both front and back pages"
            ))

        # FR = full but very weak objective
        if fr_score >= 4.5 and objective_pct <= 0.30:
            flags.append(ScoringFlag(
                question="FR",
                student_name=name,
                flag_type="fr_objective_divergence",
                severity="LOW",
                detail=f"FR score is {fr_score}/5 but objective score is {objective_total}/{objective_count} "
                       f"({objective_pct*100:.0f}%). Unusual pattern — verify FR scoring.",
                suggested_action=f"Verify {name}'s FR rubric application"
            ))

    return flags


def run_outlier_detection(results: list[dict],
                          total_possible: int = 32) -> list[ScoringFlag]:
    """Run all outlier detection checks."""
    flags = []
    flags.extend(flag_statistical_outliers(results, total_possible=total_possible))
    flags.extend(irt_residual_check(results, total_possible=total_possible))
    flags.extend(detect_fr_objective_divergence(results, total_possible=total_possible))

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: (severity_order.get(f.severity, 3), f.student_name))

    return flags
