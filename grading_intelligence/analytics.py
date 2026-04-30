"""
Learning analytics and early warning system.

Provides item-level analysis, student performance tracking,
early warning detection, concept mastery tracking, and
question quality metrics from structured grading results.
"""

from __future__ import annotations

import statistics
from typing import Optional

from pydantic import BaseModel, Field

from grading_intelligence.structured_output import BloomLevel, GradingResult


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ItemAnalysis(BaseModel):
    """Per-question statistics from a batch of graded submissions."""

    question_number: int
    difficulty_index: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of students who answered correctly (0=hardest, 1=easiest).",
    )
    discrimination_index: float = Field(
        ge=-1.0, le=1.0,
        description="Point-biserial correlation with total score (-1 to 1).",
    )
    mean_score: float
    std_dev: float
    common_errors: list[str] = Field(default_factory=list)
    bloom_level: Optional[BloomLevel] = None


class StudentTrend(BaseModel):
    """Longitudinal performance data and risk assessment for one student."""

    student_id: str
    student_name: str
    scores: list[dict] = Field(
        default_factory=list,
        description="List of dicts with keys: assignment, score, total, percentage, date.",
    )
    trend_direction: str = Field(
        description="One of 'improving', 'stable', or 'declining'.",
    )
    at_risk: bool = False
    risk_factors: list[str] = Field(default_factory=list)


class ConceptMastery(BaseModel):
    """Aggregate mastery level for a single learning objective / concept."""

    concept: str
    bloom_level: BloomLevel
    mastery_percentage: float
    student_count: int
    questions_assessed: list[int] = Field(default_factory=list)


class QuestionQualityFlag(BaseModel):
    """A quality concern raised about a specific question."""

    question_number: int
    flag_type: str = Field(
        description="One of 'too_easy', 'too_hard', 'poor_discrimination', 'high_disagreement'.",
    )
    severity: str = Field(
        description="One of 'info', 'warning', 'critical'.",
    )
    detail: str


class ClassAnalytics(BaseModel):
    """Full class-level analytics for one assignment."""

    assignment_name: str = ""
    num_students: int = 0
    mean_score: float = 0.0
    median_score: float = 0.0
    std_dev: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    item_analyses: list[ItemAnalysis] = Field(default_factory=list)
    quality_flags: list[QuestionQualityFlag] = Field(default_factory=list)
    concept_mastery: list[ConceptMastery] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length sequences.

    Returns 0.0 when the calculation is undefined (e.g. zero variance).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = (denom_x * denom_y) ** 0.5
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _safe_stdev(values: list[float]) -> float:
    """Return population stdev, or 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation.

    Expects *sorted_values* in ascending order and at least one element.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


# ---------------------------------------------------------------------------
# Core analytics functions
# ---------------------------------------------------------------------------


def compute_item_analysis(results: list[GradingResult]) -> list[ItemAnalysis]:
    """Compute per-question statistics from a batch of GradingResults.

    Uses the ``question_results`` field on each result.  Questions are
    identified by ``question_number`` and grouped across all submissions.

    Args:
        results: Graded submissions containing question-level breakdowns.

    Returns:
        A list of ``ItemAnalysis`` objects, one per distinct question number,
        sorted by question number.
    """
    if not results:
        return []

    # Gather per-question data across all students.
    # question_number -> list of (points_earned, points_possible, question_result)
    from collections import defaultdict

    q_data: dict[int, list[tuple[float, float, object]]] = defaultdict(list)
    for r in results:
        for q in r.question_results:
            q_data[q.question_number].append((q.points_earned, q.points_possible, q))

    # Total scores per student (for discrimination calculation).
    total_scores = [r.total_score for r in results]

    # Build a mapping: for each question, parallel lists of
    # (question percentage, total score) across students that answered it.
    # We need to track which student index answered which question.
    student_total_by_index: dict[int, float] = {}
    student_q_scores: dict[int, dict[int, float]] = defaultdict(dict)  # q_num -> {student_idx: pct}
    for idx, r in enumerate(results):
        student_total_by_index[idx] = r.total_score
        for q in r.question_results:
            pct = (q.points_earned / q.points_possible * 100.0) if q.points_possible > 0 else 0.0
            student_q_scores[q.question_number][idx] = pct

    analyses: list[ItemAnalysis] = []
    for q_num in sorted(q_data.keys()):
        entries = q_data[q_num]
        scores = [e[0] for e in entries]
        possibles = [e[1] for e in entries]
        q_objects = [e[2] for e in entries]

        # Difficulty index: fraction of students who got full credit.
        max_possible = max(possibles) if possibles else 1.0
        if max_possible > 0:
            difficulty = sum(1 for s, p in zip(scores, possibles) if p > 0 and s >= p) / len(entries)
        else:
            difficulty = 0.0

        # Discrimination index: Pearson r between question score and total score.
        q_score_list = list(student_q_scores[q_num].values())
        q_total_list = [student_total_by_index[i] for i in student_q_scores[q_num]]
        discrimination = _pearson_r(q_score_list, q_total_list)
        # Clamp to valid range.
        discrimination = max(-1.0, min(1.0, discrimination))

        mean_score = statistics.mean(scores) if scores else 0.0
        std_dev = _safe_stdev(scores)

        # Common errors: collect unique missing/wrong terms.
        error_counts: dict[str, int] = defaultdict(int)
        for qr in q_objects:
            for term in getattr(qr, "wrong_terms", []):
                error_counts[f"wrong: {term}"] += 1
            for term in getattr(qr, "missing_terms", []):
                error_counts[f"missing: {term}"] += 1
        # Top 5 most common errors.
        common_errors = [
            err for err, _ in sorted(error_counts.items(), key=lambda x: -x[1])[:5]
        ]

        # Bloom level: use the most common bloom level across responses.
        bloom_levels = [getattr(qr, "bloom_level", None) for qr in q_objects]
        bloom_levels = [b for b in bloom_levels if b is not None]
        bloom_level: Optional[BloomLevel] = None
        if bloom_levels:
            bloom_level = max(set(bloom_levels), key=bloom_levels.count)

        analyses.append(ItemAnalysis(
            question_number=q_num,
            difficulty_index=round(difficulty, 4),
            discrimination_index=round(discrimination, 4),
            mean_score=round(mean_score, 2),
            std_dev=round(std_dev, 2),
            common_errors=common_errors,
            bloom_level=bloom_level,
        ))

    return analyses


def compute_class_analytics(
    results: list[GradingResult],
    assignment_name: str = "",
) -> ClassAnalytics:
    """Compute full class-level analytics for an assignment.

    Args:
        results: All graded submissions for the assignment.
        assignment_name: Human-readable name for the assignment.

    Returns:
        A ``ClassAnalytics`` object with descriptive statistics, item
        analyses, quality flags, and concept mastery summaries.
    """
    if not results:
        return ClassAnalytics(assignment_name=assignment_name)

    percentages = [r.percentage for r in results]
    sorted_pcts = sorted(percentages)

    item_analyses = compute_item_analysis(results)
    quality_flags = flag_question_quality(item_analyses)
    concept_mastery = _compute_concept_mastery(item_analyses, results)

    return ClassAnalytics(
        assignment_name=assignment_name,
        num_students=len(results),
        mean_score=round(statistics.mean(percentages), 2),
        median_score=round(statistics.median(percentages), 2),
        std_dev=round(_safe_stdev(percentages), 2),
        min_score=round(min(percentages), 2),
        max_score=round(max(percentages), 2),
        q1=round(_percentile(sorted_pcts, 25), 2),
        q3=round(_percentile(sorted_pcts, 75), 2),
        item_analyses=item_analyses,
        quality_flags=quality_flags,
        concept_mastery=concept_mastery,
    )


def detect_at_risk_students(student_history: list[dict]) -> list[StudentTrend]:
    """Identify students showing concerning performance patterns.

    Args:
        student_history: Each dict must contain:
            - ``student_id`` (str)
            - ``student_name`` (str)
            - ``assignments`` (list of dicts with ``score``, ``total``, ``date``,
              and optionally ``assignment``)

    Returns:
        A list of ``StudentTrend`` objects for every student, with ``at_risk``
        set to ``True`` when any of the following hold:

        1. Declining trend across 3 or more assignments.
        2. Overall average below 60%.
        3. Most recent score is more than 20 percentage points below their
           own average.
    """
    trends: list[StudentTrend] = []

    for entry in student_history:
        student_id: str = entry.get("student_id", "")
        student_name: str = entry.get("student_name", "")
        assignments: list[dict] = entry.get("assignments", [])

        if not assignments:
            trends.append(StudentTrend(
                student_id=student_id,
                student_name=student_name,
                scores=[],
                trend_direction="stable",
                at_risk=False,
                risk_factors=[],
            ))
            continue

        # Build score records.
        score_records: list[dict] = []
        percentages: list[float] = []
        for a in assignments:
            total = float(a.get("total", 0))
            score = float(a.get("score", 0))
            pct = (score / total * 100.0) if total > 0 else 0.0
            score_records.append({
                "assignment": a.get("assignment", a.get("name", "")),
                "score": score,
                "total": total,
                "percentage": round(pct, 1),
                "date": a.get("date", ""),
            })
            percentages.append(pct)

        # Determine trend direction.
        trend_direction = _compute_trend(percentages)

        # Risk assessment.
        risk_factors: list[str] = []
        avg_pct = statistics.mean(percentages) if percentages else 0.0

        # Rule 1: declining over 3+ assignments.
        if trend_direction == "declining" and len(percentages) >= 3:
            risk_factors.append(
                f"Declining performance over {len(percentages)} assignments"
            )

        # Rule 2: overall average below 60%.
        if avg_pct < 60.0:
            risk_factors.append(f"Overall average {avg_pct:.1f}% is below 60%")

        # Rule 3: latest score > 20pp below own average.
        if percentages:
            latest = percentages[-1]
            if avg_pct - latest > 20.0:
                risk_factors.append(
                    f"Latest score {latest:.1f}% is {avg_pct - latest:.1f}pp below average"
                )

        trends.append(StudentTrend(
            student_id=student_id,
            student_name=student_name,
            scores=score_records,
            trend_direction=trend_direction,
            at_risk=len(risk_factors) > 0,
            risk_factors=risk_factors,
        ))

    return trends


def flag_question_quality(item_analyses: list[ItemAnalysis]) -> list[QuestionQualityFlag]:
    """Flag questions with potential quality issues.

    Thresholds:
        - **too_easy**: difficulty_index > 0.95 (>95% got full credit)
        - **too_hard**: difficulty_index < 0.10 (<10% got full credit)
        - **poor_discrimination**: discrimination_index < 0.15

    Args:
        item_analyses: Output from ``compute_item_analysis``.

    Returns:
        Quality flags sorted by severity (critical first) then question number.
    """
    flags: list[QuestionQualityFlag] = []

    for item in item_analyses:
        # Too easy.
        if item.difficulty_index > 0.95:
            severity = "info" if item.difficulty_index <= 0.98 else "warning"
            flags.append(QuestionQualityFlag(
                question_number=item.question_number,
                flag_type="too_easy",
                severity=severity,
                detail=(
                    f"Difficulty index {item.difficulty_index:.2f} — "
                    f"{item.difficulty_index * 100:.0f}% of students got full credit."
                ),
            ))

        # Too hard.
        if item.difficulty_index < 0.10:
            severity = "warning" if item.difficulty_index > 0.05 else "critical"
            flags.append(QuestionQualityFlag(
                question_number=item.question_number,
                flag_type="too_hard",
                severity=severity,
                detail=(
                    f"Difficulty index {item.difficulty_index:.2f} — "
                    f"only {item.difficulty_index * 100:.0f}% of students got full credit."
                ),
            ))

        # Poor discrimination.
        if item.discrimination_index < 0.15:
            severity = "warning" if item.discrimination_index >= 0.0 else "critical"
            flags.append(QuestionQualityFlag(
                question_number=item.question_number,
                flag_type="poor_discrimination",
                severity=severity,
                detail=(
                    f"Discrimination index {item.discrimination_index:.2f} — "
                    f"question does not distinguish strong from weak students."
                ),
            ))

    # Sort: critical first, then by question number.
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    flags.sort(key=lambda f: (severity_order.get(f.severity, 3), f.question_number))

    return flags


def compute_bloom_coverage(item_analyses: list[ItemAnalysis]) -> dict[str, int]:
    """Count questions at each Bloom's Taxonomy level.

    Args:
        item_analyses: Output from ``compute_item_analysis``.

    Returns:
        Mapping of Bloom level value (e.g. ``"remember"``) to the number of
        questions at that level.  Levels with zero questions are included.
    """
    coverage: dict[str, int] = {level.value: 0 for level in BloomLevel}
    for item in item_analyses:
        if item.bloom_level is not None:
            coverage[item.bloom_level.value] += 1
    return coverage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_trend(percentages: list[float]) -> str:
    """Classify a score sequence as improving, stable, or declining.

    Uses a simple linear slope heuristic.  With fewer than 3 data points
    the trend is considered stable.
    """
    if len(percentages) < 3:
        return "stable"

    # Compute simple linear regression slope.
    n = len(percentages)
    xs = list(range(n))
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(percentages)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, percentages))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return "stable"
    slope = numerator / denominator

    # Threshold: +/- 2 percentage points per assignment.
    if slope > 2.0:
        return "improving"
    elif slope < -2.0:
        return "declining"
    return "stable"


def _compute_concept_mastery(
    item_analyses: list[ItemAnalysis],
    results: list[GradingResult],
) -> list[ConceptMastery]:
    """Derive concept mastery from bloom-level groupings.

    Groups questions by their Bloom level and computes average mastery
    across students for each level.
    """
    from collections import defaultdict

    bloom_questions: dict[BloomLevel, list[int]] = defaultdict(list)
    for item in item_analyses:
        if item.bloom_level is not None:
            bloom_questions[item.bloom_level].append(item.question_number)

    if not bloom_questions:
        return []

    mastery_list: list[ConceptMastery] = []
    num_students = len(results)

    for bloom, q_nums in sorted(bloom_questions.items(), key=lambda x: list(BloomLevel).index(x[0])):
        q_set = set(q_nums)
        # Compute average percentage across all students for these questions.
        student_pcts: list[float] = []
        for r in results:
            earned = 0.0
            possible = 0.0
            for q in r.question_results:
                if q.question_number in q_set:
                    earned += q.points_earned
                    possible += q.points_possible
            if possible > 0:
                student_pcts.append(earned / possible * 100.0)

        avg_mastery = statistics.mean(student_pcts) if student_pcts else 0.0

        mastery_list.append(ConceptMastery(
            concept=f"Bloom: {bloom.value.capitalize()}",
            bloom_level=bloom,
            mastery_percentage=round(avg_mastery, 1),
            student_count=num_students,
            questions_assessed=sorted(q_nums),
        ))

    return mastery_list
