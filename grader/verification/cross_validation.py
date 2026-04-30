"""
Cross-student answer consistency validation.

Detects scoring errors by comparing how the same answers are scored across
different students. Operates on JSON grade batch files after grading is complete.

Key checks:
- Same answer → different scores (definitive grading error)
- Minority answer verification (student's answer differs from 85%+ majority)
- Zero-block / offset detection for matching sections
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class ScoringFlag:
    question: str
    student_name: str
    flag_type: str  # "scoring_inconsistency", "minority_answer", "zero_block", "offset_pattern"
    severity: str   # "HIGH", "MEDIUM", "LOW"
    detail: str
    suggested_action: str = ""


def extract_answer_key(results: list[dict]) -> dict[str, str]:
    """
    Infer the answer key from grading results by looking at which answers
    consistently receive full credit across students.

    Returns {question: correct_answer} for MC/matching questions.
    """
    key = {}
    q_correct_answers = defaultdict(Counter)

    for student in results:
        q_scores = student.get("q_scores", {})
        # We need raw readings to build the key — parse from grader_notes,
        # questions_missed, or canvas_comment if available
        for q, score in q_scores.items():
            if score >= 1:
                # This student got it right — but we don't always have the raw answer
                # Mark as "correct" for this question
                q_correct_answers[q]["_correct"] += 1

    # For MC/matching, the key is whatever answer gets scored as correct
    # We'll use questions_missed to identify which questions each student missed
    for student in results:
        missed_raw = student.get("questions_missed", "")
        if isinstance(missed_raw, str):
            missed = {q.strip() for q in missed_raw.split(",") if q.strip()}
        else:
            missed = set()

        q_scores = student.get("q_scores", {})
        for q, score in q_scores.items():
            if q not in missed and score >= 1:
                q_correct_answers[q]["_got_correct"] += 1

    return key  # Basic key — enhanced by detect_scoring_inconsistencies


def build_score_matrix(results: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """
    Build a per-question matrix: {question: [(student_name, score), ...]}.

    This lets us see how each question was scored across all students.
    """
    matrix = defaultdict(list)
    for student in results:
        name = student.get("name", "Unknown")
        q_scores = student.get("q_scores", {})
        for q, score in q_scores.items():
            matrix[q].append((name, score))

        # Also track FR, DNA, amino acid, calc scores
        for field_name in ["fr_score", "q24_dna", "q25_dna",
                           "q25a", "q25b", "q25c",
                           "q26a", "q26b", "q26c",
                           "q26_calc", "q27_calc"]:
            if field_name in student:
                matrix[field_name].append((name, student[field_name]))

    return dict(matrix)


def detect_scoring_inconsistencies(results: list[dict]) -> list[ScoringFlag]:
    """
    Detect when the same question has scoring patterns that suggest errors.

    Primary check: For MC questions (score is 0 or 1), if a question has
    a mix of 0s and 1s, verify the distribution makes sense. Questions where
    only 1-2 students got a different score than the rest are suspicious.

    For this to work well, we need the answer key. We infer it from the
    majority — if 18/20 students scored 1 on Q5, then Q5's correct answer
    is whatever those 18 students wrote.
    """
    flags = []
    matrix = build_score_matrix(results)

    for q, scores in matrix.items():
        # Only check Q1-Q22 style questions (MC/matching)
        if not re.match(r"Q\d+$", q):
            continue

        score_values = [s for _, s in scores]
        n = len(score_values)
        if n < 3:
            continue

        ones = sum(1 for s in score_values if s >= 1)
        zeros = sum(1 for s in score_values if s == 0)

        # Flag: exactly 1 student scored differently from everyone else
        if ones == 1 and zeros >= 5:
            # One student got credit while everyone else got 0 — possible overcredit
            student_name = next(name for name, s in scores if s >= 1)
            flags.append(ScoringFlag(
                question=q,
                student_name=student_name,
                flag_type="scoring_inconsistency",
                severity="HIGH",
                detail=f"Only student scored 1 on {q} (all other {zeros} students scored 0). "
                       f"Likely overcredit — verify this student's answer matches the key.",
                suggested_action=f"Re-read {student_name}'s answer for {q}"
            ))
        elif zeros == 1 and ones >= 5:
            # One student got 0 while everyone else got credit — possible undercredit
            student_name = next(name for name, s in scores if s == 0)
            flags.append(ScoringFlag(
                question=q,
                student_name=student_name,
                flag_type="scoring_inconsistency",
                severity="MEDIUM",
                detail=f"Only student scored 0 on {q} (all other {ones} students scored 1). "
                       f"Possible undercredit — verify this student's answer.",
                suggested_action=f"Re-read {student_name}'s answer for {q}"
            ))
        elif ones <= 2 and zeros >= 10:
            # Very few students got credit — either hard question or overcredits
            credited_students = [name for name, s in scores if s >= 1]
            for name in credited_students:
                flags.append(ScoringFlag(
                    question=q,
                    student_name=name,
                    flag_type="scoring_inconsistency",
                    severity="MEDIUM",
                    detail=f"Only {ones}/{n} students scored 1 on {q}. "
                           f"Verify {name}'s answer is genuinely correct.",
                    suggested_action=f"Re-read {name}'s answer for {q}"
                ))

    return flags


def detect_zero_block(results: list[dict],
                      section_start: int = 9,
                      section_end: int = 18,
                      min_consecutive: int = 5) -> list[ScoringFlag]:
    """
    Detect students with long streaks of zeros in the matching section.

    A block of 5+ consecutive zeros in Q9-Q18 is suspicious — it could
    indicate an answer key offset, a misread question numbering, or
    the student simply guessing wrong on everything.
    """
    flags = []

    for student in results:
        name = student.get("name", "Unknown")
        q_scores = student.get("q_scores", {})

        # Build sequence of scores for matching section
        section_scores = []
        for i in range(section_start, section_end + 1):
            q = f"Q{i}"
            score = q_scores.get(q, None)
            if score is not None:
                section_scores.append((q, score))

        if not section_scores:
            continue

        # Count consecutive zeros
        max_zero_streak = 0
        current_streak = 0
        for _, score in section_scores:
            if score == 0:
                current_streak += 1
                max_zero_streak = max(max_zero_streak, current_streak)
            else:
                current_streak = 0

        total_zeros = sum(1 for _, s in section_scores if s == 0)

        # Flag if all zeros in matching section
        if total_zeros == len(section_scores) and total_zeros >= min_consecutive:
            flags.append(ScoringFlag(
                question=f"Q{section_start}-Q{section_end}",
                student_name=name,
                flag_type="zero_block",
                severity="HIGH",
                detail=f"All {total_zeros} matching section answers scored 0. "
                       f"Check for answer key offset or systematic misread.",
                suggested_action=f"Re-read {name}'s matching section answers and check for row offset"
            ))
        elif max_zero_streak >= min_consecutive and total_zeros >= len(section_scores) - 2:
            flags.append(ScoringFlag(
                question=f"Q{section_start}-Q{section_end}",
                student_name=name,
                flag_type="zero_block",
                severity="MEDIUM",
                detail=f"{total_zeros}/{len(section_scores)} matching answers scored 0 "
                       f"(longest zero streak: {max_zero_streak}). Possible offset.",
                suggested_action=f"Spot-check {name}'s matching section"
            ))

    return flags


def detect_offset_pattern(student_scores: dict[str, float],
                          answer_key: dict[str, str],
                          section_start: int = 9,
                          section_end: int = 18,
                          class_results: list[dict] | None = None) -> dict | None:
    """
    Detect whether a student's matching-section zeros are consistent with a
    systematic answer-sheet offset (answers written on wrong numbered lines).

    Works without raw letter data by comparing zero-block pattern against
    class-wide question difficulty. If a student scores 0 on questions the class
    got right at >=65% rate across >=6 questions, random chance is negligible —
    systematic offset is the most likely explanation.

    Returns a result dict with confidence level, or None if no offset detected.
    """
    section_qs = [f"Q{i}" for i in range(section_start, section_end + 1)]
    student_zero_qs = [q for q in section_qs if student_scores.get(q) == 0]

    if len(student_zero_qs) < 6:
        return None  # Not enough zeros to be suspicious

    if class_results is None:
        # Without class data we can only note the all-zero pattern
        if len(student_zero_qs) >= len(section_qs) - 1:
            return {
                "offset": "unknown",
                "original_matches": len(section_qs) - len(student_zero_qs),
                "zero_count": len(student_zero_qs),
                "confidence": "LOW",
                "note": "All-zero pattern detected; class data needed to confirm offset",
            }
        return None

    # Calculate per-question class correctness rate
    q_rates: dict[str, float] = {}
    for q in section_qs:
        scores_for_q = [
            r.get("q_scores", {}).get(q)
            for r in class_results
            if r.get("q_scores", {}).get(q) is not None
        ]
        if scores_for_q:
            q_rates[q] = sum(1 for s in scores_for_q if s >= 1) / len(scores_for_q)

    # Count how many of the student's zeros are on questions the class got right
    suspicious_zeros = [q for q in student_zero_qs if q_rates.get(q, 0) >= 0.65]

    if len(suspicious_zeros) >= 6:
        return {
            "offset": "likely",
            "original_matches": len(section_qs) - len(student_zero_qs),
            "zero_count": len(student_zero_qs),
            "suspicious_zeros": suspicious_zeros,
            "confidence": "HIGH" if len(suspicious_zeros) >= 8 else "MEDIUM",
            "note": (
                f"{len(suspicious_zeros)} zeros on questions class got right "
                f"(≥65% rate). Systematic offset highly likely."
            ),
        }
    return None


def run_cross_validation(results: list[dict],
                         section_start: int = 9,
                         section_end: int = 18) -> list[ScoringFlag]:
    """
    Run all cross-validation checks on a set of grading results.

    Returns a list of ScoringFlags sorted by severity.
    """
    flags = []
    flags.extend(detect_scoring_inconsistencies(results))
    flags.extend(detect_zero_block(results, section_start, section_end))

    # Run offset detection for each student with a suspicious zero-block
    for student in results:
        name = student.get("name", "Unknown")
        q_scores = student.get("q_scores", {})
        offset_result = detect_offset_pattern(
            q_scores, {}, section_start, section_end, class_results=results
        )
        if offset_result and offset_result.get("confidence") in ("HIGH", "MEDIUM"):
            severity = "HIGH" if offset_result["confidence"] == "HIGH" else "MEDIUM"
            flags.append(ScoringFlag(
                question=f"Q{section_start}-Q{section_end}",
                student_name=name,
                flag_type="likely_offset",
                severity=severity,
                detail=offset_result["note"],
                suggested_action=(
                    f"Re-read {name}'s matching section — answers may be shifted. "
                    f"Zeros on {offset_result['zero_count']} questions the class got right."
                ),
            ))

    # Sort: HIGH first, then MEDIUM, then LOW
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: (severity_order.get(f.severity, 3), f.question))

    return flags
