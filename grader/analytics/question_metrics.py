"""
Question Calibration Analytics

Calculates Discrimination Index and Item Difficulty for each exam question.
Used to identify problematic questions, guide rubric improvements, and support
learning analytics.

Key Metrics:
- Discrimination Index (DI): difference in % correct between high/low performers
  • DI > 0.40 = excellent discriminator
  • DI 0.20-0.39 = good
  • DI 0.00-0.19 = weak
  • DI < 0 = problematic (low performers outperform high performers)

- Item Difficulty (p-value): % of students answering correctly
  • p < 0.30 = too hard (most fail)
  • p 0.30-0.70 = ideal (good spread)
  • p > 0.70 = too easy (most pass)
"""

from typing import Dict, List, Tuple
import statistics


def calculate_discrimination_index(
    results: List[Dict], question_num: int, top_percentile: float = 0.27
) -> float:
    """
    Calculate Discrimination Index for a question.

    DI = % of high performers correct - % of low performers correct

    High performers: top percentile of scores
    Low performers: bottom percentile of scores

    Args:
        results: List of student grading dicts with 'score' and 'correct' keys
        question_num: Question number (1-26)
        top_percentile: Percentile cutoff for high/low (default 27% = quartile method)

    Returns:
        DI value (range -1.0 to 1.0; >0.40 excellent, 0.20-0.39 good, <0.20 weak)
    """
    if not results or len(results) < 4:
        return 0.0

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x.get("score", 0))
    n = len(sorted_results)

    # Top/bottom groups
    cutoff = max(1, int(n * top_percentile))
    high_performers = sorted_results[-cutoff:]
    low_performers = sorted_results[:cutoff]

    # Count correct in each group
    high_correct = sum(
        1
        for r in high_performers
        if r.get("correct", {}).get(question_num, False)
    )
    low_correct = sum(
        1 for r in low_performers if r.get("correct", {}).get(question_num, False)
    )

    # Calculate percentages
    high_pct = high_correct / len(high_performers) if high_performers else 0
    low_pct = low_correct / len(low_performers) if low_performers else 0

    di = high_pct - low_pct
    return round(di, 3)


def calculate_item_difficulty(results: List[Dict], question_num: int) -> float:
    """
    Calculate Item Difficulty (p-value) for a question.

    p-value = % of students answering correctly

    Ideal range: 0.30-0.70 (indicates good discrimination)
    - p < 0.30: too hard
    - p > 0.70: too easy

    Args:
        results: List of student grading dicts
        question_num: Question number (1-26)

    Returns:
        p-value (range 0.0 to 1.0)
    """
    if not results:
        return 0.0

    correct = sum(1 for r in results if r.get("correct", {}).get(question_num, False))
    p_value = correct / len(results)
    return round(p_value, 3)


def flag_problematic_questions(
    results: List[Dict],
    question_range: Tuple[int, int] = (1, 26),
    di_threshold: float = 0.24,
    p_min: float = 0.30,
    p_max: float = 0.70,
) -> Dict[int, Dict[str, any]]:
    """
    Identify problematic questions based on DI and Item Difficulty.

    Questions flagged if:
    - DI < di_threshold (weak discriminator)
    - p-value < p_min (too hard)
    - p-value > p_max (too easy)

    Args:
        results: List of student results
        question_range: (start, end) question numbers
        di_threshold: Flag if DI below this (default 0.24)
        p_min: Flag if p-value below this (default 0.30)
        p_max: Flag if p-value above this (default 0.70)

    Returns:
        Dict mapping question_num -> {di, p_value, flags: [list of issues]}
    """
    import warnings

    # Guard: detect missing correct dicts before silently reporting zeros
    missing = sum(1 for r in results if not r.get("correct"))
    if missing == len(results):
        warnings.warn(
            f"flag_problematic_questions: no results have a 'correct' key "
            f"({missing}/{len(results)}). Analytics will report DI=0, p=0 for all questions. "
            f"Check parse_grade_text() return value.",
            stacklevel=2
        )

    problematic = {}

    for q_num in range(question_range[0], question_range[1] + 1):
        di = calculate_discrimination_index(results, q_num)
        p = calculate_item_difficulty(results, q_num)

        flags = []
        if di < di_threshold:
            flags.append(f"WEAK_DI({di:.3f})")
        if p < p_min:
            flags.append(f"TOO_HARD({p:.1%})")
        if p > p_max:
            flags.append(f"TOO_EASY({p:.1%})")

        if flags:
            problematic[q_num] = {"di": di, "p_value": p, "flags": flags}

    return problematic


def generate_question_report(
    results: List[Dict], question_range: Tuple[int, int] = (1, 26)
) -> str:
    """
    Generate human-readable report of question-level analytics.

    Args:
        results: List of student results
        question_range: (start, end) question numbers

    Returns:
        Formatted report string
    """
    report_lines = ["=== QUESTION-LEVEL ANALYTICS REPORT ===\n"]
    report_lines.append(f"Analyzed {len(results)} students, {question_range[1]} questions\n")

    # Calculate stats
    all_di = []
    all_p = []

    data_by_q = {}
    for q_num in range(question_range[0], question_range[1] + 1):
        di = calculate_discrimination_index(results, q_num)
        p = calculate_item_difficulty(results, q_num)
        data_by_q[q_num] = (di, p)
        all_di.append(di)
        all_p.append(p)

    report_lines.append(f"Mean DI: {statistics.mean(all_di):.3f}")
    report_lines.append(f"Mean p-value: {statistics.mean(all_p):.3f}\n")

    # Identify problematic
    problematic = flag_problematic_questions(results, question_range)

    if problematic:
        report_lines.append("PROBLEMATIC QUESTIONS:")
        for q_num in sorted(problematic.keys()):
            di, p = data_by_q[q_num]
            flags = ", ".join(problematic[q_num]["flags"])
            report_lines.append(f"  Q{q_num:2d}: DI={di:.3f}, p={p:.1%} — {flags}")
    else:
        report_lines.append("✓ No problematic questions identified")

    report_lines.append("\nQUESTION BREAKDOWN:")
    report_lines.append("Q#   DI     p-value  Status")
    report_lines.append("─" * 35)

    for q_num in range(question_range[0], question_range[1] + 1):
        di, p = data_by_q[q_num]
        status = "✓" if q_num not in problematic else "⚠"
        report_lines.append(
            f"{q_num:2d}   {di:+.3f}  {p:6.1%}   {status}"
        )

    return "\n".join(report_lines)
