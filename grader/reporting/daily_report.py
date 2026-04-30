"""
Daily Grading Session Report

Generates comprehensive summary after each grading session including:
- Class statistics (mean, median, std dev, range)
- Question-level analysis (DI, p-value)
- Fairness audit (disparate impact)
- Confidence distribution
- Actionable next steps
"""

from typing import Dict, List
from datetime import datetime
import statistics
from grader.analytics import (
    calculate_discrimination_index,
    calculate_item_difficulty,
    flag_problematic_questions,
)
from grader.fairness import analyze_section_grades
from grader.confidence_reporter import generate_confidence_summary


def generate_daily_report(
    results: List[Dict],
    exam_name: str = "Exam",
    section: str = "T5",
    timestamp: datetime = None,
) -> str:
    """
    Generate comprehensive daily grading report.

    Args:
        results: List of student results with 'score', 'correct', 'confidence_level'
        exam_name: Exam name (e.g., "BIOL 107 Exam 2")
        section: Section identifier (e.g., "T5")
        timestamp: Timestamp for report (defaults to now)

    Returns:
        Formatted report string
    """
    if timestamp is None:
        timestamp = datetime.now()

    scores = [r.get("score", 0) for r in results]
    n_students = len(results)
    total_possible = 0.0
    for result in results:
        total_possible = (
            result.get("total_possible")
            or result.get("total")
            or total_possible
        )
        if total_possible:
            break
    total_possible = float(total_possible or 0.0)
    below_threshold = total_possible * 0.5 if total_possible else 0.0

    report_lines = []

    # Header
    report_lines.append("╔" + "═" * 78 + "╗")
    report_lines.append(
        f"║ GRADING SESSION REPORT — {timestamp.strftime('%Y-%m-%d %H:%M')} " + " " * 37 + "║"
    )
    report_lines.append("╚" + "═" * 78 + "╝")
    report_lines.append("")

    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("─" * 80)
    report_lines.append(f"Exam:           {exam_name}")
    report_lines.append(f"Section:        {section}")
    report_lines.append(f"Students:       {n_students}")
    report_lines.append("")

    # Grade statistics
    below_50 = 0  # default for empty results; overwritten below if scores exist
    if scores:
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        below_50 = sum(1 for s in scores if total_possible and s < below_threshold)
        total_label = f"{total_possible:.0f}" if total_possible.is_integer() else f"{total_possible:g}"

        report_lines.append("GRADE STATISTICS")
        report_lines.append("─" * 80)
        if total_possible:
            report_lines.append(f"Mean:           {mean_score:6.2f} / {total_label} ({mean_score/total_possible*100:5.1f}%)")
            report_lines.append(f"Median:         {median_score:6.2f} / {total_label} ({median_score/total_possible*100:5.1f}%)")
        else:
            report_lines.append(f"Mean:           {mean_score:6.2f}")
            report_lines.append(f"Median:         {median_score:6.2f}")
        report_lines.append(f"Std Dev:        {std_dev:6.2f}")
        report_lines.append(f"Range:          {min_score:6.2f} — {max_score:6.2f}")
        report_lines.append(f"Below 50%:      {below_50} students ({below_50/n_students*100:5.1f}%)")
        report_lines.append("")

    # Question analysis
    report_lines.append("QUESTION-LEVEL ANALYSIS")
    report_lines.append("─" * 80)

    problematic = flag_problematic_questions(results)

    if problematic:
        report_lines.append(f"⚠ {len(problematic)} problematic questions identified:")
        report_lines.append("")
        for q_num in sorted(problematic.keys()):
            di = calculate_discrimination_index(results, q_num)
            p = calculate_item_difficulty(results, q_num)
            flags = ", ".join(problematic[q_num]["flags"])
            report_lines.append(f"  Q{q_num:2d}: DI={di:+.3f}, p={p:.1%}  →  {flags}")
    else:
        report_lines.append("✓ No problematic questions")

    report_lines.append("")

    # Confidence distribution
    report_lines.append("CONFIDENCE DISTRIBUTION")
    report_lines.append("─" * 80)
    report_lines.append(generate_confidence_summary(results))
    report_lines.append("")

    # Fairness audit
    report_lines.append("FAIRNESS AUDIT")
    report_lines.append("─" * 80)

    section_analysis = analyze_section_grades(results, section_key="section")
    if len(section_analysis) > 1:
        report_lines.append(f"✓ {len(section_analysis)} sections analyzed")
        for sec_name in sorted(section_analysis.keys()):
            stats = section_analysis[sec_name]
            report_lines.append(
                f"  {sec_name}: n={stats['count']:2d}, mean={stats['mean']:5.1f}, "
                f"std={stats['stdev']:4.1f}"
            )
    else:
        report_lines.append(f"✓ Single section ({section}); no cross-section comparison")

    report_lines.append("")

    # Recommendations
    report_lines.append("NEXT STEPS")
    report_lines.append("─" * 80)

    if below_50 > 3:
        report_lines.append(f"⚠ {below_50} students scored <50%. Consider:")
        report_lines.append("  • Manual spot-check of 3-5 lowest scoring students")
        report_lines.append("  • Review if questions are testing learned material vs. guess-and-check")
    else:
        report_lines.append("✓ Grade distribution within normal range")

    if problematic:
        report_lines.append("⚠ Question revision needed:")
        report_lines.append("  • Review problematic questions (marked above)")
        report_lines.append("  • Clarify wording or accept multiple correct answers")
        report_lines.append("  • Monitor in future exams for improvement")

    if sum(1 for r in results if r.get("confidence_level") == "LOW") > n_students * 0.1:
        report_lines.append(f"⚠ {sum(1 for r in results if r.get('confidence_level') == 'LOW')} grades have low confidence.")
        report_lines.append("  • Review these before posting to Canvas")

    report_lines.append("")
    report_lines.append("✓ Ready for instructor review before posting to Canvas")
    report_lines.append("")

    return "\n".join(report_lines)


def save_report_to_file(
    report: str, output_dir: str = "output/", filename: str = None
) -> str:
    """
    Save report to file.

    Args:
        report: Report text
        output_dir: Output directory (default "output/")
        filename: Filename (default: grading_session_YYYY-MM-DD_HH-MM-SS.txt)

    Returns:
        Path to saved file
    """
    import os

    if filename is None:
        filename = f"grading_session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(report)

    return filepath
