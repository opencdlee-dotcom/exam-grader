"""
Bias detection for the AI grading system.

Checks whether scan quality metrics (faint ink, enhancement strategy, DPI)
correlate with lower scores — which would indicate systematic bias from
the image processing pipeline rather than genuine student performance.

Industry standard: SMD (Standardized Mean Difference) < 0.15 indicates
acceptable fairness. SMD > 0.15 warrants investigation.

Usage:
    python main.py bias-check --results output/grades_*.xlsx
"""

import logging
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_smd(group_a: list[float], group_b: list[float]) -> float:
    """
    Compute Standardized Mean Difference (Cohen's d) between two groups.

    SMD < 0.15 = acceptable (no meaningful bias)
    SMD 0.15-0.30 = investigate (possible bias)
    SMD > 0.30 = concerning (likely bias)

    Returns absolute SMD value.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)

    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)

    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)

    if pooled_var == 0:
        return 0.0

    pooled_sd = pooled_var ** 0.5
    return abs(mean_a - mean_b) / pooled_sd


def check_quality_score_correlation(results: list[dict]) -> dict:
    """
    Check if scan quality metrics correlate with grades.

    Splits students into groups by quality characteristics and
    computes SMD for each comparison.

    Args:
        results: List of grade result dicts (from grade_exam_batch/grade_batch)

    Returns:
        Bias report dict with SMD values and flags
    """
    # Collect data
    faint_scores = []  # Students with faint ink detected
    normal_scores = []  # Students without faint ink
    enhanced_scores = []  # Students whose images were auto-enhanced
    unenhanced_scores = []  # Students with no enhancement
    poor_quality_scores = []  # Students with poor scan quality
    good_quality_scores = []  # Students with good scan quality

    for result in results:
        if result.get("error"):
            continue

        sr = result.get("structured_result")
        if not sr or sr.total_possible == 0:
            continue

        pct = (sr.total_score / sr.total_possible) * 100
        quality = result.get("quality_summary", {})

        # Faint ink grouping
        if quality.get("faint_ink_detected", 0) > 0:
            faint_scores.append(pct)
        else:
            normal_scores.append(pct)

        # Enhancement grouping
        if quality.get("auto_enhanced", 0) > 0:
            enhanced_scores.append(pct)
        else:
            unenhanced_scores.append(pct)

        # Quality grouping
        if quality.get("poor_quality", 0) > 0:
            poor_quality_scores.append(pct)
        else:
            good_quality_scores.append(pct)

    # Compute SMDs
    comparisons = {}

    if faint_scores and normal_scores:
        smd = compute_smd(faint_scores, normal_scores)
        comparisons["faint_ink_vs_normal"] = {
            "smd": round(smd, 4),
            "faint_n": len(faint_scores),
            "normal_n": len(normal_scores),
            "faint_mean": round(statistics.mean(faint_scores), 2),
            "normal_mean": round(statistics.mean(normal_scores), 2),
            "flag": "investigate" if smd > 0.15 else "ok",
            "direction": "faint scores lower" if statistics.mean(faint_scores) < statistics.mean(normal_scores) else "faint scores higher",
        }

    if enhanced_scores and unenhanced_scores:
        smd = compute_smd(enhanced_scores, unenhanced_scores)
        comparisons["enhanced_vs_unenhanced"] = {
            "smd": round(smd, 4),
            "enhanced_n": len(enhanced_scores),
            "unenhanced_n": len(unenhanced_scores),
            "enhanced_mean": round(statistics.mean(enhanced_scores), 2),
            "unenhanced_mean": round(statistics.mean(unenhanced_scores), 2),
            "flag": "investigate" if smd > 0.15 else "ok",
            "direction": "enhanced scores lower" if statistics.mean(enhanced_scores) < statistics.mean(unenhanced_scores) else "enhanced scores higher",
        }

    if poor_quality_scores and good_quality_scores:
        smd = compute_smd(poor_quality_scores, good_quality_scores)
        comparisons["poor_quality_vs_good"] = {
            "smd": round(smd, 4),
            "poor_n": len(poor_quality_scores),
            "good_n": len(good_quality_scores),
            "poor_mean": round(statistics.mean(poor_quality_scores), 2),
            "good_mean": round(statistics.mean(good_quality_scores), 2),
            "flag": "investigate" if smd > 0.15 else "ok",
            "direction": "poor quality scores lower" if statistics.mean(poor_quality_scores) < statistics.mean(good_quality_scores) else "poor quality scores higher",
        }

    # Overall assessment
    any_flagged = any(c["flag"] != "ok" for c in comparisons.values())
    max_smd = max((c["smd"] for c in comparisons.values()), default=0.0)

    report = {
        "total_students": len(results),
        "students_analyzed": len(faint_scores) + len(normal_scores),
        "comparisons": comparisons,
        "overall_flag": "investigate" if any_flagged else "ok",
        "max_smd": round(max_smd, 4),
        "recommendation": (
            "⚠ BIAS DETECTED: Scan quality metrics correlate with scores (SMD > 0.15). "
            "Review enhancement pipeline and consider manual review of affected students."
            if any_flagged else
            "✓ No significant quality-score bias detected (all SMD < 0.15)."
        ),
    }

    return report


def print_bias_report(report: dict):
    """Log a formatted bias detection report."""
    logger.info("=" * 60)
    logger.info("BIAS DETECTION REPORT")
    logger.info("=" * 60)
    logger.info("  Students analyzed: %d/%d", report["students_analyzed"], report["total_students"])
    logger.info("  Max SMD: %.4f (threshold: 0.15)", report["max_smd"])

    for name, comp in report.get("comparisons", {}).items():
        icon = "WARNING" if comp["flag"] != "ok" else "OK"
        logger.info("  [%s] %s:", icon, name)
        logger.info("    SMD = %.4f (%s)", comp["smd"], comp["flag"])

        # Log group details based on comparison type
        for key, value in comp.items():
            if key.endswith("_n"):
                group = key.replace("_n", "")
                mean_key = f"{group}_mean"
                if mean_key in comp:
                    logger.info("    %s: n=%s, mean=%.1f%%", group, value, comp[mean_key])

        if comp.get("direction"):
            logger.info("    Direction: %s", comp["direction"])

    logger.info("  %s", report["recommendation"])
    logger.info("=" * 60)
