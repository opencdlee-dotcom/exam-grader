"""
Confidence Calibration Report — post-batch summary of AI confidence accuracy.

Shows how confident the AI was on each student and which questions
needed the most human review. Estimated review time for instructors.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional


# ── Review-time constants ─────────────────────────────────────────────────────
_REVIEW_MINUTES = {"LOW": 5, "MEDIUM": 2, "HIGH": 0}


# ── Data collection ───────────────────────────────────────────────────────────

def collect_calibration_data(results: List[dict], exam_name: str) -> dict:
    """
    Aggregate calibration metrics across all graded students.

    Args:
        results: List of result dicts from grading students.  Each dict may
                 contain confidence_level (HIGH/MEDIUM/LOW) and optionally a
                 structured_result (GradingResult) with per-question confidence.
        exam_name: Human-readable exam name.

    Returns:
        Dict with keys:
          exam_name, n_students, confidence_counts,
          low_confidence_questions (q_num → count),
          estimated_review_min, calibration_history (list or None)
    """
    confidence_counts: dict = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    low_confidence_questions: dict = {}  # question_num (str) → count of LOW reads

    for r in results:
        if r.get("error"):
            continue

        # ── Student-level confidence ──────────────────────────────────────
        conf = (r.get("confidence_level") or "").upper()
        if conf not in confidence_counts:
            conf = "MEDIUM"
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        # ── Per-question confidence (from structured_result) ──────────────
        sr = r.get("structured_result")
        if sr and hasattr(sr, "question_results"):
            for q in sr.question_results:
                q_conf = ""
                if hasattr(q, "confidence"):
                    q_conf = str(q.confidence).upper()
                    if "LOW" in q_conf:
                        key = f"Q{q.question_number}"
                        low_confidence_questions[key] = low_confidence_questions.get(key, 0) + 1

    # ── Estimated review time ─────────────────────────────────────────────
    low_c = confidence_counts.get("LOW", 0)
    med_c = confidence_counts.get("MEDIUM", 0)
    estimated_review_min = low_c * _REVIEW_MINUTES["LOW"] + med_c * _REVIEW_MINUTES["MEDIUM"]

    # ── Historical calibration data (if available) ────────────────────────
    calibration_history: Optional[list] = None
    try:
        from grader.calibrated_confidence import load_calibration_data, ConfidenceCalibrator
        import numpy as np

        cal_path = os.path.join("output", "calibration", "calibrator.json")
        if os.path.exists(cal_path):
            cal = ConfidenceCalibrator.load(cal_path)
            if cal._is_fitted:
                calibration_history = [
                    {
                        "n_calibration": cal._n_calibration,
                        "mean_error": round(cal._mean_error, 2),
                        "median_error": round(cal._median_error, 2),
                        "conformal_quantile": round(cal._conformal_quantile, 2),
                        "alpha": cal.alpha,
                        "high_threshold": cal.high_threshold,
                        "medium_threshold": cal.medium_threshold,
                    }
                ]
    except Exception:
        calibration_history = None

    return {
        "exam_name": exam_name,
        "n_students": sum(confidence_counts.values()),
        "confidence_counts": confidence_counts,
        "low_confidence_questions": low_confidence_questions,
        "estimated_review_min": estimated_review_min,
        "calibration_history": calibration_history,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── Report generator ──────────────────────────────────────────────────────────

def generate_calibration_report(data: dict) -> str:
    """
    Format calibration data as a human-readable text report.

    Args:
        data: Output from collect_calibration_data().

    Returns:
        Formatted multi-line report string.
    """
    exam_name = data.get("exam_name", "Exam")
    n = data.get("n_students", 0)
    confidence_counts = data.get("confidence_counts", {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    low_conf_qs = data.get("low_confidence_questions", {})
    estimated_review_min = data.get("estimated_review_min", 0)
    calibration_history = data.get("calibration_history")
    generated_at = data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

    high_c = confidence_counts.get("HIGH", 0)
    med_c = confidence_counts.get("MEDIUM", 0)
    low_c = confidence_counts.get("LOW", 0)

    high_pct = (high_c / n * 100) if n > 0 else 0.0
    med_pct = (med_c / n * 100) if n > 0 else 0.0
    low_pct = (low_c / n * 100) if n > 0 else 0.0

    sep = "=" * 60
    thin = "\u2500" * 53  # ─────────────────────

    lines = [
        sep,
        f"CONFIDENCE CALIBRATION REPORT \u2014 {exam_name}",
        f"Generated: {generated_at}",
        sep,
        "",
        f"CONFIDENCE SUMMARY ({n} students)",
        thin,
        f"HIGH confidence  (\u226580%):   {high_c:3d} students ({high_pct:5.0f}%)  \u2014 expected 95%+ accuracy",
        f"MEDIUM confidence (50-79%): {med_c:3d} students ({med_pct:5.0f}%)  \u2014 expected 75-85% accuracy",
        f"LOW confidence  (<50%):    {low_c:3d} students ({low_pct:5.0f}%)  \u2014 expected <70% accuracy",
        "",
        "QUESTIONS NEEDING ATTENTION",
        thin,
    ]

    # Questions with more than 2 LOW confidence reads, sorted by count desc
    flagged_qs = sorted(
        ((q, cnt) for q, cnt in low_conf_qs.items() if cnt > 2),
        key=lambda x: -x[1],
    )
    if flagged_qs:
        for q_name, cnt in flagged_qs:
            lines.append(f"  {q_name}: {cnt} LOW-confidence reads")
    else:
        lines.append("  No questions with more than 2 LOW-confidence reads.")

    lines += [
        "",
        "ESTIMATED MANUAL REVIEW TIME",
        thin,
        f"LOW confidence    ({low_c} \u00d7 5 min):  {low_c * 5:4d} min",
        f"MEDIUM confidence ({med_c} \u00d7 2 min):  {med_c * 2:4d} min",
        "\u2500" * 34,
        f"Total estimated review:        {estimated_review_min:4d} min",
        "",
        "CALIBRATION STATUS",
        thin,
    ]

    if calibration_history:
        for entry in calibration_history:
            lines += [
                f"Calibration set:     {entry['n_calibration']} students",
                f"Mean absolute error: {entry['mean_error']:.2f} pts",
                f"Median abs error:    {entry['median_error']:.2f} pts",
                f"Conformal interval:  \u00b1{entry['conformal_quantile']:.2f} pts "
                f"({(1 - entry['alpha']) * 100:.0f}% coverage guarantee)",
                f"Level thresholds:    HIGH \u2265 {entry['high_threshold']}, "
                f"MEDIUM \u2265 {entry['medium_threshold']}",
            ]
    else:
        lines.append(
            "  No historical calibration data yet. Run more grading sessions to calibrate."
        )

    lines += ["", sep, ""]

    return "\n".join(lines)


# ── File persistence ──────────────────────────────────────────────────────────

def save_calibration_report(
    report: str,
    exam_name: str,
    output_dir: str = "output",
) -> str:
    """
    Save a calibration report string to a timestamped text file.

    Args:
        report: Formatted text report from generate_calibration_report().
        exam_name: Exam name used in the filename (spaces replaced with underscores).
        output_dir: Directory in which to write the file.  Created if absent.

    Returns:
        Absolute path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_name = exam_name.replace(" ", "_").replace("/", "-")
    datestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{safe_name}_calibration_{datestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(report)

    return os.path.abspath(filepath)
