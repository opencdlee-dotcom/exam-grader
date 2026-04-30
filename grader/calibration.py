"""
Calibration and accuracy measurement for the grading system.

Compares system grades against human-verified ground truth to compute:
- Overall accuracy (% agreement)
- Per-question accuracy (which questions get misgraded most)
- Cohen's kappa (inter-rater reliability)
- Per-question override tracking (diffs between original and instructor edits)

Usage:
    python main.py calibrate --dir calibration_exams/ --key answer_keys/lab1.json
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_cohens_kappa(
    system_scores: list[float],
    human_scores: list[float],
    tolerance: float = 0.5,
) -> float:
    """
    Compute Cohen's kappa for inter-rater reliability.

    Treats each score comparison as agree/disagree based on tolerance.
    Kappa ≥ 0.80 = substantial agreement (target for production).
    Kappa ≥ 0.60 = moderate agreement (acceptable with human review).

    Args:
        system_scores: Scores assigned by the grading system
        human_scores: Scores assigned by the human instructor
        tolerance: Maximum difference to count as agreement

    Returns:
        Cohen's kappa coefficient (-1 to 1)
    """
    if len(system_scores) != len(human_scores):
        raise ValueError("Score lists must be same length")

    n = len(system_scores)
    if n == 0:
        return 0.0

    # Observed agreement
    agreements = sum(
        1 for s, h in zip(system_scores, human_scores)
        if abs(s - h) <= tolerance
    )
    p_observed = agreements / n

    # Expected agreement by chance
    # Use the marginal proportions of "agree" for each rater
    # Simplified: probability both agree by chance
    p_expected = 0.5  # Conservative estimate for continuous scores

    if p_expected >= 1.0:
        return 1.0 if p_observed >= 1.0 else 0.0

    kappa = (p_observed - p_expected) / (1 - p_expected)
    return round(kappa, 4)


def compute_per_question_accuracy(
    system_results: list[dict],
    human_results: list[dict],
    tolerance: float = 0.5,
) -> dict[int, dict]:
    """
    Compute accuracy per question across calibration set.

    Args:
        system_results: List of {question_number: points_earned} dicts
        human_results: List of {question_number: points_earned} dicts
        tolerance: Maximum difference to count as correct

    Returns:
        {question_number: {
            "accuracy": float (0-1),
            "total_comparisons": int,
            "agreements": int,
            "avg_system": float,
            "avg_human": float,
            "bias": float (positive = system scores higher than human),
        }}
    """
    from collections import defaultdict
    q_data = defaultdict(lambda: {"system": [], "human": []})

    for sys_r, hum_r in zip(system_results, human_results):
        for q_num in set(list(sys_r.keys()) + list(hum_r.keys())):
            if q_num in sys_r and q_num in hum_r:
                q_data[q_num]["system"].append(sys_r[q_num])
                q_data[q_num]["human"].append(hum_r[q_num])

    results = {}
    for q_num, data in sorted(q_data.items()):
        sys_scores = data["system"]
        hum_scores = data["human"]
        n = len(sys_scores)
        if n == 0:
            continue

        agreements = sum(
            1 for s, h in zip(sys_scores, hum_scores)
            if abs(s - h) <= tolerance
        )
        avg_sys = sum(sys_scores) / n
        avg_hum = sum(hum_scores) / n

        results[q_num] = {
            "accuracy": round(agreements / n, 4),
            "total_comparisons": n,
            "agreements": agreements,
            "avg_system": round(avg_sys, 2),
            "avg_human": round(avg_hum, 2),
            "bias": round(avg_sys - avg_hum, 2),
        }

    return results


def run_calibration(
    calibration_dir: str,
    answer_key_path: str,
    human_grades_path: str,
    output_path: str = None,
) -> dict:
    """
    Run calibration against a set of pre-graded exams.

    Args:
        calibration_dir: Directory with calibration exam PDFs
        answer_key_path: Path to answer key PDF
        human_grades_path: Path to JSON file with human grades:
            [{"student_file": "exam1.pdf", "total_score": 42,
              "per_question": {1: 3, 2: 2, ...}}, ...]
        output_path: Where to save calibration results JSON

    Returns:
        Calibration results dict
    """
    from grader.scorer import grade_exam_batch

    # Load human grades
    with open(human_grades_path) as f:
        human_grades = json.load(f)

    human_lookup = {h["student_file"]: h for h in human_grades}

    # Grade the calibration set
    logger.info("Running calibration grading...")
    results = grade_exam_batch(calibration_dir, answer_key_path, show_key=True)

    # Compare
    system_total_scores = []
    human_total_scores = []
    system_per_q = []
    human_per_q = []

    for result in results:
        if result.get("error"):
            continue
        student_file = result["student_file"]
        if student_file not in human_lookup:
            logger.warning("No human grade for %s", student_file)
            continue

        human = human_lookup[student_file]
        sr = result.get("structured_result")
        if not sr:
            continue

        system_total_scores.append(sr.total_score)
        human_total_scores.append(human["total_score"])

        # Per-question comparison
        if sr.question_results and "per_question" in human:
            sys_q = {q.question_number: q.points_earned for q in sr.question_results}
            hum_q = {int(k): v for k, v in human["per_question"].items()}
            system_per_q.append(sys_q)
            human_per_q.append(hum_q)

    # Compute metrics
    n = len(system_total_scores)
    if n == 0:
        return {"error": "No comparable results found"}

    kappa = compute_cohens_kappa(system_total_scores, human_total_scores)

    # Overall accuracy (within 1 point)
    exact_match = sum(
        1 for s, h in zip(system_total_scores, human_total_scores)
        if abs(s - h) <= 0.5
    )
    within_1 = sum(
        1 for s, h in zip(system_total_scores, human_total_scores)
        if abs(s - h) <= 1.0
    )
    within_2 = sum(
        1 for s, h in zip(system_total_scores, human_total_scores)
        if abs(s - h) <= 2.0
    )

    avg_diff = sum(
        abs(s - h) for s, h in zip(system_total_scores, human_total_scores)
    ) / n

    per_q_accuracy = {}
    if system_per_q:
        per_q_accuracy = compute_per_question_accuracy(system_per_q, human_per_q)

    # Flag problematic questions (< 80% accuracy)
    problem_questions = [
        q_num for q_num, data in per_q_accuracy.items()
        if data["accuracy"] < 0.8
    ]

    calibration_result = {
        "timestamp": datetime.now().isoformat(),
        "num_exams": n,
        "cohens_kappa": kappa,
        "kappa_interpretation": (
            "substantial" if kappa >= 0.80 else
            "moderate" if kappa >= 0.60 else
            "fair" if kappa >= 0.40 else
            "poor"
        ),
        "exact_match_rate": round(exact_match / n, 4),
        "within_1pt_rate": round(within_1 / n, 4),
        "within_2pt_rate": round(within_2 / n, 4),
        "avg_absolute_diff": round(avg_diff, 2),
        "avg_system_bias": round(
            sum(s - h for s, h in zip(system_total_scores, human_total_scores)) / n, 2
        ),
        "per_question_accuracy": per_q_accuracy,
        "problem_questions": problem_questions,
    }

    # Save results
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join(
            "output",
            f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    with open(output_path, "w") as f:
        json.dump(calibration_result, f, indent=2)

    # Log summary
    logger.info("=" * 50)
    logger.info("CALIBRATION RESULTS (%d exams)", n)
    logger.info("=" * 50)
    logger.info("  Cohen's Kappa: %.4f (%s)", kappa, calibration_result["kappa_interpretation"])
    logger.info("  Exact Match (+-0.5): %d/%d (%.1f%%)", exact_match, n, exact_match / n * 100)
    logger.info("  Within 1 point:     %d/%d (%.1f%%)", within_1, n, within_1 / n * 100)
    logger.info("  Within 2 points:    %d/%d (%.1f%%)", within_2, n, within_2 / n * 100)
    logger.info("  Avg Absolute Diff:  %.2f points", avg_diff)
    logger.info("  System Bias:        %+.2f (+ = system scores higher)", calibration_result["avg_system_bias"])
    if problem_questions:
        logger.warning("  Problem Questions: %s (<80%% accuracy)", ", ".join(f"Q{q}" for q in problem_questions))
    logger.info("  Results saved: %s", output_path)
    logger.info("=" * 50)

    return calibration_result


def track_grade_edits(
    original_results: list[dict],
    edited_excel_path: str,
    output_path: str = None,
) -> dict:
    """
    Track per-question accuracy by diffing original grades against instructor edits.

    Args:
        original_results: Original grading results (from grade_exam_batch)
        edited_excel_path: Path to instructor-edited Excel file
        output_path: Where to save tracking data

    Returns:
        Tracking dict with per-question override rates
    """
    from grader.report import extract_score_structured
    from canvas.grades import read_grades_from_excel

    edited = read_grades_from_excel(edited_excel_path)
    edited_lookup = {r["name"].lower().strip(): r for r in edited}

    overrides = []
    for result in original_results:
        if result.get("error"):
            continue
        sr = result.get("structured_result")
        if not sr:
            continue

        name = result["student_file"].lower().strip()
        # Try to find in edited
        edited_record = edited_lookup.get(name)
        if not edited_record:
            continue

        orig_score, _ = extract_score_structured(result)
        edited_score = edited_record.get("score")
        if orig_score is not None and edited_score is not None and abs(orig_score - edited_score) > 0.01:
            overrides.append({
                "student": result["student_file"],
                "original_score": orig_score,
                "edited_score": edited_score,
                "difference": round(edited_score - orig_score, 2),
                "timestamp": datetime.now().isoformat(),
            })

    tracking = {
        "timestamp": datetime.now().isoformat(),
        "total_students": len(original_results),
        "overrides_count": len(overrides),
        "override_rate": round(len(overrides) / max(len(original_results), 1), 4),
        "overrides": overrides,
    }

    if output_path is None:
        os.makedirs(os.path.join("output", "calibration"), exist_ok=True)
        output_path = os.path.join("output", "calibration", "accuracy_tracking.json")

    # Append to existing tracking file if it exists
    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]

    existing.append(tracking)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(
        "Tracked %d grade overrides out of %d students (%.1f%% override rate)",
        len(overrides), len(original_results), tracking["override_rate"] * 100,
    )

    return tracking
