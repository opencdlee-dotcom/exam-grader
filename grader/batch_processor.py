"""
grader/batch_processor.py — Batch grading orchestration.

Contains the multi-student batch functions:
  - grade_batch(): batch notebook/lab report grading
  - grade_exam_batch(): batch exam grading
  - _recheck_below_50(): below-threshold recheck for exams
  - _recheck_below_50_notebook(): below-threshold recheck for notebooks
"""

import logging
import os
import sys

import click

from grader.config import CHUNK_SIZE, RECHECK_THRESHOLD, RECHECK_MARGIN
from grader.rubric import load_answer_key
from grader.exam_key import extract_answer_key, print_extracted_key
from grader.vision import new_usage_tracker, print_usage
from grader.pipeline import grade_submission, grade_exam_submission
from grader.progress_emitter import emit_student_graded, emit_batch_complete

logger = logging.getLogger(__name__)


def grade_batch(
    pdf_folder: str,
    answer_key_path: str,
    enhance: bool = False,
    ensemble_strategy: str = "single",
    strategy: str = "auto",
) -> list[dict]:
    """
    Grade all PDFs in a folder.

    Args:
        pdf_folder: Directory containing student PDF files
        answer_key_path: Path to the answer key JSON
        enhance: Whether to enhance images
        ensemble_strategy: "single", "vote", "debate", or "cascade"

    Returns:
        List of grade result dicts with structured_result and batch_analytics
    """
    pdfs = sorted([
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ])

    if not pdfs:
        logger.warning("No PDF files found in %s", pdf_folder)
        return []

    click.echo(f"Found {len(pdfs)} submissions to grade\n")
    results = []
    batch_usage = new_usage_tracker()

    def _show_current(pdf):
        return os.path.basename(pdf) if pdf else ""

    with click.progressbar(pdfs, label="Grading", item_show_func=_show_current,
                           file=sys.stderr, show_pos=True) as bar:
        for pdf_path in bar:
            try:
                result = grade_submission(pdf_path, answer_key_path, enhance,
                                         ensemble_strategy=ensemble_strategy, strategy=strategy)
                results.append(result)
                if "usage" in result:
                    batch_usage["input_tokens"] += result["usage"]["input_tokens"]
                    batch_usage["output_tokens"] += result["usage"]["output_tokens"]
                    batch_usage["api_calls"] += result["usage"]["api_calls"]
                # Emit progress event for real-time monitoring
                sr = result.get("structured_result")
                student_name = result.get("student_file", os.path.basename(pdf_path))
                emit_student_graded(
                    student_name=student_name,
                    score=sr.total_score if sr else result.get("score", 0),
                    max_score=sr.total_possible if sr else result.get("max_score", 35),
                    confidence=result.get("calibrated_confidence", result.get("confidence_level", "UNKNOWN")),
                    section=result.get("section", ""),
                )
            except Exception as e:
                click.echo(f"\n  ERROR grading {os.path.basename(pdf_path)}: {e}", err=True)
                results.append({
                    "student_file": os.path.basename(pdf_path),
                    "grade_text": f"ERROR: {e}",
                    "answer_key_title": "N/A",
                    "error": True,
                })

    if batch_usage["api_calls"] > 0:
        logger.info("=== Batch Token Summary ===")
        print_usage(batch_usage, f"{len(pdfs)} students")

    # Emit batch complete event
    successful = [r for r in results if not r.get("error")]
    if successful:
        scores = []
        for r in successful:
            sr = r.get("structured_result")
            if sr and sr.total_possible > 0:
                scores.append((sr.total_score / sr.total_possible) * 100)
        mean_pct = sum(scores) / len(scores) if scores else 0.0
        mean_score = sum(
            r["structured_result"].total_score for r in successful if r.get("structured_result")
        ) / len(successful) if successful else 0.0
        emit_batch_complete(n_students=len(successful), mean_score=mean_score, mean_pct=mean_pct)

    # Below-50% auto-recheck for notebook grading (same safety net as exams)
    results = _recheck_below_50_notebook(results, answer_key_path, enhance, strategy)

    # Run analytics on batch results
    from grader.analytics_runner import _run_batch_analytics, _run_integrity_checks
    _run_batch_analytics(results)

    # Run integrity checks (similarity detection)
    _run_integrity_checks(results)

    # Attach notebook answer key to results for downstream use (Excel Answer Key sheet)
    try:
        answer_key = load_answer_key(answer_key_path)
        for r in results:
            r["_answer_key"] = answer_key
    except Exception:
        pass  # Non-critical — Excel just won't have an Answer Key sheet

    return results


def grade_exam_batch(
    pdf_folder: str,
    answer_key_pdf: str,
    enhance: bool = False,
    show_key: bool = False,
    approve_key: bool = False,
    free_response_rubric: dict = None,
    enhanced_reading: bool = False,
    answer_page_numbers: list[int] | None = None,
    ensemble_strategy: str = "single",
    strategy: str = "auto",
    use_cache: bool = True,
    regrade_students: list[str] | None = None,
) -> list[dict]:
    """
    Grade all student exam PDFs in a folder against an answer key PDF.

    Args:
        pdf_folder: Directory containing student PDF files
        answer_key_pdf: Path to the instructor's answer key PDF
        enhance: Whether to enhance images
        show_key: Print extracted answer key before grading
        approve_key: Require manual approval of extracted key before grading
        free_response_rubric: Optional free response rubric dict
        use_cache: Whether to use cached results (default True)
        regrade_students: List of student filenames to force-regrade (others use cache)

    Returns:
        List of grade result dicts
    """
    regrade_set = set(regrade_students) if regrade_students else None

    # Extract answer key once for the whole batch
    extracted_key = extract_answer_key(answer_key_pdf, enhance=enhance, use_cache=use_cache)

    if show_key or approve_key:
        print_extracted_key(extracted_key)

    if approve_key:
        confirm = input(
            "\nDoes this extracted answer key look correct? "
            "All students will be graded against it. [y/N]: "
        )
        if confirm.strip().lower() != "y":
            logger.warning("Aborting — please review the answer key PDF and re-run.")
            return []

    pdfs = sorted([
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ])

    if not pdfs:
        logger.warning("No PDF files found in %s", pdf_folder)
        return []

    fr_points = free_response_rubric["total_points"] if free_response_rubric else 0
    total_points = extracted_key["total_points"] + fr_points

    click.echo(f"Found {len(pdfs)} submissions to grade")
    click.echo(f"Answer key: {extracted_key['total_points']} objective points" +
               (f" + {fr_points} free response points" if fr_points else "") +
               f" = {total_points} total\n")
    results = []
    batch_usage = new_usage_tracker()

    def _show_current(pdf):
        return os.path.basename(pdf) if pdf else ""

    with click.progressbar(pdfs, label="Grading", item_show_func=_show_current,
                           file=sys.stderr, show_pos=True) as bar:
        for pdf_path in bar:
            try:
                student_file = os.path.basename(pdf_path)
                force_fresh = regrade_set and student_file in regrade_set
                student_use_cache = use_cache and not force_fresh
                result = grade_exam_submission(pdf_path, extracted_key, enhance,
                                                      free_response_rubric=free_response_rubric,
                                                      enhanced_reading=enhanced_reading,
                                                      answer_page_numbers=answer_page_numbers,
                                                      ensemble_strategy=ensemble_strategy,
                                                      strategy=strategy,
                                                      use_cache=student_use_cache)
                results.append(result)
                if "usage" in result:
                    batch_usage["input_tokens"] += result["usage"]["input_tokens"]
                    batch_usage["output_tokens"] += result["usage"]["output_tokens"]
                    batch_usage["api_calls"] += result["usage"]["api_calls"]
                # Emit progress event for real-time monitoring
                sr = result.get("structured_result")
                emit_student_graded(
                    student_name=result.get("student_file", os.path.basename(pdf_path)),
                    score=sr.total_score if sr else result.get("score", 0),
                    max_score=sr.total_possible if sr else total_points,
                    confidence=result.get("calibrated_confidence", result.get("confidence_level", "UNKNOWN")),
                    section=result.get("section", ""),
                )
            except Exception as e:
                click.echo(f"\n  ERROR grading {os.path.basename(pdf_path)}: {e}", err=True)
                results.append({
                    "student_file": os.path.basename(pdf_path),
                    "grade_text": f"ERROR: {e}",
                    "total_points": total_points,
                    "error": True,
                })

    if batch_usage["api_calls"] > 0:
        logger.info("=== Batch Token Summary ===")
        print_usage(batch_usage, f"{len(pdfs)} students")

    # Emit batch complete event
    successful_exam = [r for r in results if not r.get("error")]
    if successful_exam:
        exam_scores = []
        for r in successful_exam:
            sr = r.get("structured_result")
            if sr and sr.total_possible > 0:
                exam_scores.append((sr.total_score / sr.total_possible) * 100)
        mean_pct_exam = sum(exam_scores) / len(exam_scores) if exam_scores else 0.0
        mean_score_exam = sum(
            r["structured_result"].total_score for r in successful_exam if r.get("structured_result")
        ) / len(successful_exam) if successful_exam else 0.0
        emit_batch_complete(n_students=len(successful_exam), mean_score=mean_score_exam, mean_pct=mean_pct_exam)

    # Below-50% auto-recheck: re-grade with ensemble voting
    results = _recheck_below_50(results, extracted_key, enhance, free_response_rubric,
                                enhanced_reading, answer_page_numbers, strategy)

    # Run analytics on batch results
    from grader.analytics_runner import _run_batch_analytics, _run_integrity_checks
    _run_batch_analytics(results)

    # Run integrity checks (similarity detection)
    _run_integrity_checks(results)

    # Attach extracted key to results for downstream use (Excel Answer Key sheet)
    for r in results:
        r["_extracted_key"] = extracted_key

    return results


def _recheck_below_50(
    results: list[dict],
    extracted_key: dict,
    enhance: bool,
    free_response_rubric: dict | None,
    enhanced_reading: bool,
    answer_page_numbers: list[int] | None,
    strategy: str,
) -> list[dict]:
    """
    Auto-regrade any student below 50% using ensemble voting.
    Compares with original score; uses higher score if they differ by >5%.
    """
    recheck_indices = []
    threshold = RECHECK_THRESHOLD
    margin = RECHECK_MARGIN
    for i, result in enumerate(results):
        if result.get("error"):
            continue
        sr = result.get("structured_result")
        if sr and sr.total_possible > 0:
            pct = (sr.total_score / sr.total_possible) * 100
            # Recheck below threshold AND within ±margin of threshold (boundary cases)
            if pct <= (threshold + margin):
                recheck_indices.append(i)

    if not recheck_indices:
        return results

    logger.info("=== Below-50%% Auto-Recheck: %d student(s) ===", len(recheck_indices))

    for idx in recheck_indices:
        result = results[idx]
        original_sr = result["structured_result"]
        original_score = original_sr.total_score
        student_file = result["student_file"]

        # Find the original PDF path
        pdf_path = None
        for key in ("pdf_path", "student_file"):
            val = result.get(key, "")
            if val and os.path.exists(val):
                pdf_path = val
                break

        if pdf_path is None:
            # Try to reconstruct from student_file
            logger.warning("%s: cannot locate PDF for recheck, skipping", student_file)
            result["recheck_result"] = {
                "original_score": original_score,
                "final_score": original_score,
                "status": "skipped_no_pdf",
            }
            continue

        logger.info("Rechecking %s (original: %s/%s)...", student_file, original_score, original_sr.total_possible)

        try:
            recheck = grade_exam_submission(
                pdf_path, extracted_key, enhance,
                free_response_rubric=free_response_rubric,
                enhanced_reading=enhanced_reading,
                answer_page_numbers=answer_page_numbers,
                ensemble_strategy="vote",
                strategy=strategy,
            )

            recheck_sr = recheck.get("structured_result")
            if recheck_sr:
                recheck_score = recheck_sr.total_score
                all_scores = [original_score, recheck_score]
                divergence = abs(recheck_score - original_score)
                pct_divergence = (divergence / original_sr.total_possible * 100) if original_sr.total_possible > 0 else 0

                if pct_divergence > 5:
                    # Scores diverge significantly — use median (no upward bias)
                    # and flag for mandatory human review
                    import statistics
                    final_score = statistics.median(all_scores)
                    logger.warning(
                        "Score DIVERGENT: %s vs %s (median: %s, divergence %.1f%%) — HUMAN REVIEW REQUIRED",
                        original_score, recheck_score, final_score, pct_divergence,
                    )

                    # Use whichever result is closer to median, and force human review
                    if abs(recheck_score - final_score) <= abs(original_score - final_score):
                        recheck_sr.consensus_confidence = 0.0  # Force human review
                        recheck_sr.comments += (
                            f"\n[RECHECK DIVERGENCE: Original={original_score}, "
                            f"Recheck={recheck_score}. Using median={final_score}. "
                            f"MANDATORY HUMAN REVIEW — scores diverged by {pct_divergence:.1f}%.]"
                        )
                        results[idx]["structured_result"] = recheck_sr
                        results[idx]["grade_text"] = recheck.get("grade_text", result["grade_text"])
                    else:
                        original_sr.consensus_confidence = 0.0
                        original_sr.comments += (
                            f"\n[RECHECK DIVERGENCE: Original={original_score}, "
                            f"Recheck={recheck_score}. Using median={final_score}. "
                            f"MANDATORY HUMAN REVIEW — scores diverged by {pct_divergence:.1f}%.]"
                        )
                    # Set the final score to median on whichever result we kept
                    results[idx]["structured_result"].total_score = final_score
                    results[idx]["grade_text"] = results[idx]["structured_result"].to_legacy_text()
                else:
                    final_score = original_score
                    logger.info("Score confirmed: %s (recheck: %s, within tolerance)", original_score, recheck_score)

                result["recheck_result"] = {
                    "original_score": original_score,
                    "recheck_score": recheck_score,
                    "final_score": final_score,
                    "divergence_pct": round(pct_divergence, 1),
                    "status": "divergent" if pct_divergence > 5 else "confirmed",
                }
            else:
                result["recheck_result"] = {
                    "original_score": original_score,
                    "final_score": original_score,
                    "status": "recheck_failed",
                }
        except Exception as e:
            logger.error("Recheck failed: %s", e)
            result["recheck_result"] = {
                "original_score": original_score,
                "final_score": original_score,
                "status": f"error: {e}",
            }

    return results


def _recheck_below_50_notebook(
    results: list[dict],
    answer_key_path: str,
    enhance: bool,
    strategy: str,
) -> list[dict]:
    """
    Auto-regrade notebook submissions near or below the recheck threshold.
    Re-grades with ensemble voting and uses the higher score if divergent.
    """
    threshold = RECHECK_THRESHOLD
    margin = RECHECK_MARGIN
    recheck_indices = []
    for i, result in enumerate(results):
        if result.get("error"):
            continue
        sr = result.get("structured_result")
        if sr and sr.total_possible > 0:
            pct = (sr.total_score / sr.total_possible) * 100
            if pct <= (threshold + margin):
                recheck_indices.append(i)

    if not recheck_indices:
        return results

    logger.info("=== Below-%d%% Auto-Recheck (notebooks): %d student(s) ===", threshold, len(recheck_indices))

    for idx in recheck_indices:
        result = results[idx]
        original_sr = result["structured_result"]
        original_score = original_sr.total_score
        student_file = result["student_file"]

        # Find the original PDF path
        pdf_path = None
        for key in ("pdf_path", "student_file"):
            val = result.get(key, "")
            if val and os.path.exists(val):
                pdf_path = val
                break

        if pdf_path is None:
            logger.warning("%s: cannot locate PDF for recheck, skipping", student_file)
            result["recheck_result"] = {
                "original_score": original_score,
                "final_score": original_score,
                "status": "skipped_no_pdf",
            }
            continue

        logger.info("Rechecking %s (original: %s/%s)...", student_file, original_score, original_sr.total_possible)

        try:
            recheck = grade_submission(
                pdf_path, answer_key_path, enhance,
                ensemble_strategy="vote", strategy=strategy,
            )

            recheck_sr = recheck.get("structured_result")
            if recheck_sr:
                recheck_score = recheck_sr.total_score
                all_scores = [original_score, recheck_score]
                divergence = abs(recheck_score - original_score)
                pct_divergence = (divergence / original_sr.total_possible * 100) if original_sr.total_possible > 0 else 0

                if pct_divergence > 5:
                    import statistics
                    final_score = statistics.median(all_scores)
                    logger.warning(
                        "Score DIVERGENT: %s vs %s (median: %s, divergence %.1f%%) — HUMAN REVIEW REQUIRED",
                        original_score, recheck_score, final_score, pct_divergence,
                    )

                    if abs(recheck_score - final_score) <= abs(original_score - final_score):
                        recheck_sr.consensus_confidence = 0.0
                        recheck_sr.comments += (
                            f"\n[RECHECK DIVERGENCE: Original={original_score}, "
                            f"Recheck={recheck_score}. Using median={final_score}. "
                            f"MANDATORY HUMAN REVIEW.]"
                        )
                        results[idx]["structured_result"] = recheck_sr
                        results[idx]["grade_text"] = recheck.get("grade_text", result["grade_text"])
                    else:
                        original_sr.consensus_confidence = 0.0
                        original_sr.comments += (
                            f"\n[RECHECK DIVERGENCE: Original={original_score}, "
                            f"Recheck={recheck_score}. Using median={final_score}. "
                            f"MANDATORY HUMAN REVIEW.]"
                        )
                    results[idx]["structured_result"].total_score = final_score
                    results[idx]["grade_text"] = results[idx]["structured_result"].to_legacy_text()
                else:
                    final_score = original_score
                    logger.info("Score confirmed: %s (recheck: %s, within tolerance)", original_score, recheck_score)

                result["recheck_result"] = {
                    "original_score": original_score,
                    "recheck_score": recheck_score,
                    "final_score": final_score,
                    "divergence_pct": round(pct_divergence, 1),
                    "status": "divergent" if pct_divergence > 5 else "confirmed",
                }
            else:
                result["recheck_result"] = {
                    "original_score": original_score,
                    "final_score": original_score,
                    "status": "recheck_failed",
                }
        except Exception as e:
            logger.error("Recheck failed: %s", e)
            result["recheck_result"] = {
                "original_score": original_score,
                "final_score": original_score,
                "status": f"error: {e}",
            }

    return results
