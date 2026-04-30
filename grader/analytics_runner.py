"""
grader/analytics_runner.py — Post-grading analytics and integrity checks.

Contains the post-batch analysis functions:
  - _run_batch_analytics(): item analysis, class stats, score compression detection
  - _run_integrity_checks(): cross-submission similarity detection
"""

import logging
import os

logger = logging.getLogger(__name__)


def _run_batch_analytics(results: list[dict]):
    """
    Run analytics on a batch of grading results.
    Prints item analysis, flags problematic questions, and identifies at-risk students.
    """
    try:
        from grading_intelligence.analytics import (
            compute_item_analysis,
            compute_class_analytics,
            flag_question_quality,
        )

        # Collect structured results
        structured = [r["structured_result"] for r in results
                      if r.get("structured_result") and not r.get("error")]

        if len(structured) < 2:
            return  # Need at least 2 students for meaningful analytics

        # Compute class analytics
        analytics = compute_class_analytics(structured, "Batch Grading")

        logger.info("=== Class Analytics (%d students) ===", analytics.num_students)
        logger.info(
            "  Mean: %.1f%%  |  Median: %.1f%%  |  Std Dev: %.1f",
            analytics.mean_score, analytics.median_score, analytics.std_dev,
        )
        logger.info("  Range: %.1f%% - %.1f%%", analytics.min_score, analytics.max_score)

        # Score compression detection: LLMs tend to compress distributions
        # (fewer 0s and 100s than expected). Flag if SD is suspiciously low.
        if analytics.num_students >= 5:
            percentages = [s.percentage for s in structured]
            low_tail = sum(1 for p in percentages if p <= 20)
            high_tail = sum(1 for p in percentages if p >= 90)
            n = len(percentages)
            # Expected: ~5-10% of students in tails for a typical exam
            # If SD < 8 (very compressed) or both tails empty with 10+ students, flag
            if analytics.std_dev < 8 and n >= 10:
                logger.warning("POSSIBLE SCORE COMPRESSION: SD=%.1f is unusually low.", analytics.std_dev)
                logger.warning("AI graders tend to compress scores toward the mean. Review tail cases manually.")
            if n >= 10 and low_tail == 0 and high_tail == 0:
                logger.warning("NO TAIL SCORES: 0 students below 20%% or above 90%% in %d submissions.", n)
                logger.warning("Consider spot-checking highest and lowest scorers for accuracy.")

        # Item analysis
        items = compute_item_analysis(structured)
        if items:
            flags = flag_question_quality(items)
            if flags:
                logger.info("=== Question Quality Flags ===")
                for flag in flags:
                    logger.info("  Q%d: %s — %s", flag.question_number, flag.flag_type, flag.detail)

        # Generate HTML analytics dashboard
        try:
            from grading_intelligence.report_generator import generate_analytics_html
            from datetime import datetime
            html_path = os.path.join("output", "analytics", f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            os.makedirs(os.path.join("output", "analytics"), exist_ok=True)
            generate_analytics_html(structured, html_path)
            logger.info("Analytics dashboard: %s", html_path)
        except Exception as e:
            logger.debug("Dashboard generation skipped: %s", e)

        # Review queue summary
        review_needed = [r for r in structured if r.needs_human_review]
        if review_needed:
            logger.info("=== Human Review Needed: %d submissions ===", len(review_needed))
            for r in review_needed[:5]:
                items_str = ", ".join(r.low_confidence_items[:2])
                logger.info("  %s: %s", r.student_file, items_str)
            if len(review_needed) > 5:
                logger.info("  ... and %d more", len(review_needed) - 5)

        # Auto-suggest rubric refinements
        try:
            from grading_intelligence.reasoning_engine import suggest_rubric_refinements
            # Only suggest if we have enough data and flags were computed
            if len(structured) >= 5 and items:
                all_flags = flag_question_quality(items)
                if all_flags:
                    # Note: This would call the LLM - only run if explicitly requested
                    logger.info("Rubric refinement available: run with --refine-rubric flag")
        except Exception:
            pass

        # Rubric evolution: suggest improvements for low-confidence questions
        if len(structured) >= 5:
            try:
                from grader.rubric_evolver import run_rubric_evolution  # noqa: F401
                # Note: llm_fn not available here — log suggestion for CLI use
                logger.info(
                    "Tip: Run 'python main.py evolve-rubric --results-dir <dir> --key <key.json>' "
                    "to get AI rubric improvement suggestions based on this batch."
                )
            except ImportError:
                pass

    except Exception as e:
        logger.debug("Analytics skipped: %s", e)


def _run_integrity_checks(results: list[dict]):
    """
    Run cross-submission similarity detection on batch results.
    Flags suspicious pairs for instructor review.
    """
    try:
        from grading_intelligence.integrity import batch_similarity_check

        # Extract student answers (not grade text) for similarity comparison
        # Grade text includes shared rubric boilerplate which inflates false positives
        submissions = []
        for r in results:
            if r.get("error"):
                continue
            sr = r.get("structured_result")
            if sr and sr.question_results:
                # Use actual student answers for comparison
                answer_text = " | ".join(
                    f"Q{q.question_number}: {q.student_answer}"
                    for q in sr.question_results
                    if q.student_answer
                )
            else:
                answer_text = r.get("grade_text", "")
            submissions.append({
                "student_id": r.get("student_file", "unknown"),
                "text": answer_text,
            })

        if len(submissions) < 2:
            return

        suspicious = batch_similarity_check(submissions, threshold=0.7)
        if suspicious:
            logger.warning("=== Integrity Alert: %d suspicious pair(s) ===", len(suspicious))
            for pair in suspicious[:3]:
                logger.warning("  %s <-> %s: %.0f%% similar", pair.student_a, pair.student_b, pair.overall_similarity * 100)
                for reason in pair.suspicious_reasons:
                    logger.warning("    %s", reason)
            if len(suspicious) > 3:
                logger.warning("  ... and %d more pairs", len(suspicious) - 3)

    except Exception as e:
        logger.debug("Integrity check skipped: %s", e)
