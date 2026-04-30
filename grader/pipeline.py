"""
grader/pipeline.py — Per-student grading pipeline.

Contains the single-submission grading functions:
  - grade_submission(): notebook/lab report grading
  - grade_exam_submission(): exam grading with instance-based scoring
  - apply_late_penalty(): post-grading late penalty application
  - apply_curve(): batch grade curve application
"""

import logging
import math
import os
import tempfile
from datetime import datetime

from grader.config import CHUNK_SIZE, GRADING_PROVIDER, normalize_grading_provider
from grader.pdf_reader import (
    convert_pdf,
    convert_pdf_survey,
    enhance_image,
    cleanup_images,
    get_page_count,
    detect_and_fix_orientation,
)
from grader.vision import print_usage
from grader.rubric import load_answer_key, format_rubric_prompt
from grader.exam_prompt import format_exam_grading_prompt, inject_pre_read_answers
from grader.answer_sheet_reader import (
    read_answer_sheet,
    reread_questions,
    find_alternate_scan_pdfs,
    merge_pre_read_answers,
)
from grader.context_expansion import build_context_prompt
from grader.image_optimizer import get_adaptive_settings
from grader.image_quality import assess_image, classify_handwriting_style
from grader.writer_calibration import (
    extract_writer_characteristics,
    build_writer_calibration_prompt,
)
from grader.writer_profile_session import (
    build_persistent_calibration_block,
    derive_writer_id,
    load_session_profile,
    update_session_profile,
)
from grader.answer_matcher import match_single_answer, MatchResult
from grader.chunker import should_chunk
from grading_intelligence.structured_output import ConfidenceLevel, GradingResult
from grading_intelligence.ensemble_grader import EnsembleGrader, EnsembleConfig

logger = logging.getLogger(__name__)


def _single_provider_label() -> str:
    """Human-readable label for the active single-provider grading path."""
    return f"{normalize_grading_provider(GRADING_PROVIDER)} vision"


def _extract_writer_calibration_answers(
    pre_read_answers: dict[int, str | dict | None],
    extracted_key: dict,
) -> dict[int, str]:
    """
    Use only trusted, deterministically matching answers as writer-profile anchors.
    """
    anchors: dict[int, str] = {}
    for question in extracted_key.get("questions", []):
        q_num = question.get("number")
        answer = pre_read_answers.get(q_num)
        if q_num is None or answer is None:
            continue
        if question.get("points", len(question.get("answers", []))) != 1:
            continue
        result, _ = match_single_answer(
            answer,
            question.get("answers", []),
            question.get("alternatives", {}),
            question.get("tolerance"),
        )
        if result == MatchResult.MATCH:
            if isinstance(answer, dict):
                primary = answer.get("primary")
                if primary and not answer.get("alternate") and answer.get("confidence") == "high":
                    anchors[q_num] = primary.strip()
            elif isinstance(answer, str) and answer.strip():
                anchors[q_num] = answer.strip()
    return anchors


def _apply_targeted_reread_corrections(
    structured_result: GradingResult,
    extracted_key: dict,
    reread_answers: dict[int, str | dict | None],
) -> bool:
    """
    Use focused reread evidence to confirm or correct low-confidence objective items.

    Returns True if any question score/confidence changed.
    """
    changed = False
    key_by_number = {q["number"]: q for q in extracted_key.get("questions", [])}

    for question_result in structured_result.question_results:
        if question_result.confidence != ConfidenceLevel.LOW:
            continue

        question = key_by_number.get(question_result.question_number)
        reread = reread_answers.get(question_result.question_number)
        if not question or reread is None:
            continue

        match_result, explanation = match_single_answer(
            reread,
            question.get("answers", []),
            question.get("alternatives", {}),
            question.get("tolerance"),
        )
        strong = not isinstance(reread, dict) or (
            reread.get("alternate") is None and str(reread.get("confidence", "low")).lower() == "high"
        )

        if match_result == MatchResult.MATCH and strong:
            desired_points = question_result.points_possible
            if question_result.points_earned != desired_points:
                question_result.points_earned = desired_points
                changed = True
            question_result.confidence = ConfidenceLevel.HIGH
            question_result.reading_confidence = "HIGH"
            question_result.raw_reading = reread if isinstance(reread, str) else (reread.get("primary") or "")
            question_result.student_answer = question_result.raw_reading
            question_result.reasoning = (
                f"{question_result.reasoning} [Focused reread confirmed answer: {explanation}]"
            ).strip()
            changed = True
        elif match_result == MatchResult.MISMATCH and strong and question_result.points_earned == question_result.points_possible:
            question_result.points_earned = 0
            question_result.confidence = ConfidenceLevel.HIGH
            question_result.reading_confidence = "HIGH"
            question_result.raw_reading = reread if isinstance(reread, str) else (reread.get("primary") or "")
            question_result.student_answer = question_result.raw_reading
            question_result.reasoning = (
                f"{question_result.reasoning} [Focused reread corrected over-credit: {explanation}]"
            ).strip()
            changed = True
        elif match_result == MatchResult.AMBIGUOUS and isinstance(reread, dict):
            question_result.alternative_readings = [
                alt for alt in [reread.get("alternate")] if alt
            ]
            question_result.raw_reading = reread.get("primary") or question_result.raw_reading

    if changed:
        structured_result.total_score = round(
            sum(q.points_earned for q in structured_result.question_results) +
            sum(fr.points_earned for fr in structured_result.free_response_results) +
            sum(s.points_earned for s in structured_result.section_results),
            1,
        )

    return changed


_BIDIRECTIONAL_CONFUSION_PAIRS = [
    ("B", "D"), ("O", "Q"), ("G", "C"), ("M", "N"), ("U", "V"),
    ("P", "F"), ("H", "A"), ("L", "I"), ("P", "R"), ("E", "F"),
    ("K", "R"),
]


def _collect_confirmed_letters(structured_result: GradingResult) -> dict[str, int]:
    """Count uppercase letters in HIGH-confidence single-letter readings.

    Used to update the persistent writer profile after grading. Only HIGH-
    confidence + earned-full-credit answers count, so the profile reflects
    confirmed correct readings rather than the model's first guess.
    """
    counts: dict[str, int] = {}
    for q in structured_result.question_results:
        if q.confidence != ConfidenceLevel.HIGH:
            continue
        if q.points_possible <= 0 or q.points_earned < q.points_possible:
            continue
        reading = (q.raw_reading or q.student_answer or "").strip()
        if len(reading) == 1 and reading.isalpha():
            letter = reading.upper()
            counts[letter] = counts.get(letter, 0) + 1
    return counts


def _collect_confusion_resolutions(
    structured_result: GradingResult,
    extracted_key: dict,
) -> dict[str, str]:
    """For each confusion pair, record the letter this writer formed clearly.

    Inputs are HIGH-confidence single-letter readings whose final score
    matches the answer key. The returned dict is per-pair: ``{"B/D": "B"}``
    means this writer reliably writes B in a way that doesn't get confused
    with D in this submission.
    """
    if not extracted_key:
        return {}

    confirmed = _collect_confirmed_letters(structured_result)
    if not confirmed:
        return {}

    resolutions: dict[str, str] = {}
    for a, b in _BIDIRECTIONAL_CONFUSION_PAIRS:
        ca = confirmed.get(a, 0)
        cb = confirmed.get(b, 0)
        if ca == 0 and cb == 0:
            continue
        # Only record a resolution when one side is dominant — equal counts
        # mean we have no signal to disambiguate this writer's preference.
        if ca > cb:
            resolutions[f"{a}/{b}"] = a
        elif cb > ca:
            resolutions[f"{a}/{b}"] = b
    return resolutions


def apply_late_penalty(result: dict, submission_time: str, deadline: str, penalty_config: dict) -> dict:
    """
    Apply late penalty to a grading result.

    Args:
        result: Grade result dict (must contain 'structured_result' or 'grade_text')
        submission_time: ISO 8601 timestamp of submission
        deadline: ISO 8601 timestamp of the deadline
        penalty_config: Dict with keys: penalty_per_day, max_penalty_percent, grace_period_hours

    Returns:
        Updated result dict with penalty applied and recorded.
    """
    from grading_intelligence.structured_output import LatePenaltyConfig

    # Guard against double-application of late penalty
    if result.get("_penalty_applied"):
        return result

    config = LatePenaltyConfig(**penalty_config)
    sub_dt = datetime.fromisoformat(submission_time)
    dead_dt = datetime.fromisoformat(deadline)

    # Calculate hours late (accounting for grace period)
    delta = sub_dt - dead_dt
    hours_late = max(0.0, delta.total_seconds() / 3600.0)

    if hours_late <= config.grace_period_hours:
        # Within grace period — no penalty
        result["late_penalty"] = 0.0
        result["days_late"] = 0.0
        return result

    # Calculate effective hours late after grace period
    effective_hours = hours_late - config.grace_period_hours
    days_late = math.ceil(effective_hours / 24.0)

    # Calculate penalty
    raw_penalty = days_late * config.penalty_per_day

    # Get the total possible points for capping
    sr = result.get("structured_result")
    if sr and hasattr(sr, "total_possible"):
        total_possible = sr.total_possible
    else:
        from grader.report import extract_score
        _, total_possible = extract_score(result.get("grade_text", ""))
        total_possible = total_possible or 0

    if total_possible == 0:
        # Cannot compute a meaningful penalty without a valid total.
        # Return early with an error flag rather than silently recording penalty=0,
        # which would look like the penalty was computed (it wasn't).
        logger.error(
            "apply_late_penalty: total_possible=0 for result '%s' — "
            "late penalty NOT applied. Both structured_result and regex failed. "
            "Manual review required.",
            result.get("student_file", result.get("name", "unknown"))
        )
        result["late_penalty_error"] = "total_possible=0 — penalty skipped, manual review required"
        result["_penalty_applied"] = True  # prevent retry loops
        return result

    # Cap penalty at max_penalty_percent of total
    max_penalty = (config.max_penalty_percent / 100.0) * total_possible
    penalty = min(raw_penalty, max_penalty)

    # Apply to structured result if available
    if sr and hasattr(sr, "total_score"):
        sr.total_score = max(0.0, sr.total_score - penalty)
        sr.comments += f"\n[LATE PENALTY: -{penalty:.1f} pts ({days_late} day(s) late)]"
        # Regenerate legacy text
        result["grade_text"] = sr.to_legacy_text()

    result["late_penalty"] = penalty
    result["days_late"] = days_late
    result["_penalty_applied"] = True
    return result


def apply_curve(results: list[dict], curve_config: dict) -> list[dict]:
    """
    Apply grade curve to batch results.

    Supported methods:
    - flat_boost: Add N points to every score
    - linear: Scale scores so the mean reaches target_mean (percentage)
    - sqrt: Square root curve (score = sqrt(score/total) * total)

    Args:
        results: List of grade result dicts
        curve_config: Dict with keys: method, target_mean, boost_points, drop_count

    Returns:
        Updated results list with curved scores.
    """
    from grading_intelligence.structured_output import CurveConfig

    config = CurveConfig(**curve_config)

    if config.method == "none":
        return results

    # Collect valid (score, total) pairs
    valid = []
    for r in results:
        if r.get("error"):
            continue
        sr = r.get("structured_result")
        if sr and hasattr(sr, "total_score") and hasattr(sr, "total_possible"):
            valid.append((sr.total_score, sr.total_possible, r))

    if not valid:
        return results

    if config.method == "flat_boost":
        for score, total, r in valid:
            sr = r["structured_result"]
            sr.total_score = min(total, score + config.boost_points)
            sr.comments += f"\n[CURVE: +{config.boost_points:.1f} flat boost]"
            r["grade_text"] = sr.to_legacy_text()

    elif config.method == "linear" and config.target_mean is not None:
        # Scale so mean percentage reaches target_mean
        totals = [t for _, t, _ in valid]
        if totals and totals[0] > 0:
            total_possible = totals[0]
            current_mean_pct = sum(s for s, _, _ in valid) / len(valid) / total_possible * 100
            if current_mean_pct > 0:
                scale_factor = config.target_mean / current_mean_pct
                for score, total, r in valid:
                    sr = r["structured_result"]
                    new_score = min(total, score * scale_factor)
                    sr.total_score = round(new_score, 1)
                    sr.comments += (
                        f"\n[CURVE: linear scale {scale_factor:.2f}x, "
                        f"target mean={config.target_mean}%]"
                    )
                    r["grade_text"] = sr.to_legacy_text()

    elif config.method == "sqrt":
        for score, total, r in valid:
            sr = r["structured_result"]
            if total > 0:
                normalized = score / total
                curved = math.sqrt(max(0.0, normalized))
                sr.total_score = round(curved * total, 1)
                sr.comments += f"\n[CURVE: sqrt, {score:.1f} -> {sr.total_score:.1f}]"
                r["grade_text"] = sr.to_legacy_text()

    return results


def grade_submission(
    pdf_path: str,
    answer_key_path: str,
    enhance: bool = False,
    keep_images: bool = False,
    ensemble_strategy: str = "single",
    strategy: str = "auto",
    use_cache: bool = True,
) -> dict:
    """
    Grade a single student submission.

    Args:
        pdf_path: Path to the student's PDF
        answer_key_path: Path to the answer key JSON
        enhance: Whether to enhance images for better readability
        keep_images: Keep page images after grading (for debugging)
        ensemble_strategy: "single", "vote", "debate", or "cascade"
        use_cache: Whether to use cached results (default True). When False,
            both lookup and store are disabled.

    Returns:
        Dict with student_file, grade_text, answer_key_title, usage,
        and structured_result (GradingResult) when available
    """
    # Load answer key
    answer_key = load_answer_key(answer_key_path)
    rubric_prompt = format_rubric_prompt(answer_key)

    # Checkpoint: check Tier 1 cache (full grading result)
    from grader.checkpoint import get_cache_manager
    from grader.config import get_grading_model, normalize_grading_provider, GRADING_PROVIDER
    cache = get_cache_manager(enabled=use_cache)
    active_provider = normalize_grading_provider(GRADING_PROVIDER)
    grading_settings = {
        "enhance": enhance,
        "strategy": strategy,
        "ensemble_strategy": ensemble_strategy,
        "provider": active_provider,
        "model": get_grading_model(active_provider),
    }
    grading_cache_key = cache.grading_key(pdf_path, answer_key, grading_settings)
    cached_result = cache.get_grading_result(grading_cache_key)
    if cached_result is not None:
        logger.info("Using cached grade for %s (%s)", os.path.basename(pdf_path), grading_cache_key[:8])
        return cached_result

    # RAG: ingest rubric and retrieve any relevant past grading context
    from grader.rag_client import get_rag_client
    rag = get_rag_client()
    if rag:
        rag.ingest_rubric(answer_key)
        past_context = rag.get_relevant_rubric(answer_key.get("title", "rubric"))[:3]
        if past_context:
            rubric_prompt = rag.format_context(past_context) + "\n\n" + rubric_prompt

    # Check page count for adaptive settings
    page_count = get_page_count(pdf_path)
    MAX_PAGES = 100
    if page_count > MAX_PAGES:
        raise ValueError(
            f"PDF has {page_count} pages (max {MAX_PAGES}). "
            "This looks like a combined file — split it first."
        )
    dpi, jpeg_quality, max_long_side = get_adaptive_settings(page_count)

    if page_count > CHUNK_SIZE:
        logger.info("Large submission (%d pages) — adaptive: DPI=%d, quality=%d", page_count, dpi, jpeg_quality)

    # Convert PDF to images with adaptive settings
    output_dir = tempfile.mkdtemp(prefix="exam_grader_")
    survey_dir = None
    try:
        logger.info("Converting PDF: %s", pdf_path)
        pages = convert_pdf(pdf_path, output_dir, dpi=dpi, jpeg_quality=jpeg_quality,
                            max_long_side=max_long_side, auto_enhance=enhance)
        logger.info("  %d pages converted", len(pages))

        # Fix orientation and assess quality for every page (~5ms/page)
        for page in pages:
            detect_and_fix_orientation(page["path"])
            page["quality"] = assess_image(page["path"])

        # Quality-driven enhancement
        if enhance:
            logger.info("Enhancing images...")
            for page in pages:
                if strategy == "auto":
                    s = page["quality"].get("recommended_strategy") or "llm"
                else:
                    s = strategy
                enhance_image(page["path"], s)
        else:
            # Safety net: auto-enhance faint ink even without --enhance
            for page in pages:
                if page["quality"].get("faint_ink"):
                    enhance_image(page["path"], "adaptive")
                    page["auto_enhanced"] = True
                    logger.info("Auto-enhanced faint ink: page %s", page.get("page_number", "?"))

        # Prepare survey images for large submissions
        survey_images = None
        if should_chunk(page_count):
            survey_dir = tempfile.mkdtemp(prefix="exam_grader_survey_")
            logger.info("Generating survey images...")
            survey_images = convert_pdf_survey(pdf_path, survey_dir)

        # Inject handwriting style context into the rubric prompt
        hw_style = classify_handwriting_style(pages[0]["path"]) if pages else "unknown"
        if hw_style != "unknown":
            rubric_prompt = f"[HANDWRITING STYLE: {hw_style} — adjust reading strategy accordingly]\n\n{rubric_prompt}"

        # Cross-session writer profile: load any prior calibration for this
        # student. We only LOAD here (not save) because notebook mode produces
        # section_results, not the single-letter signals that feed the profile —
        # exam-mode grading is what populates it. Loading still helps notebook
        # mode benefit from accumulated exam-mode observations for the same writer.
        student_filename = os.path.basename(pdf_path)
        persistent_profile = load_session_profile(student_filename)
        persistent_block = build_persistent_calibration_block(persistent_profile)
        if persistent_block:
            rubric_prompt = persistent_block + "\n\n" + rubric_prompt
            logger.info(
                "Injected persistent writer calibration for %s (%d prior submission(s))",
                derive_writer_id(student_filename),
                int((persistent_profile or {}).get("submissions_seen", 0)),
            )

        # Use ensemble grader for structured output
        config = EnsembleConfig(strategy=ensemble_strategy)
        grader = EnsembleGrader(config)

        logger.info("Grading with %s...", "ensemble" if ensemble_strategy != "single" else _single_provider_label())
        structured_result = grader.grade_notebook(
            pages, rubric_prompt, answer_key,
            survey_images=survey_images,
            max_long_side=max_long_side,
            jpeg_quality=jpeg_quality,
            student_file=os.path.basename(pdf_path),
        )

        # Generate legacy text for backward compatibility
        grade_text = structured_result.to_legacy_text()

        # Build usage tracker from structured result
        tracker = {
            "input_tokens": structured_result.total_input_tokens,
            "output_tokens": structured_result.total_output_tokens,
            "api_calls": structured_result.total_api_calls,
        }
        print_usage(tracker, os.path.basename(pdf_path))

        if structured_result.needs_human_review:
            logger.warning("NEEDS HUMAN REVIEW: %s", ", ".join(structured_result.low_confidence_items[:3]))

        quality_summary = {
            "pages_assessed": len(pages),
            "poor_quality": sum(1 for p in pages if p.get("quality", {}).get("quality") == "poor"),
            "faint_ink_detected": sum(1 for p in pages if p.get("quality", {}).get("faint_ink")),
            "auto_enhanced": sum(1 for p in pages if p.get("auto_enhanced")),
        }

        result_dict = {
            "student_file": os.path.basename(pdf_path),
            "grade_text": grade_text,
            "answer_key_title": answer_key["title"],
            "images_dir": output_dir if keep_images else None,
            "usage": tracker,
            "structured_result": structured_result,
            "quality_summary": quality_summary,
        }

        # Checkpoint: store results and rubric snapshot to cache
        cache.store_grading_result(grading_cache_key, result_dict)
        cache.store_rubric_snapshot(
            grading_cache_key,
            answer_key,
            rubric_path=answer_key_path,
            student_file=os.path.basename(pdf_path),
            free_response_rubric=None,
        )

        return result_dict
    finally:
        # Always clean up temp directories, even on exception
        if not keep_images:
            cleanup_images(output_dir)
        if survey_dir:
            cleanup_images(survey_dir)


def grade_exam_submission(
    pdf_path: str,
    extracted_key: dict,
    enhance: bool = False,
    keep_images: bool = False,
    free_response_rubric: dict = None,
    enhanced_reading: bool = False,
    answer_page_numbers: list[int] | None = None,
    ensemble_strategy: str = "single",
    strategy: str = "auto",
    use_cache: bool = True,
) -> dict:
    """
    Grade a single student exam submission using instance-based scoring.

    Args:
        pdf_path: Path to the student's PDF
        extracted_key: Pre-extracted answer key dict from extract_answer_key()
        enhance: Whether to enhance images for better readability
        keep_images: Keep page images after grading
        free_response_rubric: Optional free response rubric dict
        enhanced_reading: Use 300 DPI answer cropping for better handwriting accuracy
        answer_page_numbers: 0-indexed pages with answer sheets (default [0])
        ensemble_strategy: "single", "vote", "debate", or "cascade"
        use_cache: Whether to use cached results (default True)

    Returns:
        Dict with student_file, grade_text, total_points, images_dir, usage,
        and structured_result (GradingResult)
    """
    # Validate answer key structure before grading
    if "questions" not in extracted_key or not isinstance(extracted_key["questions"], list):
        raise ValueError("Invalid answer key: missing or invalid 'questions' list")
    if "total_points" not in extracted_key:
        raise ValueError("Invalid answer key: missing 'total_points'")
    # Validate sequential numbering
    q_numbers = [q["number"] for q in extracted_key["questions"]]
    expected = list(range(1, len(q_numbers) + 1))
    if q_numbers != expected:
        raise ValueError(
            f"Answer key questions not sequentially numbered. "
            f"Got {q_numbers}, expected {expected}"
        )
    # Validate point total matches sum
    computed_pts = sum(q.get("points", len(q.get("answers", []))) for q in extracted_key["questions"])
    if computed_pts != extracted_key["total_points"]:
        raise ValueError(
            f"Answer key point mismatch: total_points={extracted_key['total_points']} "
            f"but sum of per-question points={computed_pts}"
        )

    # Checkpoint: check Tier 1 cache (full grading result)
    from grader.checkpoint import get_cache_manager
    from grader.config import get_grading_model, normalize_grading_provider, GRADING_PROVIDER
    cache = get_cache_manager(enabled=use_cache)
    active_provider = normalize_grading_provider(GRADING_PROVIDER)
    grading_settings = {
        "enhance": enhance, "strategy": strategy,
        "ensemble_strategy": ensemble_strategy,
        "enhanced_reading": enhanced_reading,
        "answer_page_numbers": answer_page_numbers,
        "free_response_rubric": free_response_rubric,
        "provider": active_provider,
        "model": get_grading_model(active_provider),
    }
    grading_cache_key = cache.grading_key(pdf_path, extracted_key, grading_settings)
    cached_result = cache.get_grading_result(grading_cache_key)
    if cached_result is not None:
        logger.info("Using cached grade for %s (%s)", os.path.basename(pdf_path), grading_cache_key[:8])
        return cached_result

    grading_prompt = format_exam_grading_prompt(extracted_key, free_response_rubric)

    # RAG: ingest answer key and retrieve relevant context
    from grader.rag_client import get_rag_client
    rag = get_rag_client()
    if rag:
        rag.ingest_rubric(extracted_key)
        key_title = extracted_key.get("title", "exam key")
        past_context = rag.get_relevant_rubric(key_title)[:3]
        if past_context:
            grading_prompt = rag.format_context(past_context) + "\n\n" + grading_prompt

    # Check page count for adaptive settings
    page_count = get_page_count(pdf_path)
    MAX_PAGES = 100
    if page_count > MAX_PAGES:
        raise ValueError(
            f"PDF has {page_count} pages (max {MAX_PAGES}). "
            "This looks like a combined file — split it first."
        )
    dpi, jpeg_quality, max_long_side = get_adaptive_settings(page_count)

    if page_count > CHUNK_SIZE:
        logger.info("Large submission (%d pages) — adaptive: DPI=%d, quality=%d", page_count, dpi, jpeg_quality)

    # Checkpoint: check Tier 2 cache (page images)
    images_cache_key = cache.images_key(pdf_path, dpi, jpeg_quality, max_long_side, enhance, strategy)
    cached_pages = cache.get_page_images(images_cache_key)
    images_from_cache = cached_pages is not None

    if images_from_cache:
        pages = cached_pages
        output_dir = cache.images_dir(images_cache_key)
        logger.info("Using cached page images (%d pages, %s)", len(pages), images_cache_key[:8])
    else:
        output_dir = tempfile.mkdtemp(prefix="exam_grader_")

    try:
        if not images_from_cache:
            logger.info("Converting PDF: %s", pdf_path)
            pages = convert_pdf(pdf_path, output_dir, dpi=dpi, jpeg_quality=jpeg_quality,
                                max_long_side=max_long_side, auto_enhance=enhance)
            logger.info("  %d pages converted", len(pages))

            # Fix orientation and assess quality for every page (~5ms/page)
            for page in pages:
                detect_and_fix_orientation(page["path"])
                page["quality"] = assess_image(page["path"])

        # Auto-enable enhanced reading for poor/fair quality answer sheet pages
        # Must run BEFORE pre-read pipeline so the flag is set in time
        if not enhanced_reading:
            for pg in (answer_page_numbers or [0]):
                if pg < len(pages) and pages[pg].get("quality", {}).get("quality") in ("poor", "fair"):
                    enhanced_reading = True
                    logger.info("Auto-enabling enhanced reading (poor quality on page %d)", pg)
                    break

        # Enhanced reading: crop answer sheet at 300 DPI and pre-read answers
        pre_read_answers = {}
        pre_read_sources = [pdf_path]
        pre_read_sources.extend(find_alternate_scan_pdfs(pdf_path))
        if enhanced_reading:
            answer_pages = answer_page_numbers or [0]
            num_questions = len(extracted_key.get("questions", []))
            logger.info(
                "Enhanced reading: cropping %d answers at 300 DPI from %d scan source(s)...",
                num_questions, len(pre_read_sources),
            )
            per_source_answers = []
            for source_pdf in pre_read_sources:
                source_answers = {}
                for page_num in answer_pages:
                    pre_read = read_answer_sheet(source_pdf, page_num, num_questions)
                    source_answers.update(pre_read)
                if source_answers:
                    per_source_answers.append(source_answers)
            if per_source_answers:
                pre_read_answers = merge_pre_read_answers(*per_source_answers)
            if pre_read_answers:
                grading_prompt = inject_pre_read_answers(grading_prompt, pre_read_answers)
                grading_prompt = build_context_prompt().strip() + "\n\n" + grading_prompt

                anchor_answers = _extract_writer_calibration_answers(pre_read_answers, extracted_key)
                if anchor_answers:
                    profile = extract_writer_characteristics(anchor_answers)
                    writer_calibration = build_writer_calibration_prompt(profile)
                    if writer_calibration:
                        grading_prompt = writer_calibration + "\n\n" + grading_prompt
                        logger.info(
                            "Injected writer calibration from %d anchor pre-reads",
                            len(anchor_answers),
                        )
                logger.info("Injected %d pre-read answers into grading prompt", len(pre_read_answers))

        # Cross-session writer profile: load before grading, save after.
        # This persists per-student style observations across labs so the same
        # writer accumulates calibration over time (MetaWriter pattern).
        student_filename = os.path.basename(pdf_path)
        persistent_profile = load_session_profile(student_filename)
        persistent_block = build_persistent_calibration_block(persistent_profile)
        if persistent_block:
            grading_prompt = persistent_block + "\n\n" + grading_prompt
            logger.info(
                "Injected persistent writer calibration for %s (%d prior submission(s))",
                derive_writer_id(student_filename),
                int((persistent_profile or {}).get("submissions_seen", 0)),
            )

        # Quality-driven enhancement (skip if images came from cache — already applied)
        if not images_from_cache:
            if enhance:
                logger.info("Enhancing images...")
                for page in pages:
                    if strategy == "auto":
                        s = page["quality"].get("recommended_strategy") or "llm"
                    else:
                        s = strategy
                    enhance_image(page["path"], s)
            else:
                for page in pages:
                    if page["quality"].get("faint_ink"):
                        enhance_image(page["path"], "adaptive")
                        page["auto_enhanced"] = True
                        logger.info("Auto-enhanced faint ink: page %s", page.get("page_number", "?"))

        # Inject handwriting style context into the grading prompt
        hw_style = classify_handwriting_style(pages[0]["path"]) if pages else "unknown"
        if hw_style != "unknown":
            grading_prompt = f"[HANDWRITING STYLE: {hw_style} — adjust reading strategy accordingly]\n\n{grading_prompt}"

        # Use ensemble grader for structured output
        fr_points = free_response_rubric["total_points"] if free_response_rubric else 0
        total_points = extracted_key["total_points"] + fr_points

        config = EnsembleConfig(strategy=ensemble_strategy)
        grader = EnsembleGrader(config)

        logger.info("Grading with %s...", "ensemble" if ensemble_strategy != "single" else _single_provider_label())
        structured_result = grader.grade_exam(
            pages, grading_prompt,
            max_long_side=max_long_side,
            jpeg_quality=jpeg_quality,
            student_file=os.path.basename(pdf_path),
            total_points=total_points,
        )

        # Focused reread pass for low-confidence objective items.
        low_conf_objective_qs = [
            q.question_number for q in structured_result.question_results
            if q.confidence == ConfidenceLevel.LOW and q.points_possible > 0
        ]
        if enhanced_reading and low_conf_objective_qs:
            answer_pages = answer_page_numbers or [0]
            reread_candidates = {}
            for source_pdf in pre_read_sources:
                for page_num in answer_pages:
                    reread_answers = reread_questions(
                        source_pdf,
                        page_num,
                        low_conf_objective_qs,
                        output_dir=output_dir,
                        passes=2,
                    )
                    reread_candidates = merge_pre_read_answers(reread_candidates, reread_answers)
            if reread_candidates:
                changed = _apply_targeted_reread_corrections(
                    structured_result,
                    extracted_key,
                    reread_candidates,
                )
                if changed:
                    logger.info(
                        "Focused reread updated low-confidence objective questions: %s",
                        ", ".join(f"Q{q}" for q in sorted(reread_candidates)),
                    )

        # Bias-aligned-error guard: defends against the most common VLM failure
        # mode — confident emission of a memorized canonical answer when image
        # evidence is weak. Runs after grading and after the focused reread so
        # only persistently low-confidence canonical-looking answers are caught.
        from grading_intelligence.bias_guard import (
            apply_bias_guard,
            collect_high_risk_questions,
        )
        high_risk = collect_high_risk_questions(structured_result, extracted_key, pages)
        if high_risk:
            apply_bias_guard(structured_result, extracted_key, high_risk)
            logger.info("Bias-aligned-error guard flagged %d high-risk question(s)", len(high_risk))

        # Persist writer profile across submissions for this student.
        try:
            student_filename = os.path.basename(pdf_path)
            confirmed_letters = _collect_confirmed_letters(structured_result)
            confusion_map = _collect_confusion_resolutions(structured_result, extracted_key)
            update_session_profile(
                student_filename,
                confirmed_letters=confirmed_letters,
                confusion_resolutions=confusion_map,
                formation_style=hw_style if hw_style != "unknown" else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Persistent writer profile update failed: %s", exc)

        # Generate legacy text for backward compatibility
        grade_text = structured_result.to_legacy_text()

        # Build usage tracker from structured result
        tracker = {
            "input_tokens": structured_result.total_input_tokens,
            "output_tokens": structured_result.total_output_tokens,
            "api_calls": structured_result.total_api_calls,
        }
        print_usage(tracker, os.path.basename(pdf_path))

        if structured_result.needs_human_review:
            logger.warning("NEEDS HUMAN REVIEW: %s", ", ".join(structured_result.low_confidence_items[:3]))

        # Run post-grading validators
        from grader.validators import run_all as run_validators
        validation_flags = run_validators(structured_result, {})
        if validation_flags:
            for flag in validation_flags:
                logger.warning("%s", flag)

        quality_summary = {
            "pages_assessed": len(pages),
            "poor_quality": sum(1 for p in pages if p.get("quality", {}).get("quality") == "poor"),
            "faint_ink_detected": sum(1 for p in pages if p.get("quality", {}).get("faint_ink")),
            "auto_enhanced": sum(1 for p in pages if p.get("auto_enhanced")),
        }

        result_dict = {
            "student_file": os.path.basename(pdf_path),
            "pdf_path": pdf_path,
            "grade_text": grade_text,
            "total_points": total_points,
            "images_dir": output_dir if keep_images else None,
            "usage": tracker,
            "structured_result": structured_result,
            "quality_summary": quality_summary,
            "validation_flags": validation_flags,
        }

        # Checkpoint: store results and images to cache
        cache.store_grading_result(grading_cache_key, result_dict)
        cache.store_rubric_snapshot(
            grading_cache_key,
            extracted_key,
            rubric_path="",
            student_file=os.path.basename(pdf_path),
            free_response_rubric=free_response_rubric,
        )
        if not images_from_cache:
            cache.store_page_images(images_cache_key, pages, output_dir)

        return result_dict
    finally:
        if not keep_images and not images_from_cache:
            cleanup_images(output_dir)
