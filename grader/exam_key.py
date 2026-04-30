"""
Answer key extraction from scanned PDF.
Sends the instructor's filled-out answer sheet to the active grading provider,
returns structured question→answers mapping with per-question point values.
"""

import json
import logging
import re
import tempfile

from grader.pdf_reader import convert_pdf, enhance_image, cleanup_images
from grader.vision import _encode_image, _grade_chunk, _get_provider
from grader.config import get_grading_model

logger = logging.getLogger(__name__)


def _build_extraction_prompt() -> str:
    """Build the prompt that instructs the active provider to extract answers from the key PDF."""
    return """You are reading an instructor's filled-out answer sheet / answer key.
Each question has one or more expected answer terms written by the instructor.

Extract ALL questions and their expected answers. Return ONLY valid JSON in this exact format:
{
  "questions": [
    {
      "number": 1,
      "answers": ["term1", "term2", "term3"],
      "alternatives": {
        "term1": ["synonym1", "common_variant1"],
        "term2": ["abbreviation1"]
      }
    },
    {"number": 2, "answers": ["single term"], "alternatives": {}},
    {"number": 3, "answers": ["answer A", "answer B"], "alternatives": {}}
  ]
}

Rules:
- Each distinct term, word, or phrase the instructor wrote for a question is one answer entry
- If a question has multiple blanks or multiple expected terms, list each separately
- Preserve the instructor's wording exactly as written
- If a question appears blank/unanswered on the key, include it with an empty answers array
- Number questions sequentially as they appear on the sheet
- Include ALL questions, even single-word answers
- For multiple-choice questions, include BOTH the letter (A, B, C, D) AND the full text
  of the correct answer option. Students may write just the letter, just the word/phrase,
  or both — all should be accepted.
  Example: if answer is "B) mitosis", answers should be ["mitosis"] with alternatives
  {"mitosis": ["B", "b"]}
- For each answer term, include an "alternatives" dict mapping the term to commonly
  accepted equivalent expressions that a student might write instead (synonyms,
  abbreviations, chemical names vs formulas, common student phrasings, AND the
  corresponding letter choice if it's a multiple-choice question).
  If no obvious alternatives exist, use an empty dict {}
- Return ONLY the JSON object, no other text"""


def _parse_extraction_response(response_text: str) -> dict:
    """
    Parse the provider response into a structured answer key dict.
    Handles markdown code fences and extra text around JSON.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try to find JSON object in the text
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse answer key extraction. Provider returned:\n"
            f"{response_text[:500]}\n\nJSON error: {e}"
        )

    # Validate structure
    if "questions" not in data:
        raise ValueError("Extracted answer key missing 'questions' field")

    if not isinstance(data["questions"], list):
        raise ValueError("'questions' must be a list")

    total_points = 0
    for q in data["questions"]:
        if "number" not in q or "answers" not in q:
            raise ValueError(f"Each question needs 'number' and 'answers'. Got: {q}")
        if not isinstance(q["answers"], list):
            raise ValueError(f"Q{q['number']} 'answers' must be a list")
        q["points"] = len(q["answers"])
        total_points += q["points"]
        # Ensure alternatives field exists (optional from Claude)
        if "alternatives" not in q:
            q["alternatives"] = {}

    data["total_points"] = total_points

    # Validate sequential question numbering
    q_numbers = [q["number"] for q in data["questions"]]
    expected = list(range(1, len(q_numbers) + 1))
    if q_numbers != expected:
        # Try sorting first — Claude might return them out of order
        data["questions"].sort(key=lambda q: q["number"])
        q_numbers = [q["number"] for q in data["questions"]]
        if q_numbers != expected:
            missing = set(expected) - set(q_numbers)
            extra = set(q_numbers) - set(expected)
            raise ValueError(
                f"Answer key questions not sequentially numbered 1-{len(expected)}. "
                f"Missing: {sorted(missing) if missing else 'none'}, "
                f"Unexpected: {sorted(extra) if extra else 'none'}"
            )

    return data


def extract_answer_key(pdf_path: str, enhance: bool = False, use_cache: bool = True) -> dict:
    """
    Extract structured answer key from an instructor's filled-out PDF.

    Args:
        pdf_path: Path to the answer key PDF
        enhance: Whether to enhance images for better readability
        use_cache: Whether to use cached results (default True)

    Returns:
        Dict with 'questions' list and 'total_points'
    """
    from grader.config import MAX_IMAGE_LONG_SIDE
    from grader.checkpoint import get_cache_manager

    cache = get_cache_manager(enabled=use_cache)
    cache_key = cache.answer_key_key(pdf_path, enhance)
    cached = cache.get_answer_key(cache_key)
    if cached is not None:
        logger.info("Using cached answer key (%s)", cache_key[:8])
        logger.info("  %d questions, %d total points", len(cached["questions"]), cached["total_points"])
        return cached

    output_dir = tempfile.mkdtemp(prefix="exam_key_")
    logger.info("Extracting answer key from: %s", pdf_path)
    pages = convert_pdf(pdf_path, output_dir, max_long_side=MAX_IMAGE_LONG_SIDE, auto_enhance=enhance)
    logger.info("  %d pages converted", len(pages))

    if enhance:
        for page in pages:
            enhance_image(page["path"], "medium")

    # Build image blocks for the active provider
    image_blocks = []
    for page in pages:
        b64_data, media_type = _encode_image(page["path"])
        image_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64_data},
        })
    provider = _get_provider()
    logger.info(
        "Reading answer key with %s (%s)...",
        provider,
        get_grading_model(provider),
    )
    try:
        raw_text = _grade_chunk(
            None,
            image_blocks,
            _build_extraction_prompt(),
            system_prompt="You extract structured answer keys from instructor answer sheets.",
            max_tokens=4096,
            retries=5,
            provider=provider,
        )
    finally:
        cleanup_images(output_dir)

    result = _parse_extraction_response(raw_text)
    cache.store_answer_key(cache_key, result, source_pdf=pdf_path)
    logger.info("Extracted %d questions, %d total points", len(result["questions"]), result["total_points"])

    # Cross-validation: log summary for instructor verification
    logger.info("--- Answer Key Summary (verify before grading) ---")
    for q in result["questions"]:
        ans_str = ", ".join(q["answers"]) if q["answers"] else "(blank)"
        logger.info("  Q%d (%d pt%s): %s", q["number"], q["points"], "s" if q["points"] != 1 else "", ans_str)
    logger.info("--- Total: %d points across %d questions ---", result["total_points"], len(result["questions"]))

    return result


def print_extracted_key(extracted_key: dict) -> None:
    """Log the extracted answer key for instructor verification."""
    logger.info("=" * 50)
    logger.info("EXTRACTED ANSWER KEY — %d total points", extracted_key["total_points"])
    logger.info("=" * 50)
    for q in extracted_key["questions"]:
        answers_str = ", ".join(q["answers"]) if q["answers"] else "(blank)"
        logger.info("  Q%d (%d pt%s): %s", q["number"], q["points"], "s" if q["points"] != 1 else "", answers_str)
    logger.info("=" * 50)
