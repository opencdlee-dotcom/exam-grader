"""
High-accuracy answer sheet reading using cropped answer regions.

Renders the answer sheet at 300 DPI, crops individual answer regions,
and sends focused images to the active grading provider for character-level reading.
"""

import json
import re
from pathlib import Path
from grader.answer_cropper import crop_answer_sheet, AnswerSheetLayout
from grader.image_optimizer import validate_and_prepare_image
from grader.vision import _grade_chunk, new_usage_tracker, print_usage
from grader.config import CROP_JPEG_QUALITY


READING_SYSTEM_PROMPT = (
    "You are an expert at reading handwritten text from scanned exam answer sheets. "
    "Each image shows one answer line from a student's exam. The question number is visible. "
    "Read the handwritten answer carefully.\n\n"
    "CRITICAL DISAMBIGUATION for single letters:\n"
    "- B vs D: B has TWO bumps/loops stacked on the right. D has ONE smooth curve. Count the loops.\n"
    "- H vs K: H has two verticals connected by a HORIZONTAL bar. K has one vertical with DIAGONAL strokes.\n"
    "- H vs A: H has two full-height PARALLEL verticals. A has two DIAGONALS meeting at a point at top.\n"
    "- B vs R vs N: B has closed bumps/loops on the right. R has one bump + a diagonal leg. "
    "N has a diagonal stroke between two verticals with no loops.\n"
    "- P vs R: P has a closed loop with no leg extending down-right. R has a loop + a leg.\n"
    "- P vs F: P has a closed loop at top-right. F has open horizontal strokes with no loop.\n"
    "- M vs N: M has two humps (or W-shape inverted). N has one diagonal between verticals.\n"
    "- D vs O: D has a flat left side (vertical stroke). O is fully rounded.\n"
    "- O vs Q: O is a fully closed oval. Q has a tail/slash extending from bottom-right.\n"
    "- G vs C: G has an inward horizontal bar at mid-right. C is fully open on the right (no bar).\n"
    "- I vs L: I is a single vertical stroke (may have serifs). L has a vertical + horizontal foot.\n"
    "- I vs J: I is straight. J curves left at the bottom.\n"
    "- E vs F: E has THREE horizontal strokes. F has TWO (missing bottom horizontal).\n"
    "- K vs R: K has diagonal strokes from mid-vertical, no loop. R has a loop at top-right.\n"
    "- G vs Q: G has a horizontal inward stroke. Q has a tail extending outward.\n\n"
    "DUAL-READ PROTOCOL for single-letter answers:\n"
    "For every single-letter answer, report your PRIMARY reading. If you are less than 90% "
    "confident, ALSO report an alternate reading in parentheses: e.g. \"B (or D?)\".\n"
    "Look at the student's OTHER letters on the page to calibrate their handwriting style.\n\n"
    "FAINT INK: If a page or answer region appears faded, washed out, or written in light "
    "pencil, increase your scrutiny. Do NOT default to blank/null — look harder. Adjust your "
    "contrast perception mentally and report what you can see, flagging low confidence.\n\n"
    "Count strokes carefully. Look at whether loops are open or closed. "
    "Track the student's handwriting style across all images for consistency."
)


def build_reading_prompt(num_questions: int) -> str:
    """Build the prompt for reading cropped answer images."""
    return (
        f"You are reading a student's handwritten exam answer sheet. "
        f"There are {num_questions} answer images below, one per question.\n\n"
        "For each question, read the handwritten answer. Answers may be:\n"
        "- A single letter (A, B, C, D for multiple choice)\n"
        "- A single letter from the matching bank (A through Q)\n"
        "- A word or phrase (for fill-in questions)\n"
        "- A number or set of numbers (for temperature questions)\n"
        "- A checkmark or blank (for freebie questions)\n\n"
        "Return ONLY a JSON object mapping question numbers to answers:\n"
        '{"1": "D", "2": "B", "3": "A", ...}\n\n'
        "Rules:\n"
        "- For single letters: report the capital letter. If <90% confident, add alternate: \"B (or D?)\"\n"
        "- For words/phrases, transcribe as written\n"
        "- For blank lines, use null — but ONLY if genuinely blank. Faint writing is NOT blank.\n"
        "- For faint/light pencil answers, try harder before marking null. Report best reading with [faint]\n"
        "- For checkmarks, use \"✓\"\n"
        "- Be precise — do not guess. If truly illegible after maximum effort, use \"[?]\""
    )


def build_crop_image_blocks(crops: dict[int, str]) -> list[dict]:
    """Build API content blocks from cropped answer images."""
    blocks = []
    for q_num in sorted(crops.keys()):
        result = validate_and_prepare_image(
            crops[q_num],
            max_long_side=1568,
            jpeg_quality=CROP_JPEG_QUALITY,
        )
        if result is None:
            continue
        data, media_type = result
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
        blocks.append({
            "type": "text",
            "text": f"[Question {q_num}]",
        })
    return blocks


def reading_strength(answer: str | dict | None) -> float:
    """Map a parsed reading to a comparable evidence score."""
    if answer is None:
        return 0.0
    if isinstance(answer, dict):
        confidence = str(answer.get("confidence", "low")).lower()
        alternate = answer.get("alternate")
        if confidence == "high" and not alternate:
            return 0.95
        if confidence == "illegible":
            return 0.10
        return 0.55 if alternate else 0.65
    return 0.90 if str(answer).strip() else 0.0


def merge_pre_read_answers(*answer_sets: dict[int, str | dict | None]) -> dict[int, str | dict | None]:
    """
    Merge multiple pre-read passes/scans, keeping the strongest evidence per question.

    If equally strong readings disagree, preserve ambiguity as a structured dict so
    downstream code knows not to treat the result as a hard prior.
    """
    merged: dict[int, str | dict | None] = {}
    strengths: dict[int, float] = {}

    for answers in answer_sets:
        for q_num, answer in answers.items():
            strength = reading_strength(answer)
            if q_num not in merged or strength > strengths[q_num]:
                merged[q_num] = answer
                strengths[q_num] = strength
                continue

            if strength == strengths[q_num] and answer != merged[q_num]:
                current = merged[q_num]
                current_primary = current.get("primary") if isinstance(current, dict) else current
                new_primary = answer.get("primary") if isinstance(answer, dict) else answer
                merged[q_num] = {
                    "primary": current_primary,
                    "alternate": new_primary,
                    "confidence": "low",
                }

    return merged


def parse_dual_read(answer: str) -> dict:
    """
    Parse a dual-read answer string into structured form.

    Examples:
        "B" -> {"primary": "B", "alternate": None, "confidence": "high"}
        "B (or D?)" -> {"primary": "B", "alternate": "D", "confidence": "low"}
        "mitosis" -> {"primary": "mitosis", "alternate": None, "confidence": "high"}
        "A [faint]" -> {"primary": "A", "alternate": None, "confidence": "low"}
        "[?]" -> {"primary": None, "alternate": None, "confidence": "illegible"}
    """
    if answer is None:
        return {"primary": None, "alternate": None, "confidence": "illegible"}

    answer = str(answer).strip()

    # Illegible marker
    if answer in ("[?]", "?", "[illegible]", ""):
        return {"primary": None, "alternate": None, "confidence": "illegible"}

    # Dual-read format: "B (or D?)"
    dual_match = re.match(r'^([A-Za-z])\s*\(or\s+([A-Za-z])\??\)', answer)
    if dual_match:
        return {
            "primary": dual_match.group(1).upper(),
            "alternate": dual_match.group(2).upper(),
            "confidence": "low",
        }

    # Faint marker: "A [faint]"
    faint_match = re.match(r'^(.+?)\s*\[faint\]', answer, re.IGNORECASE)
    if faint_match:
        return {
            "primary": faint_match.group(1).strip(),
            "alternate": None,
            "confidence": "low",
        }

    # Normal confident reading
    return {"primary": answer, "alternate": None, "confidence": "high"}


def parse_reading_response(text: str, num_questions: int) -> dict[int, str | dict]:
    """
    Parse the provider JSON response into {question_number: answer_or_structured}.

    Returns raw string answers for simple cases, structured dicts for dual-reads.
    """
    # Extract JSON from response (may be wrapped in markdown code block)
    # Use brace counting to find the outermost matching {} pair
    raw = None
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        raw = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        if raw is not None:
            break
        start = text.find('{', start + 1)

    if raw is None:
        return {}

    # Normalize keys to ints and parse dual-reads
    result = {}
    for k, v in raw.items():
        try:
            q_num = int(k)
        except (ValueError, TypeError):
            continue

        if v is None or (isinstance(v, str) and v.strip() in ("[?]", "?", "")):
            result[q_num] = v
            continue

        # Check if this is a dual-read format
        parsed = parse_dual_read(str(v))
        if parsed["alternate"] is not None or parsed["confidence"] != "high":
            # Store structured dict for downstream answer matching
            result[q_num] = parsed
        else:
            # Simple confident answer — store as string
            result[q_num] = v

    return result


def read_answer_sheet(
    pdf_path: str,
    page_number: int = 0,
    num_questions: int = 23,
    output_dir: str | None = None,
    layout: AnswerSheetLayout | None = None,
    tracker: dict | None = None,
) -> dict[int, str | None]:
    """
    Full pipeline: render, crop, read all answers from an answer sheet page.

    Args:
        pdf_path: Path to the student exam PDF
        page_number: 0-indexed page of the answer sheet
        num_questions: How many questions to read
        output_dir: Where to save crop images (temp dir if None)
        layout: Custom layout or None for default
        tracker: Optional usage tracker

    Returns:
        {question_number: answer_string} for each question
    """
    if tracker is None:
        tracker = new_usage_tracker()

    # Crop all answer regions at 300 DPI
    crops = crop_answer_sheet(pdf_path, page_number, num_questions, output_dir, layout)

    # Build image blocks from crops
    image_blocks = build_crop_image_blocks(crops)
    if not image_blocks:
        return {}

    # Send to the active grading provider
    prompt = build_reading_prompt(num_questions)

    response_text = _grade_chunk(
        None,
        image_blocks,
        prompt,
        system_prompt=READING_SYSTEM_PROMPT,
        max_tokens=2048,
        tracker=tracker,
    )

    answers = parse_reading_response(response_text, num_questions)
    print_usage(tracker, "answer-sheet-reader")

    return answers


def reread_questions(
    pdf_path: str,
    page_number: int,
    question_numbers: list[int],
    output_dir: str | None = None,
    tracker: dict | None = None,
    passes: int = 2,
) -> dict[int, str | dict | None]:
    """
    Perform focused rereads for a subset of questions and merge the strongest result.
    """
    if tracker is None:
        tracker = new_usage_tracker()
    if not question_numbers:
        return {}

    question_numbers = sorted(set(question_numbers))
    pass_results: list[dict[int, str | dict | None]] = []
    tmp_dir = output_dir or str(Path(output_dir or ".").resolve())

    for idx in range(max(1, passes)):
        pass_dir = output_dir
        if output_dir:
            pass_dir = str(Path(output_dir) / f"reread_pass_{idx + 1}")
        answers = read_answer_sheet(
            pdf_path=pdf_path,
            page_number=page_number,
            num_questions=max(question_numbers),
            output_dir=pass_dir or tmp_dir,
            tracker=tracker,
        )
        pass_results.append({q: answers.get(q) for q in question_numbers if q in answers})

    return merge_pre_read_answers(*pass_results)


def find_alternate_scan_pdfs(pdf_path: str) -> list[str]:
    """
    Find sibling alternate scan PDFs, preferring clearer scanner-style uploads.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        return []

    alternates = []
    for candidate in pdf.parent.rglob("*.pdf"):
        if candidate.resolve() == pdf.resolve():
            continue
        name = candidate.name.lower()
        if any(token in name for token in ("camscanner", "scan", "scanner", "clear")):
            alternates.append(str(candidate))
    return sorted(dict.fromkeys(alternates))
