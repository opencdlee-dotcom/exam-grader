"""
Context Expansion for Handwriting Recognition — Week 1 Improvement

Expands answer box crops to include surrounding context:
- Question prompt text above
- Answer margins (left/right)
- Writing style below answer line

Context dramatically improves ambiguous letter disambiguation.
"""

from PIL import Image
from typing import Tuple, Optional


def crop_answer_with_context(
    image: Image.Image,
    answer_bbox: Tuple[int, int, int, int],
    question_region: Optional[Tuple[int, int, int, int]] = None,
    margin_px: int = 50,
    context_above: int = 30,
    context_below: int = 25,
) -> Image.Image:
    """
    Crop answer box with surrounding context for better disambiguation.

    Args:
        image: PIL Image (full page at 300 DPI)
        answer_bbox: (x1, y1, x2, y2) of answer box
        question_region: (qx1, qy1, qx2, qy2) of question text (optional)
        margin_px: pixels to expand left/right of answer box
        context_above: pixels above question text to include
        context_below: pixels below answer line to show writing style

    Returns:
        PIL Image containing answer + context
    """
    x1, y1, x2, y2 = answer_bbox

    # Expand left/right with margins
    crop_x1 = max(0, x1 - margin_px)
    crop_x2 = min(image.width, x2 + margin_px)

    # If question region provided, include it
    if question_region:
        qx1, qy1, qx2, qy2 = question_region
        crop_y1 = max(0, qy1 - context_above)
    else:
        # Otherwise, include space above answer for context
        crop_y1 = max(0, y1 - context_above)

    # Include space below answer to see writing style
    crop_y2 = min(image.height, y2 + context_below)

    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return cropped


def expand_multiple_answers(
    image: Image.Image,
    answer_bboxes: dict,  # {q_num: (x1, y1, x2, y2), ...}
    question_regions: dict = None,  # {q_num: (qx1, qy1, qx2, qy2), ...}
) -> dict:
    """
    Expand multiple answer crops at once (batch operation).

    Args:
        image: Full page image at 300 DPI
        answer_bboxes: {question_number: bbox_tuple, ...}
        question_regions: {question_number: question_bbox, ...}

    Returns:
        {question_number: PIL.Image, ...}
    """
    crops = {}
    question_regions = question_regions or {}

    for q_num, bbox in answer_bboxes.items():
        q_region = question_regions.get(q_num)
        crops[q_num] = crop_answer_with_context(
            image,
            bbox,
            question_region=q_region,
            margin_px=50,
            context_above=30,
            context_below=25,
        )

    return crops


def build_context_prompt() -> str:
    """
    Build instruction text for Claude/Gemini when grading with context.
    Tells the model to use surrounding information for disambiguation.
    """
    return """
=== CONTEXT-AWARE READING ===
The image shows the answer box with surrounding context:
- Question text or option labels above
- Student's response space below
- Margins showing writing style

Use this context to disambiguate similar letters:
- B vs D: look at loop shape, extension below baseline
- O vs Q: look for tail below
- G vs C: look for horizontal bar closure
- M vs N: count peaks (2 = M, 1 = N)
- U vs V: look at bottom shape (rounded = U, pointed = V)

The surrounding context reveals this student's writing habits.
Apply observed patterns consistently.
"""
