"""
Answer Space Priors for Handwriting Recognition — Week 4 Improvement

Includes surrounding answer space (margins, spacing, alignment cues) in the
reading context. Student spacing, alignment, and margins provide visual priors
for letter disambiguation:

- Wide spacing between strokes → likely different letters, not ligature
- Right-aligned vs left-aligned → affects B/D reading (D often right-aligned)
- Vertical alignment → helps with O/Q (Q's tail often drops below line)
- Margin usage → indicates confidence level and hand control
"""

from PIL import Image
from typing import Tuple, Optional


def expand_answer_crop_with_spacing(
    image: Image.Image,
    answer_bbox: Tuple[int, int, int, int],
    margin_px: int = 75,
    context_above: int = 40,
    context_below: int = 40,
) -> Image.Image:
    """
    Expand answer crop to include full answer space context:
    - Left/right margins showing spacing and alignment
    - Space above for question reference
    - Space below showing writing style and baseline

    Args:
        image: PIL Image (full page at 300 DPI)
        answer_bbox: (x1, y1, x2, y2) of answer box
        margin_px: pixels to expand left/right (increased from 50 to 75)
        context_above: pixels above answer for question text (increased from 30 to 40)
        context_below: pixels below answer to show baseline and style (increased from 25 to 40)

    Returns:
        PIL Image with expanded spacing context
    """
    x1, y1, x2, y2 = answer_bbox

    # Expand with larger margins to show student's spacing choices
    crop_x1 = max(0, x1 - margin_px)
    crop_x2 = min(image.width, x2 + margin_px)

    # Include space above for question reference
    crop_y1 = max(0, y1 - context_above)

    # Include space below to show writing baseline and style
    crop_y2 = min(image.height, y2 + context_below)

    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return cropped


def build_spacing_context_prompt() -> str:
    """
    Build instruction text for Claude/Gemini when using spacing context.
    Tells the model to use margins and alignment as reading priors.
    """
    return """
=== ANSWER SPACE PRIORS ===
The image includes surrounding margins and spacing that provide reading context:

**Spacing Cues**:
- Wide gaps between strokes → likely separate letters, not connected loops
- Strokes that touch or overlap → likely same letter or ligature
- Letter extends into left/right margin → incomplete letter or pen pressure start/stop
- Consistent baseline alignment → high confidence in reading

**Alignment Cues**:
- Right-aligned answer (rightmost stroke near right margin) → often D, not B (D tends right)
- Left-aligned answer (leftmost stroke near left margin) → often B, not D (B is solid)
- Centered answer → letter is intentional, not margin artifact

**Confidence Indicators**:
- Clean spacing, centered answer, consistent baseline → HIGH confidence
- Crowded or offset spacing, unclear margins → LOWER confidence
- Answer that bleeds into margin → possible handwriting error or forced fit

Use these spatial cues to disambiguate similar letters (B/D, O/Q, etc.)
and adjust confidence accordingly.
"""


def build_spacing_analysis(
    answer_bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> dict:
    """
    Analyze the spacing of an answer box within its containing image.
    Returns metrics about margins, alignment, and spacing context.

    Args:
        answer_bbox: (x1, y1, x2, y2) of the answer box
        image_width: Width of the full image
        image_height: Height of the full image

    Returns:
        {
            'left_margin': pixels,
            'right_margin': pixels,
            'top_margin': pixels,
            'bottom_margin': pixels,
            'horizontal_alignment': 'left'|'center'|'right',
            'vertical_alignment': 'top'|'center'|'bottom',
            'width': answer_width,
            'height': answer_height,
        }
    """
    x1, y1, x2, y2 = answer_bbox
    width = x2 - x1
    height = y2 - y1

    left_margin = x1
    right_margin = image_width - x2
    top_margin = y1
    bottom_margin = image_height - y2

    # Determine horizontal alignment
    if left_margin < right_margin * 0.7:
        h_align = "left"
    elif right_margin < left_margin * 0.7:
        h_align = "right"
    else:
        h_align = "center"

    # Determine vertical alignment
    if top_margin < bottom_margin * 0.7:
        v_align = "top"
    elif bottom_margin < top_margin * 0.7:
        v_align = "bottom"
    else:
        v_align = "center"

    return {
        "left_margin": left_margin,
        "right_margin": right_margin,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "horizontal_alignment": h_align,
        "vertical_alignment": v_align,
        "width": width,
        "height": height,
    }


def apply_spacing_context_to_prompt(base_prompt: str) -> str:
    """
    Inject spacing context guidance into the grading prompt.

    Args:
        base_prompt: Original grading prompt

    Returns:
        Modified prompt with spacing context instructions
    """
    spacing_text = build_spacing_context_prompt()

    # Find insertion point — right before answer key so spacing context informs reading
    insertion_point = "=== ANSWER KEY ==="
    if insertion_point in base_prompt:
        return base_prompt.replace(
            insertion_point,
            spacing_text + "\n\n" + insertion_point
        )

    # Fallback: append before OUTPUT FORMAT
    insertion_point = "=== OUTPUT FORMAT ==="
    if insertion_point in base_prompt:
        return base_prompt.replace(
            insertion_point,
            spacing_text + "\n\n" + insertion_point
        )

    # Last resort: append at end
    return base_prompt + "\n\n" + spacing_text
