"""
Prompt Library — Centralized prompt management for grading intelligence.

Migrates and upgrades prompts from exam_prompt.py and rubric.py.
Adds structured JSON output instructions for machine-readable results.
"""

# JSON output format that Claude should return for structured grading
STRUCTURED_EXAM_OUTPUT_SCHEMA = """\
=== OUTPUT FORMAT ===
You MUST respond with valid JSON in EXACTLY this structure (no markdown fencing, no extra text):

{
  "student_name": "full name as read from exam",
  "name_confidence": "HIGH or LOW",
  "shift_detected": true/false,
  "shift_details": "description or null",
  "total_score": <number>,
  "total_possible": <number>,
  "questions": [
    {
      "question_number": <int>,
      "raw_reading": "exactly what you see written, before any interpretation",
      "alternative_readings": ["other plausible readings of the handwriting"],
      "reading_confidence": "HIGH or LOW (how confident you are in the raw reading)",
      "student_answer": "final interpretation after applying tolerance/synonym rules",
      "points_earned": <number>,
      "points_possible": <number>,
      "confidence": "HIGH or LOW (overall scoring confidence)",
      "correct_terms": ["term1", "term2"],
      "missing_terms": ["term3"],
      "wrong_terms": [],
      "reasoning": "brief explanation of scoring decision",
      "illegible": true/false,
      "faint_content": true/false,
      "answer_hunted": true/false
    }
  ],
  "free_response": [
    {
      "question_number": <int>,
      "raw_reading": "verbatim transcription of student's written response",
      "reading_confidence": "HIGH or LOW",
      "points_earned": <number>,
      "points_possible": <number>,
      "confidence": "HIGH or LOW",
      "rubric_level_matched": "which rubric level",
      "reasoning": "full deduction explanation"
    }
  ],
  "comments": "brief feedback",
  "scan_format_notes": "any scan quality issues or null"
}

CRITICAL: Return ONLY valid JSON. No markdown code fences. No text before or after the JSON.
"""

STRUCTURED_NOTEBOOK_OUTPUT_SCHEMA = """\
=== OUTPUT FORMAT ===
You MUST respond with valid JSON in EXACTLY this structure (no markdown fencing, no extra text):

{
  "total_score": <number>,
  "total_possible": <number>,
  "sections": [
    {
      "section_name": "Section Name",
      "points_earned": <number>,
      "points_possible": <number>,
      "confidence": "HIGH" or "LOW",
      "deductions": [
        {
          "points": <number>,
          "reason": "specific reason",
          "criterion": "which criterion",
          "propagated": true/false
        }
      ],
      "reasoning": "overall assessment of section"
    }
  ],
  "comments": "brief feedback starting with positives",
  "scan_format_notes": "any scan quality issues or null"
}

CRITICAL: Return ONLY valid JSON. No markdown code fences. No text before or after the JSON.
"""


def get_structured_output_instructions(mode: str = "exam") -> str:
    """Get the appropriate structured output schema for the grading mode."""
    if mode == "exam":
        return STRUCTURED_EXAM_OUTPUT_SCHEMA
    return STRUCTURED_NOTEBOOK_OUTPUT_SCHEMA


def wrap_prompt_for_structured_output(original_prompt: str, mode: str = "exam") -> str:
    """
    Replace the legacy OUTPUT FORMAT section with structured JSON instructions.

    Keeps all the tolerance rules, scoring rules, and rubric content intact.
    Only swaps out the output format section at the end.
    """
    # Find and replace the OUTPUT FORMAT section
    marker = "=== OUTPUT FORMAT ==="
    if marker in original_prompt:
        # Everything before OUTPUT FORMAT
        before = original_prompt[:original_prompt.index(marker)]
        return before + get_structured_output_instructions(mode)

    # No marker found — append structured instructions
    return original_prompt + "\n\n" + get_structured_output_instructions(mode)
