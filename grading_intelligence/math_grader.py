"""
Math Step-by-Step Grading Engine.

Decomposes student math work into numbered steps, verifies each against
the expected solution, classifies errors (arithmetic, algebraic, unit,
conceptual, propagated), and applies the single-penalty rule so that
downstream errors caused by one mistake don't snowball the deduction.

Supports: stoichiometry, molarity, dilution, pH, Beer-Lambert,
unit conversions, and general algebra.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from grading_intelligence.llm_executor import execute_prompt, _extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MathStep(BaseModel):
    """A single step in a student's mathematical work."""

    step_number: int
    expression: str  # What the student wrote
    expected: str = ""  # What was expected
    is_correct: bool = True
    error_type: str | None = None  # "arithmetic", "algebraic", "unit", "conceptual", "propagated"
    partial_credit: float = 1.0  # 0.0–1.0 credit fraction for this step
    explanation: str = ""


class MathGradingResult(BaseModel):
    """Complete result of step-by-step math grading."""

    total_steps: int
    correct_steps: int
    first_error_step: int | None = None
    propagated_after: int = 0  # Steps wrong only due to propagation from first error
    methodology_correct: bool = True  # Right approach, possibly wrong arithmetic
    answer_correct: bool = False
    points_earned: float
    points_possible: float
    steps: list[MathStep] = Field(default_factory=list)
    feedback: str = ""


# ---------------------------------------------------------------------------
# Step-grading dataclasses (new — rubric-driven, per-step point allocation)
# ---------------------------------------------------------------------------


@dataclass
class RubricMathStep:
    """A single step extracted from a student's handwritten math work."""

    step_num: int
    expression: str           # e.g. "5g NaCl × (1 mol / 58.4g)"
    intermediate_result: str  # e.g. "= 0.0856 mol"
    is_correct: bool
    error_type: str           # "wrong_formula" | "arithmetic" | "units" | "none"
    carried_forward: bool     # True if this step used a wrong result from a prior step
    points_awarded: float


@dataclass
class RubricMathGradingResult:
    """Complete result of rubric-driven step-by-step math grading."""

    steps: list[RubricMathStep]
    total_points: float
    max_points: float
    has_carried_forward_errors: bool
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

MATH_STEP_GRADING_PROMPT = """\
You are an expert science/math grading assistant. Your job is to grade a
student's multi-step calculation by decomposing their work into discrete
steps and verifying each one.

=== QUESTION ===
{question}

=== EXPECTED SOLUTION (reference) ===
{expected_solution}

=== STUDENT'S WORK ===
{student_work}

=== TOTAL POINTS ===
{points}

=== GRADING INSTRUCTIONS ===

1. **Decompose** the student's work into numbered steps (each formula
   substitution, arithmetic operation, or unit conversion is one step).
2. **For each step**, compare against the expected solution:
   - If correct, mark is_correct = true, partial_credit = 1.0.
   - If wrong, classify the error:
     * "arithmetic" — correct setup but computation mistake (e.g. 2×3 = 7)
     * "algebraic" — incorrect rearrangement or simplification
     * "unit" — wrong or missing units, dimensional error
     * "conceptual" — wrong formula, wrong approach, misunderstands the science
     * "propagated" — this step would be correct IF the earlier error value
       were used; the student carried forward a prior mistake consistently.
3. **Single-penalty rule**: propagated errors get partial_credit = 0.75
   (they show correct methodology despite the wrong input). Only the
   *original* error gets full deduction. Mark error_type = "propagated".
4. **Methodology**: set methodology_correct = true if the student chose
   the right formulas/approach even if arithmetic went wrong.
5. **Points allocation**: distribute {points} points across steps
   proportionally. Sum partial_credit fractions × per-step points to get
   points_earned. Round to 2 decimal places.

=== SUPPORTED PROBLEM TYPES ===
Stoichiometry, molarity/dilution (M1V1=M2V2), pH/pOH, Beer-Lambert
(A=εbc), unit conversions, percent composition, limiting reagent,
general algebra.

=== OUTPUT FORMAT ===
Respond with ONLY valid JSON matching this schema (no markdown fences):

{{
  "total_steps": <int>,
  "correct_steps": <int>,
  "first_error_step": <int or null>,
  "propagated_after": <int>,
  "methodology_correct": <bool>,
  "answer_correct": <bool>,
  "points_earned": <float>,
  "points_possible": {points},
  "steps": [
    {{
      "step_number": <int>,
      "expression": "what the student wrote",
      "expected": "what was expected",
      "is_correct": <bool>,
      "error_type": null or "arithmetic" or "algebraic" or "unit" or "conceptual" or "propagated",
      "partial_credit": <float 0.0-1.0>,
      "explanation": "brief note"
    }}
  ],
  "feedback": "2-3 sentence summary of performance and key errors"
}}
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_math_grading_prompt(
    question: str,
    expected_solution: str,
    student_work: str,
    points: float,
) -> str:
    """Build the full prompt for step-by-step math grading."""
    return MATH_STEP_GRADING_PROMPT.format(
        question=question,
        expected_solution=expected_solution,
        student_work=student_work,
        points=points,
    )


def parse_math_grading_response(raw_json: str) -> MathGradingResult | None:
    """Parse raw LLM JSON into a MathGradingResult.

    Returns None if parsing fails.
    """
    data = _extract_json(raw_json)
    if data is None:
        return None

    try:
        steps = [MathStep(**s) for s in data.get("steps", [])]

        # Recompute propagated_after from steps for consistency
        first_error = data.get("first_error_step")
        propagated_count = sum(
            1 for s in steps
            if s.error_type == "propagated"
        )

        return MathGradingResult(
            total_steps=data.get("total_steps", len(steps)),
            correct_steps=data.get("correct_steps", sum(1 for s in steps if s.is_correct)),
            first_error_step=first_error,
            propagated_after=propagated_count,
            methodology_correct=data.get("methodology_correct", True),
            answer_correct=data.get("answer_correct", False),
            points_earned=float(data.get("points_earned", 0)),
            points_possible=float(data.get("points_possible", 0)),
            steps=steps,
            feedback=data.get("feedback", ""),
        )
    except (TypeError, ValueError, KeyError):
        return None


def grade_math_problem(
    question: str,
    expected_solution: str,
    student_work: str,
    points: float,
    image_blocks: list[dict] | None = None,
    provider: str = "claude",
) -> MathGradingResult | None:
    """Grade a multi-step math problem with step-by-step analysis.

    Sends the question, expected solution, and student work to an LLM,
    then parses the structured JSON response into a MathGradingResult.

    Args:
        question: The problem statement.
        expected_solution: Reference solution with steps.
        student_work: What the student wrote (transcribed or OCR'd).
        points: Total points for this problem.
        image_blocks: Optional image content blocks (for handwritten work).
        provider: LLM provider to use.

    Returns:
        MathGradingResult or None if the LLM call or parsing fails.
    """
    prompt = build_math_grading_prompt(question, expected_solution, student_work, points)

    system = (
        "You are a precise math/science grading engine. "
        "Return ONLY valid JSON. No explanation outside the JSON."
    )

    raw = execute_prompt(
        prompt,
        system_prompt=system,
        image_blocks=image_blocks,
        max_tokens=4096,
        temperature=0,
        provider=provider,
    )

    return parse_math_grading_response(raw)


# ---------------------------------------------------------------------------
# Rubric-driven step grading (new API)
# ---------------------------------------------------------------------------

_STEP_GRADING_PROMPT_TEMPLATE = """\
You are an expert science/math grading assistant. Grade a student's handwritten \
calculation step-by-step against the provided rubric.

=== QUESTION ===
{question}

=== STUDENT'S ANSWER (transcribed) ===
{student_answer}

=== RUBRIC STEPS ===
{rubric_steps_text}

=== TOTAL POINTS ===
{max_points}

=== GRADING INSTRUCTIONS ===
1. Extract each mathematical step the student wrote (formula setup, substitution, \
arithmetic, unit conversion, final answer).
2. For each step, determine:
   - expression: what the student wrote for this step
   - result: the intermediate value they calculated
   - correct: true if formula choice AND arithmetic AND units are all right
   - error_type: one of "wrong_formula", "arithmetic", "units", "none"
   - carried_forward: true ONLY if (a) this step is mathematically correct given \
the prior wrong result AND (b) a prior step had an error — meaning the student \
applied the right logic but carried a wrong number from before
   - points: points awarded for this step per rubric (carried-forward steps still \
earn their points — the error penalty was already applied at the source step)
3. Return ONLY valid JSON (no markdown fences):

{{
  "steps": [
    {{
      "num": <int>,
      "expression": "<what student wrote>",
      "result": "<intermediate value or final answer>",
      "correct": <true|false>,
      "error_type": "<wrong_formula|arithmetic|units|none>",
      "carried_forward": <true|false>,
      "points": <float>
    }}
  ],
  "total": <float>,
  "reasoning": "<2-3 sentence summary>"
}}
"""


def build_step_grading_prompt(
    question: str,
    student_answer: str,
    rubric_steps: list[dict],
    max_points: float,
) -> str:
    """Build a prompt instructing the LLM to grade each math step individually.

    Args:
        question: The problem statement.
        student_answer: Transcription of the student's handwritten work.
        rubric_steps: List of dicts, each with at minimum a "description" key
            and optionally a "points" key.
        max_points: Total points available for this problem.

    Returns:
        Formatted prompt string ready to send to an LLM.
    """
    lines = []
    for i, step in enumerate(rubric_steps, start=1):
        desc = step.get("description", f"Step {i}")
        pts = step.get("points", "")
        pts_str = f" ({pts} pts)" if pts != "" else ""
        lines.append(f"Step {i}{pts_str}: {desc}")
    rubric_steps_text = "\n".join(lines) if lines else "(No rubric steps provided)"

    return _STEP_GRADING_PROMPT_TEMPLATE.format(
        question=question,
        student_answer=student_answer,
        rubric_steps_text=rubric_steps_text,
        max_points=max_points,
    )


def _parse_rubric_step(raw: dict, idx: int) -> RubricMathStep:
    """Convert a raw JSON step dict from the LLM into a RubricMathStep."""
    return RubricMathStep(
        step_num=int(raw.get("num", idx)),
        expression=str(raw.get("expression", "")),
        intermediate_result=str(raw.get("result", "")),
        is_correct=bool(raw.get("correct", False)),
        error_type=str(raw.get("error_type", "none")),
        carried_forward=bool(raw.get("carried_forward", False)),
        points_awarded=float(raw.get("points", 0.0)),
    )


def detect_error_carried_forward(steps: list[RubricMathStep]) -> list[RubricMathStep]:
    """Post-process steps to identify and tag carried-forward errors.

    If step N is incorrect and step N+1 applies the correct subsequent
    operation to step N's wrong result, mark step N+1 as carried_forward=True
    so it retains its points.  The LLM sets this flag initially; this function
    enforces the rule defensively for any step whose carried_forward flag may
    have been missed.

    Rules applied:
    - A step can only be carried_forward if a previous step has is_correct=False
      and error_type != "none".
    - A step that itself has a *new* error (wrong_formula) is NOT carried_forward,
      even if a prior error exists.

    Returns the (possibly modified) list in place.
    """
    any_prior_error = False
    for step in steps:
        if any_prior_error:
            # A step with a brand-new conceptual/formula error is not just
            # carrying forward — it introduced a new mistake.
            if step.is_correct is False and step.error_type == "wrong_formula":
                step.carried_forward = False
            elif step.is_correct is False and step.error_type in ("arithmetic", "units"):
                # Independent arithmetic error on top of a prior error —
                # do NOT mark as carried_forward.
                step.carried_forward = False
            elif step.is_correct is False and step.error_type == "none":
                # LLM said not correct but no error type — treat conservatively.
                pass
            # If LLM already set carried_forward, honour it (step is logically
            # correct given prior wrong input).
        else:
            # No prior error yet — carried_forward cannot legitimately be True.
            if step.carried_forward:
                step.carried_forward = False

        if step.is_correct is False and step.error_type not in ("none", ""):
            any_prior_error = True

    return steps


def grade_math_submission(
    question: str,
    student_answer: str,
    rubric_steps: list[dict],
    max_points: float,
    llm_fn: Callable[[str], str],
) -> RubricMathGradingResult:
    """Grade a student's math submission using rubric-driven step analysis.

    Args:
        question: The problem statement.
        student_answer: Transcription of the student's handwritten work.
        rubric_steps: Rubric step dicts with "description" and optional "points".
        max_points: Total points available.
        llm_fn: Callable that accepts a prompt string and returns the LLM's
            raw text response (typically containing JSON).

    Returns:
        RubricMathGradingResult with per-step scores and error-carried-forward
        detection applied.
    """
    prompt = build_step_grading_prompt(question, student_answer, rubric_steps, max_points)
    raw = llm_fn(prompt)

    data = _extract_json(raw)
    if data is None:
        logger.warning("grade_math_submission: failed to parse LLM JSON response")
        return RubricMathGradingResult(
            steps=[],
            total_points=0.0,
            max_points=max_points,
            has_carried_forward_errors=False,
            reasoning="[Parse error — LLM did not return valid JSON]",
        )

    raw_steps = data.get("steps", [])
    steps = [_parse_rubric_step(s, i + 1) for i, s in enumerate(raw_steps)]
    steps = detect_error_carried_forward(steps)

    total = round(float(data.get("total", sum(s.points_awarded for s in steps))), 2)
    total = min(total, max_points)

    has_cf = any(s.carried_forward for s in steps)

    return RubricMathGradingResult(
        steps=steps,
        total_points=total,
        max_points=max_points,
        has_carried_forward_errors=has_cf,
        reasoning=str(data.get("reasoning", "")),
    )
