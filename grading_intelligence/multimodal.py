"""
Multi-modal grading engine for heterogeneous page content.

Segments exam pages into typed regions (text, diagram, table, graph, code,
chemical structure) and routes each region to a specialized grading pipeline.
All inputs and outputs are typed Pydantic models, and every LLM interaction
uses structured JSON prompts with deterministic parse functions.

Phase 5 -- Multi-Modal Grading.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RegionType(str, Enum):
    """Content types that can appear on a scanned exam page."""

    TEXT = "TEXT"
    DIAGRAM = "DIAGRAM"
    TABLE = "TABLE"
    GRAPH = "GRAPH"
    CODE = "CODE"
    CHEMICAL = "CHEMICAL"


# ---------------------------------------------------------------------------
# Page segmentation models
# ---------------------------------------------------------------------------


class PageRegion(BaseModel):
    """A single detected region on an exam page."""

    region_type: RegionType = Field(
        ..., description="Semantic type of this page region."
    )
    bounding_box: tuple[float, float, float, float] = Field(
        ...,
        description=(
            "Normalised bounding box as (x_min, y_min, x_max, y_max) "
            "where values are in [0, 1] relative to page dimensions."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence that the region classification is correct.",
    )
    content_description: str = Field(
        ...,
        description="Brief plain-text description of what this region contains.",
    )


class SegmentationResult(BaseModel):
    """Full segmentation output for a single page."""

    page_number: int = Field(..., ge=1, description="1-based page index.")
    regions: list[PageRegion] = Field(
        default_factory=list,
        description="All detected regions on the page.",
    )


# ---------------------------------------------------------------------------
# Diagram grading models
# ---------------------------------------------------------------------------


class DiagramGradingResult(BaseModel):
    """Result of comparing a student diagram to a reference."""

    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall similarity between student and reference diagrams.",
    )
    correct_elements: list[str] = Field(
        default_factory=list,
        description="Diagram elements the student drew correctly.",
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Required elements absent from the student diagram.",
    )
    extra_elements: list[str] = Field(
        default_factory=list,
        description="Elements present in the student diagram but not in the reference.",
    )
    feedback: str = Field(
        ...,
        description="Human-readable feedback on the diagram.",
    )


# ---------------------------------------------------------------------------
# Table grading models
# ---------------------------------------------------------------------------


class TableCellError(BaseModel):
    """A single incorrect cell in a student data table."""

    row: int = Field(..., description="0-based row index of the error.")
    col: int = Field(..., description="0-based column index of the error.")
    expected: str = Field(..., description="Expected cell value.")
    actual: str = Field(..., description="Value the student wrote.")


class TableGradingResult(BaseModel):
    """Result of evaluating a student data table against expected values."""

    correct_cells: int = Field(
        ..., ge=0, description="Number of cells that match the answer key."
    )
    total_cells: int = Field(
        ..., ge=0, description="Total number of evaluated cells."
    )
    errors: list[TableCellError] = Field(
        default_factory=list,
        description="List of cell-level errors with positions and values.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of correct cells (correct_cells / total_cells).",
    )


# ---------------------------------------------------------------------------
# Code grading models
# ---------------------------------------------------------------------------


class CodeGradingResult(BaseModel):
    """Result of grading handwritten pseudocode or code."""

    logic_correct: bool = Field(
        ...,
        description="Whether the core logic/algorithm is correct.",
    )
    syntax_issues: list[str] = Field(
        default_factory=list,
        description="Syntax or formatting issues found in the code.",
    )
    algorithm_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for algorithmic correctness (0 = wrong, 1 = perfect).",
    )
    edge_cases_handled: list[str] = Field(
        default_factory=list,
        description="Edge cases the student's code correctly handles.",
    )
    feedback: str = Field(
        ...,
        description="Human-readable feedback on the code answer.",
    )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> Optional[dict | list]:
    """Extract a JSON object or array from *raw*, tolerating markdown fences.

    Handles three common LLM output patterns:
      1. Raw JSON (``{...}`` or ``[...]``)
      2. Markdown-fenced JSON (`` ```json ... ``` ``)
      3. JSON embedded in prose (first ``{`` to last ``}``)

    Returns ``None`` when no valid JSON can be recovered.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip markdown fences if present.
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL
    )
    if fence_match:
        text = fence_match.group(1).strip()

    # Attempt direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost braces / brackets.
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.warning("Failed to extract JSON from LLM response (length=%d).", len(raw))
    return None


# ---------------------------------------------------------------------------
# Segmentation prompt & parsing
# ---------------------------------------------------------------------------


SEGMENTATION_PROMPT: str = textwrap.dedent("""\
    You are an expert document-layout analyser. Examine the provided exam page
    image and identify every distinct content region.

    For EACH region return a JSON object with:
      - "region_type": one of TEXT, DIAGRAM, TABLE, GRAPH, CODE, CHEMICAL
      - "bounding_box": [x_min, y_min, x_max, y_max] normalised to 0-1
      - "confidence": float 0-1
      - "content_description": brief description of the content

    Region-type guidance:
      TEXT      — prose, fill-in-the-blank, short/long answer paragraphs
      DIAGRAM   — hand-drawn figures, labelled diagrams, flowcharts
      TABLE     — any grid / tabular data including lab data tables
      GRAPH     — plotted graphs, charts, coordinate grids
      CODE      — pseudocode, programming code, algorithmic steps
      CHEMICAL  — molecular structures, reaction equations, Lewis structures

    Return ONLY a JSON object matching this schema:
    {
      "page_number": <int>,
      "regions": [ { "region_type": "...", "bounding_box": [...], "confidence": ..., "content_description": "..." }, ... ]
    }

    If no non-text regions are found, return an empty regions list.
    Do NOT wrap output in markdown fences.
""")


def build_segmentation_prompt() -> str:
    """Return the system prompt for page-region segmentation.

    The prompt instructs the vision model to identify all content regions
    on a scanned exam page and return structured JSON with bounding boxes,
    types, and confidence scores.
    """
    return SEGMENTATION_PROMPT


def parse_segmentation_response(raw_json: str) -> Optional[SegmentationResult]:
    """Parse a segmentation JSON response into a ``SegmentationResult``.

    Handles markdown-fenced JSON and malformed output gracefully.

    Args:
        raw_json: Raw string returned by the LLM.

    Returns:
        A validated ``SegmentationResult`` or ``None`` if parsing fails.
    """
    data = _extract_json(raw_json)
    if data is None or not isinstance(data, dict):
        return None

    try:
        # Normalise bounding_box lists into tuples for Pydantic.
        for region in data.get("regions", []):
            if isinstance(region.get("bounding_box"), list):
                region["bounding_box"] = tuple(region["bounding_box"])
        return SegmentationResult.model_validate(data)
    except Exception as exc:
        logger.warning("Segmentation parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Diagram grading prompt & parsing
# ---------------------------------------------------------------------------


def build_diagram_grading_prompt(reference_description: str) -> str:
    """Build a prompt to grade a student diagram against a reference.

    Args:
        reference_description: Plain-text description of the correct /
            expected diagram, including required labels, connections,
            and structural features.

    Returns:
        A system prompt string for the vision model.
    """
    return textwrap.dedent(f"""\
        You are an expert diagram evaluator for science exams.

        REFERENCE DIAGRAM DESCRIPTION:
        {reference_description}

        Examine the student's hand-drawn diagram in the provided image and
        compare it against the reference description above.

        Evaluate:
          1. Which required elements are present and correctly drawn.
          2. Which required elements are missing.
          3. Any extra elements the student added that are NOT in the reference.
          4. Overall structural similarity (0 = completely wrong, 1 = perfect).

        Return ONLY a JSON object:
        {{
          "similarity_score": <float 0-1>,
          "correct_elements": ["element1", "element2", ...],
          "missing_elements": ["element3", ...],
          "extra_elements": ["element4", ...],
          "feedback": "<concise paragraph of feedback>"
        }}

        Be generous with hand-drawn quality — focus on conceptual correctness,
        not artistic precision.  Labels may be abbreviated or misspelled;
        accept reasonable variants.
    """)


def parse_diagram_grading_response(
    raw_json: str,
) -> Optional[DiagramGradingResult]:
    """Parse a diagram-grading LLM response into a ``DiagramGradingResult``.

    Args:
        raw_json: Raw string returned by the LLM.

    Returns:
        A validated ``DiagramGradingResult`` or ``None`` on failure.
    """
    data = _extract_json(raw_json)
    if data is None or not isinstance(data, dict):
        return None

    try:
        return DiagramGradingResult.model_validate(data)
    except Exception as exc:
        logger.warning("Diagram grading parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Table grading prompt & parsing
# ---------------------------------------------------------------------------


def build_table_grading_prompt(expected_table: dict) -> str:
    """Build a prompt to grade a student data table.

    Args:
        expected_table: Dictionary describing the correct table. Expected
            keys are ``headers`` (list[str]) and ``rows`` (list[list[str]]).
            Tolerances or notes can be included under an optional
            ``tolerances`` key.

    Returns:
        A system prompt string for the vision model.
    """
    table_json = json.dumps(expected_table, indent=2)
    return textwrap.dedent(f"""\
        You are an expert lab-data evaluator for science exams.

        EXPECTED TABLE (JSON):
        {table_json}

        Examine the student's handwritten data table in the provided image.
        Compare each cell against the expected values above.

        Rules:
          - Numeric values: accept if within any stated tolerance or within
            +-1 in the last significant digit (instrument precision).
          - Units: must match expected unless tolerance note says otherwise.
          - Empty / illegible cells count as incorrect.
          - Headers are NOT counted as data cells.

        Return ONLY a JSON object:
        {{
          "correct_cells": <int>,
          "total_cells": <int>,
          "errors": [
            {{"row": <int 0-based>, "col": <int 0-based>, "expected": "...", "actual": "..."}},
            ...
          ],
          "score": <float 0-1, correct_cells / total_cells>
        }}
    """)


def parse_table_grading_response(
    raw_json: str,
) -> Optional[TableGradingResult]:
    """Parse a table-grading LLM response into a ``TableGradingResult``.

    Args:
        raw_json: Raw string returned by the LLM.

    Returns:
        A validated ``TableGradingResult`` or ``None`` on failure.
    """
    data = _extract_json(raw_json)
    if data is None or not isinstance(data, dict):
        return None

    try:
        # Validate error dicts into TableCellError models.
        if "errors" in data and isinstance(data["errors"], list):
            data["errors"] = [
                TableCellError.model_validate(e) if isinstance(e, dict) else e
                for e in data["errors"]
            ]
        return TableGradingResult.model_validate(data)
    except Exception as exc:
        logger.warning("Table grading parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Code grading prompt & parsing
# ---------------------------------------------------------------------------


def build_code_grading_prompt(expected_behavior: str) -> str:
    """Build a prompt to grade handwritten pseudocode or code.

    Args:
        expected_behavior: Plain-text description of what the code should
            accomplish, including input/output contract and any required
            algorithmic approach.

    Returns:
        A system prompt string for the vision model.
    """
    return textwrap.dedent(f"""\
        You are an expert code evaluator for computer science exams.

        EXPECTED BEHAVIOR:
        {expected_behavior}

        Examine the student's handwritten code or pseudocode in the provided
        image and evaluate it against the expected behavior above.

        Evaluation criteria:
          1. **Logic correctness** — Does the algorithm achieve the goal?
          2. **Syntax** — Note syntax issues but weigh them lightly for
             handwritten code (minor typos, missing semicolons are acceptable).
          3. **Algorithm quality** — Efficiency, clarity, correct use of
             data structures (0 = fundamentally wrong, 1 = optimal).
          4. **Edge cases** — Which boundary / special cases does the code
             handle (e.g., empty input, negative numbers, off-by-one)?

        Return ONLY a JSON object:
        {{
          "logic_correct": <bool>,
          "syntax_issues": ["issue1", "issue2", ...],
          "algorithm_score": <float 0-1>,
          "edge_cases_handled": ["case1", "case2", ...],
          "feedback": "<concise paragraph of feedback>"
        }}

        Be lenient on handwriting artifacts — focus on whether the student
        understands the algorithm, not penmanship.
    """)


def parse_code_grading_response(
    raw_json: str,
) -> Optional[CodeGradingResult]:
    """Parse a code-grading LLM response into a ``CodeGradingResult``.

    Args:
        raw_json: Raw string returned by the LLM.

    Returns:
        A validated ``CodeGradingResult`` or ``None`` on failure.
    """
    data = _extract_json(raw_json)
    if data is None or not isinstance(data, dict):
        return None

    try:
        return CodeGradingResult.model_validate(data)
    except Exception as exc:
        logger.warning("Code grading parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# LLM Execution Functions
# ---------------------------------------------------------------------------


def segment_page(image_blocks: list[dict], page_number: int = 1, provider: str = "claude") -> SegmentationResult | None:
    """Actually segment a page image into regions via LLM vision."""
    from grading_intelligence.llm_executor import execute_prompt
    prompt = build_segmentation_prompt()
    raw = execute_prompt(prompt, system_prompt="You are a document layout analysis expert.", image_blocks=image_blocks, provider=provider)
    return parse_segmentation_response(raw)


def grade_diagram(image_blocks: list[dict], reference_description: str, provider: str = "claude") -> DiagramGradingResult | None:
    """Actually grade a hand-drawn diagram via LLM vision."""
    from grading_intelligence.llm_executor import execute_prompt
    prompt = build_diagram_grading_prompt(reference_description)
    raw = execute_prompt(prompt, system_prompt="You are grading a student's hand-drawn diagram.", image_blocks=image_blocks, provider=provider)
    return parse_diagram_grading_response(raw)


def grade_table(image_blocks: list[dict], expected_table: dict, provider: str = "claude") -> TableGradingResult | None:
    """Actually grade a data table via LLM vision."""
    from grading_intelligence.llm_executor import execute_prompt
    prompt = build_table_grading_prompt(expected_table)
    raw = execute_prompt(prompt, system_prompt="You are grading a student's data table.", image_blocks=image_blocks, provider=provider)
    return parse_table_grading_response(raw)


def grade_code(image_blocks: list[dict], expected_behavior: str, provider: str = "claude") -> CodeGradingResult | None:
    """Actually grade handwritten code via LLM vision."""
    from grading_intelligence.llm_executor import execute_prompt
    prompt = build_code_grading_prompt(expected_behavior)
    raw = execute_prompt(prompt, system_prompt="You are grading a student's handwritten code/pseudocode.", image_blocks=image_blocks, provider=provider)
    return parse_code_grading_response(raw)
