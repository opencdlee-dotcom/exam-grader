"""
Answer key cross-validation: extract the key twice with different prompts,
compare results, and flag conflicts before they cascade to all students.

Usage:
    from grader.key_cross_validator import validate_answer_key
    consensus = validate_answer_key("path/to/key.pdf", llm_fn)
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_BASE_KEY_EXTRACTION_PROMPT = """\
You are reading an instructor's filled-out answer sheet / answer key.
Each question has one or more expected answer terms written by the instructor.

Extract ALL questions and their correct answers. For multiple-choice questions,
include both the letter (A/B/C/D) and the answer text when visible.

Return ONLY valid JSON in this exact format — no markdown fences, no extra text:
{
  "questions": {
    "1": "B",
    "2": "mitosis",
    "3": ["term1", "term2"]
  }
}

Rules:
- Keys are question numbers as strings ("1", "2", ...).
- Values are the correct answer: a string for single answers, a list for
  multi-part answers.
- Include every question that appears on the sheet.
- Preserve the instructor's wording exactly.
- For blank/unanswered questions, use null.
"""

_ALT_KEY_EXTRACTION_PROMPT = """\
You are reviewing an answer key written by an instructor on a printed exam sheet.
Your task: identify every question number and its correct answer(s).

For each question found, record:
  - The question number
  - The answer the instructor marked or wrote

Output ONLY a JSON object mapping question numbers (as strings) to answers.
For multi-part answers, use a JSON array. For single answers, use a string.
Example format (no markdown fences):
{
  "questions": {
    "1": "C",
    "2": ["diffusion", "osmosis"],
    "3": "transcription"
  }
}

Additional guidance:
- Multiple-choice: record the letter AND the option text if legible.
- Fill-in: record the exact word(s) the instructor wrote.
- If a question is blank, use null.
- Do not skip any question, even if you are uncertain — record your best read.
"""


def build_key_extraction_prompt_v2(pdf_description: str) -> str:
    """Build an alternate extraction prompt with different wording.

    Using a different angle for the second pass improves the likelihood that
    genuine ambiguities surface as conflicts rather than being masked by
    identical prompt wording.

    Args:
        pdf_description: Optional plain-text description of the PDF (e.g.
            page count, subject matter).  Appended to the prompt for context.

    Returns:
        Formatted alternate prompt string.
    """
    suffix = f"\n\nDocument context: {pdf_description}" if pdf_description.strip() else ""
    return _ALT_KEY_EXTRACTION_PROMPT + suffix


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _normalize_answer(ans: object) -> list[str]:
    """Normalize an answer value to a sorted list of lowercase strings."""
    if ans is None:
        return []
    if isinstance(ans, list):
        return sorted(str(a).strip().lower() for a in ans)
    return [str(ans).strip().lower()]


def compare_extractions(key1: dict, key2: dict) -> dict:
    """Compare two extracted answer keys question by question.

    Args:
        key1: First extraction — expected shape ``{"questions": {str: answer}}``.
        key2: Second extraction — same shape.

    Returns:
        Dict with keys:
        - ``"conflicts"``: list of ``{"question": int, "extraction1": ...,
          "extraction2": ...}`` dicts for disagreements.
        - ``"agreed"``: sorted list of question numbers where both extractions
          agree.
        - ``"confidence"``: float in [0, 1] — fraction of questions agreed on.
    """
    qs1: dict = key1.get("questions", {})
    qs2: dict = key2.get("questions", {})

    all_questions = sorted(
        set(qs1.keys()) | set(qs2.keys()),
        key=lambda q: (int(q) if q.isdigit() else float("inf"), q),
    )

    conflicts: list[dict] = []
    agreed: list[int] = []

    for q in all_questions:
        ans1 = qs1.get(q)
        ans2 = qs2.get(q)
        norm1 = _normalize_answer(ans1)
        norm2 = _normalize_answer(ans2)

        if norm1 == norm2:
            try:
                agreed.append(int(q))
            except ValueError:
                agreed.append(q)  # type: ignore[arg-type]
        else:
            try:
                q_display: object = int(q)
            except ValueError:
                q_display = q
            conflicts.append({
                "question": q_display,
                "extraction1": ans1,
                "extraction2": ans2,
            })

    total = len(all_questions)
    confidence = round(len(agreed) / total, 4) if total > 0 else 1.0

    return {
        "conflicts": conflicts,
        "agreed": sorted(agreed),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Consensus building
# ---------------------------------------------------------------------------

def build_consensus_key(key1: dict, key2: dict, conflicts: list[dict]) -> dict:
    """Merge two extractions into a single consensus key.

    Where both agree: use that answer directly.
    Where they conflict: store both candidates and flag for instructor review.

    Args:
        key1: First extraction dict.
        key2: Second extraction dict.
        conflicts: List of conflict dicts from :func:`compare_extractions`.

    Returns:
        Consensus key dict with shape::

            {
              "questions": {
                "1": "B",
                "4": {"answer": ["B", "C"], "needs_review": True, "confidence": "LOW"},
                ...
              },
              "total_questions": int,
              "conflicts_count": int,
            }
    """
    qs1: dict = key1.get("questions", {})
    qs2: dict = key2.get("questions", {})
    conflict_nums = {str(c["question"]) for c in conflicts}

    all_questions = sorted(
        set(qs1.keys()) | set(qs2.keys()),
        key=lambda q: (int(q) if q.isdigit() else float("inf"), q),
    )

    consensus_questions: dict = {}
    for q in all_questions:
        if q not in conflict_nums:
            # Both agree — use key1's value (they're identical after normalization)
            consensus_questions[q] = qs1.get(q) if q in qs1 else qs2.get(q)
        else:
            # Conflict — store both and flag
            candidates = [qs1.get(q), qs2.get(q)]
            # Deduplicate while preserving order
            seen: list = []
            for c in candidates:
                if c not in seen:
                    seen.append(c)
            consensus_questions[q] = {
                "answer": seen if len(seen) > 1 else seen[0],
                "needs_review": True,
                "confidence": "LOW",
            }

    return {
        "questions": consensus_questions,
        "total_questions": len(all_questions),
        "conflicts_count": len(conflicts),
    }


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def _extract_key_from_pdf(pdf_path: str, prompt: str, llm_fn: Callable[[str], str]) -> dict:
    """Use llm_fn to extract an answer key from pdf_path using the given prompt.

    The llm_fn receives the prompt; it is the caller's responsibility to inject
    image content from the PDF (e.g. via a wrapper that reads the PDF and passes
    images alongside the prompt).  This function only handles JSON parsing.
    """
    import json
    import re

    raw = llm_fn(prompt)

    # Strip markdown fences if present
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Find JSON object
    if not text.startswith("{"):
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            text = m2.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Key extraction JSON parse error: %s — raw: %.200s", exc, raw)
        return {"questions": {}}

    # Normalise: some LLMs return {"1": "B"} at root level instead of wrapped
    if "questions" not in data and isinstance(data, dict):
        # Check if all keys look like question numbers
        if all(k.isdigit() for k in data.keys()):
            return {"questions": data}

    return data


def validate_answer_key(
    pdf_path: str,
    llm_fn: Callable[[str], str],
    answer_key_path: str | None = None,
) -> dict:
    """Extract and cross-validate an answer key from a PDF using two prompts.

    Performs two independent extractions with different prompt wordings, then
    compares them to surface conflicts before grading begins.

    Args:
        pdf_path: Path to the answer key PDF.
        llm_fn: Callable that accepts a prompt string and returns the LLM's
            raw response. The caller is responsible for attaching PDF images
            to the LLM call (e.g. via a closure or wrapper).
        answer_key_path: Unused; reserved for future cache integration.

    Returns:
        Consensus key dict (see :func:`build_consensus_key`) with added keys:
        - ``"extraction1"``: raw first extraction
        - ``"extraction2"``: raw second extraction
        - ``"comparison"``: output of :func:`compare_extractions`
    """
    logger.info("Extracting answer key (pass 1) from: %s", pdf_path)
    key1 = _extract_key_from_pdf(pdf_path, _BASE_KEY_EXTRACTION_PROMPT, llm_fn)

    logger.info("Extracting answer key (pass 2) from: %s", pdf_path)
    key2 = _extract_key_from_pdf(
        pdf_path, build_key_extraction_prompt_v2(""), llm_fn
    )

    comparison = compare_extractions(key1, key2)
    conflicts = comparison["conflicts"]
    total_q = len(comparison["agreed"]) + len(conflicts)

    consensus = build_consensus_key(key1, key2, conflicts)

    # Human-readable summary
    if conflicts:
        conflict_summary = ", ".join(
            f"Q{c['question']} ({c['extraction1']!r} vs {c['extraction2']!r})"
            for c in conflicts[:5]
        )
        if len(conflicts) > 5:
            conflict_summary += f" ... and {len(conflicts) - 5} more"
        print(
            f"Key extracted: {total_q} questions. "
            f"{len(conflicts)} conflict{'s' if len(conflicts) != 1 else ''}: "
            f"{conflict_summary}. Please verify before grading."
        )
    else:
        print(
            f"Key extracted: {total_q} questions. "
            f"No conflicts — both extractions agreed on all answers."
        )

    consensus["extraction1"] = key1
    consensus["extraction2"] = key2
    consensus["comparison"] = comparison

    return consensus
