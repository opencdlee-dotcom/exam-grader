"""
Deterministic answer matching layer for objective questions.

Runs AFTER OCR but BEFORE LLM grading. Provides programmatic matching
for multiple choice, true/false, fill-in-the-blank, and temperature
answers. Only routes AMBIGUOUS or subjective questions to the LLM.

This eliminates the black-box problem where "Is the answer B?" costs
$0.03 and relies on LLM judgment for deterministic questions.
"""

import re
from enum import Enum


class MatchResult(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    AMBIGUOUS = "ambiguous"  # Route to LLM


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison: lowercase, strip, collapse whitespace."""
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    # Collapse whitespace
    answer = " ".join(answer.split())
    # Remove trailing punctuation
    answer = answer.rstrip(".,;:!?")
    return answer


def normalize_letter(answer: str) -> str | None:
    """Extract a single letter answer if present."""
    answer = str(answer).strip()
    # Handle "B)" or "B." or just "B"
    m = re.match(r'^([A-Za-z])\s*[).:]?\s*$', answer)
    if m:
        return m.group(1).upper()
    return None


def match_single_answer(
    student_answer: str | dict,
    expected_answers: list[str],
    alternatives: dict[str, list[str]] | None = None,
    tolerance: str | None = None,
) -> tuple[MatchResult, str]:
    """
    Match a single student answer against expected answers.

    Args:
        student_answer: Student's answer (string or dual-read dict from answer_sheet_reader)
        expected_answers: List of correct answers for this question
        alternatives: Dict mapping answer terms to accepted alternatives
        tolerance: Optional tolerance string (e.g., "±1°C")

    Returns:
        (MatchResult, explanation_string)
    """
    alternatives = alternatives or {}

    # Handle dual-read structured dict from answer_sheet_reader
    if isinstance(student_answer, dict):
        primary = student_answer.get("primary")
        alternate = student_answer.get("alternate")
        confidence = student_answer.get("confidence", "high")

        if confidence == "illegible" or primary is None:
            return MatchResult.AMBIGUOUS, "Answer illegible — route to LLM"

        # Try primary first
        primary_result, primary_reason = _match_text(
            primary, expected_answers, alternatives, tolerance
        )
        if primary_result == MatchResult.MATCH:
            if confidence == "low":
                return MatchResult.MATCH, f"Primary reading '{primary}' matches (low confidence)"
            return MatchResult.MATCH, f"'{primary}' matches"

        # Try alternate if available
        if alternate:
            alt_result, alt_reason = _match_text(
                alternate, expected_answers, alternatives, tolerance
            )
            if alt_result == MatchResult.MATCH:
                return MatchResult.AMBIGUOUS, (
                    f"Primary '{primary}' doesn't match, but alternate '{alternate}' does. "
                    f"Route to LLM for visual verification."
                )

        # Neither matched
        if confidence == "low" or alternate:
            return MatchResult.AMBIGUOUS, (
                f"Low confidence reading: primary='{primary}', alternate='{alternate}'. "
                f"Route to LLM for verification."
            )
        return MatchResult.MISMATCH, f"'{primary}' does not match expected {expected_answers}"

    # Simple string answer
    if student_answer is None or str(student_answer).strip() in ("", "[?]", "?"):
        return MatchResult.MISMATCH, "No answer provided"

    return _match_text(str(student_answer), expected_answers, alternatives, tolerance)


def _match_text(
    student_text: str,
    expected_answers: list[str],
    alternatives: dict[str, list[str]],
    tolerance: str | None,
) -> tuple[MatchResult, str]:
    """Core text matching logic."""
    student_norm = normalize_answer(student_text)
    student_letter = normalize_letter(student_text)

    # Build set of all acceptable answers (expected + alternatives)
    acceptable = set()
    for ans in expected_answers:
        acceptable.add(normalize_answer(ans))
        # Add the letter form if it's a single letter
        letter = normalize_letter(ans)
        if letter:
            acceptable.add(letter.lower())
        # Add alternatives for this answer
        for alt in alternatives.get(ans, []):
            acceptable.add(normalize_answer(alt))
            alt_letter = normalize_letter(alt)
            if alt_letter:
                acceptable.add(alt_letter.lower())

    # Direct text match
    if student_norm in acceptable:
        return MatchResult.MATCH, f"'{student_text}' matches"

    # Letter match (student wrote just "B", expected has "B" as alternative)
    if student_letter and student_letter.lower() in acceptable:
        return MatchResult.MATCH, f"Letter '{student_letter}' matches"

    # True/False normalization
    tf_map = {
        "true": {"t", "true", "yes", "y", "correct"},
        "false": {"f", "false", "no", "n", "incorrect"},
    }
    for canonical, variants in tf_map.items():
        if student_norm in variants:
            for ans in expected_answers:
                if normalize_answer(ans) in variants:
                    return MatchResult.MATCH, f"'{student_text}' matches (T/F)"

    # Temperature tolerance: ±1°C
    if tolerance and "°C" in (tolerance or ""):
        return _match_temperature(student_text, expected_answers, tolerance)

    # Numeric tolerance check — only when tolerance is explicitly set
    student_num = _try_parse_number(student_norm)
    if student_num is not None and tolerance is not None:
        for ans in expected_answers:
            expected_num = _try_parse_number(normalize_answer(ans))
            if expected_num is not None:
                # Check ±1 tolerance for temperature-like values
                if abs(student_num - expected_num) <= 1.0:
                    return MatchResult.MATCH, f"{student_num} ≈ {expected_num} (within ±1 tolerance)"

    return MatchResult.MISMATCH, f"'{student_text}' does not match expected {expected_answers}"


def _match_temperature(
    student_text: str,
    expected_answers: list[str],
    tolerance: str,
) -> tuple[MatchResult, str]:
    """Match temperature answers with ±1°C tolerance."""
    student_num = _try_parse_number(student_text)
    if student_num is None:
        return MatchResult.AMBIGUOUS, f"Cannot parse '{student_text}' as temperature"

    for ans in expected_answers:
        expected_num = _try_parse_number(ans)
        if expected_num is not None:
            if abs(student_num - expected_num) <= 1.0:
                return MatchResult.MATCH, f"{student_num}°C ≈ {expected_num}°C (within ±1°C)"

    return MatchResult.MISMATCH, f"{student_num}°C outside tolerance of expected values"


def _try_parse_number(text: str) -> float | None:
    """Try to parse a number from text, stripping units."""
    text = re.sub(r'[°℃℉CFcf%]', '', text).strip()
    text = text.replace(',', '')
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def match_exam_answers(
    student_answers: dict[int, str | dict],
    extracted_key: dict,
) -> dict[int, dict]:
    """
    Match all student answers against the answer key deterministically.

    Args:
        student_answers: {question_number: answer_string_or_dual_read_dict}
        extracted_key: Answer key dict with 'questions' list

    Returns:
        {question_number: {
            "result": MatchResult,
            "explanation": str,
            "points_awarded": float,
            "points_possible": float,
            "needs_llm": bool,
        }}
    """
    results = {}

    for q in extracted_key.get("questions", []):
        q_num = q["number"]
        expected = q.get("answers", [])
        alts = q.get("alternatives", {})
        tolerance = q.get("tolerance")
        points_possible = q.get("points", len(expected))

        student_ans = student_answers.get(q_num)
        if student_ans is None:
            results[q_num] = {
                "result": MatchResult.MISMATCH,
                "explanation": "No answer found",
                "points_awarded": 0,
                "points_possible": points_possible,
                "needs_llm": False,
            }
            continue

        # For multi-term questions (e.g., "list 3 types"), match each term
        if len(expected) > 1 and isinstance(student_ans, str):
            matched_count, total, explanations, any_ambiguous = _match_multi_term(
                student_ans, expected, alts, tolerance
            )
            if any_ambiguous:
                results[q_num] = {
                    "result": MatchResult.AMBIGUOUS,
                    "explanation": "; ".join(explanations),
                    "points_awarded": matched_count,
                    "points_possible": points_possible,
                    "needs_llm": True,
                }
            else:
                results[q_num] = {
                    "result": MatchResult.MATCH if matched_count == total else MatchResult.MISMATCH,
                    "explanation": "; ".join(explanations),
                    "points_awarded": matched_count,
                    "points_possible": points_possible,
                    "needs_llm": False,
                }
        else:
            # Single answer question
            result, explanation = match_single_answer(
                student_ans, expected, alts, tolerance
            )
            results[q_num] = {
                "result": result,
                "explanation": explanation,
                "points_awarded": points_possible if result == MatchResult.MATCH else 0,
                "points_possible": points_possible,
                "needs_llm": result == MatchResult.AMBIGUOUS,
            }

    return results


def _match_multi_term(
    student_text: str,
    expected: list[str],
    alternatives: dict[str, list[str]],
    tolerance: str | None,
) -> tuple[int, int, list[str], bool]:
    """
    Match a multi-term answer (student provided text, expected has multiple terms).
    Returns (matched_count, total_expected, explanations, any_ambiguous).
    """
    # Split student text by common delimiters
    student_terms = re.split(r'[,;\n]+', student_text)
    student_terms = [t.strip() for t in student_terms if t.strip()]

    matched = 0
    explanations = []
    any_ambiguous = False
    matched_expected = set()

    for exp_ans in expected:
        best_result = MatchResult.MISMATCH
        best_explanation = f"'{exp_ans}' not found"

        for st in student_terms:
            result, explanation = _match_text(st, [exp_ans], alternatives, tolerance)
            if result == MatchResult.MATCH:
                best_result = result
                best_explanation = explanation
                break
            elif result == MatchResult.AMBIGUOUS:
                best_result = result
                best_explanation = explanation
                any_ambiguous = True

        if best_result == MatchResult.MATCH:
            matched += 1
            matched_expected.add(exp_ans)
        explanations.append(best_explanation)

    return matched, len(expected), explanations, any_ambiguous
