"""
Adaptive Rubric Evolver — auto-refine rubrics based on batch grading results.

After every batch of 5+ students, analyzes low-confidence questions and
generates rubric improvement suggestions via Claude. Saves versioned rubrics.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


def identify_weak_questions(results: list[dict], confidence_threshold: float = 0.5) -> list[int]:
    """
    Identify question numbers where >30% of students had LOW confidence.

    Args:
        results: List of grading result dicts, each with a "structured_result" key.
        confidence_threshold: Unused numeric threshold; LOW confidence is detected
            by the string label "LOW" in question-level confidence fields.

    Returns:
        Sorted list of question numbers where >30% of students had LOW confidence.
    """
    # Tally per-question LOW confidence counts
    low_counts: dict[int, int] = {}
    total_students = 0

    for result in results:
        if result.get("error"):
            continue
        sr = result.get("structured_result")
        if sr is None:
            continue
        total_students += 1

        # structured_result may be a Pydantic model or a dict
        question_results = (
            sr.question_results if hasattr(sr, "question_results") else sr.get("question_results", [])
        )
        for qr in (question_results or []):
            q_num = (
                qr.question_number if hasattr(qr, "question_number") else qr.get("question_number")
            )
            confidence = (
                qr.confidence if hasattr(qr, "confidence") else qr.get("confidence", "")
            )
            if q_num is None:
                continue
            # ConfidenceLevel may be an enum or a string; normalise to string
            conf_str = str(confidence).upper()
            if "LOW" in conf_str:
                low_counts[q_num] = low_counts.get(q_num, 0) + 1

    if total_students == 0:
        return []

    threshold_count = total_students * 0.30
    weak = [q for q, count in low_counts.items() if count > threshold_count]
    return sorted(weak)


def collect_student_answers_for_question(results: list[dict], question_num: int) -> list[str]:
    """
    Extract student answers for a specific question from results.

    Returns up to 5 non-empty answer strings to keep prompts small.
    """
    answers: list[str] = []

    for result in results:
        if result.get("error"):
            continue
        sr = result.get("structured_result")
        if sr is None:
            continue

        question_results = (
            sr.question_results if hasattr(sr, "question_results") else sr.get("question_results", [])
        )
        for qr in (question_results or []):
            q_num = (
                qr.question_number if hasattr(qr, "question_number") else qr.get("question_number")
            )
            if q_num != question_num:
                continue
            answer = (
                qr.student_answer if hasattr(qr, "student_answer") else qr.get("student_answer", "")
            ) or ""
            if answer.strip():
                answers.append(answer.strip())
            break  # Only one answer per student per question

        if len(answers) >= 5:
            break

    return answers


def build_evolution_prompt(
    answer_key: dict,
    weak_questions: list[int],
    sample_answers: dict[int, list[str]],
) -> str:
    """
    Build a prompt for Claude asking it to suggest rubric improvements.

    For each weak question shows current rubric criteria + sample student answers,
    then asks Claude how to refine the rubric for more consistent grading.

    Returns a prompt string.
    """
    lines = [
        "You are an expert assessment designer reviewing an exam rubric.",
        "",
        f"Exam: {answer_key.get('title', 'Unknown')}",
        "",
        "After a batch grading session, the following questions had high rates of LOW-confidence"
        " grading (>30% of students). This suggests the rubric criteria are ambiguous or incomplete.",
        "",
        "For each weak question below, you will see:",
        "  1. The current rubric criteria",
        "  2. Sample student answers that caused grading uncertainty",
        "",
        "Your task: Suggest targeted rubric improvements to reduce ambiguity.",
        "",
        "=== WEAK QUESTIONS ===",
        "",
    ]

    # Build a lookup of criteria by question number
    # The rubric JSON has sections → criteria; criteria may have question numbers
    # or we use positional numbering across all criteria
    criteria_map: dict[int, dict] = {}
    q_counter = 0
    for section in answer_key.get("sections", []):
        for criterion in section.get("criteria", []):
            q_counter += 1
            criteria_map[q_counter] = {
                "section": section.get("name", "Unknown"),
                "criterion": criterion,
            }

    for q_num in weak_questions:
        lines.append(f"--- Question {q_num} ---")
        entry = criteria_map.get(q_num)
        if entry:
            crit = entry["criterion"]
            lines.append(f"Section: {entry['section']}")
            lines.append(f"Item: {crit.get('item', '?')} ({crit.get('points', '?')} pts)")
            lines.append(f"Type: {crit.get('type', 'content')}")
            if crit.get("expected"):
                lines.append(f"Expected: {crit['expected']}")
            if crit.get("alternatives"):
                lines.append(f"Also accept: {', '.join(crit['alternatives'])}")
            if crit.get("anchor_examples"):
                anchors = crit["anchor_examples"]
                if anchors.get("full_credit"):
                    lines.append(f"Full credit example: \"{anchors['full_credit']}\"")
                if anchors.get("partial_credit"):
                    lines.append(f"Partial credit example: \"{anchors['partial_credit']}\"")
                if anchors.get("no_credit"):
                    lines.append(f"No credit example: \"{anchors['no_credit']}\"")
        else:
            lines.append(f"(No rubric entry found for question {q_num})")

        answers = sample_answers.get(q_num, [])
        if answers:
            lines.append("Sample student answers that caused uncertainty:")
            for i, ans in enumerate(answers[:5], 1):
                lines.append(f"  {i}. \"{ans}\"")
        else:
            lines.append("(No sample answers collected)")
        lines.append("")

    lines.extend([
        "=== INSTRUCTIONS ===",
        "",
        "For each question above, suggest how the rubric criteria should be refined to handle"
        " the observed student answers more consistently.",
        "",
        "Focus on:",
        "- Clarifying ambiguous wording",
        "- Adding missing edge cases or alternative phrasings",
        "- Adjusting partial credit levels to match real student work",
        "- Making acceptance/rejection criteria more explicit",
        "",
        "Return ONLY valid JSON in this exact format:",
        '{',
        '  "suggestions": [',
        '    {',
        '      "question": <question_number as int>,',
        '      "current_criteria": "<brief description of current criteria>",',
        '      "suggested_change": "<specific wording change or addition>",',
        '      "rationale": "<why this change reduces ambiguity based on the sample answers>"',
        '    }',
        '  ]',
        '}',
        "",
        "If a question has no clear improvement to suggest, omit it from the suggestions list.",
        "Return ONLY valid JSON — no preamble, no markdown code fences.",
    ])

    return "\n".join(lines)


def apply_suggestions_to_rubric(rubric: dict, suggestions: list[dict]) -> dict:
    """
    Apply Claude's rubric improvement suggestions to the rubric dict.

    Does NOT save to disk. Returns a new (shallow-copied) rubric dict with
    modified criteria and _evolved_from metadata on each changed criterion.

    Args:
        rubric: Original rubric dict (will not be mutated).
        suggestions: List of suggestion dicts from Claude's JSON response.

    Returns:
        Updated rubric dict.
    """
    import copy
    evolved = copy.deepcopy(rubric)

    # Build positional index of criteria
    criteria_by_pos: dict[int, dict] = {}
    q_counter = 0
    for section in evolved.get("sections", []):
        for criterion in section.get("criteria", []):
            q_counter += 1
            criteria_by_pos[q_counter] = criterion

    changes: list[dict] = []

    for suggestion in suggestions:
        q_num = suggestion.get("question")
        if not isinstance(q_num, int):
            continue
        criterion = criteria_by_pos.get(q_num)
        if criterion is None:
            logger.warning("Suggestion for Q%d but no criterion found at that position", q_num)
            continue

        suggested_change = suggestion.get("suggested_change", "")
        rationale = suggestion.get("rationale", "")

        # Append the suggestion as an anchor example note so it's visible in the rubric
        if "anchor_examples" not in criterion:
            criterion["anchor_examples"] = {}
        existing_note = criterion["anchor_examples"].get("evolution_note", "")
        separator = " | " if existing_note else ""
        criterion["anchor_examples"]["evolution_note"] = (
            existing_note + separator + suggested_change
        )

        # Tag the criterion with evolution provenance
        criterion["_evolved_from"] = {
            "suggested_change": suggested_change,
            "rationale": rationale,
        }

        changes.append({
            "question": q_num,
            "item": criterion.get("item", "?"),
            "suggested_change": suggested_change,
            "rationale": rationale,
        })

    # Store change list at top level for metadata extraction later
    evolved["_evolution_changes"] = changes

    return evolved


def create_rubric_version(original_path: str, evolved_rubric: dict, n_students: int) -> str:
    """
    Save the evolved rubric as a versioned JSON file.

    Version numbers are inferred by scanning existing files with the same base name.
    E.g. ex2T5.json → ex2T5_v2.json, then ex2T5_v3.json on the next call.

    Adds a _metadata block recording provenance.

    Returns the new file path as a string.
    """
    original = Path(original_path)
    stem = original.stem
    suffix = original.suffix
    parent = original.parent

    # Strip any existing _vN suffix so we can re-version cleanly
    base_stem = re.sub(r"_v\d+$", "", stem)

    # Find existing version files to determine next version number
    existing_versions = list(parent.glob(f"{base_stem}_v*.json"))
    version_numbers: list[int] = []
    for p in existing_versions:
        m = re.search(r"_v(\d+)$", p.stem)
        if m:
            version_numbers.append(int(m.group(1)))

    next_version = max(version_numbers, default=1) + 1

    new_filename = f"{base_stem}_v{next_version}{suffix}"
    new_path = parent / new_filename

    changes = evolved_rubric.pop("_evolution_changes", [])

    # Inject metadata block
    evolved_rubric["_metadata"] = {
        "parent": original.name,
        "version": next_version,
        "evolved_from_n_students": n_students,
        "date": date.today().isoformat(),
        "changes": changes,
    }

    new_path.write_text(json.dumps(evolved_rubric, indent=2), encoding="utf-8")
    logger.info("Versioned rubric saved: %s", new_path)

    return str(new_path)


def display_rubric_diff(original: dict, evolved: dict) -> None:
    """
    Print a human-readable diff of rubric changes to stdout.

    Shows before/after for each changed criterion and the rationale.
    """
    changes = evolved.get("_metadata", {}).get("changes") or evolved.get("_evolution_changes", [])

    print("=== Rubric Evolution Suggestions ===")

    if not changes:
        print("No changes suggested.")
        return

    # Build a map of question → original criterion text for "BEFORE"
    original_criteria: dict[int, dict] = {}
    q_counter = 0
    for section in original.get("sections", []):
        for criterion in section.get("criteria", []):
            q_counter += 1
            original_criteria[q_counter] = criterion

    for change in changes:
        q_num = change.get("question")
        item = change.get("item", "?")
        suggested = change.get("suggested_change", "")
        rationale = change.get("rationale", "")

        orig_crit = original_criteria.get(q_num, {})
        before_text = orig_crit.get("expected") or orig_crit.get("item", "")

        print(f"Q{q_num} ({item}):")
        print(f"  BEFORE: \"{before_text}\"")
        print(f"  AFTER:  \"{suggested}\"")
        print(f"  Reason: {rationale}")

    # List questions that were weak but had no change suggested
    metadata = evolved.get("_metadata", {})
    changed_questions = {c.get("question") for c in changes}
    # We don't have the full weak_questions list here; just note the ones that changed
    if changed_questions:
        print(f"\n{len(changed_questions)} question(s) updated.")


def run_rubric_evolution(
    results: list[dict],
    answer_key_path: str,
    llm_fn: callable,
) -> str | None:
    """
    Main entry point for adaptive rubric evolution.

    Workflow:
      1. Skip if batch < 5 students.
      2. Identify weak questions (>30% LOW confidence).
      3. Collect sample student answers for each weak question.
      4. Build evolution prompt and call llm_fn(prompt) → raw JSON string.
      5. Parse suggestions, display diff, save versioned rubric.
      6. Return new rubric path, or None if nothing to do.

    Args:
        results: Grading results list (same format used by _run_batch_analytics).
        answer_key_path: Path to the original answer key JSON file.
        llm_fn: Callable that accepts a prompt string and returns a raw response string.

    Returns:
        Path to the new versioned rubric file, or None if skipped/no suggestions.
    """
    # Gate: need at least 5 students for meaningful analysis
    valid_results = [r for r in results if not r.get("error")]
    if len(valid_results) < 5:
        logger.info(
            "Rubric evolution skipped: only %d valid results (need 5+).", len(valid_results)
        )
        return None

    # Load original rubric
    try:
        with open(answer_key_path, encoding="utf-8") as f:
            answer_key = json.load(f)
    except Exception as e:
        logger.error("Could not load answer key for rubric evolution: %s", e)
        return None

    # Step 1: Find weak questions
    weak_questions = identify_weak_questions(results)
    if not weak_questions:
        logger.info("No weak questions found — rubric appears consistent across all students.")
        return None

    logger.info("Weak questions (>30%% LOW confidence): %s", weak_questions)

    # Step 2: Collect sample answers
    sample_answers: dict[int, list[str]] = {}
    for q_num in weak_questions:
        sample_answers[q_num] = collect_student_answers_for_question(results, q_num)

    # Step 3: Build and call evolution prompt
    prompt = build_evolution_prompt(answer_key, weak_questions, sample_answers)

    try:
        raw_response = llm_fn(prompt)
    except Exception as e:
        logger.error("LLM call failed during rubric evolution: %s", e)
        return None

    # Step 4: Parse suggestions
    suggestions = _parse_evolution_response(raw_response)
    if not suggestions:
        logger.info("Rubric evolution: LLM returned no actionable suggestions.")
        return None

    # Step 5: Apply suggestions and create versioned rubric
    evolved_rubric = apply_suggestions_to_rubric(answer_key, suggestions)
    display_rubric_diff(answer_key, evolved_rubric)

    new_path = create_rubric_version(answer_key_path, evolved_rubric, n_students=len(valid_results))
    logger.info("Evolved rubric saved to: %s", new_path)
    return new_path


# ─── Internal helpers ────────────────────────────────────────────────────────


def _parse_evolution_response(raw_response: str) -> list[dict]:
    """Parse Claude's rubric evolution JSON response. Returns empty list on failure."""
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract embedded JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse rubric evolution JSON response.")
                return []
        else:
            logger.warning("No JSON found in rubric evolution response.")
            return []

    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        logger.warning("Unexpected suggestions format: %s", type(suggestions))
        return []

    return suggestions
