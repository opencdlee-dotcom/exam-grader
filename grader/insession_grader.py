"""
In-Session Grading Module — No API Calls

Provides grading functions that work directly with student page data
without making any API calls. Claude Code reads images via Read tool
and provides grades, which are then aggregated.

This implements the protocol from CLAUDE.md:
"Never use the Anthropic API key for grading. Always grade directly in-session."
"""

import re
from typing import Dict, List, Any

def grade_student_answers(
    answer_responses: Dict[int, str],
    answer_key: Dict,
    total_points: float,
    student_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Score a student's answers against the key.

    Args:
        answer_responses: {q_num: answer_text, ...} from Claude Code's reading
        answer_key: Dict with "questions" list, each having "number", "answers", "points"
        total_points: Total points possible
        student_name: Student's name

    Returns:
        {
            "name": str,
            "score": float,
            "pct": float,
            "total": float,
            "answered": {q_num: answer_text, ...},
            "correct": {q_num: bool, ...},
            "flags": str,
            "missed": str,
        }
    """

    correct_count = 0
    score = 0
    missed = []
    flags = []

    correct_dict = {}
    answered_dict = dict(answer_responses)

    # Score each question
    for q in answer_key["questions"]:
        q_num = q["number"]
        q_points = q.get("points", 1)
        q_answers = q.get("answers", [])
        q_alts = q.get("alternatives", {})

        # Get student's answer
        student_ans = answer_responses.get(q_num, "").strip().upper()

        if not student_ans or student_ans == "?":
            missed.append(f"Q{q_num}")
            correct_dict[q_num] = False
            continue

        # Check primary answer
        is_correct = False
        if q_answers:
            primary = q_answers[0].upper()
            # Direct match
            if student_ans == primary:
                is_correct = True
            # Check alternatives
            else:
                for alt_group, alt_list in q_alts.items():
                    if primary.upper() == alt_group.upper():
                        for variant in alt_list:
                            if student_ans.upper() == variant.upper():
                                is_correct = True
                                break
                if not is_correct and len(primary) == 1:
                    # Single letter with tight tolerance
                    if student_ans == primary:
                        is_correct = True

        if is_correct:
            score += q_points
            correct_count += 1
            correct_dict[q_num] = True
        else:
            correct_dict[q_num] = False
            if q_points > 0:
                missed.append(f"Q{q_num}")

    pct = round((score / total_points * 100), 1) if total_points > 0 else 0

    # Build result
    result = {
        "name": student_name,
        "score": score,
        "pct": pct,
        "total": total_points,
        "answered": answered_dict,
        "correct": correct_dict,
        "flags": ", ".join(flags) if flags else "",
        "missed": ", ".join(missed) if missed else "",
        "raw": "",
    }

    return result

def format_grading_instruction() -> str:
    """Return instruction text for grading a student in-session."""
    return """
=== IN-SESSION STUDENT GRADING ===

For each student exam, provide answers in this format:

Student Name: [name from exam]
Q1: [answer]
Q2: [answer]
...
Q26: [answer]
Q23-FR: [free response evaluation]

Where:
- Q1-8: Single letter (A-Z)
- Q9-18: Single letter (A-Z)
- Q19-22: Text answer
- Q23: Free response (score 0-5, or "NOT ATTEMPTED")
- Q24: DNA sequence
- Q25: Three amino acids
- Q26: Numerical answer with units

Confidence flags:
- Add (AMBIGUOUS) if answer is unclear/borderline
- Add (FLAGGED) if needs human review

Example response:
---
Student Name: Trsn Lang
Q1: B
Q2: C
Q3: C
Q4: B
...
Q23-FR: 3 (PCR steps - partial work shown)
---

After each student, output a summary line:
STUDENT {NUM}: {NAME} - {SCORE}/{TOTAL} ({PCT}%) {FLAGS}
"""
