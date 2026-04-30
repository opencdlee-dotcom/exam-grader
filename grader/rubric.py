"""
Answer key / rubric loading and management.
Supports JSON answer key files that define sections, criteria, and point values.
"""

import json
import os
import warnings
from grader.handwriting import get_grading_handwriting_rules


def load_answer_key(path: str) -> dict:
    """Load an answer key from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Answer key not found: {path}")

    with open(path) as f:
        key = json.load(f)

    # Validate required fields
    required = ["title", "total_points", "sections"]
    missing = [f for f in required if f not in key]
    if missing:
        raise ValueError(f"Answer key missing required fields: {', '.join(missing)}")

    # Validate sections and check deduction caps
    for section in key["sections"]:
        if "name" not in section or "points" not in section:
            raise ValueError(f"Each section needs 'name' and 'points'. Got: {section}")

        for criterion in section.get("criteria", []):
            if "deductions" in criterion:
                max_pts = criterion.get("points", 0)
                total_ded = sum(d["points"] for d in criterion["deductions"])
                if total_ded > max_pts:
                    # Enforce cap: keep only the largest deductions that fit within the criterion's value.
                    # This mutates the rubric in-place so the formatted prompt never contains excess deductions.
                    criterion["deductions"].sort(key=lambda d: d["points"], reverse=True)
                    kept, running = [], 0.0
                    for d in criterion["deductions"]:
                        if running + d["points"] <= max_pts:
                            kept.append(d)
                            running += d["points"]
                    removed = len(criterion["deductions"]) - len(kept)
                    criterion["deductions"] = kept
                    warnings.warn(
                        f"Rubric cap enforced on '{criterion.get('item', '?')}': "
                        f"removed {removed} deduction(s) that would exceed the "
                        f"{max_pts}pt criterion cap. Kept: {[d['points'] for d in kept]}"
                    )

    # Validate total_points matches sum of section points
    section_sum = sum(s["points"] for s in key["sections"])
    if section_sum != key["total_points"]:
        raise ValueError(
            f"Answer key total_points ({key['total_points']}) doesn't match "
            f"sum of section points ({section_sum}). Fix the JSON before grading."
        )

    # Check for duplicate section names
    section_names = [s["name"] for s in key["sections"]]
    dupes = [n for n in section_names if section_names.count(n) > 1]
    if dupes:
        raise ValueError(f"Duplicate section names in answer key: {', '.join(set(dupes))}")

    return key


def format_rubric_prompt(answer_key: dict) -> str:
    """
    Convert an answer key into a grading prompt for Claude.
    This tells Claude exactly what to look for and how to score.
    """
    lines = [
        "You are grading a student's handwritten lab notebook/report.",
        f"Assignment: {answer_key['title']}",
        f"Total points: {answer_key['total_points']}",
        "",
        "Grade using a DEDUCTION-BASED system. Start at full points and subtract for issues.",
        "",
        "=== GRADING PHILOSOPHY ===",
        "Err on the side of the student. Your role is to verify understanding, not penalize",
        "imprecision. Only deduct points when the student demonstrably lacks understanding",
        "or missed a requirement.",
        "",
        "=== ANSWER TOLERANCE RULES ===",
        "1. HANDWRITING: When handwriting is ambiguous, attempt all reasonable readings.",
        "   If any plausible reading satisfies the criterion, do NOT deduct.",
        "   Note ambiguities in SCAN/FORMAT NOTES but do not penalize for them.",
        "   If 50-80% confident in reading: note but do not deduct.",
        "   Below 50% (genuinely illegible): mark as ILLEGIBLE in notes, do not deduct or credit.",
        "",
        get_grading_handwriting_rules(domain="biology"),
        "",
        "",
        "2. EQUIVALENT EXPRESSION: Accept scientifically equivalent phrasing.",
        "   'The absorbance increased' = 'Absorbance went up' = 'A higher reading was observed'",
        "   Do not deduct for informal language if the scientific concept is correct.",
        "",
        "3. FORMAT: Do not deduct for:",
        "   - Unit formatting variations (mL vs ml vs ML)",
        "   - Minor capitalization issues",
        "   - Spelling errors where the intended word is recognizable",
        "   - Missing articles or prepositions in technical descriptions",
        "   - Handwriting that is messy but readable",
        "",
        "4. CONTENT DEPTH: For content-type criteria, the student must demonstrate understanding.",
        "   Deduct only when:",
        "   - Key concepts are missing entirely",
        "   - Scientific reasoning is fundamentally wrong",
        "   - Required calculations/data are absent",
        "   NOT when the student expresses the right idea with different words.",
        "",
        "5. SINGLE-PENALTY RULE: If a single error propagates through multiple criteria",
        "   (e.g., wrong calculation leads to wrong conclusion), deduct only ONCE for the",
        "   root error. Note the propagation but do not stack deductions.",
        "",
        "6. ANSWER HUNTING: Before marking ANY section as missing or incomplete, scan ALL",
        "   submitted pages — front AND back. Students may:",
        "   - Write sections out of order",
        "   - Continue work on backs of pages or extra sheets",
        "   - Use margins or spaces between other sections",
        "   Look for labels, section headers, or contextual clues that indicate which rubric",
        "   section the work belongs to. Only mark something as missing after checking every page.",
        "",
        "7. DEDUCTION CAP: Total deductions for any single criterion MUST NOT exceed that",
        "   criterion's point value. If multiple deductions could apply within one criterion,",
        "   apply only the largest applicable one. Similarly, total deductions for a section",
        "   must never exceed that section's total points. A section worth 5 points cannot",
        "   lose more than 5 points.",
        "",
        "8. CROSSED-OUT WORK: Students often cross out, scribble over, or erase and rewrite.",
        "   ALWAYS evaluate the student's FINAL version — the one NOT crossed out.",
        "   - If work is crossed out with new work beside/below it, evaluate the new version only",
        "   - NEVER penalize a student for changing their mind — only the final work matters",
        "",
        "9. DIAGRAMS AND DRAWINGS: If a student includes a diagram, flowchart, or drawing as part",
        "   of their work, award PARTIAL credit if it demonstrates understanding. A diagram alone",
        "   is not a substitute for required written analysis or explanations — but it shows effort",
        "   and partial understanding. Credit what the diagram communicates, but note that written",
        "   explanation is still expected for full marks on content-type criteria.",
        "",
        "10. ESL / NON-NATIVE PHRASING: Accept grammatically imperfect but scientifically correct",
        "    writing. Focus on whether the student demonstrates understanding, not whether their",
        "    English is polished. Do not deduct for awkward phrasing, missing articles, or",
        "    non-standard sentence structure if the scientific content is correct.",
        "",
        "=== ANTI-BIAS PROTOCOL ===",
        "When evaluating each criterion:",
        "1. First determine WHAT the student wrote (content extraction)",
        "2. Then determine if it satisfies the criterion (scoring)",
        "3. NEVER let handwriting neatness, organization, or apparent effort influence scoring.",
        "   A messy but correct answer gets full credit. A neat but wrong answer gets zero.",
        "",
        "=== SCORE GUARDRAILS ===",
        f"- The total score MUST NOT exceed {answer_key['total_points']}.",
        "- If NO student work is visible on any page (blank pages, wrong document, or only printed",
        f"  materials with no student writing), return GRADE: 0 / {answer_key['total_points']} with a note",
        "  explaining that no student work was found. Do NOT guess from ambiguous marks.",
        "",
        "=== RUBRIC ===",
    ]

    for section in answer_key["sections"]:
        lines.append(f"\n### {section['name']} — {section['points']} points")

        if "criteria" in section:
            for criterion in section["criteria"]:
                ctype = criterion.get("type", "content")
                lines.append(f"  - [{ctype}] {criterion['item']} ({criterion['points']} pts)")

                if "expected" in criterion:
                    lines.append(f"    Expected: {criterion['expected']}")

                if "alternatives" in criterion:
                    lines.append(f"    Also accept: {', '.join(criterion['alternatives'])}")

                if "anchor_examples" in criterion:
                    anchors = criterion["anchor_examples"]
                    if "full_credit" in anchors:
                        lines.append(f"    Example of full credit: \"{anchors['full_credit']}\"")
                    if "partial_credit" in anchors:
                        lines.append(f"    Example of partial credit: \"{anchors['partial_credit']}\"")
                    if "no_credit" in anchors:
                        lines.append(f"    Example of no credit: \"{anchors['no_credit']}\"")

                if "tolerance" in criterion:
                    lines.append(f"    Tolerance: {criterion['tolerance']}")

                if "deductions" in criterion:
                    for ded in criterion["deductions"]:
                        lines.append(f"    Deduction: -{ded['points']} if {ded['reason']}")

        if "notes" in section:
            lines.append(f"  Note: {section['notes']}")

    lines.extend([
        "",
        "=== OUTPUT FORMAT ===",
        "Respond in EXACTLY this format:",
        "",
        f"GRADE: [score] / {answer_key['total_points']}",
        "",
        "BREAKDOWN:",
    ])

    for section in answer_key["sections"]:
        lines.append(f"- {section['name']}: __/{section['points']} [confidence: HIGH/LOW]")

    lines.extend([
        "",
        "CONFIDENCE KEY: HIGH = confident in reading and scoring. LOW = handwriting ambiguous,",
        "criterion borderline, or scoring required significant interpretation. Flag LOW items",
        "for human review.",
        "",
        "DEDUCTIONS:",
        "[List each deduction with point value and specific reason]",
        "",
        "COMMENTS:",
        "[Start with what was done well, then specific constructive feedback]",
        "[Reference specific content from the student's work]",
        "[Use a firm but supportive tone]",
        "",
        "SCAN/FORMAT NOTES (if applicable):",
        "[Any issues with page orientation, scan quality, or organization]",
    ])

    return "\n".join(lines)


def list_answer_keys(directory: str = "answer_keys") -> list[str]:
    """List available answer key files."""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".json")]
