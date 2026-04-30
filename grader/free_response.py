"""
Free response rubric management.
Handles loading, saving, listing, and formatting free response rubrics
for mixed exams (objective + free response questions).

Rubric format (JSON):
{
  "name": "PCR Steps",
  "total_points": 5,
  "sections": [
    {
      "number": 1,
      "title": "Identifies all three steps",
      "points": 1.0,
      "criteria": {
        "1": "Names denaturation, annealing, extension",
        "0.5": "Missing one step or unclear naming",
        "0": "Two or more steps missing"
      }
    }
  ]
}
"""

import json
import os
import re

RUBRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rubrics")


def list_saved_rubrics() -> list[dict]:
    """List all saved rubrics with name and path."""
    if not os.path.isdir(RUBRICS_DIR):
        return []

    rubrics = []
    for f in sorted(os.listdir(RUBRICS_DIR)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(RUBRICS_DIR, f)
        try:
            with open(path) as fh:
                data = json.load(fh)
            rubrics.append({
                "name": data.get("name", f),
                "total_points": data.get("total_points", "?"),
                "sections": len(data.get("sections", [])),
                "path": path,
                "filename": f,
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return rubrics


def load_rubric(path: str) -> dict:
    """Load a rubric JSON file and validate its structure."""
    with open(path) as f:
        data = json.load(f)

    if "sections" not in data or not isinstance(data["sections"], list):
        raise ValueError("Rubric must have a 'sections' list")

    total = 0
    for section in data["sections"]:
        if "number" not in section or "points" not in section:
            raise ValueError(f"Each section needs 'number' and 'points'. Got: {section}")
        if "criteria" not in section or not isinstance(section["criteria"], dict):
            raise ValueError(f"Section {section['number']} needs a 'criteria' dict")
        total += float(section["points"])

    data["total_points"] = total
    return data


def parse_rubric_text(text: str) -> dict:
    """
    Parse a pasted rubric in natural text format into structured JSON.

    Handles formats like:
        Use this 5-point rubric. Score each section. Add up to 5.
        1  Identifies all three steps (1 point)
        *  1: Names denaturation, annealing, extension
        *  0.5: Missing one step or unclear naming
        *  0: Two or more steps missing
    """
    lines = text.strip().split("\n")
    sections = []
    current_section = None
    name = ""
    total_from_header = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Try to extract a header/name line (first non-section line)
        header_match = re.match(r".*?(\d+)[- ]?point\s+rubric", stripped, re.IGNORECASE)
        if header_match and not sections and current_section is None:
            total_from_header = float(header_match.group(1))
            name = stripped
            continue

        # Match section header: "1  Title (N point)" or "1  Title (N points)" or just "1  Title"
        section_match = re.match(
            r"^(\d+)\s+(.+?)(?:\s*\((\d+(?:\.\d+)?)\s*points?\))?\s*$", stripped
        )
        if section_match:
            if current_section:
                sections.append(current_section)
            num = int(section_match.group(1))
            title = section_match.group(2).strip()
            points = float(section_match.group(3)) if section_match.group(3) else None
            current_section = {
                "number": num,
                "title": title,
                "points": points,
                "criteria": {},
            }
            continue

        # Match criteria line: "* 1: description" or "- 0.5: description" or just "1: description"
        criteria_match = re.match(
            r"^[\s*\-\u2022]*(\d+(?:\.\d+)?)\s*:\s*(.+)$", stripped
        )
        if criteria_match and current_section is not None:
            score = criteria_match.group(1)
            description = criteria_match.group(2).strip()
            current_section["criteria"][score] = description

            # Infer section points from the highest criteria score
            score_val = float(score)
            if current_section["points"] is None or score_val > current_section["points"]:
                current_section["points"] = score_val
            continue

        # Match "Total: ___ / N" line
        total_match = re.match(r"^Total\s*:.*?/\s*(\d+(?:\.\d+)?)", stripped, re.IGNORECASE)
        if total_match:
            total_from_header = float(total_match.group(1))
            continue

        # If we're in a section but line doesn't match criteria, treat as continuation
        # of the section title or skip
        if current_section is None and not name:
            name = stripped

    if current_section:
        sections.append(current_section)

    if not sections:
        raise ValueError(
            "Could not parse any rubric sections from the text. "
            "Expected format like:\n"
            "  1  Section title (1 point)\n"
            "  * 1: Full credit description\n"
            "  * 0.5: Partial credit description\n"
            "  * 0: No credit description"
        )

    # Fill in missing points from criteria
    for section in sections:
        if section["points"] is None:
            if section["criteria"]:
                section["points"] = max(float(s) for s in section["criteria"].keys())
            else:
                section["points"] = 1.0

    total = sum(s["points"] for s in sections)

    return {
        "name": name or "Untitled Rubric",
        "total_points": total_from_header if total_from_header else total,
        "sections": sections,
    }


def save_rubric(rubric: dict, filename: str = None) -> str:
    """
    Save a rubric to the rubrics/ directory.

    Args:
        rubric: Parsed rubric dict
        filename: Optional filename (without .json). If None, derived from name.

    Returns:
        Path to the saved file
    """
    os.makedirs(RUBRICS_DIR, exist_ok=True)

    if not filename:
        # Derive from rubric name
        filename = re.sub(r"[^\w\s-]", "", rubric.get("name", "rubric")).strip()
        filename = re.sub(r"\s+", "_", filename).lower()

    if not filename.endswith(".json"):
        filename += ".json"

    path = os.path.join(RUBRICS_DIR, filename)
    with open(path, "w") as f:
        json.dump(rubric, f, indent=2)

    return path


def format_free_response_prompt(rubric: dict) -> str:
    """
    Format a free response rubric into a grading prompt section
    that gets appended to the exam grading prompt.
    """
    lines = [
        "",
        "=== FREE RESPONSE RUBRIC ===",
        f"This exam includes free response questions graded with the following rubric.",
        f"Free response total: {rubric['total_points']} points",
        "",
        "=== FREE RESPONSE SCORING RULES ===",
        "For each free response section, use the criteria below to assign a score.",
        "Award the HIGHEST score whose description matches the student's answer.",
        "If the student's answer falls between two criteria levels, round in the student's favor.",
        "",
        "=== FREE RESPONSE DEDUCTION EXPLANATION (CRITICAL) ===",
        "For EVERY free response question where the student does NOT earn full credit,",
        "you MUST provide a detailed explanation of WHY points were lost. This is non-negotiable.",
        "",
        "Your deduction explanation must include ALL of the following:",
        "",
        "1. WHAT THE RUBRIC REQUIRED: State the specific criterion or criteria the student",
        "   needed to satisfy for full credit. Quote or paraphrase the rubric level directly.",
        "",
        "2. WHAT THE STUDENT WROTE: Quote or closely paraphrase the student's actual answer.",
        "   Reference specific words, phrases, or content from their response. Do NOT summarize",
        "   vaguely — cite what they actually put on the page.",
        "",
        "3. WHAT WAS MISSING OR INCORRECT: Explicitly identify the gap between what was required",
        "   and what was provided. Be specific:",
        "   - If a key term or concept was missing, name it",
        "   - If the explanation was incomplete, state what part was lacking",
        "   - If the reasoning was wrong, explain the error",
        "   - If the answer was too vague, explain what specificity was needed",
        "",
        "4. WHICH RUBRIC LEVEL WAS MATCHED: State which criteria level the student's answer",
        "   actually matched and why that level (not a higher one) was appropriate.",
        "",
        "Example of GOOD deduction explanation:",
        "  FR1 (3 pts): 2/3 [confidence: HIGH] — Rubric requires naming all three PCR steps",
        "  (denaturation, annealing, extension) for full credit. Student wrote 'first the DNA is",
        "  heated to separate strands, then primers attach, then...' and did not complete the",
        "  third step. Student correctly identified denaturation ('heated to separate strands')",
        "  and annealing ('primers attach') but left extension blank. Matches 2-point criterion:",
        "  'Missing one step or unclear naming.' Deducted 1 point for missing extension step.",
        "",
        "Example of BAD deduction explanation (DO NOT do this):",
        "  FR1 (3 pts): 2/3 [confidence: HIGH] — Incomplete answer, missing details.",
        "",
        "=== FREE RESPONSE TOLERANCE RULES ===",
        "Apply these tolerances BEFORE deciding on deductions:",
        "",
        "- EQUIVALENT PHRASING: Accept scientifically equivalent expressions. 'DNA unzips'",
        "  = 'strands separate' = 'denaturation occurs'. Do not deduct for informal language",
        "  if the scientific concept is clearly demonstrated.",
        "",
        "- PARTIAL KNOWLEDGE: If a student demonstrates understanding of a concept using",
        "  imprecise but not incorrect language, credit it. Only deduct when the answer is",
        "  fundamentally wrong, demonstrates a misconception, or is too vague to distinguish",
        "  from other plausible answers.",
        "",
        "- ESL / NON-NATIVE PHRASING: Focus on scientific content, not grammar or polish.",
        "  Awkward but correct = full credit. Do not deduct for missing articles, non-standard",
        "  sentence structure, or informal phrasing if the science is right.",
        "",
        "- CROSSED-OUT WORK: Only evaluate the student's FINAL answer (not crossed out).",
        "  Never penalize for changing their mind.",
        "",
        "- SINGLE-PENALTY RULE: If one error propagates across multiple FR sections",
        "  (e.g., wrong premise leads to wrong conclusion), deduct only ONCE at the root.",
        "  Note the propagation but do not stack deductions across sections.",
        "",
        "- BENEFIT OF THE DOUBT: When a student's answer could reasonably be read as",
        "  matching a higher rubric level, award the higher level. Default posture is to",
        "  give credit, not withhold it.",
        "",
    ]

    for section in rubric["sections"]:
        lines.append(
            f"FR{section['number']}: {section.get('title', 'Untitled')} "
            f"({section['points']} pt{'s' if section['points'] != 1 else ''})"
        )
        # Sort criteria by score descending so full credit appears first
        sorted_criteria = sorted(
            section["criteria"].items(),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        for score, description in sorted_criteria:
            lines.append(f"  {score}: {description}")
        lines.append("")

    lines.extend([
        "=== FREE RESPONSE OUTPUT FORMAT ===",
        "After the QUESTION BREAKDOWN for objective questions, add:",
        "",
        "FREE RESPONSE BREAKDOWN:",
    ])

    for section in rubric["sections"]:
        lines.append(
            f"FR{section['number']} ({section['points']} pt{'s' if section['points'] != 1 else ''}): "
            f"[earned]/[{section['points']}] [confidence: HIGH/LOW] — [FULL deduction explanation: "
            f"what rubric required → what student wrote → what was missing/wrong → which rubric level matched]"
        )

    lines.extend([
        "",
        "For any FR section where full credit is awarded, a brief confirmation is sufficient:",
        "  Example: FR1 (3 pts): 3/3 [confidence: HIGH] — Student correctly identified all three",
        "  PCR steps: denaturation, annealing, and extension. Matches full-credit criterion.",
        "",
    ])

    return "\n".join(lines)


def prompt_for_rubric() -> dict | None:
    """
    Interactive prompt to get a free response rubric from the user.
    Returns parsed rubric dict or None if user skips.
    """
    import click

    has_fr = click.confirm(
        "\nDoes this exam have free response questions?", default=False
    )
    if not has_fr:
        return None

    # Check for saved rubrics
    saved = list_saved_rubrics()
    if saved:
        click.echo("\nSaved rubrics:")
        for i, r in enumerate(saved, 1):
            click.echo(f"  {i}. {r['name']} ({r['total_points']} pts, {r['sections']} sections)")
        click.echo(f"  {len(saved) + 1}. Enter a new rubric")
        click.echo()

        choice = click.prompt(
            "Choose a rubric",
            type=int,
            default=len(saved) + 1,
        )

        if 1 <= choice <= len(saved):
            rubric = load_rubric(saved[choice - 1]["path"])
            click.echo(f"Loaded: {rubric['name']} ({rubric['total_points']} pts)")
            return rubric

    # Enter new rubric
    click.echo(
        "\nPaste your free response rubric below, then press Enter twice when done."
    )
    click.echo(
        "Format: numbered sections with point criteria (e.g., '1: full credit description')"
    )
    click.echo("---")

    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        click.echo("No rubric entered, skipping free response grading.")
        return None

    try:
        rubric = parse_rubric_text(text)
    except ValueError as e:
        click.echo(f"\nError parsing rubric: {e}")
        click.echo("Skipping free response grading.")
        return None

    # Show parsed result
    click.echo(f"\nParsed rubric: {rubric['name']}")
    click.echo(f"Total points: {rubric['total_points']}")
    for s in rubric["sections"]:
        click.echo(f"  {s['number']}. {s.get('title', 'Untitled')} ({s['points']} pts)")

    # Offer to save
    if click.confirm("\nSave this rubric for future use?", default=True):
        name = click.prompt("Rubric name", default=rubric["name"])
        rubric["name"] = name
        path = save_rubric(rubric)
        click.echo(f"Saved to: {path}")

    return rubric
