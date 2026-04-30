"""
Chunking and merging logic for large PDF submissions.

Splits large submissions into manageable chunks for the Claude API,
then merges per-chunk grading results into a unified output.
"""

import re
from grader.config import CHUNK_SIZE


def should_chunk(page_count: int) -> bool:
    """Return True if a submission needs chunked grading."""
    return page_count > CHUNK_SIZE


def plan_chunks_exam(pages: list[dict], chunk_size: int = CHUNK_SIZE) -> list[list[dict]]:
    """
    Split pages into sequential chunks for exam grading.
    Questions are typically in page order, so sequential chunking works well.

    Args:
        pages: List of page dicts from pdf_reader.convert_pdf()
        chunk_size: Max pages per chunk

    Returns:
        List of page groups (each group is a list of page dicts)
    """
    chunks = []
    for i in range(0, len(pages), chunk_size):
        chunks.append(pages[i:i + chunk_size])
    return chunks


def plan_chunks_notebook(
    pages: list[dict],
    survey_result: dict,
    answer_key: dict,
) -> list[tuple[str, list[dict]]]:
    """
    Group pages by rubric section based on survey results.
    Each section gets the pages that the survey identified as containing
    content for that section.

    Args:
        pages: Full-resolution page images from pdf_reader
        survey_result: Output from parse_survey_response()
        answer_key: The answer key dict

    Returns:
        List of (section_name, [page_dicts]) tuples
    """
    section_pages = survey_result.get("section_pages", {})

    # Build a page_number -> page_dict lookup
    page_lookup = {p["page_number"]: p for p in pages}

    section_chunks = []
    assigned_pages = set()

    for section in answer_key["sections"]:
        section_name = section["name"]
        page_nums = section_pages.get(section_name, [])

        if page_nums:
            section_page_dicts = [page_lookup[n] for n in page_nums if n in page_lookup]
            if section_page_dicts:
                section_chunks.append((section_name, section_page_dicts))
                assigned_pages.update(page_nums)

    # If any pages weren't assigned to any section, add them as "Unassigned"
    # so they still get graded (student may have used unlabeled pages)
    unassigned_nums = [p["page_number"] for p in pages if p["page_number"] not in assigned_pages]
    if unassigned_nums:
        unassigned_pages = [page_lookup[n] for n in unassigned_nums]
        section_chunks.append(("Unassigned Pages", unassigned_pages))

    # If survey returned nothing useful, fall back to all pages in one chunk
    if not section_chunks:
        section_chunks = [("Full Submission", pages)]

    return section_chunks


def _extract_score(text: str) -> tuple[float | None, float | None]:
    """Extract score and total from a GRADE: X/Y line."""
    match = re.search(r"GRADE\s*:\s*([\d.]+)\s*/\s*([\d.]+)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def _extract_section(text: str, header: str) -> str:
    """Extract a named section from grading output (e.g., COMMENTS:, DEDUCTIONS:)."""
    pattern = rf"{header}\s*:?\s*\n(.*?)(?=\n[A-Z]+\s*:|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def merge_exam_results(chunk_results: list[str]) -> str:
    """
    Merge per-chunk exam grading results into a single unified output.

    Combines GRADE scores, QUESTION BREAKDOWN sections, and COMMENTS.

    Args:
        chunk_results: List of Claude responses, one per chunk

    Returns:
        Merged grading output in the standard format
    """
    total_score = 0.0
    total_possible = 0.0
    all_breakdowns = []
    all_comments = []
    all_deductions = []
    confidence_notes = []

    for i, result in enumerate(chunk_results):
        score, possible = _extract_score(result)
        if score is not None:
            total_score += score
            total_possible = max(total_possible, possible)

        # Extract question breakdown
        breakdown = _extract_section(result, "(?:QUESTION )?BREAKDOWN")
        if breakdown:
            all_breakdowns.append(f"--- Chunk {i + 1} ---\n{breakdown}")

        # Extract comments
        comments = _extract_section(result, "COMMENTS")
        if comments:
            all_comments.append(comments)

        # Extract deductions
        deductions = _extract_section(result, "DEDUCTIONS")
        if deductions:
            all_deductions.append(deductions)

        # Extract confidence
        confidence = _extract_section(result, "CONFIDENCE")
        if confidence:
            confidence_notes.append(confidence)

    # Build merged output
    lines = [
        f"GRADE: {total_score:.0f} / {total_possible:.0f}",
        "",
        "BREAKDOWN:",
        "\n".join(all_breakdowns) if all_breakdowns else "See individual chunk results",
        "",
    ]

    if all_deductions:
        lines.extend(["DEDUCTIONS:", "\n".join(all_deductions), ""])

    if all_comments:
        lines.extend(["COMMENTS:", "\n".join(all_comments), ""])

    if confidence_notes:
        lines.extend(["CONFIDENCE:", "\n".join(confidence_notes), ""])

    return "\n".join(lines)


def merge_notebook_results(
    section_results: list[tuple[str, str]],
    answer_key: dict,
) -> str:
    """
    Merge per-section notebook grading results into a single unified output.

    Args:
        section_results: List of (section_name, claude_response) tuples
        answer_key: The answer key dict (for total points and section caps)

    Returns:
        Merged grading output in the standard format
    """
    total_points = answer_key["total_points"]
    section_caps = {s["name"]: s["points"] for s in answer_key["sections"]}

    total_score = 0.0
    breakdown_lines = []
    all_deductions = []
    all_comments = []

    for section_name, result in section_results:
        score, _ = _extract_score(result)

        if score is not None:
            # Apply section cap
            cap = section_caps.get(section_name, float("inf"))
            capped_score = min(score, cap)
            total_score += capped_score
            breakdown_lines.append(f"- {section_name}: {capped_score:.0f}/{cap:.0f}")
        else:
            # Couldn't parse score for this section
            breakdown_lines.append(f"- {section_name}: [could not parse score]")

        deductions = _extract_section(result, "DEDUCTIONS")
        if deductions:
            all_deductions.append(f"[{section_name}]\n{deductions}")

        comments = _extract_section(result, "COMMENTS")
        if comments:
            all_comments.append(f"[{section_name}]\n{comments}")

    # Cap total score
    total_score = min(total_score, total_points)

    lines = [
        f"GRADE: {total_score:.0f} / {total_points}",
        "",
        "BREAKDOWN:",
    ]
    lines.extend(breakdown_lines)
    lines.append("")

    if all_deductions:
        lines.extend(["DEDUCTIONS:", "\n".join(all_deductions), ""])

    if all_comments:
        lines.extend(["COMMENTS:", "\n".join(all_comments), ""])

    return "\n".join(lines)
