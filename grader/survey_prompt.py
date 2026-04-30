"""
Survey pass prompt for large notebook submissions.
Sends low-resolution page images to Claude to identify which pages
contain which rubric sections, enabling targeted per-section grading.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def format_survey_prompt(answer_key: dict) -> str:
    """
    Build a prompt that asks Claude to map pages to rubric sections.

    Args:
        answer_key: The answer key dict with 'sections' list

    Returns:
        Prompt string for the survey pass
    """
    section_names = [s["name"] for s in answer_key["sections"]]
    sections_list = "\n".join(f"  - {name}" for name in section_names)

    return (
        "You are analyzing a student's handwritten lab notebook/report.\n"
        "Your task is to identify which pages contain content for which rubric sections.\n\n"
        "The rubric sections are:\n"
        f"{sections_list}\n\n"
        "For each page, identify:\n"
        "1. What rubric section(s) the page content belongs to\n"
        "2. A brief description of what's on the page (e.g., 'data table', 'graph', 'written analysis')\n\n"
        "Students may write sections out of order, continue on backs of pages, or use margins.\n"
        "A single page may contain content for multiple sections.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        "```json\n"
        "{\n"
        '  "page_map": {\n'
        '    "1": {"sections": ["Section Name"], "content": "brief description"},\n'
        '    "2": {"sections": ["Section Name", "Other Section"], "content": "brief description"}\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "Use the exact section names from the rubric. If a page is blank or contains only "
        "printed headers with no student work, use an empty sections list."
    )


def parse_survey_response(response_text: str) -> dict:
    """
    Parse Claude's survey response into a section-to-pages mapping.

    Args:
        response_text: Claude's JSON response from the survey pass

    Returns:
        Dict with:
        - 'section_pages': {section_name: [page_numbers]}
        - 'page_map': {page_num: [section_names]}
    """
    # Extract JSON from markdown code fence if present
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try the whole response as JSON
        json_str = response_text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: return empty mapping so caller can fall back to sequential chunking
        logger.warning("Could not parse survey response as JSON")
        return {"section_pages": {}, "page_map": {}}

    page_map_raw = data.get("page_map", {})

    # Build section_pages: {section_name: [page_numbers]}
    section_pages = {}
    page_map = {}

    for page_str, info in page_map_raw.items():
        page_num = int(page_str)
        sections = info.get("sections", []) if isinstance(info, dict) else []
        page_map[page_num] = sections

        for section_name in sections:
            if section_name not in section_pages:
                section_pages[section_name] = []
            section_pages[section_name].append(page_num)

    # Sort page numbers within each section
    for section_name in section_pages:
        section_pages[section_name].sort()

    return {"section_pages": section_pages, "page_map": page_map}
