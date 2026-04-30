"""
ExamSchema — derive all exam structure from an answer key JSON at runtime.

Replaces every hardcoded value in the codebase:
  - total_possible (was 32)
  - fr_total_pts (was 5)
  - matching_range (was Q9-Q18)
  - below_50_threshold (was 17.5)
  - section names and types

Usage:
    from grader.exam_schema import build_exam_schema
    schema = build_exam_schema(answer_key)
    # schema.total_possible, schema.fr_total_pts, schema.matching_range, ...
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ExamSchema:
    """
    Fully-derived exam structure. Built once from the answer key JSON.
    All downstream grading functions (verification, spot-checking,
    reporting, prompts) read from this object — never from hardcoded constants.
    """
    total_possible: float          # Total points (objective + FR)
    objective_total: float         # Objective-only points
    fr_total_pts: float            # Free-response points
    matching_range: tuple[int, int]   # (first_q, last_q) inclusive — e.g. (9, 18)
    mc_range: tuple[int, int] | None  # MC question range, or None
    fill_in_range: tuple[int, int] | None  # Fill-in range, or None
    section_names: list[str]          # e.g. ["MC", "Matching", "Fill-In", "FR"]
    section_map: dict[str, str]       # {"Q9": "matching", "Q1": "mc", ...}
    below_50_threshold: float         # total_possible * 0.50
    question_count: int               # total objective question count


def build_exam_schema(answer_key: dict, fr_rubric: dict | None = None) -> ExamSchema:
    """
    Derive all exam structure from the answer key JSON (and optional FR rubric).

    The answer key JSON has a "sections" list. Each section has:
      - "name": e.g. "MC", "Matching", "Fill-In"
      - "points": total points for the section
      - "questions": list of question dicts with "number" fields
      - "type" (optional): "mc" | "matching" | "fill_in" | "fr"

    FR rubric is a separate JSON loaded by load_rubric().
    If not provided, the schema looks for an "FR" or "Free Response" section
    in the answer key itself.

    Returns a fully-populated ExamSchema.
    """
    sections = answer_key.get("sections", [])

    # ── Identify section types ────────────────────────────────────────────────
    # Detection order: explicit "type" field → name-based heuristic
    mc_qs: list[int] = []
    matching_qs: list[int] = []
    fill_in_qs: list[int] = []
    fr_pts_from_key: float = 0.0
    section_names: list[str] = []
    section_map: dict[str, str] = {}
    objective_total: float = 0.0

    for sec in sections:
        sec_name = sec.get("name", "")
        sec_pts = float(sec.get("points", 0))
        sec_type = sec.get("type", "").lower()
        questions = sec.get("questions", [])
        q_numbers = [int(q.get("number", q)) if isinstance(q, dict) else int(q)
                     for q in questions if q]

        # Classify section type
        if not sec_type:
            name_lower = sec_name.lower()
            if "fr" in name_lower or "free response" in name_lower:
                sec_type = "fr"
            elif "match" in name_lower:
                sec_type = "matching"
            elif "fill" in name_lower or "blank" in name_lower:
                sec_type = "fill_in"
            elif "mc" in name_lower or "multiple" in name_lower or "choice" in name_lower:
                sec_type = "mc"
            else:
                sec_type = "other"

        section_names.append(sec_name)

        if sec_type == "fr":
            fr_pts_from_key += sec_pts
        else:
            objective_total += sec_pts
            for q_num in q_numbers:
                q_key = f"Q{q_num}"
                section_map[q_key] = sec_type
                if sec_type == "mc":
                    mc_qs.append(q_num)
                elif sec_type == "matching":
                    matching_qs.append(q_num)
                elif sec_type == "fill_in":
                    fill_in_qs.append(q_num)

    # ── FR total: prefer explicit rubric, fall back to key ────────────────────
    if fr_rubric is not None:
        fr_pts = float(fr_rubric.get("total_points", fr_pts_from_key))
    else:
        fr_pts = fr_pts_from_key

    # ── Derive ranges from sorted question lists ──────────────────────────────
    def _to_range(q_list: list[int]) -> tuple[int, int] | None:
        if not q_list:
            return None
        return (min(q_list), max(q_list))

    matching_range = _to_range(matching_qs) or (9, 18)  # fallback to historical default
    mc_range = _to_range(mc_qs)
    fill_in_range = _to_range(fill_in_qs)

    total_possible = float(answer_key.get("total_points", objective_total)) + fr_pts

    return ExamSchema(
        total_possible=total_possible,
        objective_total=objective_total,
        fr_total_pts=fr_pts,
        matching_range=matching_range,
        mc_range=mc_range,
        fill_in_range=fill_in_range,
        section_names=section_names,
        section_map=section_map,
        below_50_threshold=total_possible * 0.50,
        question_count=len(section_map),
    )
