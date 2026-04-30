"""
Free Response Anchor Library — calibrate FR scoring with pre-scored examples.
Injects example answers (full/partial/zero credit) into grading prompt for
consistency.

Usage:
    from grader.fr_anchors import load_anchors, build_anchor_context, save_anchor
    anchors = load_anchors("exam2")
    context = build_anchor_context(anchors.get(4, []), question_num=4)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Valid score level identifiers
_SCORE_LEVELS = ("full", "partial", "zero")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FRAnchor:
    """A single pre-scored example for a free response question.

    Attributes:
        question_num: Question number this anchor applies to.
        score_level: One of ``"full"``, ``"partial"``, or ``"zero"``.
        example_answer: The example student answer text.
        points: Points awarded for this example.
        max_points: Maximum points possible for the question.
        grading_notes: Explanation of why this score was awarded.
        common_mistakes: List of typical student errors at this score level.
    """

    question_num: int
    score_level: str          # "full" | "partial" | "zero"
    example_answer: str
    points: float
    max_points: float
    grading_notes: str = ""
    common_mistakes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.score_level not in _SCORE_LEVELS:
            raise ValueError(
                f"score_level must be one of {_SCORE_LEVELS}, got {self.score_level!r}"
            )
        if self.points < 0 or self.points > self.max_points:
            raise ValueError(
                f"points ({self.points}) must be in [0, max_points ({self.max_points})]"
            )


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _anchor_file_path(exam_name: str, anchor_dir: str) -> Path:
    return Path(anchor_dir) / f"{exam_name}_anchors.json"


def load_anchors(
    exam_name: str,
    anchor_dir: str = "answer_keys",
) -> dict[int, list[FRAnchor]]:
    """Load pre-scored FR anchors from disk.

    Args:
        exam_name: Logical exam identifier (e.g. ``"exam2"``).
        anchor_dir: Directory containing anchor JSON files.

    Returns:
        Dict mapping question number → list of :class:`FRAnchor`.
        Returns an empty dict if the file does not exist (graceful degradation).
    """
    path = _anchor_file_path(exam_name, anchor_dir)
    if not path.exists():
        logger.debug("No anchor file found at %s — returning empty dict", path)
        return {}

    try:
        raw_list: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load anchors from %s: %s", path, exc)
        return {}

    result: dict[int, list[FRAnchor]] = {}
    for raw in raw_list:
        try:
            anchor = FRAnchor(
                question_num=int(raw["question_num"]),
                score_level=str(raw["score_level"]),
                example_answer=str(raw["example_answer"]),
                points=float(raw["points"]),
                max_points=float(raw["max_points"]),
                grading_notes=str(raw.get("grading_notes", "")),
                common_mistakes=list(raw.get("common_mistakes", [])),
            )
            result.setdefault(anchor.question_num, []).append(anchor)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed anchor entry %s: %s", raw, exc)

    total = sum(len(v) for v in result.values())
    logger.debug(
        "Loaded %d anchors for %d question(s) from %s", total, len(result), path
    )
    return result


def save_anchor(
    exam_name: str,
    anchor: FRAnchor,
    anchor_dir: str = "answer_keys",
) -> None:
    """Append a single anchor to the exam's anchor file.

    Creates the file (and parent directory) if it doesn't exist.

    Args:
        exam_name: Logical exam identifier.
        anchor: The :class:`FRAnchor` to persist.
        anchor_dir: Directory containing anchor JSON files.
    """
    path = _anchor_file_path(exam_name, anchor_dir)
    os.makedirs(path.parent, exist_ok=True)

    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                logger.warning(
                    "Anchor file %s has unexpected shape — overwriting", path
                )
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(asdict(anchor))
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Saved anchor for Q%d (%s) to %s", anchor.question_num, anchor.score_level, path)


# ---------------------------------------------------------------------------
# Prompt context builder
# ---------------------------------------------------------------------------

_LEVEL_LABELS = {
    "full": "FULL CREDIT",
    "partial": "PARTIAL CREDIT",
    "zero": "ZERO CREDIT",
}

_LEVEL_ORDER = {"full": 0, "partial": 1, "zero": 2}


def build_anchor_context(anchors: list[FRAnchor], question_num: int) -> str:
    """Format anchor examples as grading prompt context.

    Injects pre-scored examples into a grading prompt so the LLM can calibrate
    its scoring to match instructor intent.

    Args:
        anchors: List of :class:`FRAnchor` objects for the question (may be
            from different score levels).
        question_num: Question number used in the header label.

    Returns:
        Formatted multi-line string ready to embed in a grading prompt.
        Returns an empty string if ``anchors`` is empty.
    """
    if not anchors:
        return ""

    # Sort by level (full → partial → zero) for consistent presentation
    sorted_anchors = sorted(anchors, key=lambda a: _LEVEL_ORDER.get(a.score_level, 9))

    lines: list[str] = [f"GRADING ANCHORS FOR Q{question_num}:"]

    for anchor in sorted_anchors:
        label = _LEVEL_LABELS.get(anchor.score_level, anchor.score_level.upper())
        pts_str = f"{anchor.points:.4g}/{anchor.max_points:.4g} pts"
        lines.append(f'{label} ({pts_str}): "{anchor.example_answer}"')

        if anchor.grading_notes:
            lines.append(f"  Notes: {anchor.grading_notes}")

        if anchor.common_mistakes:
            mistakes_str = "; ".join(anchor.common_mistakes)
            lines.append(f"  COMMON MISTAKES: {mistakes_str}")

    return "\n".join(lines)
