"""
Confusion Pair Recovery for Handwriting Recognition — Week 3 Improvement

Implements explicit decision trees for disambiguating commonly-confused letter pairs:
- B vs D: loop position, baseline extension
- O vs Q: presence of tail below baseline
- G vs C: horizontal bar closure at top
- M vs N: number of peaks (2 = M, 1 = N)
- U vs V: bottom shape (rounded = U, pointed = V)
- P vs F: crossbar position and length
- H vs A: crossbar height relative to middle
- L vs I: horizontal foot

Each pair has specific visual heuristics derived from letter morphology.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class ConfusionPairAnalysis:
    """Result of analyzing a potentially-confused letter pair."""
    pair: Tuple[str, str]  # (letter1, letter2)
    primary: str  # Most likely letter
    confidence: float  # 0.0-1.0 confidence in distinction
    reasoning: List[str]  # List of observations supporting the decision


# Decision trees for each confusion pair
CONFUSION_HEURISTICS = {
    ("B", "D"): {
        "name": "B vs D discrimination",
        "questions": [
            "Does the leftmost vertical stroke have a loop that goes LEFT at the middle? → B",
            "Does the loop bulge OUTWARD at the middle, then curve back to the stem? → B",
            "Is the loop positioned in the UPPER half, not the middle? → D",
            "Is there a straight diagonal line curving LEFT from the stem? → B",
            "If you cover the top half, does the bottom half look symmetric (bulge on BOTH sides)? → B",
            "If you cover the bottom half, does it look like a 'D' on top? → D",
        ],
        "visual_checks": [
            "B: tight loop at middle-left, bottom bulge smaller than top",
            "D: loop at top, extends down the right side, uniform right curve",
        ]
    },
    ("O", "Q"): {
        "name": "O vs Q discrimination",
        "questions": [
            "Is there a small tail extending BELOW the baseline? → Q",
            "Below the oval, is there a diagonal or curved stroke? → Q",
            "Does the shape end at the baseline with no extension below? → O",
            "Is the tail present but very small or faint? → Likely Q with weak tail",
        ],
        "visual_checks": [
            "Q: oval with tail extending down-right from bottom-right",
            "O: complete oval, contained within baseline-to-top region",
        ]
    },
    ("G", "C"): {
        "name": "G vs C discrimination",
        "questions": [
            "Is there a small horizontal bar or serif on the INSIDE at the middle height? → G",
            "At ~2/3 height, does a horizontal line extend INWARD from the right curve? → G",
            "Is it a simple open curve with no internal features? → C",
            "At the bottom, does the right side have a spur extending LEFT (into the curve)? → G",
        ],
        "visual_checks": [
            "G: C-shape with inward horizontal bar or spur at middle-right",
            "C: simple arc opening to the right, no internal features",
        ]
    },
    ("M", "N"): {
        "name": "M vs N discrimination",
        "questions": [
            "Count the number of peaks (high points touching top). How many? → (2=M, 1=N)",
            "In the middle, do the strokes form a VALLEY (dip down)? → M",
            "In the middle, is there a single diagonal stroke from bottom-left to top-right? → N",
            "At the top, are there two distinct peaks with space between them? → M",
            "Is there only one peak (at the top of the diagonal)? → N",
        ],
        "visual_checks": [
            "M: ∧∧ shape (two peaks), valley in middle",
            "N: / shape with two verticals at ends (one peak overall)",
        ]
    },
    ("U", "V"): {
        "name": "U vs V discrimination",
        "questions": [
            "At the bottom, is the corner ROUNDED (gentle curve)? → U",
            "At the bottom, is the corner SHARP (pointed point)? → V",
            "Do the bottom sides extend horizontally (parallel) after the curve? → U",
            "Does the shape come to a single point at the bottom-center? → V",
            "Is the base flat or slightly rounded? → U",
        ],
        "visual_checks": [
            "U: rounded bottom, sides stay parallel at base",
            "V: sharp pointed bottom, sides converge to point",
        ]
    },
    ("P", "F"): {
        "name": "P vs F discrimination",
        "questions": [
            "Below the loop/bulge, does the vertical stem extend down? → P",
            "Does the stem go DOWN PAST the middle? → P",
            "Does the stem STOP at the middle (no extension below)? → F",
            "Is there a horizontal bar below the middle? → P",
            "Are there horizontal bars only at the top two positions? → F",
        ],
        "visual_checks": [
            "P: loop at top, stem extends full height down",
            "F: loop at top, stem stops at middle (truncated)",
        ]
    },
    ("H", "A"): {
        "name": "H vs A discrimination",
        "questions": [
            "Do the two vertical strokes reach the SAME height at the top? → H",
            "Do the two strokes converge to a POINT or apex at the top? → A",
            "Is the crossbar positioned at the TRUE MIDDLE of the height? → H",
            "Is the crossbar positioned ABOVE the middle? → A (apex above)",
            "At the top, is there a gap between the two strokes (they don't touch)? → H",
        ],
        "visual_checks": [
            "H: two equal-height verticals, crossbar at middle",
            "A: two strokes converging to apex, crossbar below apex",
        ]
    },
    ("L", "I"): {
        "name": "L vs I discrimination",
        "questions": [
            "At the bottom, is there a horizontal foot extending RIGHT? → L",
            "Is the bottom a simple straight end (no foot)? → I",
            "Below the vertical, is there any horizontal element? → L",
            "Is it a single vertical line with possibly serifs at top/bottom? → I",
            "Does the base form an 'L' shape (right angle)? → L",
        ],
        "visual_checks": [
            "L: vertical with horizontal foot at bottom-right",
            "I: single vertical line, possibly with serif tops/bottoms",
        ]
    },
}


def disambiguate_pair(letter_candidate: str, alt_candidate: str, image_description: str = "") -> ConfusionPairAnalysis:
    """
    Disambiguate between two similar letters using visual heuristics.

    Args:
        letter_candidate: Primary reading (e.g., "B")
        alt_candidate: Alternative reading (e.g., "D")
        image_description: Optional textual description of visual features

    Returns:
        ConfusionPairAnalysis with decision and confidence
    """
    pair = tuple(sorted([letter_candidate, alt_candidate]))
    if pair not in CONFUSION_HEURISTICS:
        # Unknown pair, return primary as-is
        return ConfusionPairAnalysis(
            pair=pair,
            primary=letter_candidate,
            confidence=0.70,
            reasoning=[f"No specific heuristic for {pair}, using primary reading"],
        )

    heuristics = CONFUSION_HEURISTICS[pair]
    questions = heuristics["questions"]
    visual_checks = heuristics["visual_checks"]

    # Without visual analysis, we can't apply full heuristics
    # Instead, return guidance for the LLM to apply
    reasoning = [
        f"Apply these visual checks to distinguish {pair[0]} from {pair[1]}:",
    ] + visual_checks

    # Default: return primary with moderate confidence (70%)
    # The LLM will apply the heuristics based on the image
    return ConfusionPairAnalysis(
        pair=pair,
        primary=letter_candidate,
        confidence=0.70,
        reasoning=reasoning,
    )


def build_confusion_recovery_prompt() -> str:
    """
    Generate instruction text for handling confusion pairs.
    This guides the LLM when reading ambiguous letters.
    """
    lines = [
        "=== CONFUSION PAIR RECOVERY ===",
        "When you encounter any of these letter pairs, apply these visual heuristics:",
        "",
    ]

    for (letter1, letter2), heuristics in sorted(CONFUSION_HEURISTICS.items()):
        lines.append(f"**{letter1} vs {letter2}**: {heuristics['name']}")
        lines.append("Visual checks:")
        for check in heuristics["visual_checks"]:
            lines.append(f"  - {check}")
        lines.append("")

    lines.append("Apply these distinctions when reading Q9-18 matching questions.")
    lines.append("If the image is ambiguous, report both letters with your confidence in each.")

    return "\n".join(lines)


def apply_confusion_recovery_to_prompt(base_prompt: str) -> str:
    """
    Inject confusion recovery heuristics into the grading prompt.

    Args:
        base_prompt: Original grading prompt

    Returns:
        Modified prompt with confusion recovery guidance
    """
    recovery_text = build_confusion_recovery_prompt()

    # Find a good insertion point — right before answer key so heuristics inform reading
    insertion_point = "=== ANSWER KEY ==="
    if insertion_point in base_prompt:
        return base_prompt.replace(
            insertion_point,
            recovery_text + "\n\n" + insertion_point
        )

    # Fallback: append before OUTPUT FORMAT
    insertion_point = "=== OUTPUT FORMAT ==="
    if insertion_point in base_prompt:
        return base_prompt.replace(
            insertion_point,
            recovery_text + "\n\n" + insertion_point
        )

    # Last resort: append at end
    return base_prompt + "\n\n" + recovery_text
