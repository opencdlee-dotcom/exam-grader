"""
Weighted Confidence Voting for Handwriting Recognition — Week 1 Improvement

Instead of simple majority voting (3/3 agree = HIGH, 2/3 = LOW),
use confidence-weighted voting where each read is weighted by how confident
the model is in that specific reading.

Example:
- Read 1: "B" with 95% confidence
- Read 2: "D" with 50% confidence
- Read 3: "B" with 92% confidence

Weighted consensus: B = 95 + 92 = 187, D = 50
Result: B (HIGH confidence) — not just "tie broken randomly"
"""

import re
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Optional


@dataclass
class ConfidentAnswer:
    """Answer with explicit confidence score and reasoning."""
    answer: str
    confidence: float  # 0.0 to 1.0
    reads: List[Tuple[str, float]]  # [(answer, confidence_score), ...]
    is_ambiguous: bool  # True if close tie
    reasoning: str


def extract_confidence_score(response_text: str) -> float:
    """
    Extract confidence from Claude/Gemini/OpenAI response.

    Looks for patterns like:
    - "I'm very confident this is B"
    - "with high confidence, this is..."
    - "confidence: 95%"
    - "~90% confident"
    - "this clearly appears to be"

    Returns:
        float between 0.0 and 1.0
    """
    response_lower = response_text.lower()

    # Explicit percentage patterns: "95%", "~80%", etc.
    percent_match = re.search(r'~?(\d+)\s*%\s*(?:confid|sure|certain)', response_lower)
    if percent_match:
        return float(percent_match.group(1)) / 100.0

    # Explicit confidence score: "confidence: 0.87"
    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', response_lower)
    if conf_match:
        score = float(conf_match.group(1))
        return min(1.0, max(0.0, score))  # clamp to [0, 1]

    # Keyword-based scoring
    confidence_keywords = {
        r'\b(very confident|absolutely certain|definitely|clearly|obviously)\b': 0.95,
        r'\b(confident|likely|probably|certain)\b': 0.85,
        r'\b(somewhat confident|somewhat likely|think|appears to be)\b': 0.70,
        r'\b(unsure|ambiguous|could be|might be|possibly)\b': 0.50,
        r'\b(cannot read|illegible|unclear)\b': 0.20,
    }

    for pattern, score in confidence_keywords.items():
        if re.search(pattern, response_lower):
            return score

    # Default: neutral confidence
    return 0.70


def weighted_consensus_vote(
    reads: List[Tuple[str, float]],
    confidence_weight: float = 1.0,
    tie_threshold: float = 0.10,
) -> ConfidentAnswer:
    """
    Compute weighted consensus from multiple reads.

    Args:
        reads: list of (answer, confidence) tuples
        confidence_weight: how much to weight confidence (1.0 = full weight, 0.5 = half)
        tie_threshold: if top 2 answers within this margin, mark as ambiguous

    Returns:
        ConfidentAnswer with weighted consensus
    """
    if not reads:
        return ConfidentAnswer(
            answer="?",
            confidence=0.0,
            reads=[],
            is_ambiguous=True,
            reasoning="No reads provided",
        )

    # Aggregate scores by answer
    answer_scores = defaultdict(float)
    for answer, conf in reads:
        answer_scores[answer] += conf * confidence_weight

    # Sort by total confidence
    sorted_answers = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)
    best_answer, best_score = sorted_answers[0]

    # Normalize to [0, 1] range
    max_possible_score = len(reads) * confidence_weight
    normalized_confidence = min(1.0, best_score / max_possible_score)

    # Check for ambiguous tie
    is_ambiguous = False
    if len(sorted_answers) > 1:
        second_best_score = sorted_answers[1][1]
        margin = best_score - second_best_score
        margin_normalized = margin / max_possible_score
        is_ambiguous = margin_normalized < tie_threshold

    return ConfidentAnswer(
        answer=best_answer,
        confidence=normalized_confidence,
        reads=reads,
        is_ambiguous=is_ambiguous,
        reasoning=(
            f"Weighted consensus from {len(reads)} reads. "
            f"Scores: {dict(answer_scores)}. "
            f"{'AMBIGUOUS - close tie.' if is_ambiguous else 'Clear consensus.'}"
        ),
    )


def ensemble_reread_with_confidence(
    reads_list: List[str],
) -> ConfidentAnswer:
    """
    Convert raw read outputs (with confidence markers) into weighted consensus.

    This is the integration point with the grading pipeline.

    Args:
        reads_list: list of response texts from multiple independent reads

    Returns:
        ConfidentAnswer with weighted consensus
    """
    parsed_reads = []

    for read_text in reads_list:
        # Extract primary reading (ignore alternates like "H (or M)")
        primary_match = re.search(r'^([A-Z?])\s*(?:\(|$)', read_text.strip())
        if primary_match:
            answer = primary_match.group(1)
            confidence = extract_confidence_score(read_text)
            parsed_reads.append((answer, confidence))

    if not parsed_reads:
        return ConfidentAnswer(
            answer="?",
            confidence=0.0,
            reads=[],
            is_ambiguous=True,
            reasoning="Failed to parse reads",
        )

    return weighted_consensus_vote(parsed_reads, confidence_weight=1.0, tie_threshold=0.10)
