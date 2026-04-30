"""
Answer key consistency validation across grading batches.

Detects when different grading agents used inconsistent answer keys,
and when the FR rubric topic doesn't match what students actually wrote.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

from .cross_validation import ScoringFlag


def load_batch_files(batch_dir: str, pattern: str) -> list[tuple[str, list[dict]]]:
    """Load all batch JSON files from a directory. Returns [(filename, records)]."""
    batch_dir = Path(batch_dir)
    files = sorted(batch_dir.glob(pattern))
    batches = []
    for f in files:
        try:
            records = json.loads(f.read_text())
            batches.append((f.name, records))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "CORRUPTED BATCH FILE %s: %s — students from this file will be "
                "missing from validation. Manual review required.", f, e
            )
            # Inject a sentinel record so callers know a file failed
            batches.append((f.name, [{"_batch_load_error": True, "_file": str(f),
                                       "_error": str(e),
                                       "name": f"[LOAD ERROR: {f.name}]"}]))
    return batches


def infer_key_per_batch(batch_records: list[dict]) -> dict[str, str]:
    """
    Infer the answer key used by a grading batch.

    Strategy: For each question, find the answer that was scored as correct
    by the majority of students in this batch. This works because the agent
    applies the same key to all students in a batch.

    Returns {question: inferred_correct_answer} — may be incomplete if
    raw_reading data is not available.
    """
    # For MC questions (Q1-Q8), we can try to infer from questions_missed
    # and canvas_comment patterns
    key = {}

    # Count how many students got each question right vs wrong
    q_stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for student in batch_records:
        q_scores = student.get("q_scores", {})
        for q, score in q_scores.items():
            if score >= 1:
                q_stats[q]["correct"] += 1
            else:
                q_stats[q]["wrong"] += 1

    # A question where ALL students got 0 might indicate a key error
    for q, stats in q_stats.items():
        if stats["correct"] == 0 and stats["wrong"] >= 3:
            key[q] = "_ALL_WRONG"  # Flag: no student got this right

    return key


def validate_key_consistency(batch_dir: str, pattern: str) -> list[ScoringFlag]:
    """
    Cross-check that all grading batches used the same answer key.

    Compares the scoring patterns between batches. If batch 1 scores
    Q5 as easy (80%+ correct) but batch 3 scores Q5 as hard (20% correct),
    the agents may have used different keys.
    """
    flags = []
    batches = load_batch_files(batch_dir, pattern)

    if len(batches) < 2:
        return flags

    # Calculate per-question correctness rate for each batch
    batch_rates = {}
    for filename, records in batches:
        rates = {}
        for student in records:
            q_scores = student.get("q_scores", {})
            for q, score in q_scores.items():
                if q not in rates:
                    rates[q] = {"correct": 0, "total": 0}
                rates[q]["total"] += 1
                if score >= 1:
                    rates[q]["correct"] += 1

        batch_rates[filename] = {
            q: v["correct"] / v["total"] if v["total"] > 0 else 0
            for q, v in rates.items()
        }

    # Compare rates between batches
    all_questions = set()
    for rates in batch_rates.values():
        all_questions.update(rates.keys())

    for q in sorted(all_questions, key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        rates_for_q = {fname: rates.get(q) for fname, rates in batch_rates.items()
                       if rates.get(q) is not None}

        if len(rates_for_q) < 2:
            continue

        values = list(rates_for_q.values())
        max_rate = max(values)
        min_rate = min(values)

        gap = max_rate - min_rate
        high_batch = max(rates_for_q, key=rates_for_q.get)
        low_batch = min(rates_for_q, key=rates_for_q.get)

        # Tier 1: severe inconsistency — very likely different answer keys used
        if max_rate >= 0.70 and min_rate <= 0.20 and gap >= 0.50:
            flags.append(ScoringFlag(
                question=q,
                student_name="[ALL]",
                flag_type="key_inconsistency",
                severity="HIGH",
                detail=f"{q} correctness rate varies wildly between batches: "
                       f"{high_batch}={max_rate*100:.0f}% vs {low_batch}={min_rate*100:.0f}%. "
                       f"Agents may have used different answer keys.",
                suggested_action=f"Verify answer key for {q} across all batches"
            ))
        # Tier 2: moderate inconsistency — possible rubric drift or batch error
        elif gap >= 0.30:
            flags.append(ScoringFlag(
                question=q,
                student_name="[ALL]",
                flag_type="key_inconsistency_moderate",
                severity="MEDIUM",
                detail=f"{q} correctness gap of {gap*100:.0f}% between batches "
                       f"({high_batch}={max_rate*100:.0f}% vs {low_batch}={min_rate*100:.0f}%). "
                       f"Possible rubric drift — verify key application.",
                suggested_action=f"Compare answer key application across batches for {q}"
            ))

    return flags


def detect_fr_topic_mismatch(results: list[dict],
                              expected_topic: str = "") -> list[ScoringFlag]:
    """
    Detect when the FR rubric topic doesn't match what students wrote.

    If the expected topic is "Griffith" but student fr_full texts consistently
    mention "PCR", "denaturation", "annealing", "extension" — the rubric
    was probably swapped.

    If no expected_topic is given, detect when ALL students wrote about
    the same topic but their FR scores are mostly 0.
    """
    flags = []

    if not results:
        return flags

    # Keywords for topic detection
    topic_keywords = {
        "pcr": ["pcr", "denaturation", "annealing", "extension", "polymerase",
                "primer", "taq", "amplif", "cycle"],
        "griffith": ["griffith", "smooth", "rough", "virulent", "transform",
                    "pneumoniae", "mice", "mouse", "heat-killed", "capsule"],
    }

    # Analyze FR content across all students
    topic_counts = Counter()
    fr_scores = []
    students_with_fr = 0

    for student in results:
        fr_text = (student.get("fr_full", "") or "").lower()
        fr_detail = (student.get("fr_detail", "") or "").lower()
        fr_score = student.get("fr_score", None)
        combined = fr_text + " " + fr_detail

        if not combined.strip() or fr_score is None:
            continue

        students_with_fr += 1
        fr_scores.append(fr_score)

        for topic, keywords in topic_keywords.items():
            matches = sum(1 for kw in keywords if kw in combined)
            if matches >= 2:
                topic_counts[topic] += 1

    if students_with_fr < 3:
        return flags

    # Check for topic mismatch
    if expected_topic:
        expected_lower = expected_topic.lower()
        wrong_topic = None
        for topic, count in topic_counts.items():
            if topic != expected_lower and count >= students_with_fr * 0.70:
                wrong_topic = topic

        if wrong_topic:
            avg_fr = sum(fr_scores) / len(fr_scores) if fr_scores else 0
            flags.append(ScoringFlag(
                question="FR",
                student_name="[ALL]",
                flag_type="fr_topic_mismatch",
                severity="HIGH",
                detail=f"Expected FR topic '{expected_topic}' but "
                       f"{topic_counts[wrong_topic]}/{students_with_fr} students wrote "
                       f"about '{wrong_topic}'. Average FR score: {avg_fr:.1f}/5. "
                       f"Likely rubric swap — grade with correct rubric.",
                suggested_action="Re-grade ALL FR answers with the correct rubric topic"
            ))

    # Also flag if FR scores are very low across the board (possible rubric issue)
    if fr_scores:
        avg_fr = sum(fr_scores) / len(fr_scores)
        zero_count = sum(1 for s in fr_scores if s == 0)

        if avg_fr <= 0.5 and zero_count >= len(fr_scores) * 0.60:
            flags.append(ScoringFlag(
                question="FR",
                student_name="[ALL]",
                flag_type="fr_mass_zero",
                severity="HIGH",
                detail=f"Average FR score is {avg_fr:.1f}/5 with "
                       f"{zero_count}/{len(fr_scores)} students at 0. "
                       f"Possible rubric mismatch or reading failure.",
                suggested_action="Check FR rubric topic matches student answers"
            ))

    return flags


def run_key_validation(batch_dir: str, pattern: str,
                       results: list[dict],
                       expected_fr_topic: str = "") -> list[ScoringFlag]:
    """Run all key validation checks."""
    flags = []
    flags.extend(validate_key_consistency(batch_dir, pattern))
    flags.extend(detect_fr_topic_mismatch(results, expected_fr_topic))

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: (severity_order.get(f.severity, 3), f.question))

    return flags
