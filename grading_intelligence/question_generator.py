"""
Error Pattern -> Question Generation.

Analyzes batch grading results to extract common misconception patterns,
then generates targeted practice questions via LLM that specifically
probe those misconceptions. Exports to QTI 2.1 XML for Canvas import.
"""

from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from grading_intelligence.structured_output import BloomLevel, GradingResult
from grading_intelligence.analytics import ItemAnalysis
from grading_intelligence.llm_executor import execute_prompt, _extract_json


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MisconceptionPattern(BaseModel):
    """A misconception detected across multiple student submissions."""

    concept: str  # e.g., "exocytosis vs endocytosis"
    error_description: str  # "Students confuse the direction of vesicle transport"
    frequency: float  # 0.0-1.0, what fraction of students made this error
    question_numbers: list[int] = Field(default_factory=list)
    bloom_level: BloomLevel | None = None


class GeneratedQuestion(BaseModel):
    """A single practice question generated from misconception data."""

    question_text: str
    correct_answer: str
    distractors: list[str] = Field(default_factory=list)  # For MC questions
    question_type: str = "short_answer"  # "mc", "short_answer", "free_response"
    targets_misconception: str  # Which misconception this addresses
    bloom_level: BloomLevel = BloomLevel.UNDERSTAND
    difficulty: str = "medium"  # "easy", "medium", "hard"
    explanation: str = ""  # Why this answer is correct


class QuestionSet(BaseModel):
    """A collection of generated questions targeting specific misconceptions."""

    title: str
    target_misconceptions: list[str] = Field(default_factory=list)
    questions: list[GeneratedQuestion] = Field(default_factory=list)
    total_points: int = 0


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """\
You are an expert science educator. Given a set of student misconceptions
detected from exam grading, generate targeted practice questions that
specifically test whether students still hold those misconceptions.

=== MISCONCEPTIONS ===
{misconceptions_json}

=== INSTRUCTIONS ===

Generate {num_questions} practice questions with these constraints:
- Allowed question types: {question_types}
- Each question MUST target one specific misconception from the list.
- Prioritize misconceptions with higher frequency.
- For "mc" (multiple choice): include exactly 4 options (1 correct + 3 distractors).
  At least one distractor MUST match the common wrong answer from the misconception.
- For "short_answer": the correct answer should be concise (1-3 words or a number).
- For "free_response": require 2-4 sentences explaining reasoning.
- Vary Bloom's taxonomy levels (remember, understand, apply, analyze).
- Vary difficulty (easy, medium, hard) — weight toward medium.
- Include a brief explanation of why the correct answer is right and why
  the misconception leads to the wrong answer.

=== OUTPUT FORMAT ===
Respond with ONLY valid JSON (no markdown fences):

{{
  "questions": [
    {{
      "question_text": "...",
      "correct_answer": "...",
      "distractors": ["...", "...", "..."],
      "question_type": "mc" or "short_answer" or "free_response",
      "targets_misconception": "concept name from input",
      "bloom_level": "remember" or "understand" or "apply" or "analyze" or "evaluate" or "create",
      "difficulty": "easy" or "medium" or "hard",
      "explanation": "Why this is correct and how the misconception leads astray"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def extract_misconceptions(
    item_analyses: list[ItemAnalysis],
    results: list[GradingResult],
) -> list[MisconceptionPattern]:
    """Extract common misconception patterns from batch grading results.

    Looks at common_errors, missing_terms, and wrong_terms across all
    results to cluster similar errors and rank by frequency.

    Args:
        item_analyses: Per-question statistics from compute_item_analysis.
        results: Individual student grading results.

    Returns:
        List of MisconceptionPattern objects, sorted by frequency (highest first).
    """
    if not results:
        return []

    num_students = len(results)

    # Collect error signals per question
    # error_key -> {count, question_numbers, bloom_level}
    error_clusters: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "question_numbers": set(), "bloom_level": None}
    )

    # From item analyses: use common_errors field
    item_map = {ia.question_number: ia for ia in item_analyses}
    for ia in item_analyses:
        for error_str in ia.common_errors:
            # common_errors are like "wrong: term" or "missing: term"
            key = error_str.strip().lower()
            error_clusters[key]["count"] += 1
            error_clusters[key]["question_numbers"].add(ia.question_number)
            error_clusters[key]["bloom_level"] = ia.bloom_level

    # From individual results: aggregate wrong_terms and missing_terms
    term_counts: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "question_numbers": set(), "bloom_level": None}
    )

    for r in results:
        for q in r.question_results:
            bloom = item_map.get(q.question_number, None)
            bl = bloom.bloom_level if bloom else None

            for term in q.wrong_terms:
                key = f"wrong: {term.lower()}"
                term_counts[key]["count"] += 1
                term_counts[key]["question_numbers"].add(q.question_number)
                term_counts[key]["bloom_level"] = bl

            for term in q.missing_terms:
                key = f"missing: {term.lower()}"
                term_counts[key]["count"] += 1
                term_counts[key]["question_numbers"].add(q.question_number)
                term_counts[key]["bloom_level"] = bl

    # Merge term_counts into error_clusters
    for key, data in term_counts.items():
        if key in error_clusters:
            error_clusters[key]["count"] = max(error_clusters[key]["count"], data["count"])
            error_clusters[key]["question_numbers"] |= data["question_numbers"]
        else:
            error_clusters[key] = data

    # Convert to MisconceptionPattern objects
    patterns: list[MisconceptionPattern] = []
    for error_key, data in error_clusters.items():
        # Parse error type and concept from key
        if ": " in error_key:
            error_type, concept = error_key.split(": ", 1)
        else:
            error_type = "error"
            concept = error_key

        if error_type == "wrong":
            description = f"Students incorrectly used or identified '{concept}'"
        elif error_type == "missing":
            description = f"Students failed to include or mention '{concept}'"
        else:
            description = f"Common error related to '{concept}'"

        frequency = data["count"] / num_students if num_students > 0 else 0.0

        patterns.append(MisconceptionPattern(
            concept=concept,
            error_description=description,
            frequency=round(min(frequency, 1.0), 4),
            question_numbers=sorted(data["question_numbers"]),
            bloom_level=data.get("bloom_level"),
        ))

    # Sort by frequency descending
    patterns.sort(key=lambda p: -p.frequency)
    return patterns


def generate_practice_questions(
    misconceptions: list[MisconceptionPattern],
    num_questions: int = 10,
    question_types: list[str] | None = None,
    provider: str = "claude",
) -> QuestionSet | None:
    """Generate targeted practice questions from misconception patterns via LLM.

    Args:
        misconceptions: Ranked list of misconception patterns.
        num_questions: How many questions to generate.
        question_types: Allowed types (default: ["mc", "short_answer"]).
        provider: LLM provider.

    Returns:
        QuestionSet or None if generation fails.
    """
    if question_types is None:
        question_types = ["mc", "short_answer"]

    if not misconceptions:
        return QuestionSet(
            title="Practice Questions",
            target_misconceptions=[],
            questions=[],
            total_points=0,
        )

    # Serialize misconceptions for the prompt
    misconceptions_data = [
        {
            "concept": m.concept,
            "error_description": m.error_description,
            "frequency": m.frequency,
            "question_numbers": m.question_numbers,
        }
        for m in misconceptions
    ]

    prompt = QUESTION_GENERATION_PROMPT.format(
        misconceptions_json=json.dumps(misconceptions_data, indent=2),
        num_questions=num_questions,
        question_types=", ".join(question_types),
    )

    system = (
        "You are a science education expert who creates assessment items. "
        "Return ONLY valid JSON. No explanation outside the JSON."
    )

    raw = execute_prompt(
        prompt,
        system_prompt=system,
        max_tokens=8192,
        temperature=0.3,
        provider=provider,
    )

    data = _extract_json(raw)
    if data is None:
        return None

    try:
        questions = []
        for q_data in data.get("questions", []):
            # Normalize bloom_level
            bl_str = q_data.get("bloom_level", "understand")
            try:
                bl = BloomLevel(bl_str.lower())
            except ValueError:
                bl = BloomLevel.UNDERSTAND

            questions.append(GeneratedQuestion(
                question_text=q_data.get("question_text", ""),
                correct_answer=q_data.get("correct_answer", ""),
                distractors=q_data.get("distractors", []),
                question_type=q_data.get("question_type", "short_answer"),
                targets_misconception=q_data.get("targets_misconception", ""),
                bloom_level=bl,
                difficulty=q_data.get("difficulty", "medium"),
                explanation=q_data.get("explanation", ""),
            ))

        target_concepts = sorted(set(m.concept for m in misconceptions))

        return QuestionSet(
            title="Targeted Practice: Common Misconceptions",
            target_misconceptions=target_concepts,
            questions=questions,
            total_points=len(questions),  # 1 point each by default
        )
    except (TypeError, ValueError, KeyError):
        return None


def export_qti(question_set: QuestionSet) -> str:
    """Export questions as QTI 2.1 XML for Canvas quiz import.

    Generates a valid IMS QTI 2.1 assessment XML document that Canvas
    can import via the course import tool.

    Args:
        question_set: The questions to export.

    Returns:
        QTI 2.1 XML string.
    """
    # QTI 2.1 namespace
    ns = "http://www.imsglobal.org/xsd/imsqti_v2p1"

    # Build assessment test
    root = ET.Element("assessmentTest", {
        "xmlns": ns,
        "identifier": f"test_{uuid.uuid4().hex[:8]}",
        "title": question_set.title,
    })

    # Test part
    test_part = ET.SubElement(root, "testPart", {
        "identifier": "part1",
        "navigationMode": "linear",
        "submissionMode": "individual",
    })

    # Assessment section
    section = ET.SubElement(test_part, "assessmentSection", {
        "identifier": "section1",
        "title": question_set.title,
        "visible": "true",
    })

    for i, q in enumerate(question_set.questions, 1):
        item_id = f"item_{uuid.uuid4().hex[:8]}"

        if q.question_type == "mc" and q.distractors:
            _add_mc_item(section, item_id, i, q, ns)
        elif q.question_type == "free_response":
            _add_extended_text_item(section, item_id, i, q, ns)
        else:
            # short_answer
            _add_text_entry_item(section, item_id, i, q, ns)

    # Serialize to string
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _add_mc_item(
    parent: ET.Element,
    item_id: str,
    num: int,
    q: GeneratedQuestion,
    ns: str,
) -> None:
    """Add a multiple-choice assessmentItem to the section."""
    item = ET.SubElement(parent, "assessmentItem", {
        "identifier": item_id,
        "title": f"Question {num}",
        "adaptive": "false",
        "timeDependent": "false",
    })

    # Response declaration
    correct_id = "choice_correct"
    resp_decl = ET.SubElement(item, "responseDeclaration", {
        "identifier": "RESPONSE",
        "cardinality": "single",
        "baseType": "identifier",
    })
    correct_resp = ET.SubElement(resp_decl, "correctResponse")
    value_el = ET.SubElement(correct_resp, "value")
    value_el.text = correct_id

    # Outcome declaration
    outcome = ET.SubElement(item, "outcomeDeclaration", {
        "identifier": "SCORE",
        "cardinality": "single",
        "baseType": "float",
    })
    default_val = ET.SubElement(outcome, "defaultValue")
    dv = ET.SubElement(default_val, "value")
    dv.text = "0"

    # Item body
    body = ET.SubElement(item, "itemBody")
    prompt_el = ET.SubElement(body, "choiceInteraction", {
        "responseIdentifier": "RESPONSE",
        "shuffle": "true",
        "maxChoices": "1",
    })
    prompt_text = ET.SubElement(prompt_el, "prompt")
    prompt_text.text = q.question_text

    # Correct choice
    choice = ET.SubElement(prompt_el, "simpleChoice", {"identifier": correct_id})
    choice.text = q.correct_answer

    # Distractor choices
    for j, d in enumerate(q.distractors):
        d_choice = ET.SubElement(prompt_el, "simpleChoice", {
            "identifier": f"choice_{j}",
        })
        d_choice.text = d

    # Response processing
    rp = ET.SubElement(item, "responseProcessing", {
        "template": "http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct",
    })


def _add_text_entry_item(
    parent: ET.Element,
    item_id: str,
    num: int,
    q: GeneratedQuestion,
    ns: str,
) -> None:
    """Add a short-answer (text entry) assessmentItem."""
    item = ET.SubElement(parent, "assessmentItem", {
        "identifier": item_id,
        "title": f"Question {num}",
        "adaptive": "false",
        "timeDependent": "false",
    })

    resp_decl = ET.SubElement(item, "responseDeclaration", {
        "identifier": "RESPONSE",
        "cardinality": "single",
        "baseType": "string",
    })
    correct_resp = ET.SubElement(resp_decl, "correctResponse")
    value_el = ET.SubElement(correct_resp, "value")
    value_el.text = q.correct_answer

    outcome = ET.SubElement(item, "outcomeDeclaration", {
        "identifier": "SCORE",
        "cardinality": "single",
        "baseType": "float",
    })
    default_val = ET.SubElement(outcome, "defaultValue")
    dv = ET.SubElement(default_val, "value")
    dv.text = "0"

    body = ET.SubElement(item, "itemBody")
    p = ET.SubElement(body, "p")
    p.text = q.question_text
    ET.SubElement(body, "textEntryInteraction", {
        "responseIdentifier": "RESPONSE",
        "expectedLength": "20",
    })

    rp = ET.SubElement(item, "responseProcessing", {
        "template": "http://www.imsglobal.org/question/qti_v2p1/rptemplates/match_correct",
    })


def _add_extended_text_item(
    parent: ET.Element,
    item_id: str,
    num: int,
    q: GeneratedQuestion,
    ns: str,
) -> None:
    """Add a free-response (extended text) assessmentItem."""
    item = ET.SubElement(parent, "assessmentItem", {
        "identifier": item_id,
        "title": f"Question {num}",
        "adaptive": "false",
        "timeDependent": "false",
    })

    resp_decl = ET.SubElement(item, "responseDeclaration", {
        "identifier": "RESPONSE",
        "cardinality": "single",
        "baseType": "string",
    })

    outcome = ET.SubElement(item, "outcomeDeclaration", {
        "identifier": "SCORE",
        "cardinality": "single",
        "baseType": "float",
    })
    default_val = ET.SubElement(outcome, "defaultValue")
    dv = ET.SubElement(default_val, "value")
    dv.text = "0"

    body = ET.SubElement(item, "itemBody")
    p = ET.SubElement(body, "p")
    p.text = q.question_text
    ET.SubElement(body, "extendedTextInteraction", {
        "responseIdentifier": "RESPONSE",
        "expectedLines": "5",
    })
