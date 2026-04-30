"""
Automated spot-check protocol.

Replaces manual spot-checks with structured risk scoring, smart question
selection, and verification prompt generation. Designed for use with
parallel Claude Code agents in the direct grading workflow.

Usage:
    from grader.verification.spot_checker import build_spot_checks
    checks = build_spot_checks(results, verification_flags)
    # Each check contains: student info, questions to verify, prompt template
"""

import hashlib
import random
import re
from dataclasses import dataclass, field

from .cross_validation import ScoringFlag


@dataclass
class SpotCheck:
    student_name: str
    risk_score: int
    risk_reasons: list[str]
    questions_to_verify: list[str]
    verification_prompt: str
    priority: str = "MEDIUM"
    recommended_source: str = "original scan"
    student_record: dict = field(default_factory=dict)


def calculate_risk_score(student: dict,
                          verification_flags: list[ScoringFlag]) -> tuple[int, list[str]]:
    """
    Calculate a cumulative risk score for a student.

    Returns (score, [reasons]).
    Students with risk >= 3 should be spot-checked.
    """
    score = 0
    reasons = []
    name = student.get("name", "")

    # Normalize flags: parse_grade_text stores flags as a comma-joined string;
    # future code may store as a list. Handle both to avoid silent mismatches.
    flags_raw = student.get("flags", "")
    if isinstance(flags_raw, list):
        flags = [f.strip() for f in flags_raw if f.strip()]
    else:
        flags = [f.strip() for f in str(flags_raw).split(",") if f.strip()]

    # Check student-level flags using exact list membership
    if "BELOW 50%" in flags:
        score += 3
        reasons.append("BELOW 50%")

    if "SHIFT DETECTED" in flags:
        score += 3
        reasons.append("SHIFT DETECTED")

    if "FAINT CONTENT" in flags:
        score += 2
        reasons.append("FAINT CONTENT")

    if "NAME UNCERTAIN" in flags:
        score += 1
        reasons.append("NAME UNCERTAIN")

    # HUMAN REVIEW may appear with trailing text (e.g. "HUMAN REVIEW flag")
    if any("HUMAN REVIEW" in f for f in flags):
        score += 1
        reasons.append("HUMAN REVIEW flag")

    # Check verification flags for this student
    student_flags = [f for f in verification_flags if f.student_name == name]
    high_flags = [f for f in student_flags if f.severity == "HIGH"]
    medium_flags = [f for f in student_flags if f.severity == "MEDIUM"]

    if high_flags:
        score += 3
        reasons.append(f"{len(high_flags)} HIGH verification flags")

    if medium_flags:
        score += 1
        reasons.append(f"{len(medium_flags)} MEDIUM verification flags")

    # Check for zero-block in matching section
    q_scores = student.get("q_scores", {})
    matching_zeros = sum(1 for i in range(9, 19)
                        if q_scores.get(f"Q{i}", None) == 0)
    if matching_zeros >= 8:
        score += 2
        reasons.append(f"Matching section: {matching_zeros}/10 zeros")

    return score, reasons


def select_questions(student: dict,
                     verification_flags: list[ScoringFlag],
                     target_count: int = 7,
                     seed: str | None = None) -> list[str]:
    """
    Select 5-8 questions for spot-checking using priority-based selection.

    Priority order:
    1. Questions flagged by verification pipeline (cross-validation, IRT)
    2. All zero-scored matching questions (Q9-Q18)
    3. 2-3 random "correct" questions (check for overcredit)
    4. Random wrong questions for undercredit check

    Uses deterministic seeded RNG per student name for reproducible audits.
    """
    name = student.get("name", "Unknown")
    q_scores = student.get("q_scores", {})

    # Seed RNG with student name for reproducibility
    if seed is None:
        seed = name
    rng = random.Random(hashlib.md5(seed.encode()).hexdigest())

    selected = []
    selected_set = set()

    def add(q):
        if q not in selected_set and q in q_scores:
            selected.append(q)
            selected_set.add(q)

    # Priority 1: Verification-flagged questions
    student_flags = [f for f in verification_flags
                     if f.student_name == name and re.match(r"Q\d+$", f.question)]
    for f in student_flags:
        add(f.question)

    # Priority 2: Zero-scored matching questions
    for i in range(9, 19):
        q = f"Q{i}"
        if q_scores.get(q, None) == 0:
            add(q)
        if len(selected) >= target_count:
            break

    # Priority 3: Random correct questions (overcredit check)
    correct_qs = [q for q, s in q_scores.items() if s >= 1]
    rng.shuffle(correct_qs)
    for q in correct_qs[:3]:
        add(q)
        if len(selected) >= target_count:
            break

    # Priority 4: Random wrong questions (undercredit check)
    wrong_qs = [q for q, s in q_scores.items() if s == 0 and q not in selected_set]
    rng.shuffle(wrong_qs)
    for q in wrong_qs[:2]:
        add(q)
        if len(selected) >= target_count:
            break

    # Sort by question number
    selected.sort(key=lambda q: int(re.sub(r'\D', '', q) or 0))

    return selected[:target_count]


def build_verification_prompt(student: dict,
                               questions: list[str],
                               page_info: str = "") -> str:
    """
    Build a focused verification prompt for a spot-check agent.

    The prompt asks the agent to re-read only the selected questions
    and compare against the recorded scores.
    """
    name = student.get("name", "Unknown")
    q_scores = student.get("q_scores", {})
    total = student.get("total", 0)

    # Build question details
    q_details = []
    for q in questions:
        recorded = q_scores.get(q, "N/A")
        q_details.append(f"  - {q}: recorded score = {recorded}")

    q_detail_str = "\n".join(q_details)
    q_list = ", ".join(questions)

    total_possible = student.get("total_possible", total)  # use explicit field if set, else fall back to total
    prompt = f"""You are verifying specific questions for student "{name}" (recorded total: {total}/{total_possible}).

Read the student's exam pages and verify ONLY these questions: {q_list}

For each question, report:
1. What the student actually wrote (exact text or letter)
2. What the correct answer is
3. Whether the recorded score is correct

Recorded scores to verify:
{q_detail_str}

{f'Page location: {page_info}' if page_info else ''}

Output format (JSON):
{{
  "student_name": "{name}",
  "verifications": {{
    "Q5": {{
      "student_wrote": "B",
      "correct_answer": "D",
      "recorded_score": 1,
      "verified_score": 0,
      "match": false,
      "note": "Student wrote B, key is D — overcredit"
    }}
  }},
  "mismatches": 0,
  "recommendation": "NONE"  // or "SPOT_CHECK_MORE" or "FULL_REGRADE"
}}

Rules:
- Report ONLY the questions listed above
- If recorded_score matches verified_score → match: true
- If they differ → match: false, explain in note
- recommendation: NONE if 0-1 mismatches, SPOT_CHECK_MORE if 2, FULL_REGRADE if 3+"""

    return prompt


def build_spot_checks(results: list[dict],
                       verification_flags: list[ScoringFlag],
                       risk_threshold: int = 3) -> list[SpotCheck]:
    """
    Build spot-check tasks for all students above the risk threshold.

    Returns a list of SpotCheck objects, each containing the student info,
    selected questions, and a ready-to-use verification prompt.
    """
    checks = []

    for student in results:
        risk_score, risk_reasons = calculate_risk_score(student, verification_flags)

        if risk_score < risk_threshold:
            continue

        questions = select_questions(student, verification_flags)

        if not questions:
            continue

        # Build page info from student record
        pages = student.get("pages", student.get("student_file", ""))
        page_info = f"pages: {pages}" if pages else ""

        prompt = build_verification_prompt(student, questions, page_info)
        priority = "HIGH" if risk_score >= 6 else "MEDIUM"
        recommended_source = "alternate scan if available" if any(
            reason in {"FAINT CONTENT", "SHIFT DETECTED"} for reason in risk_reasons
        ) else "original scan"

        checks.append(SpotCheck(
            student_name=student.get("name", "Unknown"),
            risk_score=risk_score,
            risk_reasons=risk_reasons,
            questions_to_verify=questions,
            verification_prompt=prompt,
            priority=priority,
            recommended_source=recommended_source,
            student_record=student,
        ))

    # Sort by risk score (highest first)
    checks.sort(key=lambda c: -c.risk_score)

    return checks


def resolve_discrepancies(original: dict,
                          verification: dict,
                          threshold: int = 2) -> dict:
    """
    Compare original scores to verification scores and determine action.

    Args:
        original: Original student grade record
        verification: Verification agent output (parsed JSON)
        threshold: Number of mismatches to trigger full regrade

    Returns:
        {
            "action": "NONE" | "APPLY_CORRECTIONS" | "FULL_REGRADE",
            "corrections": {question: new_score},
            "mismatch_count": int,
            "details": [str]
        }
    """
    verifications = verification.get("verifications", {})
    corrections = {}
    details = []
    mismatch_count = 0

    q_scores = original.get("q_scores", {})

    for q, v in verifications.items():
        if not v.get("match", True):
            mismatch_count += 1
            recorded = v.get("recorded_score", q_scores.get(q))
            verified = v.get("verified_score", recorded)
            corrections[q] = verified
            details.append(
                f"{q}: {recorded}→{verified} ({v.get('note', 'mismatch')})"
            )

    if mismatch_count >= threshold:
        action = "FULL_REGRADE"
    elif mismatch_count > 0:
        action = "APPLY_CORRECTIONS"
    else:
        action = "NONE"

    return {
        "action": action,
        "corrections": corrections,
        "mismatch_count": mismatch_count,
        "details": details,
    }
