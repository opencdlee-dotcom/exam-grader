"""
Completeness Gate — Structural enforcement that every question has a response.

Prevents the "missed section" failure mode where parallel grading agents skip
entire answer sections (matching, fill-in, FR), causing 10+ point grade reversals.

Works with ExamSchema to know what questions exist, then validates that every
question has either a transcription or an explicit BLANK/UNREADABLE marker.

Usage:
    from grader.completeness_gate import validate_completeness, CompletenessReport

    report = validate_completeness(student_result, schema)
    if not report.is_complete:
        print(report.retry_prompt)  # targeted prompt for missing questions only

Integration:
    Call after grading, before Excel generation. If incomplete, the retry_prompt
    tells the agent exactly which questions to re-read — no full re-read needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from grader.exam_schema import ExamSchema


@dataclass
class SectionCoverage:
    """Coverage report for a single exam section."""
    name: str
    section_type: str
    expected: list[str]
    found: list[str]
    missing: list[str]
    blank: list[str]

    @property
    def coverage_pct(self) -> float:
        if not self.expected:
            return 1.0
        return len(self.found) / len(self.expected)

    @property
    def is_complete(self) -> bool:
        return len(self.missing) == 0


@dataclass
class CompletenessReport:
    """Result of completeness validation for one student."""
    student_name: str
    is_complete: bool
    total_expected: int
    total_found: int
    total_missing: int
    total_blank: int
    coverage_pct: float
    sections: list[SectionCoverage]
    missing_questions: list[str]
    missing_sections: list[str]        # sections with 0% coverage
    partial_sections: list[str]        # sections with >0% but <100% coverage
    retry_prompt: Optional[str] = None

    @property
    def has_missing_sections(self) -> bool:
        """True if any entire section was skipped."""
        return len(self.missing_sections) > 0


def validate_completeness(
    student: dict,
    schema: ExamSchema,
    *,
    answered_key: str = "answered",
    correct_key: str = "correct",
    fr_details_key: str = "fr_details",
) -> CompletenessReport:
    """
    Validate that a student result covers every question in the exam schema.

    Checks three sources for evidence that a question was addressed:
    1. The 'answered' dict (explicit transcriptions)
    2. The 'correct' dict (scoring results — if scored, it was read)
    3. The 'fr_details' string (section log lines mention questions)

    A question is "found" if it appears in ANY of these sources.
    A question is "blank" if it appears but with empty/BLANK/UNREADABLE value.
    A question is "missing" if it appears in NONE of them.

    Args:
        student: Student result dict from grading
        schema: ExamSchema defining all expected questions
        answered_key: Key in student dict for answer transcriptions
        correct_key: Key in student dict for correctness flags
        fr_details_key: Key in student dict for score breakdown logs

    Returns:
        CompletenessReport with coverage details and retry prompt if incomplete
    """
    name = student.get("name", "Unknown")
    answered = student.get(answered_key, {})
    correct = student.get(correct_key, {})
    fr_details = student.get(fr_details_key, "")

    # Normalize keys to strings like "Q5"
    answered_qs = _normalize_q_keys(answered)
    correct_qs = _normalize_q_keys(correct)
    fr_mentioned = _extract_questions_from_fr_details(fr_details)

    # Union of all sources = everything we have evidence for
    all_found = answered_qs | correct_qs | fr_mentioned

    # Identify blanks: present in answered but empty/blank/unreadable
    blank_markers = {"", "?", "blank", "unreadable", "not attempted", "n/a", "none"}
    blank_qs = set()
    for q_key, val in answered.items():
        q_norm = _normalize_q(q_key)
        if q_norm and str(val).strip().lower() in blank_markers:
            blank_qs.add(q_norm)

    # Build per-section coverage
    sections = []
    all_expected = set()
    all_missing = []

    for q_key, sec_type in schema.section_map.items():
        all_expected.add(q_key)

    # Group by section type for reporting
    section_groups: dict[str, list[str]] = {}
    for q_key, sec_type in schema.section_map.items():
        section_groups.setdefault(sec_type, []).append(q_key)

    # Also check FR if schema has fr_total_pts > 0
    if schema.fr_total_pts > 0:
        # FR is tracked via fr_details logs, not individual Q keys
        # Check if FR log line exists
        has_fr_log = _has_section_log(fr_details, "FR")
        fr_section = SectionCoverage(
            name="FR",
            section_type="fr",
            expected=["FR"],
            found=["FR"] if has_fr_log else [],
            missing=[] if has_fr_log else ["FR"],
            blank=[],
        )
        sections.append(fr_section)

    for sec_type, q_keys in sorted(section_groups.items()):
        q_keys_sorted = sorted(q_keys, key=lambda q: int(q[1:]))
        found = [q for q in q_keys_sorted if q in all_found]
        missing = [q for q in q_keys_sorted if q not in all_found]
        blank = [q for q in q_keys_sorted if q in blank_qs]
        all_missing.extend(missing)

        sec_name = sec_type.upper().replace("_", "-")
        sections.append(SectionCoverage(
            name=sec_name,
            section_type=sec_type,
            expected=q_keys_sorted,
            found=found,
            missing=missing,
            blank=blank,
        ))

    # Compute totals
    total_expected = len(all_expected) + (1 if schema.fr_total_pts > 0 else 0)
    total_found = sum(len(s.found) for s in sections)
    total_missing = sum(len(s.missing) for s in sections)
    total_blank = sum(len(s.blank) for s in sections)
    coverage_pct = total_found / total_expected if total_expected > 0 else 1.0

    missing_sections = [s.name for s in sections if s.coverage_pct == 0.0 and len(s.expected) > 0]
    partial_sections = [
        f"{s.name} ({len(s.found)}/{len(s.expected)})"
        for s in sections
        if 0.0 < s.coverage_pct < 1.0
    ]

    is_complete = total_missing == 0

    # Build retry prompt if incomplete
    retry_prompt = None
    if not is_complete:
        retry_prompt = _build_retry_prompt(name, sections, all_missing)

    return CompletenessReport(
        student_name=name,
        is_complete=is_complete,
        total_expected=total_expected,
        total_found=total_found,
        total_missing=total_missing,
        total_blank=total_blank,
        coverage_pct=coverage_pct,
        sections=sections,
        missing_questions=sorted(all_missing, key=lambda q: int(q[1:]) if q.startswith("Q") else 999),
        missing_sections=missing_sections,
        partial_sections=partial_sections,
        retry_prompt=retry_prompt,
    )


def check_batch_completeness(
    results: list[dict],
    schema: ExamSchema,
) -> dict:
    """
    Check completeness for an entire batch. Returns summary + list of incomplete students.

    Returns:
        {
            "all_complete": bool,
            "total_students": int,
            "complete_count": int,
            "incomplete_count": int,
            "incomplete_students": [
                {"name": str, "missing": [str], "missing_sections": [str], "retry_prompt": str}
            ],
            "section_coverage": {section_name: {"complete": int, "total": int, "pct": float}},
        }
    """
    reports = []
    incomplete = []
    section_stats: dict[str, dict] = {}

    for student in results:
        report = validate_completeness(student, schema)
        reports.append(report)

        if not report.is_complete:
            incomplete.append({
                "name": report.student_name,
                "missing": report.missing_questions,
                "missing_sections": report.missing_sections,
                "coverage_pct": report.coverage_pct,
                "retry_prompt": report.retry_prompt,
            })

        # Track per-section coverage across all students
        for sec in report.sections:
            if sec.name not in section_stats:
                section_stats[sec.name] = {"complete": 0, "total": 0}
            section_stats[sec.name]["total"] += 1
            if sec.is_complete:
                section_stats[sec.name]["complete"] += 1

    for stats in section_stats.values():
        stats["pct"] = round(stats["complete"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0

    return {
        "all_complete": len(incomplete) == 0,
        "total_students": len(results),
        "complete_count": len(results) - len(incomplete),
        "incomplete_count": len(incomplete),
        "incomplete_students": incomplete,
        "section_coverage": section_stats,
    }


def format_completeness_report(batch_result: dict) -> str:
    """Format batch completeness check as a human-readable report."""
    lines = ["COMPLETENESS GATE", "=" * 50]

    if batch_result["all_complete"]:
        lines.append(
            f"PASS: All {batch_result['total_students']} students have "
            f"complete coverage across all sections."
        )
    else:
        lines.append(
            f"FAIL: {batch_result['incomplete_count']}/{batch_result['total_students']} "
            f"students have missing questions."
        )
        lines.append("")

        for student in batch_result["incomplete_students"]:
            missing_str = ", ".join(student["missing"])
            lines.append(f"  {student['name']:<30} missing: {missing_str}")
            if student["missing_sections"]:
                lines.append(f"  {'':30} ENTIRE SECTIONS: {', '.join(student['missing_sections'])}")

    lines.append("")
    lines.append("Section Coverage:")
    for sec_name, stats in batch_result["section_coverage"].items():
        bar = "PASS" if stats["pct"] == 100 else f"INCOMPLETE ({stats['pct']}%)"
        lines.append(f"  {sec_name:<12} {stats['complete']}/{stats['total']} students  {bar}")

    lines.append("=" * 50)
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_q(key) -> Optional[str]:
    """Normalize a question key to 'Q5' format."""
    if isinstance(key, int):
        return f"Q{key}"
    s = str(key).strip()
    if s.startswith("Q") and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"Q{s}"
    return None


def _normalize_q_keys(d: dict) -> set[str]:
    """Extract normalized Q-keys from a dict."""
    result = set()
    for key in d:
        q = _normalize_q(key)
        if q:
            result.add(q)
    return result


def _extract_questions_from_fr_details(fr_details: str) -> set[str]:
    """Extract all question references (Q1, Q10, etc.) mentioned in fr_details."""
    import re
    return set(re.findall(r'Q\d+', fr_details or ""))


def _has_section_log(fr_details: str, section_name: str) -> bool:
    """Check if fr_details contains a log line for the named section."""
    for line in (fr_details or "").splitlines():
        if line.strip().upper().startswith(section_name.upper() + ":"):
            return True
    return False


def _build_retry_prompt(student_name: str, sections: list[SectionCoverage],
                         missing_qs: list[str]) -> str:
    """Build a targeted retry prompt for missing questions only."""
    lines = [
        f"INCOMPLETE EXTRACTION for {student_name}",
        f"Missing {len(missing_qs)} questions. Re-read ONLY these:",
        "",
    ]

    for sec in sections:
        if sec.missing:
            q_list = ", ".join(sec.missing)
            lines.append(f"  {sec.name} section: {q_list}")

    lines.extend([
        "",
        "For each missing question, report:",
        "  - Question number",
        "  - What the student wrote (exact text/letter)",
        "  - If blank, write BLANK",
        "  - If unreadable, write UNREADABLE",
        "",
        "Do NOT re-read questions that are already scored.",
        "Do NOT re-grade — just transcribe what is on the page.",
    ])

    return "\n".join(lines)
