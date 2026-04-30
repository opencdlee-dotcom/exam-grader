"""
Verification pipeline orchestrator.

Runs all post-grading checks on JSON batch files and produces a
VerificationReport with status (CLEAN/REVIEW/INVESTIGATE) and
actionable flags.

Usage:
    # As module
    from grader.verification.run_verification import run_all_checks
    report = run_all_checks("~/Desktop/ex/ex2T5", "grades_T5_b*.json")

    # As standalone script
    python -m grader.verification.run_verification --dir ~/Desktop/ex/ex2T5 --pattern "grades_T5_b*.json"
"""

import json
import glob
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from .cross_validation import ScoringFlag, run_cross_validation
from .outlier_detection import run_outlier_detection
from .key_validator import run_key_validation
from .calc_validator import run_calc_validation


@dataclass
class VerificationReport:
    status: str  # "CLEAN", "REVIEW", "INVESTIGATE"
    flags: list[ScoringFlag] = field(default_factory=list)
    summary: str = ""

    @property
    def flag_count(self) -> int:
        return len(self.flags)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "HIGH")

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "MEDIUM")

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "LOW")

    def format_summary(self) -> str:
        lines = [
            f"Verification Status: {self.status}",
            f"Total flags: {self.flag_count} "
            f"(HIGH: {self.high_count}, MEDIUM: {self.medium_count}, LOW: {self.low_count})",
            "",
        ]

        if self.flags:
            lines.append("Flags:")
            for f in self.flags:
                lines.append(
                    f"  [{f.severity}] {f.flag_type} | {f.student_name} | "
                    f"{f.question}: {f.detail}"
                )
                if f.suggested_action:
                    lines.append(f"    → {f.suggested_action}")

        return "\n".join(lines)


def load_all_results(batch_dir: str, pattern: str) -> list[dict]:
    """Load and merge all batch JSON files into a single results list.

    Corrupted or unreadable files inject a sentinel record so downstream checks
    can report the gap rather than silently treating it as zero students.
    """
    batch_dir = Path(batch_dir)
    files = sorted(batch_dir.glob(pattern))
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            results.extend(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "CORRUPTED BATCH FILE %s: %s — students from this file excluded "
                "from all verification checks. Manual review required.", f, e
            )
            results.append({
                "_batch_load_error": True,
                "_file": str(f),
                "_error": str(e),
                "name": f"[LOAD ERROR: {f.name}]",
            })
    return results


def run_all_checks(batch_dir: str, pattern: str,
                   total_possible: int = 32,
                   expected_fr_topic: str = "",
                   matching_range: tuple[int, int] = (9, 18)) -> VerificationReport:
    """
    Run all verification checks and produce a report.

    Args:
        batch_dir: Directory containing grade batch JSON files
        pattern: Glob pattern for batch files (e.g., "grades_T5_b*.json")
        total_possible: Total points possible on the exam
        expected_fr_topic: Expected FR topic ("griffith" or "pcr")
        matching_range: Question range for matching section (start, end)

    Returns:
        VerificationReport with CLEAN/REVIEW/INVESTIGATE status
    """
    results = load_all_results(batch_dir, pattern)

    if not results:
        return VerificationReport(
            status="CLEAN",
            summary="No results to verify."
        )

    all_flags = []

    # 1A: Cross-student answer consistency
    all_flags.extend(run_cross_validation(
        results,
        section_start=matching_range[0],
        section_end=matching_range[1]
    ))

    # 1B: Statistical outlier detection
    all_flags.extend(run_outlier_detection(results, total_possible=total_possible))

    # 1C: Answer key consistency across batches
    all_flags.extend(run_key_validation(
        batch_dir, pattern, results, expected_fr_topic
    ))

    # 1D: Calculation scoring consistency
    all_flags.extend(run_calc_validation(results))

    # Determine status
    high_flags = sum(1 for f in all_flags if f.severity == "HIGH")
    total_flags = len(all_flags)

    if high_flags > 0 or total_flags >= 5:
        status = "INVESTIGATE"
    elif total_flags > 0:
        status = "REVIEW"
    else:
        status = "CLEAN"

    report = VerificationReport(status=status, flags=all_flags)
    report.summary = report.format_summary()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run post-grading verification on JSON batch files"
    )
    parser.add_argument("--dir", required=True, help="Directory with batch JSON files")
    parser.add_argument("--pattern", required=True, help="Glob pattern for batch files")
    parser.add_argument("--total", type=int, default=32, help="Total points possible")
    parser.add_argument("--fr-topic", default="", help="Expected FR topic (griffith/pcr)")
    parser.add_argument("--matching-start", type=int, default=9)
    parser.add_argument("--matching-end", type=int, default=18)

    args = parser.parse_args()

    report = run_all_checks(
        batch_dir=args.dir,
        pattern=args.pattern,
        total_possible=args.total,
        expected_fr_topic=args.fr_topic,
        matching_range=(args.matching_start, args.matching_end)
    )

    print(report.format_summary())
    return 0 if report.status == "CLEAN" else 1


if __name__ == "__main__":
    exit(main())
