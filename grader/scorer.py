"""
grader/scorer.py — backward-compatibility shim.

All grading functions have been moved to focused modules:
  - grader.pipeline: per-student grading
  - grader.batch_processor: batch orchestration
  - grader.analytics_runner: post-grading analytics

This shim re-exports everything for backward compatibility.
Remove in a future version once all callers are updated.
"""

from grader.pipeline import (  # noqa: F401
    apply_late_penalty,
    apply_curve,
    grade_submission,
    grade_exam_submission,
)

from grader.batch_processor import (  # noqa: F401
    grade_batch,
    grade_exam_batch,
)

from grader.analytics_runner import (  # noqa: F401
    _run_batch_analytics,
    _run_integrity_checks,
)
