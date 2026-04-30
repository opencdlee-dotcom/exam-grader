"""
Rubric drift detection for the grading pipeline.

When a professor edits an answer-key rubric mid-batch, the grading pipeline
silently invalidates the cache and re-grades against the new key. This module
gives the user visibility into systematic score shifts caused by such an edit.

Workflow:
  1. Walk <cache_dir>/grading_results/*.rubric.json snapshots.
  2. Compare each snapshot's rubric_hash against the current rubric's hash
     (computed via the same algorithm as CacheManager._hash_dict).
  3. For drifted snapshots, simulate a re-grade of OBJECTIVE questions only
     using the cached `raw_reading` re-matched against the new rubric — no
     LLM calls are made. Free-response items keep their prior points.
  4. Report deltas via DriftReport.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from grader.answer_matcher import MatchResult, match_single_answer
from grading_intelligence.structured_output import GradingResult

logger = logging.getLogger(__name__)


# ── Hashing ─────────────────────────────────────────────────────────────

def compute_rubric_hash(rubric: dict) -> str:
    """SHA-256 of canonical JSON, first 16 hex chars.

    MUST exactly match the algorithm in hub/grader/checkpoint.py:
    CacheManager._hash_dict:
        json.dumps(rubric, sort_keys=True, default=str).encode()
        -> sha256 -> hexdigest()[:16]
    """
    raw = json.dumps(rubric, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class RubricSnapshot:
    """A point-in-time copy of the rubric used for one cached grading run."""
    grading_key: str         # 16-char cache key the snapshot was stored under
    rubric_hash: str         # 16-char sha256 of the canonical rubric JSON
    rubric: dict             # full extracted_key dict at the time of grading
    student_file: str        # original PDF filename
    snapshot_at: float       # unix timestamp
    # Free-response rubric snapshot (added in FR-drift extension). Older
    # snapshots predate these fields, so they default to None and are treated
    # as "no FR info available".
    free_response_rubric: dict | None = None
    fr_rubric_hash: str | None = None


@dataclass
class DriftItem:
    """A single student whose rubric snapshot differs from the current rubric."""
    student_file: str
    grading_key: str
    prior_hash: str
    current_hash: str
    prior_score: float
    prior_total: float
    simulated_score: float | None    # None if no simulation possible
    simulated_total: float | None
    delta: float | None              # simulated_score - prior_score, None if no sim
    affected_questions: list[int] = field(default_factory=list)
    # Free-response drift signal — purely a flag for the human, since the
    # simulator can't re-grade FR without an LLM call.
    fr_rubric_changed: bool = False
    fr_questions_changed: list[int] = field(default_factory=list)


@dataclass
class DriftReport:
    """Aggregate report of rubric drift across a batch."""
    current_rubric_hash: str
    snapshots_scanned: int
    unchanged_count: int
    drifted: list[DriftItem] = field(default_factory=list)
    # Number of drifted items whose free-response rubric also changed.
    fr_rubric_changed_count: int = 0

    @property
    def has_drift(self) -> bool:
        return len(self.drifted) > 0

    @property
    def mean_delta(self) -> float:
        """Average of simulated deltas across drifted items that have a simulation.

        Returns 0.0 when no drifted items have a delta.
        """
        deltas = [d.delta for d in self.drifted if d.delta is not None]
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)


# ── Snapshot reader ─────────────────────────────────────────────────────

def iter_rubric_snapshots(cache_dir: str | Path) -> list[RubricSnapshot]:
    """Read all <cache_dir>/grading_results/*.rubric.json files.

    Skips malformed entries silently (logs a warning). Each file has schema:
      {
        "rubric_hash": str,
        "rubric": dict,
        "rubric_path": str,
        "student_file": str,
        "snapshot_at": float
      }
    The grading_key is the filename stem with the trailing ".rubric" stripped.
    """
    cache_path = Path(cache_dir)
    results_dir = cache_path / "grading_results"
    if not results_dir.exists():
        return []

    snapshots: list[RubricSnapshot] = []
    for f in sorted(results_dir.glob("*.rubric.json")):
        try:
            data = json.loads(f.read_text())
            # Strip the trailing ".rubric" from "<key>.rubric"
            stem = f.name[: -len(".rubric.json")]
            snapshots.append(RubricSnapshot(
                grading_key=stem,
                rubric_hash=data["rubric_hash"],
                rubric=data["rubric"],
                student_file=data.get("student_file", ""),
                snapshot_at=float(data.get("snapshot_at", 0.0)),
                free_response_rubric=data.get("free_response_rubric"),
                fr_rubric_hash=data.get("fr_rubric_hash"),
            ))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed rubric snapshot %s: %s", f, exc)
            continue
    return snapshots


# ── Objective re-grade simulator (no LLM) ───────────────────────────────

def _index_questions_by_number(rubric: dict) -> dict[int, dict]:
    """Build a lookup of question_number -> question dict from a rubric."""
    out: dict[int, dict] = {}
    for q in rubric.get("questions", []) or []:
        try:
            n = int(q.get("number"))
        except (TypeError, ValueError):
            continue
        out[n] = q
    return out


def simulate_objective_regrade(
    structured_result: Union[dict, GradingResult],
    current_rubric: dict,
) -> tuple[float, float, list[int]]:
    """Re-grade the cached objective questions against the new rubric without
    calling any LLM.

    For each cached QuestionResult:
      - Look up the question in current_rubric by question_number.
      - If absent in the new rubric -> 0 points, included in affected list.
      - If present, run grader.answer_matcher.match_single_answer on the
        cached raw_reading (or student_answer fallback).
        * MATCH      -> award the new question's points
        * MISMATCH   -> 0 points
        * AMBIGUOUS  -> keep prior points_earned (don't make a hard call)
    Free-response items keep their prior points_earned (cannot simulate
    without an LLM).

    Returns (new_total_score, new_total_possible, affected_question_numbers).

    Accepts either a dict (deserialized cache) or a GradingResult model.
    """
    # Coerce to GradingResult so we have a uniform shape
    if isinstance(structured_result, dict):
        result = GradingResult.model_validate(structured_result)
    else:
        result = structured_result

    new_q_index = _index_questions_by_number(current_rubric)
    affected: list[int] = []

    new_score = 0.0
    new_total = 0.0

    # Objective questions
    for q in result.question_results:
        qn = q.question_number
        prior_earned = float(q.points_earned)
        prior_possible = float(q.points_possible)

        new_q = new_q_index.get(qn)
        if new_q is None:
            # Question removed from rubric -> 0 points awarded
            # Total possible drops to 0 for this question.
            if prior_earned != 0:
                affected.append(qn)
            # Contributes 0 to both new_score and new_total
            continue

        new_points = float(new_q.get("points", prior_possible))
        new_total += new_points

        # Use raw_reading if available, else fall back to student_answer
        cached_reading: str = q.raw_reading or q.student_answer or ""

        match, _explanation = match_single_answer(
            cached_reading,
            new_q.get("answers", []) or [],
            new_q.get("alternatives", {}) or {},
            new_q.get("tolerance"),
        )

        if match == MatchResult.MATCH:
            awarded = new_points
        elif match == MatchResult.MISMATCH:
            awarded = 0.0
        else:  # AMBIGUOUS
            # Keep prior points — but cap at the new possible to avoid overflow
            awarded = min(prior_earned, new_points)

        new_score += awarded
        if awarded != prior_earned:
            affected.append(qn)

    # Free-response items: cannot simulate without an LLM, keep prior values
    for fr in result.free_response_results:
        new_score += float(fr.points_earned)
        new_total += float(fr.points_possible)

    # Section results (notebook-style): also keep prior values
    for s in result.section_results:
        new_score += float(s.points_earned)
        new_total += float(s.points_possible)

    return new_score, new_total, affected


# ── Free-response rubric comparison ─────────────────────────────────────

def _fr_questions_list(fr_rubric: dict) -> list[dict] | None:
    """Pick the per-question list out of an FR rubric dict.

    Tries "questions", then "free_response", then "items". Returns None when
    none of those keys are present (or are not lists).
    """
    for key in ("questions", "free_response", "items"):
        val = fr_rubric.get(key)
        if isinstance(val, list):
            return val
    return None


def _q_number(q: dict) -> int | None:
    """Best-effort extraction of a question number from an FR question dict."""
    for key in ("number", "question_number", "q_number"):
        if key in q:
            try:
                return int(q[key])
            except (TypeError, ValueError):
                return None
    return None


def _compare_fr_rubrics(
    snapshot_fr: dict | None,
    current_fr: dict | None,
) -> tuple[bool, list[int]]:
    """Determine whether the free-response rubric changed between two states.

    Returns (changed: bool, affected_fr_question_numbers: list[int]).

    See module docstring / spec for full comparison logic.
    """
    # Treat empty dicts as equivalent to None.
    snap_empty = snapshot_fr is None or snapshot_fr == {}
    curr_empty = current_fr is None or current_fr == {}

    if snap_empty and curr_empty:
        return (False, [])

    if snap_empty != curr_empty:
        # One side has FR rubric, the other doesn't -> changed.
        present = current_fr if snap_empty else snapshot_fr
        qs = _fr_questions_list(present or {}) or []
        affected = [n for n in (_q_number(q) for q in qs) if n is not None]
        return (True, affected)

    # Both present.
    assert snapshot_fr is not None and current_fr is not None
    if compute_rubric_hash(snapshot_fr) == compute_rubric_hash(current_fr):
        return (False, [])

    snap_qs = _fr_questions_list(snapshot_fr)
    curr_qs = _fr_questions_list(current_fr)

    if snap_qs is None or curr_qs is None:
        # Hashes differ but we can't pinpoint per-question deltas.
        return (True, [])

    snap_by_num: dict[int, dict] = {}
    for q in snap_qs:
        n = _q_number(q)
        if n is not None:
            snap_by_num[n] = q

    curr_by_num: dict[int, dict] = {}
    for q in curr_qs:
        n = _q_number(q)
        if n is not None:
            curr_by_num[n] = q

    affected: list[int] = []
    all_nums = set(snap_by_num.keys()) | set(curr_by_num.keys())
    for n in all_nums:
        snap_q = snap_by_num.get(n)
        curr_q = curr_by_num.get(n)
        if snap_q is None or curr_q is None:
            affected.append(n)
            continue
        if compute_rubric_hash(snap_q) != compute_rubric_hash(curr_q):
            affected.append(n)

    return (True, affected)


# ── Top-level entry point ───────────────────────────────────────────────

def _load_grading_result_file(cache_dir: Path, grading_key: str) -> dict | None:
    """Load <cache_dir>/grading_results/<key>.json (the full result file).

    Returns the deserialized dict (with structured_result still as a dict),
    or None if the file is missing or malformed.
    """
    path = cache_dir / "grading_results" / f"{grading_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load grading result %s: %s", path, exc)
        return None


def detect_drift(
    current_rubric: dict,
    cache_dir: str | Path,
    current_fr_rubric: dict | None = None,
) -> DriftReport:
    """Read snapshots, compare each against current_rubric. For drifted
    snapshots, load the corresponding grading_results/<key>.json (NOT the
    .rubric.json), pull the structured_result + total_score + total_possible,
    run simulate_objective_regrade, and produce a DriftReport.

    A snapshot is "drifted" iff
        snapshot.rubric_hash != current_rubric_hash
        OR (snapshot.fr_rubric_hash is not None
            AND current_fr_rubric is not None
            AND snapshot.fr_rubric_hash != compute_rubric_hash(current_fr_rubric)).

    A snapshot is "unchanged" iff neither condition above is true.
    Snapshots whose corresponding grading result is missing on disk are still
    counted as drifted but have simulated_score/simulated_total/delta = None.

    For each drifted snapshot we additionally compute fr_rubric_changed and
    fr_questions_changed via _compare_fr_rubrics; the report's
    fr_rubric_changed_count is the number of drifted items whose
    fr_rubric_changed is True.
    """
    cache_path = Path(cache_dir)
    current_hash = compute_rubric_hash(current_rubric)
    current_fr_hash = (
        compute_rubric_hash(current_fr_rubric)
        if current_fr_rubric is not None else None
    )

    snapshots = iter_rubric_snapshots(cache_path)
    unchanged = 0
    drifted: list[DriftItem] = []

    for snap in snapshots:
        objective_drift = snap.rubric_hash != current_hash
        fr_drift_by_hash = (
            snap.fr_rubric_hash is not None
            and current_fr_hash is not None
            and snap.fr_rubric_hash != current_fr_hash
        )

        if not objective_drift and not fr_drift_by_hash:
            unchanged += 1
            continue

        # Compute FR comparison once per drifted item.
        fr_changed, fr_qs = _compare_fr_rubrics(
            snap.free_response_rubric, current_fr_rubric,
        )

        # Drifted: try to load the full grading result
        result_data = _load_grading_result_file(cache_path, snap.grading_key)
        if result_data is None or "structured_result" not in result_data:
            drifted.append(DriftItem(
                student_file=snap.student_file,
                grading_key=snap.grading_key,
                prior_hash=snap.rubric_hash,
                current_hash=current_hash,
                prior_score=0.0,
                prior_total=0.0,
                simulated_score=None,
                simulated_total=None,
                delta=None,
                affected_questions=[],
                fr_rubric_changed=fr_changed,
                fr_questions_changed=fr_qs,
            ))
            continue

        structured = result_data["structured_result"]
        try:
            prior_result = GradingResult.model_validate(structured)
        except Exception as exc:
            logger.warning(
                "Failed to deserialize structured_result for %s: %s",
                snap.grading_key, exc,
            )
            drifted.append(DriftItem(
                student_file=snap.student_file,
                grading_key=snap.grading_key,
                prior_hash=snap.rubric_hash,
                current_hash=current_hash,
                prior_score=float(result_data.get("total_score", 0.0)),
                prior_total=float(result_data.get("total_possible", 0.0)),
                simulated_score=None,
                simulated_total=None,
                delta=None,
                affected_questions=[],
                fr_rubric_changed=fr_changed,
                fr_questions_changed=fr_qs,
            ))
            continue

        prior_score = float(
            result_data.get("total_score", prior_result.total_score)
        )
        prior_total = float(
            result_data.get("total_possible", prior_result.total_possible)
        )

        if objective_drift:
            sim_score, sim_total, affected = simulate_objective_regrade(
                prior_result, current_rubric,
            )
            delta = sim_score - prior_score
        else:
            # Only FR drift — no objective re-grade is meaningful.
            sim_score = prior_score
            sim_total = prior_total
            delta = 0.0
            affected = []

        drifted.append(DriftItem(
            student_file=snap.student_file or prior_result.student_file,
            grading_key=snap.grading_key,
            prior_hash=snap.rubric_hash,
            current_hash=current_hash,
            prior_score=prior_score,
            prior_total=prior_total,
            simulated_score=sim_score,
            simulated_total=sim_total,
            delta=delta,
            affected_questions=affected,
            fr_rubric_changed=fr_changed,
            fr_questions_changed=fr_qs,
        ))

    fr_changed_count = sum(1 for d in drifted if d.fr_rubric_changed)

    return DriftReport(
        current_rubric_hash=current_hash,
        snapshots_scanned=len(snapshots),
        unchanged_count=unchanged,
        drifted=drifted,
        fr_rubric_changed_count=fr_changed_count,
    )


# Suppress unused-import warning for `time` when none of the helpers above need it.
_ = time
