"""
Calibrated Confidence — Isotonic Regression + Conformal Prediction.

Replaces the fixed HIGH/MEDIUM/LOW bins (0.80/0.70 thresholds) with
empirically calibrated confidence derived from historical grading data.

Two complementary techniques:

1. **Isotonic Regression**: Learns a monotonic mapping from raw model
   confidence to actual accuracy. Halves Expected Calibration Error (ECE)
   at zero marginal inference cost. Requires a calibration set of
   (predicted_score, actual_score) pairs from instructor-verified exams.

2. **Conformal Prediction**: Provides prediction intervals with statistical
   coverage guarantees. "This grade is 22.5 +/- 3 with 90% probability"
   instead of arbitrary bins. Works with any calibration set size n >= 1.

Both calibrate on your 66 graded exams (or any future verified data).
As more exams are graded and verified, calibration improves automatically.

Usage:
    from grader.calibrated_confidence import (
        ConfidenceCalibrator,
        load_calibration_data,
        save_calibration_data,
    )

    cal = ConfidenceCalibrator()
    cal.fit(predicted_scores, actual_scores, max_score=32)

    # For a new student:
    result = cal.predict("student_name", predicted_score=22.5, max_score=32)
    # result.calibrated_confidence = 0.87
    # result.interval = (19.5, 25.5)
    # result.level = "HIGH"

Integration:
    Replaces assign_confidence_level() in confidence_reporter.py.
    The calibrator is fitted once per exam (or loaded from disk), then
    applied to every student in the batch.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Confidence result for one student, with calibrated score and interval."""
    student_name: str
    raw_confidence: float              # original heuristic confidence (0-1)
    calibrated_confidence: float       # after isotonic regression (0-1)
    level: str                         # HIGH / MEDIUM / LOW
    interval_lower: float              # conformal lower bound
    interval_upper: float              # conformal upper bound
    interval_width: float              # upper - lower
    predicted_score: float
    max_score: float
    suspicion_flags: list[str] = field(default_factory=list)

    @property
    def interval_pct(self) -> float:
        """Interval width as percentage of max score."""
        return (self.interval_width / self.max_score * 100) if self.max_score > 0 else 0


class ConfidenceCalibrator:
    """
    Empirical confidence calibration using isotonic regression + conformal prediction.

    Learns from historical (predicted, actual) score pairs. Once fitted,
    produces calibrated confidence scores and prediction intervals for new students.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        high_threshold: float = 0.80,
        medium_threshold: float = 0.65,
    ):
        """
        Args:
            alpha: Miscoverage rate for conformal intervals.
                   0.10 = 90% coverage. With n=66, actual coverage in [88.5%, 91.5%].
            high_threshold: Calibrated confidence >= this → HIGH
            medium_threshold: Calibrated confidence >= this → MEDIUM (else LOW)
        """
        self.alpha = alpha
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        # Isotonic regression state
        self._iso_x: Optional[np.ndarray] = None  # sorted raw error fractions
        self._iso_y: Optional[np.ndarray] = None  # corresponding calibrated accuracy
        self._is_fitted = False

        # Conformal prediction state
        self._conformal_quantile: float = 0.0
        self._residuals: Optional[np.ndarray] = None
        self._n_calibration: int = 0

        # Metadata
        self._max_score_fitted: float = 0.0
        self._mean_error: float = 0.0
        self._median_error: float = 0.0

    def fit(
        self,
        predicted_scores: list[float],
        actual_scores: list[float],
        max_score: float,
    ) -> dict:
        """
        Fit the calibrator on historical grading data.

        Args:
            predicted_scores: Model's first-pass scores for each student
            actual_scores: Instructor-verified final scores
            max_score: Maximum possible score on the exam

        Returns:
            Fit statistics dict
        """
        predicted = np.array(predicted_scores, dtype=float)
        actual = np.array(actual_scores, dtype=float)

        if len(predicted) != len(actual):
            raise ValueError(f"Length mismatch: {len(predicted)} predicted vs {len(actual)} actual")
        if len(predicted) < 2:
            raise ValueError(f"Need at least 2 calibration examples, got {len(predicted)}")

        n = len(predicted)
        self._max_score_fitted = max_score
        self._n_calibration = n

        # ── Isotonic regression ──────────────────────────────────────────
        # Map error magnitude to accuracy.
        # error_frac = |predicted - actual| / max_score  (0 = perfect, 1 = max error)
        # accuracy = 1 - error_frac
        residuals = np.abs(predicted - actual)
        error_fracs = residuals / max_score
        accuracies = 1.0 - error_fracs

        # Sort by error fraction for isotonic regression
        sort_idx = np.argsort(error_fracs)
        sorted_errors = error_fracs[sort_idx]
        sorted_acc = accuracies[sort_idx]

        # Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
        # We want: as raw confidence increases, calibrated confidence should not decrease
        # Here we map: lower error → higher accuracy (monotone decreasing relationship)
        # So we apply PAVA to ensure accuracy is non-increasing as error increases
        self._iso_x, self._iso_y = _pava_decreasing(sorted_errors, sorted_acc)

        # ── Conformal prediction ─────────────────────────────────────────
        self._residuals = residuals
        # Conformal quantile: ceil((n+1)(1-alpha)) / n -th quantile
        q_level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self._conformal_quantile = float(np.quantile(residuals, q_level))

        self._mean_error = float(np.mean(residuals))
        self._median_error = float(np.median(residuals))
        self._is_fitted = True

        stats = {
            "n_calibration": n,
            "max_score": max_score,
            "mean_absolute_error": round(self._mean_error, 2),
            "median_absolute_error": round(self._median_error, 2),
            "conformal_quantile": round(self._conformal_quantile, 2),
            "alpha": self.alpha,
            "coverage_guarantee": f"{(1-self.alpha)*100:.0f}%",
            "ece_before": round(_compute_ece(error_fracs, accuracies), 4),
        }
        logger.info("Calibrator fitted: n=%d, MAE=%.2f, conformal_q=%.2f",
                     n, self._mean_error, self._conformal_quantile)
        return stats

    def predict(
        self,
        student_name: str,
        predicted_score: float,
        max_score: float,
        raw_confidence: Optional[float] = None,
        easy_questions_wrong: Optional[list[str]] = None,
        question_difficulties: Optional[dict[str, float]] = None,
    ) -> CalibrationResult:
        """
        Get calibrated confidence and prediction interval for a student.

        Args:
            student_name: Student identifier
            predicted_score: Model's predicted score
            max_score: Maximum possible score
            raw_confidence: Original heuristic confidence (0-1). If None,
                           estimated from score position in calibration distribution.
            easy_questions_wrong: List of easy questions this student got wrong
            question_difficulties: {Q1: p_value, ...} for suspicion scoring

        Returns:
            CalibrationResult with calibrated confidence, interval, and level
        """
        if not self._is_fitted:
            # Fallback: use fixed thresholds (pre-calibration behavior)
            return self._fallback_predict(student_name, predicted_score, max_score, raw_confidence)

        # ── Conformal interval ───────────────────────────────────────────
        lower = max(0.0, predicted_score - self._conformal_quantile)
        upper = min(max_score, predicted_score + self._conformal_quantile)
        interval_width = upper - lower

        # ── Calibrated confidence ────────────────────────────────────────
        # Base: confidence from interval width (narrow = high confidence)
        interval_confidence = 1.0 - (interval_width / max_score) if max_score > 0 else 0.5

        # If raw_confidence provided, calibrate it via isotonic regression
        if raw_confidence is not None:
            error_est = 1.0 - raw_confidence
            calibrated = float(_isotonic_lookup(self._iso_x, self._iso_y, error_est))
        else:
            calibrated = interval_confidence

        # ── Suspicion adjustment ─────────────────────────────────────────
        suspicion_flags = []
        suspicion_penalty = 0.0

        if easy_questions_wrong and question_difficulties:
            for q in easy_questions_wrong:
                p_val = question_difficulties.get(q, 0.5)
                if p_val > 0.80:
                    suspicion_penalty += 0.05
                    suspicion_flags.append(f"{q} (p={p_val:.2f}, very easy)")
                elif p_val > 0.70:
                    suspicion_penalty += 0.02
                    suspicion_flags.append(f"{q} (p={p_val:.2f}, easy)")

        adjusted = max(0.0, min(1.0, calibrated - suspicion_penalty))

        # ── Level assignment ─────────────────────────────────────────────
        if adjusted >= self.high_threshold:
            level = "HIGH"
        elif adjusted >= self.medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        return CalibrationResult(
            student_name=student_name,
            raw_confidence=raw_confidence if raw_confidence is not None else interval_confidence,
            calibrated_confidence=round(adjusted, 3),
            level=level,
            interval_lower=round(lower, 2),
            interval_upper=round(upper, 2),
            interval_width=round(interval_width, 2),
            predicted_score=predicted_score,
            max_score=max_score,
            suspicion_flags=suspicion_flags,
        )

    def _fallback_predict(
        self, student_name: str, predicted_score: float,
        max_score: float, raw_confidence: Optional[float],
    ) -> CalibrationResult:
        """Pre-calibration fallback using fixed thresholds."""
        conf = raw_confidence if raw_confidence is not None else 0.70
        if conf >= 0.80:
            level = "HIGH"
        elif conf >= 0.70:
            level = "MEDIUM"
        else:
            level = "LOW"

        return CalibrationResult(
            student_name=student_name,
            raw_confidence=conf,
            calibrated_confidence=conf,
            level=level,
            interval_lower=0.0,
            interval_upper=max_score,
            interval_width=max_score,
            predicted_score=predicted_score,
            max_score=max_score,
        )

    def compute_ece(self, n_bins: int = 10) -> Optional[float]:
        """Compute Expected Calibration Error on the calibration set."""
        if not self._is_fitted or self._residuals is None:
            return None
        error_fracs = self._residuals / self._max_score_fitted
        accuracies = 1.0 - error_fracs
        return _compute_ece(error_fracs, accuracies, n_bins)

    def save(self, path: str) -> None:
        """Save calibrator state to JSON for reuse across sessions."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted calibrator")

        data = {
            "alpha": self.alpha,
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
            "conformal_quantile": self._conformal_quantile,
            "n_calibration": self._n_calibration,
            "max_score_fitted": self._max_score_fitted,
            "mean_error": self._mean_error,
            "median_error": self._median_error,
            "iso_x": self._iso_x.tolist() if self._iso_x is not None else [],
            "iso_y": self._iso_y.tolist() if self._iso_y is not None else [],
            "residuals": self._residuals.tolist() if self._residuals is not None else [],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> ConfidenceCalibrator:
        """Load calibrator state from JSON."""
        with open(path) as f:
            data = json.load(f)

        cal = cls(
            alpha=data["alpha"],
            high_threshold=data["high_threshold"],
            medium_threshold=data["medium_threshold"],
        )
        cal._conformal_quantile = data["conformal_quantile"]
        cal._n_calibration = data["n_calibration"]
        cal._max_score_fitted = data["max_score_fitted"]
        cal._mean_error = data["mean_error"]
        cal._median_error = data["median_error"]
        cal._iso_x = np.array(data["iso_x"])
        cal._iso_y = np.array(data["iso_y"])
        cal._residuals = np.array(data["residuals"])
        cal._is_fitted = True

        logger.info("Calibrator loaded from %s (n=%d)", path, cal._n_calibration)
        return cal


# ── Calibration data management ──────────────────────────────────────────────

@dataclass
class CalibrationRecord:
    """One student's calibration data point."""
    student_name: str
    section: str
    predicted_score: float
    actual_score: float
    max_score: float
    exam_name: str = ""


def load_calibration_data(path: str) -> list[CalibrationRecord]:
    """Load calibration records from a JSON file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [CalibrationRecord(**r) for r in data.get("records", [])]


def save_calibration_data(records: list[CalibrationRecord], path: str) -> None:
    """Save calibration records to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"records": [_record_to_dict(r) for r in records]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_calibration_from_excel(
    results: list[dict],
    verified_results: list[dict],
    section: str = "",
    exam_name: str = "",
) -> list[CalibrationRecord]:
    """
    Build calibration records by matching model results to verified results.

    Args:
        results: Model's first-pass grading results (with "name" and "score")
        verified_results: Instructor-corrected final results
        section: Section identifier (T5, W2, W5)
        exam_name: Exam name for metadata

    Returns:
        List of CalibrationRecord for students that appear in both sets
    """
    verified_map = {_normalize_name(r["name"]): r for r in verified_results}
    records = []

    for r in results:
        name_norm = _normalize_name(r.get("name", ""))
        if name_norm in verified_map:
            v = verified_map[name_norm]
            records.append(CalibrationRecord(
                student_name=r.get("name", ""),
                section=section,
                predicted_score=float(r.get("score", 0)),
                actual_score=float(v.get("score", 0)),
                max_score=float(r.get("total", r.get("total_possible", 32))),
                exam_name=exam_name,
            ))

    return records


def format_calibration_summary(cal: ConfidenceCalibrator) -> str:
    """Format calibrator state as a human-readable summary."""
    if not cal._is_fitted:
        return "Calibrator not fitted. No historical data available."

    lines = [
        "CONFIDENCE CALIBRATION SUMMARY",
        "=" * 50,
        f"Calibration set:     {cal._n_calibration} students",
        f"Max score:           {cal._max_score_fitted}",
        f"Mean absolute error: {cal._mean_error:.2f} pts",
        f"Median abs error:    {cal._median_error:.2f} pts",
        f"Conformal quantile:  +/- {cal._conformal_quantile:.2f} pts ({(1-cal.alpha)*100:.0f}% coverage)",
        f"",
        f"Interpretation:",
        f"  A predicted score of X has a {(1-cal.alpha)*100:.0f}% chance of being",
        f"  within +/- {cal._conformal_quantile:.1f} pts of the true score.",
        f"",
        f"Thresholds: HIGH >= {cal.high_threshold}, MEDIUM >= {cal.medium_threshold}",
    ]

    ece = cal.compute_ece()
    if ece is not None:
        lines.append(f"ECE (calibration set): {ece:.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pava_decreasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool Adjacent Violators Algorithm for isotonic (non-increasing) regression.

    Given sorted x values and corresponding y values, returns adjusted y values
    that are monotonically non-increasing (higher error → lower accuracy).

    Pure numpy implementation — no sklearn dependency.
    """
    n = len(y)
    if n == 0:
        return x, y

    # Work on a copy
    result = y.astype(float).copy()
    weights = np.ones(n, dtype=float)

    # Blocks: each block is [start, end, weighted_mean]
    blocks = [[i, i, result[i], 1.0] for i in range(n)]

    # Merge adjacent blocks that violate non-increasing order
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] < blocks[i + 1][2]:  # violation: left < right
            # Merge blocks i and i+1
            w1 = blocks[i][3]
            w2 = blocks[i + 1][3]
            new_mean = (blocks[i][2] * w1 + blocks[i + 1][2] * w2) / (w1 + w2)
            blocks[i] = [blocks[i][0], blocks[i + 1][1], new_mean, w1 + w2]
            blocks.pop(i + 1)
            # Check backward
            if i > 0:
                i -= 1
        else:
            i += 1

    # Write merged means back
    for block in blocks:
        start, end, mean, _ = block
        result[start:end + 1] = mean

    return x, result


def _isotonic_lookup(iso_x: np.ndarray, iso_y: np.ndarray, error_est: float) -> float:
    """Look up calibrated accuracy for a given error estimate via interpolation."""
    if len(iso_x) == 0:
        return 1.0 - error_est

    error_est = max(0.0, min(1.0, error_est))

    # Find position in sorted x
    idx = np.searchsorted(iso_x, error_est)

    if idx == 0:
        return float(iso_y[0])
    if idx >= len(iso_x):
        return float(iso_y[-1])

    # Linear interpolation between adjacent points
    x0, x1 = iso_x[idx - 1], iso_x[idx]
    y0, y1 = iso_y[idx - 1], iso_y[idx]

    if x1 == x0:
        return float(y0)

    t = (error_est - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


def _compute_ece(error_fracs: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Bins predictions by confidence, computes gap between average confidence
    and actual accuracy per bin, returns weighted average.
    Lower ECE = better calibrated. Well-calibrated: ECE < 0.05.
    """
    confidences = 1.0 - error_fracs  # confidence = 1 - error_fraction
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (count / total) * abs(avg_conf - avg_acc)

    return float(ece)


def _normalize_name(name: str) -> str:
    """Normalize student name for matching."""
    return " ".join(name.strip().lower().split())


def _record_to_dict(r: CalibrationRecord) -> dict:
    return {
        "student_name": r.student_name,
        "section": r.section,
        "predicted_score": r.predicted_score,
        "actual_score": r.actual_score,
        "max_score": r.max_score,
        "exam_name": r.exam_name,
    }
