"""
Cross-session writer profile persistence for the hub grader.

The hub grades the same students repeatedly across labs. Within a single
submission, ``writer_calibration.py`` already extracts a profile from the
high-trust anchor pre-reads (Q1-3) and injects it into the grading prompt.

This module persists those observations across sessions, keyed by a stable
writer ID derived from the submission filename. The next time the same
student submits, the prior profile is loaded and prepended as a
"PRIOR-SESSION CALIBRATION" block.

Pattern: arXiv MetaWriter (CVPR 2025) — personalization that captures
writer-specific styles by updating <1% of parameters. Here, the analog is a
tiny per-writer prompt prefix updated after each successfully graded
submission.

Storage delegates to ``handwriting_engine.writer_profile_store.WriterProfileStore``
when available, with a small JSON fallback when the engine is not installed
(unit tests, lightweight environments).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Tokens that scanners and the upload UI commonly append to the original
# filename. Stripping them yields a stable ID for the same student across
# multiple submissions. Order matters: regex alternation matches the FIRST
# alternative that fits, so longer / more-specific tokens come before their
# prefixes (``_resubmission`` before ``_resub``, ``_scanner`` before ``_scan``).
_FILENAME_NOISE = re.compile(
    r"(?ix)("
    r"_resubmission|_resub|_camscanner|_scanner|_scan|_clear|"
    r"_attempt[-_]?\d+|_v\d+|_revised|_revise|_final|_corrected|_redo|"
    r"_page[-_]?\d+|_pp?\d+|_section[-_]?\d+|"
    r"_lab[-_]?\d+|_exam[-_]?\d+|_quiz[-_]?\d+|"
    r"_\d{4}[-_]?\d{2}[-_]?\d{2}|_\d{8}|_\d{10,}"
    r")"
)


def derive_writer_id(student_file: str) -> str:
    """
    Derive a stable writer ID from a submission filename.

    Strips common scanner/version/page suffixes and keeps the alphanumeric
    portion. Returns ``"unknown"`` for empty or unparseable input — the
    store will then write to a generic profile that won't bias future runs.
    """
    if not student_file:
        return "unknown"

    stem = Path(student_file).stem
    # Path treats bare-suffix names like ``.pdf`` as having empty suffix and a
    # ".pdf" stem; treat that as no real name to avoid keying everyone with a
    # blank filename to the same writer profile.
    if not stem or stem.startswith("."):
        return "unknown"

    # Run the noise regex repeatedly so chained noise tokens (``_lab_4`` then
    # ``_resubmission``) all get stripped — re.sub is non-recursive so a single
    # pass can leave residual tokens behind once an outer one is removed.
    prev = None
    while prev != stem:
        prev = stem
        stem = _FILENAME_NOISE.sub("", stem)

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")
    return safe.lower() or "unknown"


def _get_store():
    """Return a WriterProfileStore-like object, with a JSON fallback."""
    try:
        from handwriting_engine.writer_profile_store import WriterProfileStore
        return WriterProfileStore()
    except ImportError:
        return _LocalStore()


class _LocalStore:
    """Tiny fallback used when handwriting_engine is not installed."""

    def __init__(self, profiles_dir: Path | None = None):
        self._dir = profiles_dir or (Path.home() / ".handwriting-engine" / "writer-profiles")
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, writer_id: str) -> Path:
        safe_id = "".join(c for c in writer_id if c.isalnum() or c in "-_")
        return self._dir / f"{safe_id}.json"

    def load(self, writer_id: str) -> dict | None:
        path = self._path(writer_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load writer profile %s: %s", writer_id, exc)
            return None

    def save(self, writer_id: str, profile: dict) -> Path:
        path = self._path(writer_id)
        with open(path, "w") as f:
            json.dump({"writer_id": writer_id, **profile}, f, indent=2)
        return path

    def build_calibration_block(self, profile: dict) -> str:
        return _fallback_calibration_block(profile)


def _fallback_calibration_block(profile: dict) -> str:
    """Best-effort calibration block when the engine's renderer is unavailable."""
    if not profile:
        return ""
    lines = ["=== PRIOR-SESSION WRITER CALIBRATION ==="]
    confusion = profile.get("confusion_resolutions") or {}
    for pair, resolution in sorted(confusion.items()):
        lines.append(f"- For this writer, '{pair}' typically reads as '{resolution}'.")
    if profile.get("formation_style"):
        lines.append(f"- Style observed in prior sessions: {profile['formation_style']}")
    notes = profile.get("notes") or []
    for note in notes[:5]:
        lines.append(f"- {note}")
    if len(lines) == 1:
        return ""
    lines.append("Apply these observations consistently — they came from previously-confirmed answers for this same writer.")
    return "\n".join(lines)


def load_session_profile(student_file: str) -> dict | None:
    """Load the persisted writer profile for this student, if one exists."""
    writer_id = derive_writer_id(student_file)
    if writer_id == "unknown":
        return None
    profile = _get_store().load(writer_id)
    if profile:
        logger.info("Loaded persistent writer profile for %s (%s)", writer_id, profile.get("submissions_seen", 0))
    return profile


def build_persistent_calibration_block(profile: dict | None) -> str:
    """Render a persistent profile into a prompt-ready calibration block."""
    if not profile:
        return ""
    store = _get_store()
    block = store.build_calibration_block(profile)
    if not block:
        return ""
    if not block.startswith("=== PRIOR-SESSION"):
        block = "=== PRIOR-SESSION WRITER CALIBRATION ===\n" + block
    return block


def _merge_confusion_resolutions(prior: dict, fresh: dict) -> dict:
    """Per-pair majority vote across sessions; ties keep the prior reading."""
    merged: dict[str, str] = {}
    keys = set(prior) | set(fresh)
    for pair in keys:
        prior_val = prior.get(pair)
        fresh_val = fresh.get(pair)
        if prior_val and fresh_val and prior_val != fresh_val:
            # Disagreement across sessions — prefer the more recent reading
            # (the writer may have changed pen / style), but record both.
            merged[pair] = fresh_val
        else:
            merged[pair] = fresh_val or prior_val
    return merged


def update_session_profile(
    student_file: str,
    confirmed_letters: dict[str, int] | None = None,
    confusion_resolutions: dict[str, str] | None = None,
    formation_style: str | None = None,
    notes: list[str] | None = None,
) -> dict | None:
    """
    Merge this session's observations into the persistent writer profile.

    Args:
        student_file: Submission filename — used to derive a stable writer ID.
        confirmed_letters: ``{letter: count_seen_in_high_confidence_answers}``.
        confusion_resolutions: ``{"B/D": "B", ...}`` — per-pair winning
            interpretation for this writer, derived from confirmed answers.
        formation_style: ``"print"`` / ``"cursive"`` / ``"mixed"``.
        notes: Free-form observations (kept short — top 5 retained).

    Returns:
        The updated profile dict (after merge), or ``None`` if writer ID
        could not be derived.
    """
    writer_id = derive_writer_id(student_file)
    if writer_id == "unknown":
        return None

    store = _get_store()
    prior = store.load(writer_id) or {}

    merged_letters = Counter(prior.get("confirmed_letters") or {})
    merged_letters.update(confirmed_letters or {})

    merged_confusion = _merge_confusion_resolutions(
        prior.get("confusion_resolutions") or {},
        confusion_resolutions or {},
    )

    merged_notes = list(prior.get("notes") or [])
    for note in notes or []:
        if note not in merged_notes:
            merged_notes.append(note)
    # Keep the most recent observations; cap to keep prompt size bounded.
    merged_notes = merged_notes[-5:]

    profile = {
        "writer_id": writer_id,
        "submissions_seen": int(prior.get("submissions_seen", 0)) + 1,
        "confirmed_letters": dict(merged_letters),
        "confusion_resolutions": merged_confusion,
        "formation_style": formation_style or prior.get("formation_style"),
        "notes": merged_notes,
    }

    try:
        store.save(writer_id, profile)
        logger.info(
            "Updated persistent writer profile for %s (now %d submission(s) seen)",
            writer_id, profile["submissions_seen"],
        )
    except OSError as exc:
        logger.warning("Failed to save writer profile for %s: %s", writer_id, exc)
        return profile

    return profile
