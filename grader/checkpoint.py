"""
Checkpoint / save-point system for the grading pipeline.

3-tier content-addressed cache:
  Tier 1 — Grading results (full result dicts with GradingResult)
  Tier 2 — Page images (converted + enhanced JPEGs with quality metadata)
  Tier 3 — Answer keys (extracted answer key dicts)

Cache keys are SHA-256 truncated to 16 hex chars, derived from
the actual file bytes + relevant settings so any change auto-invalidates.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

from grading_intelligence.structured_output import GradingResult

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = ".grader_cache"

# Module-level singleton
_manager = None


def get_cache_manager(enabled: bool = True, cache_dir: str = _DEFAULT_CACHE_DIR) -> "CacheManager":
    """Get or create the singleton CacheManager."""
    global _manager
    if _manager is None or _manager.enabled != enabled:
        _manager = CacheManager(cache_dir=cache_dir, enabled=enabled)
    return _manager


class CacheManager:
    """Content-addressed cache for the grading pipeline."""

    TIER_DIRS = {
        "results": "grading_results",
        "images": "page_images",
        "keys": "answer_keys",
    }

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._pdf_hash_cache: dict[str, str] = {}

    def _ensure_dir(self, tier: str) -> Path:
        d = self.cache_dir / self.TIER_DIRS[tier]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Hashing ──────────────────────────────────────────────────

    def _hash_pdf(self, pdf_path: str) -> str:
        """SHA-256 of PDF file bytes, memoized per session."""
        abs_path = os.path.abspath(pdf_path)
        if abs_path not in self._pdf_hash_cache:
            h = hashlib.sha256(Path(abs_path).read_bytes()).hexdigest()[:16]
            self._pdf_hash_cache[abs_path] = h
        return self._pdf_hash_cache[abs_path]

    def _hash_dict(self, d: dict) -> str:
        """Deterministic hash of a JSON-serializable dict."""
        raw = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def answer_key_key(self, pdf_path: str, enhance: bool) -> str:
        """Cache key for Tier 3 — extracted answer key."""
        pdf_h = self._hash_pdf(pdf_path)
        settings_h = self._hash_dict({"enhance": enhance})
        combined = f"{pdf_h}:{settings_h}".encode()
        return hashlib.sha256(combined).hexdigest()[:16]

    def images_key(self, pdf_path: str, dpi: int, jpeg_quality: int,
                   max_long_side: int, enhance: bool, strategy: str) -> str:
        """Cache key for Tier 2 — converted page images."""
        pdf_h = self._hash_pdf(pdf_path)
        settings_h = self._hash_dict({
            "dpi": dpi, "jpeg_quality": jpeg_quality,
            "max_long_side": max_long_side, "enhance": enhance,
            "strategy": strategy,
        })
        combined = f"{pdf_h}:{settings_h}".encode()
        return hashlib.sha256(combined).hexdigest()[:16]

    def grading_key(self, pdf_path: str, extracted_key: dict, settings: dict) -> str:
        """
        Cache key for Tier 1 — grading result.

        Settings should include: enhance, strategy, ensemble_strategy,
        enhanced_reading, answer_page_numbers, free_response_rubric, model.
        """
        pdf_h = self._hash_pdf(pdf_path)
        key_h = self._hash_dict(extracted_key)
        settings_h = self._hash_dict(settings)
        combined = f"{pdf_h}:{key_h}:{settings_h}".encode()
        return hashlib.sha256(combined).hexdigest()[:16]

    # ── Tier 3: Answer Keys ─────────────────────────────────────

    def get_answer_key(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self.cache_dir / self.TIER_DIRS["keys"] / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def store_answer_key(self, key: str, data: dict, source_pdf: str = ""):
        if not self.enabled:
            return
        d = self._ensure_dir("keys")
        (d / f"{key}.json").write_text(json.dumps(data, indent=2, default=str))
        (d / f"{key}.meta.json").write_text(json.dumps({
            "source_pdf": os.path.basename(source_pdf),
            "stored_at": time.time(),
            "question_count": len(data.get("questions", [])),
            "total_points": data.get("total_points", 0),
        }))

    # ── Tier 2: Page Images ─────────────────────────────────────

    def images_dir(self, key: str) -> str:
        return str(self.cache_dir / self.TIER_DIRS["images"] / key)

    def get_page_images(self, key: str) -> list[dict] | None:
        """Load cached page images. Returns list of page dicts or None."""
        if not self.enabled:
            return None
        d = self.cache_dir / self.TIER_DIRS["images"] / key
        manifest = d / "manifest.json"
        if not manifest.exists():
            return None
        try:
            meta = json.loads(manifest.read_text())
            pages = []
            for pg in meta["pages"]:
                img_path = d / pg["filename"]
                if not img_path.exists():
                    return None  # Corrupted cache entry
                pages.append({
                    "page_number": pg["page_number"],
                    "path": str(img_path),
                    "quality": pg.get("quality"),
                    "auto_enhanced": pg.get("auto_enhanced", False),
                })
            return pages
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def store_page_images(self, key: str, pages: list[dict], source_dir: str):
        """Copy page images to cache and save manifest with quality data."""
        if not self.enabled:
            return
        d = self._ensure_dir("images") / key
        d.mkdir(parents=True, exist_ok=True)

        manifest_pages = []
        for pg in pages:
            src = pg["path"]
            filename = os.path.basename(src)
            dst = d / filename
            if not dst.exists():
                shutil.copy2(src, dst)
            manifest_pages.append({
                "page_number": pg.get("page_number"),
                "filename": filename,
                "quality": pg.get("quality"),
                "auto_enhanced": pg.get("auto_enhanced", False),
            })

        (d / "manifest.json").write_text(json.dumps({
            "page_count": len(pages),
            "stored_at": time.time(),
            "pages": manifest_pages,
        }, indent=2))

    # ── Tier 1: Grading Results ─────────────────────────────────

    def get_grading_result(self, key: str) -> dict | None:
        """Load a cached grading result. Reconstructs GradingResult from JSON."""
        if not self.enabled:
            return None
        path = self.cache_dir / self.TIER_DIRS["results"] / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            # Reconstruct GradingResult from serialized dict
            if "structured_result" in data and isinstance(data["structured_result"], dict):
                data["structured_result"] = GradingResult.model_validate(data["structured_result"])
            return data
        except (json.JSONDecodeError, OSError, Exception):
            return None

    def store_grading_result(self, key: str, result: dict):
        """Store a grading result. Serializes GradingResult to JSON."""
        if not self.enabled:
            return
        d = self._ensure_dir("results")
        # Deep copy to avoid mutating the original
        data = {}
        for k, v in result.items():
            if k == "structured_result" and isinstance(v, GradingResult):
                data[k] = v.model_dump(mode="json")
            elif k == "images_dir":
                continue  # Don't cache temp dir paths
            elif k.startswith("_"):
                continue  # Skip internal keys like _extracted_key, _answer_key
            else:
                data[k] = v
        (d / f"{key}.json").write_text(json.dumps(data, indent=2, default=str))

    # ── Rubric Snapshots ────────────────────────────────────────

    def store_rubric_snapshot(
        self,
        grading_key: str,
        rubric: dict,
        rubric_path: str = "",
        student_file: str = "",
        free_response_rubric: dict | None = None,
    ) -> None:
        """Persist the rubric used for a given grading_key.

        Writes <cache_dir>/grading_results/<grading_key>.rubric.json.
        No-op when cache is disabled.

        When ``free_response_rubric`` is None or empty, both
        ``free_response_rubric`` and ``fr_rubric_hash`` are written as null.
        """
        if not self.enabled:
            return
        d = self._ensure_dir("results")
        if free_response_rubric:
            fr_rubric_value = free_response_rubric
            fr_rubric_hash = self._hash_dict(free_response_rubric)
        else:
            fr_rubric_value = None
            fr_rubric_hash = None
        snapshot = {
            "rubric_hash": self._hash_dict(rubric),
            "rubric": rubric,
            "free_response_rubric": fr_rubric_value,
            "fr_rubric_hash": fr_rubric_hash,
            "rubric_path": rubric_path,
            "student_file": student_file,
            "snapshot_at": time.time(),
        }
        (d / f"{grading_key}.rubric.json").write_text(
            json.dumps(snapshot, indent=2, default=str)
        )

    def get_rubric_snapshot(self, grading_key: str) -> dict | None:
        """Return the snapshot dict, or None if missing/malformed."""
        path = self.cache_dir / self.TIER_DIRS["results"] / f"{grading_key}.rubric.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def iter_rubric_snapshots(self) -> list[dict]:
        """Return every persisted snapshot dict. Each dict additionally has a
        "grading_key" field (filename stem with the .rubric suffix removed) so
        callers can reconcile against the corresponding grading_results/<key>.json.

        Skip and log a warning for malformed JSON files. Return [] when the
        grading_results dir does not exist.
        """
        results_dir = self.cache_dir / self.TIER_DIRS["results"]
        if not results_dir.exists():
            return []
        snapshots: list[dict] = []
        for path in results_dir.glob("*.rubric.json"):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping malformed rubric snapshot %s: %s", path, exc)
                continue
            # Strip ".rubric" suffix from stem (path.stem strips only ".json")
            stem = path.stem
            if stem.endswith(".rubric"):
                stem = stem[: -len(".rubric")]
            data["grading_key"] = stem
            snapshots.append(data)
        return snapshots

    # ── Management ──────────────────────────────────────────────

    def status(self) -> dict:
        """Return cache statistics."""
        stats = {"total_size_mb": 0.0, "tiers": {}}
        for tier_name, subdir in self.TIER_DIRS.items():
            tier_path = self.cache_dir / subdir
            if not tier_path.exists():
                stats["tiers"][tier_name] = {"entries": 0, "size_mb": 0.0}
                continue
            size = sum(f.stat().st_size for f in tier_path.rglob("*") if f.is_file())
            if tier_name == "images":
                entries = sum(1 for d in tier_path.iterdir() if d.is_dir())
            else:
                entries = sum(1 for f in tier_path.glob("*.json") if not f.name.endswith(".meta.json"))
            size_mb = round(size / (1024 * 1024), 2)
            stats["tiers"][tier_name] = {"entries": entries, "size_mb": size_mb}
            stats["total_size_mb"] += size_mb
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    def clear(self, tier: str = "all", older_than_days: int | None = None):
        """Clear cached data. tier='all' or 'results'|'images'|'keys'."""
        tiers = self.TIER_DIRS.keys() if tier == "all" else [tier]
        cutoff = time.time() - (older_than_days * 86400) if older_than_days else None

        removed = 0
        for t in tiers:
            tier_path = self.cache_dir / self.TIER_DIRS[t]
            if not tier_path.exists():
                continue
            if cutoff is None:
                shutil.rmtree(tier_path)
                removed += 1
            else:
                for item in tier_path.iterdir():
                    if item.stat().st_mtime < cutoff:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        removed += 1
        return removed
