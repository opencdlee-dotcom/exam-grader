"""
Cross-process provider exhaustion store.

Backs the in-process exhaustion tracking with SQLite so multi-worker uvicorn
deployments stay consistent. Without this, each worker independently learned
"claude is out" and /providers/reset only cleared one worker's view.

Single table, three columns. WAL mode for concurrent reader/writer safety.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _resolve_db_path() -> Path:
    """Locate a writable SQLite path for the exhaustion table.

    Prefers the directory of DATABASE_URL when it is sqlite (so all hub state
    lives together). Falls back to the hub package directory otherwise.
    """
    try:
        from grader.settings import settings
        url = (settings.database_url or "").strip()
    except Exception:  # noqa: BLE001
        url = ""

    if url.startswith("sqlite:///"):
        rel = url[len("sqlite:///"):].split("?", 1)[0]
        try:
            db_dir = Path(rel).resolve().parent
            if db_dir.exists() or db_dir.parent.exists():
                return db_dir / "provider_exhaustion.db"
        except OSError:
            pass

    return Path(__file__).resolve().parent / "provider_exhaustion.db"


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        path = _resolve_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
            timeout=5.0,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS provider_exhaustion ("
            "  provider TEXT PRIMARY KEY,"
            "  marked_at REAL NOT NULL,"
            "  reason TEXT"
            ")"
        )
        _conn = conn
        logger.info("Provider exhaustion store ready at %s", path)
        return conn


def mark(provider: str, reason: str = "") -> None:
    """Record that `provider` is exhausted as of now. Upserts on provider name."""
    _get_conn().execute(
        "INSERT INTO provider_exhaustion(provider, marked_at, reason) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(provider) DO UPDATE SET "
        "  marked_at=excluded.marked_at, reason=excluded.reason",
        (provider, time.time(), reason),
    )


def is_exhausted(provider: str, ttl_seconds: float) -> bool:
    """True when `provider` was marked within the TTL window. Side effect:
    expired rows are deleted on read so the table stays small."""
    row = _get_conn().execute(
        "SELECT marked_at FROM provider_exhaustion WHERE provider=?",
        (provider,),
    ).fetchone()
    if not row:
        return False
    marked_at = row[0]
    if time.time() - marked_at > ttl_seconds:
        _get_conn().execute(
            "DELETE FROM provider_exhaustion WHERE provider=?",
            (provider,),
        )
        return False
    return True


def get_all() -> dict[str, float]:
    """Snapshot of provider → unix-ts of mark. Does NOT filter expired rows;
    the caller should pair with is_exhausted() if it needs liveness."""
    rows = _get_conn().execute(
        "SELECT provider, marked_at FROM provider_exhaustion"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def reset() -> int:
    """Delete all marks. Returns number of rows removed."""
    cur = _get_conn().execute("DELETE FROM provider_exhaustion")
    return cur.rowcount or 0


def _close_for_tests() -> None:
    """Test helper: close and forget the connection so the next call reopens it."""
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None
