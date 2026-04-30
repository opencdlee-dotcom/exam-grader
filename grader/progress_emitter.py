"""
Thread-safe progress emitter for real-time grading progress.
Uses a simple observer pattern — WebSocket handlers subscribe to events.
"""

import threading
from typing import Callable, Any

_lock = threading.Lock()
_subscribers: list[Callable] = []


def subscribe(callback: Callable[[str, dict], None]) -> None:
    """Subscribe to progress events. callback(event_type, data)."""
    with _lock:
        _subscribers.append(callback)


def unsubscribe(callback: Callable) -> None:
    with _lock:
        _subscribers.remove(callback)


def emit(event_type: str, data: dict) -> None:
    """Emit a progress event to all subscribers."""
    with _lock:
        callbacks = list(_subscribers)
    for cb in callbacks:
        try:
            cb(event_type, data)
        except Exception:
            pass  # Never crash the grader due to a subscriber error


def emit_student_graded(student_name: str, score: float, max_score: float,
                         confidence: str, section: str = "", error: str = None) -> None:
    emit("student_graded", {
        "student": student_name,
        "score": score,
        "max_score": max_score,
        "percentage": round(score / max_score * 100, 1) if max_score > 0 else 0,
        "confidence": confidence,
        "section": section,
        "error": error,
    })


def emit_batch_complete(n_students: int, mean_score: float, mean_pct: float) -> None:
    emit("batch_complete", {
        "n_students": n_students,
        "mean_score": mean_score,
        "mean_percentage": mean_pct,
    })
