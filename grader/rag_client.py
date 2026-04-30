"""
HTTP client for the RAG engine (Node.js service on port 3000).

All methods degrade gracefully — if the RAG engine is unavailable, callers
fall back to full rubric injection (current behavior). Never raises on
connectivity errors; returns None/[] instead.

Auth: set RAG_API_KEY in .env to send Bearer token with every request.

Usage:
    from grader.rag_client import get_rag_client

    rag = get_rag_client()          # None if RAG is not running
    if rag:
        doc_id = rag.ingest_text("Rubric", rubric_text)
        sources = rag.query("denaturation temperature tolerance")
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Any

import requests

import logging

from grader.config import RAG_ENGINE_URL, RAG_API_KEY

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 2    # seconds — fast fail if RAG not running
_REQUEST_TIMEOUT = 8    # seconds — per query/ingest call


class RAGClient:
    """Thin HTTP client for the RAG engine's /api/v1 endpoints."""

    def __init__(self, base_url: str = RAG_ENGINE_URL, api_key: str = RAG_API_KEY):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    # ── Health ──

    def health_check(self) -> bool:
        """Return True if the RAG engine is reachable."""
        try:
            r = self._session.get(
                f"{self.base_url}/query/health",
                timeout=_CONNECT_TIMEOUT,
            )
            return r.ok
        except (requests.exceptions.RequestException, OSError):
            return False

    # ── Ingest ──

    def ingest_text(self, title: str, content: str, metadata: dict | None = None) -> str | None:
        """
        Ingest plain text into the RAG engine.

        Sends as base64-encoded .txt file so the ingestion workflow can parse it.
        Returns the created document ID, or None on failure.
        """
        try:
            filename = f"{_slug(title)}.txt"
            payload: dict[str, Any] = {
                "filename": filename,
                "fileBase64": base64.b64encode(content.encode()).decode(),
                "metadata": metadata or {},
            }
            r = self._session.post(
                f"{self.base_url}/documents",
                data=json.dumps(payload),
                timeout=_REQUEST_TIMEOUT,
            )
            if r.ok:
                return r.json().get("documentId")
            return None
        except Exception as e:
            logger.warning("RAG ingest_text failed for %r: %s", title, e, exc_info=True)
            return None

    def ingest_rubric(self, rubric_json: dict, title: str | None = None) -> str | None:
        """
        Ingest an answer key/rubric dict.

        Returns the created document ID, or None on failure.
        """
        label = title or rubric_json.get("title", "Unnamed Rubric")
        content = _rubric_to_text(rubric_json)
        return self.ingest_text(
            title=f"Rubric: {label}",
            content=content,
            metadata={"type": "rubric", "source_hash": _hash(content)},
        )

    def ingest_grading_result(self, result: dict) -> str | None:
        """
        Ingest a grading result for future consistency lookups.

        Extracts question-level patterns: question text, student answer,
        awarded score, reasoning.
        """
        lines: list[str] = []
        student = result.get("student_file", "unknown")
        lines.append(f"Student: {student}")

        sr = result.get("structured_result")
        if sr is None:
            return None

        # Handle both GradingResult object and plain dict (loaded from JSON)
        if hasattr(sr, "sections"):
            for sec in sr.sections:
                for item in getattr(sec, "items", []):
                    lines.append(
                        f"Q: {getattr(item, 'criterion', '')} | "
                        f"Score: {getattr(item, 'score', 0)}/{getattr(item, 'max_score', 0)} | "
                        f"Answer: {getattr(item, 'student_response', '')} | "
                        f"Reasoning: {getattr(item, 'feedback', '')}"
                    )
        elif isinstance(sr, dict):
            for sec in sr.get("sections", []):
                for item in sec.get("items", []):
                    lines.append(
                        f"Q: {item.get('criterion', '')} | "
                        f"Score: {item.get('score', 0)}/{item.get('max_score', 0)} | "
                        f"Answer: {item.get('student_response', '')} | "
                        f"Reasoning: {item.get('feedback', '')}"
                    )

        if len(lines) <= 1:
            return None

        content = "\n".join(lines)
        return self.ingest_text(
            title=f"Grading: {student}",
            content=content,
            metadata={"type": "grading_result", "student_file": student},
        )

    # ── Query ──

    def query(
        self,
        query: str,
        mode: str = "balanced",
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query string.

        Returns a list of source dicts with 'content'/'snippet', 'score', etc.
        Returns [] on failure or if RAG is unavailable.
        """
        try:
            payload: dict[str, Any] = {
                "query": query,
                "mode": mode,
            }
            if filters:
                payload["filters"] = filters
            r = self._session.post(
                f"{self.base_url}/query",
                data=json.dumps(payload),
                timeout=_REQUEST_TIMEOUT,
            )
            if r.ok:
                data = r.json()
                # Response may be QueryResponse (has 'sources') or raw chunk list
                sources = data.get("sources", [])
                # Normalize: ensure each source has a 'content' key
                # (QueryResponse uses 'snippet'; direct chunk search uses 'content')
                for s in sources:
                    if "content" not in s:
                        s["content"] = s.get("snippet", "")
                return sources
            return []
        except Exception as e:
            logger.warning("RAG query failed for %r: %s", query[:50], e, exc_info=True)
            return []

    def get_similar_answers(self, question: str, student_answer: str) -> list[dict]:
        """Retrieve past grading decisions similar to this student answer."""
        q = f"Student answer for: {question}\nAnswer: {student_answer}"
        return self.query(q, mode="fast")

    def get_misconceptions(self, concept: str) -> list[dict]:
        """Retrieve common misconceptions and partial credit patterns for a concept."""
        return self.query(f"common misconceptions about: {concept}", mode="fast")

    def get_relevant_rubric(self, section_or_question: str) -> list[dict]:
        """Retrieve rubric sections most relevant to a given question or section label."""
        return self.query(section_or_question, mode="balanced")

    def format_context(self, sources: list[dict], max_chars: int = 2000) -> str:
        """
        Format retrieved sources into a compact context string for prompt injection.

        Truncates to max_chars to avoid bloating prompts.
        Sanitizes snippets to prevent prompt injection from stored rubric content.
        """
        if not sources:
            return ""
        lines = ["[RAG CONTEXT START — relevant rubric/grading patterns]"]
        for s in sources:
            snippet = s.get("content", s.get("snippet", "")).strip()[:400]
            if snippet:
                snippet = _sanitize_rag_snippet(snippet)
                lines.append(f"  - {snippet}")
        lines.append("[RAG CONTEXT END]")
        text = "\n".join(lines)
        return text[:max_chars]


# ── Module-level singleton ──

_rag_instance: RAGClient | None = None
_rag_check_time: float = 0
RAG_CHECK_TTL = 300  # Re-check health every 5 minutes


def get_rag_client() -> RAGClient | None:
    """
    Return a live RAGClient if the engine is running, else None.

    Result is cached with a 5-minute TTL so the engine can be
    started/stopped mid-session.
    """
    global _rag_instance, _rag_check_time
    if time.time() - _rag_check_time < RAG_CHECK_TTL:
        return _rag_instance

    _rag_check_time = time.time()
    client = RAGClient()
    if client.health_check():
        _rag_instance = client
        logger.info("[RAG] Engine connected at %s", RAG_ENGINE_URL)
    else:
        _rag_instance = None

    return _rag_instance


# ── Helpers ──

def _rubric_to_text(rubric: dict) -> str:
    """Convert a rubric/answer-key dict to readable text for ingestion."""
    lines: list[str] = [f"# {rubric.get('title', 'Rubric')}"]
    lines.append(f"Total points: {rubric.get('total_points', '?')}")

    for section in rubric.get("sections", []):
        lines.append(f"\n## {section.get('name', '')} ({section.get('points', '?')} pts)")
        for criterion in section.get("criteria", []):
            lines.append(f"- {criterion.get('item', '')}: {criterion.get('points', '?')} pts")
            if criterion.get("expected"):
                lines.append(f"  Expected: {criterion['expected']}")
            for alt in criterion.get("alternatives", []):
                lines.append(f"  Also accepted: {alt}")
            for ded in criterion.get("deductions", []):
                lines.append(f"  Deduct {ded.get('points', 0)}: {ded.get('reason', '')}")

    for q in rubric.get("questions", []):
        lines.append(f"\nQ{q.get('number', '?')}: {q.get('text', '')} ({q.get('points', '?')} pts)")
        lines.append(f"  Answer: {q.get('answer', '')}")
        for alt in q.get("alternatives", []):
            lines.append(f"  Also: {alt}")

    return "\n".join(lines)


import re as _re

# Patterns that could influence the grading prompt if injected via RAG
_INJECTION_PATTERNS = _re.compile(
    r"^\s*(IMPORTANT|IGNORE|SYSTEM|OVERRIDE|NOTE:|INSTRUCTION|YOU MUST|ALWAYS|NEVER|FORGET|DISREGARD)",
    _re.IGNORECASE | _re.MULTILINE,
)


def _sanitize_rag_snippet(snippet: str) -> str:
    """Strip instruction-like patterns from RAG snippets to prevent prompt injection."""
    # Remove lines that look like injected instructions
    clean_lines = []
    for line in snippet.split("\n"):
        if _INJECTION_PATTERNS.match(line):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:40]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
