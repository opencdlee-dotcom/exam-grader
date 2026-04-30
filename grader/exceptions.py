"""
Typed exception hierarchy for the Exam Grader platform.

Replaces bare `except Exception: return None` patterns with
specific, logged exceptions that give callers meaningful context.
"""


class GraderError(Exception):
    """Base class for all exam-grader exceptions."""


class RagClientError(GraderError):
    """Raised when the RAG engine request fails (network, auth, bad response)."""


class VisionError(GraderError):
    """Raised when the vision/LLM API call fails after all retries."""


class GradingPipelineError(GraderError):
    """Raised when the end-to-end grading pipeline fails for a student."""

    def __init__(self, message: str, student_file: str = "", stage: str = ""):
        super().__init__(message)
        self.student_file = student_file
        self.stage = stage


class CacheError(GraderError):
    """Raised when the grading cache cannot be read or written."""


class AnswerKeyError(GraderError):
    """Raised when the answer key is missing, malformed, or fails validation."""


class PDFConversionError(GraderError):
    """Raised when PDF-to-image conversion fails."""
