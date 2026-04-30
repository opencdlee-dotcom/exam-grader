"""
Analytics module for question-level metrics and learning analytics.

Provides:
- Discrimination Index (DI): measure of question quality
- Item Difficulty (p-value): % of students answering correctly
- Flagging of problematic questions
"""

from .question_metrics import (
    calculate_discrimination_index,
    calculate_item_difficulty,
    flag_problematic_questions,
    generate_question_report,
)

__all__ = [
    "calculate_discrimination_index",
    "calculate_item_difficulty",
    "flag_problematic_questions",
    "generate_question_report",
]
