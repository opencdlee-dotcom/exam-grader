"""Shim — delegates to handwriting_engine.

Falls back to call-time stubs / empty constants when the engine isn't
installed so transitive importers (e.g. tests that patch the grader) can
still load this module.
"""


def _missing(symbol: str):
    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"grader.handwriting.{symbol} requires the 'handwriting_engine' "
            "package. Install it from the engine repo and reload."
        )
    _stub.__name__ = symbol
    return _stub


try:
    from handwriting_engine.handwriting import (
        get_reading_strategies,
        get_grading_handwriting_rules,
        get_disambiguation_pairs,
        CHARACTER_DISAMBIGUATION,
        MULTI_PASS_READING_INSTRUCTIONS,
        NUMBER_READING_RULES,
        SCIENTIFIC_NOTATION,
        BIOLOGY_DOMAIN_RULES,
        TABLE_DETECTION,
        SINGLE_LETTER_PROTOCOL,
        ANTI_HALLUCINATION_PROTOCOL,
    )
except ImportError:
    # The handwriting prompt-content helpers return strings/lists that
    # downstream code joins into a grading prompt. When the engine is
    # absent, returning empty content yields a less-helpful prompt but
    # doesn't crash the pipeline — the engine being missing surfaces
    # later when an actual LLM grade is attempted.
    def get_reading_strategies(*_args, **_kwargs) -> str:
        return ""

    def get_grading_handwriting_rules(*_args, **_kwargs) -> str:
        return ""

    def get_disambiguation_pairs(*_args, **_kwargs) -> list:
        return []

    CHARACTER_DISAMBIGUATION = ""
    MULTI_PASS_READING_INSTRUCTIONS = ""
    NUMBER_READING_RULES = ""
    SCIENTIFIC_NOTATION = ""
    BIOLOGY_DOMAIN_RULES = ""
    TABLE_DETECTION = ""
    SINGLE_LETTER_PROTOCOL = ""
    ANTI_HALLUCINATION_PROTOCOL = ""
