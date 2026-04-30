"""Shim — delegates to handwriting_engine.

Falls back to call-time stubs when the engine isn't installed so callers
that import these names (and tests that patch them) still load.
"""


def _missing(symbol: str):
    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"grader.image_optimizer.{symbol} requires the 'handwriting_engine' "
            "package. Install it from the engine repo and reload."
        )
    _stub.__name__ = symbol
    return _stub


try:
    from handwriting_engine.optimize import (
        validate_and_prepare_image,
        build_image_blocks,
        batch_pages_by_size,
        get_adaptive_settings,
    )
except ImportError:
    validate_and_prepare_image = _missing("validate_and_prepare_image")
    build_image_blocks = _missing("build_image_blocks")
    batch_pages_by_size = _missing("batch_pages_by_size")

    def get_adaptive_settings(page_count: int = 1):
        """Fallback when handwriting_engine isn't installed.

        Returns (dpi, jpeg_quality, max_long_side). Real engine adapts
        these to page count; this default is a single conservative tier
        that works for short submissions and lets the pipeline progress
        far enough to exercise tests / non-grading code paths.
        """
        return (200, 80, 1568)
