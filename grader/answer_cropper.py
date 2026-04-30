"""Shim — delegates to handwriting_engine.

Falls back to call-time stubs when the engine isn't installed.
"""


def _missing(symbol: str):
    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"grader.answer_cropper.{symbol} requires the 'handwriting_engine' "
            "package. Install it from the engine repo and reload."
        )
    _stub.__name__ = symbol
    return _stub


try:
    from handwriting_engine.crop import (
        AnswerSheetLayout,
        compute_answer_regions,
        preprocess_crop,
        crop_and_enhance,
        crop_answer_sheet,
    )
except ImportError:
    # AnswerSheetLayout is used as a type annotation downstream (e.g.
    # `layout: AnswerSheetLayout | None`), so the fallback must be a real
    # class — a function won't satisfy `cls | None` PEP 604 unions.
    class AnswerSheetLayout:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "grader.answer_cropper.AnswerSheetLayout requires the "
                "'handwriting_engine' package. Install it from the engine "
                "repo and reload."
            )

    compute_answer_regions = _missing("compute_answer_regions")
    preprocess_crop = _missing("preprocess_crop")
    crop_and_enhance = _missing("crop_and_enhance")
    crop_answer_sheet = _missing("crop_answer_sheet")
