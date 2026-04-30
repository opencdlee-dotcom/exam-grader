"""Shim — delegates to handwriting_engine. Original code consolidated into engine.

Falls back to call-time stubs when the engine isn't installed.
"""


def _missing(symbol: str):
    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"grader.image_quality.{symbol} requires the 'handwriting_engine' "
            "package. Install it from the engine repo and reload."
        )
    _stub.__name__ = symbol
    return _stub


try:
    from handwriting_engine.quality import (
        assess_blur,
        assess_contrast,
        assess_brightness,
        assess_image,
        classify_handwriting_style,
    )
    from handwriting_engine.enhance import (
        smart_enhance,
        _enhance_sharpen,
        _enhance_contrast,
        _enhance_brighten,
        _enhance_denoise,
    )
except ImportError:
    assess_blur = _missing("assess_blur")
    assess_contrast = _missing("assess_contrast")
    assess_brightness = _missing("assess_brightness")
    assess_image = _missing("assess_image")
    classify_handwriting_style = _missing("classify_handwriting_style")
    smart_enhance = _missing("smart_enhance")
    _enhance_sharpen = _missing("_enhance_sharpen")
    _enhance_contrast = _missing("_enhance_contrast")
    _enhance_brighten = _missing("_enhance_brighten")
    _enhance_denoise = _missing("_enhance_denoise")
