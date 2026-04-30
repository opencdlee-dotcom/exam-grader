"""
Shim — delegates core PDF logic to handwriting_engine.
Keeps thin wrappers for convert_pdf_survey and enhance_image so existing
imports (from grader.pdf_reader import convert_pdf) continue to work.

When `handwriting_engine` isn't installed, the engine-backed symbols fall
back to call-time stubs that raise a clear ImportError. This lets tests
that patch these names (`patch.object(pipeline, "convert_pdf", …)`)
succeed without a real engine install — the production grader path still
fails fast if the engine is genuinely missing.
"""


def _missing(symbol: str):
    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"grader.pdf_reader.{symbol} requires the 'handwriting_engine' "
            "package. Install it from the engine repo and reload."
        )
    _stub.__name__ = symbol
    return _stub


try:
    from handwriting_engine.pdf import (
        convert_pdf,
        detect_and_fix_orientation,
        get_page_count,
    )
    from handwriting_engine.pdf import convert_pdf_survey as _engine_convert_pdf_survey
    from handwriting_engine.enhance import enhance_image as _engine_enhance_image

    def convert_pdf_survey(pdf_path: str, output_dir: str) -> list[dict]:
        """Low-resolution PDF conversion for the survey pass."""
        return _engine_convert_pdf_survey(pdf_path, output_dir)

    def enhance_image(image_path: str, strength: str = "smart") -> str:
        """Enhance a page image for better handwriting readability."""
        return _engine_enhance_image(image_path, strength=strength)

except ImportError:
    convert_pdf = _missing("convert_pdf")
    detect_and_fix_orientation = _missing("detect_and_fix_orientation")
    get_page_count = _missing("get_page_count")
    convert_pdf_survey = _missing("convert_pdf_survey")
    enhance_image = _missing("enhance_image")


def cleanup_images(output_dir: str):
    """Remove all generated page images. Pure stdlib — no engine dep."""
    import os
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith((".jpg", ".png")):
                os.remove(os.path.join(output_dir, f))
        try:
            os.rmdir(output_dir)
        except OSError:
            pass
