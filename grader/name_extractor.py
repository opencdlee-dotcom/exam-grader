"""
Name field ROI extraction and enhancement.

Crops the student name region from exam pages, applies targeted
enhancement (upscale + CLAHE with small tile grid), and provides
utilities for multi-provider name consensus.
"""

from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_name_region(
    page_image: Image.Image,
    x_frac: float = 0.0,
    y_frac: float = 0.0,
    w_frac: float = 0.45,
    h_frac: float = 0.07,
    padding: int = 10,
    upscale: int = 3,
) -> Image.Image:
    """
    Crop and enhance the name field from the top of an exam page.

    Default fractions assume the name is in the top-left ~45% of page width,
    top ~7% of height.  Adjust per exam form layout.

    Parameters
    ----------
    page_image : PIL.Image.Image
        Full exam page image.
    x_frac, y_frac : float
        Top-left corner of name region as fraction of page dimensions.
    w_frac, h_frac : float
        Width and height of name region as fraction of page dimensions.
    padding : int
        Extra pixels around the crop for context.
    upscale : int
        Upscale factor.  3x is recommended for small handwritten name fields.
    """
    w, h = page_image.size
    x = int(x_frac * w)
    y = int(y_frac * h)
    crop_w = int(w_frac * w)
    crop_h = int(h_frac * h)

    crop = page_image.crop((
        max(0, x - padding),
        max(0, y - padding),
        min(w, x + crop_w + padding),
        min(h, y + crop_h + padding),
    ))

    # Upscale for legibility
    crop = crop.resize(
        (crop.width * upscale, crop.height * upscale),
        Image.LANCZOS,
    )

    return crop


def enhance_name_field(name_img: Image.Image) -> Image.Image:
    """
    Enhancement pipeline optimized for name field legibility.

    Uses CLAHE with small tile grid (4x4 for narrow regions),
    then light unsharp mask.  Falls back to PIL-only if cv2 unavailable.
    """
    if HAS_CV2:
        arr = cv2.cvtColor(np.array(name_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(arr)
        result = Image.fromarray(enhanced).convert("RGB")
    else:
        result = ImageOps.autocontrast(name_img.convert("RGB"), cutoff=2)

    result = result.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=2)
    )
    return result


def extract_and_enhance_name(
    page_image: Image.Image,
    **crop_kwargs,
) -> Image.Image:
    """Crop the name region and enhance it in one call."""
    crop = extract_name_region(page_image, **crop_kwargs)
    return enhance_name_field(crop)


def name_consensus(reads: list[str]) -> dict:
    """
    Compute consensus from multiple name readings.

    Parameters
    ----------
    reads : list[str]
        Name readings from different providers or passes.

    Returns
    -------
    dict
        {
            "name": str,         # best guess
            "confidence": str,   # "HIGH", "MEDIUM", "LOW"
            "all_reads": list,
            "flag": bool,        # True if NAME UNCERTAIN
            "agreement": float,  # 0.0-1.0
        }
    """
    if not reads:
        return {
            "name": "UNKNOWN",
            "confidence": "LOW",
            "all_reads": [],
            "flag": True,
            "agreement": 0.0,
        }

    # Normalize for comparison
    normalized = [r.strip().lower() for r in reads if r and r.strip()]
    if not normalized:
        # All reads were empty/whitespace — return UNKNOWN, not an empty string
        first_non_empty = next((r.strip() for r in reads if r and r.strip()), "UNKNOWN")
        return {
            "name": first_non_empty,
            "confidence": "LOW",
            "all_reads": reads,
            "flag": True,
            "agreement": 0.0,
        }

    from collections import Counter
    counts = Counter(normalized)
    most_common, count = counts.most_common(1)[0]
    agreement = count / len(normalized)

    # Find the original-cased version of the winning name
    # Guard against None values in reads (e.g. from failed OCR passes)
    best_name = next((r.strip() for r in reads if r and r.strip()), "UNKNOWN")
    for r in reads:
        if r and r.strip() and r.strip().lower() == most_common:
            best_name = r.strip()
            break

    if agreement >= 0.67:
        confidence = "HIGH"
        flag = False
    elif agreement >= 0.50:
        confidence = "MEDIUM"
        flag = True
    else:
        confidence = "LOW"
        flag = True

    return {
        "name": best_name,
        "confidence": confidence,
        "all_reads": reads,
        "flag": flag,
        "agreement": agreement,
    }
