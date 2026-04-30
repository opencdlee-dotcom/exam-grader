"""
Accessibility module for WCAG AA compliance.

Provides:
- High-contrast color schemes (4.5:1 minimum)
- Dyslexia-friendly fonts
- Semantic HTML structure
- Alt text for icons and abbreviations
"""

from .wcag_compliance import (
    apply_wcag_aa_formatting,
    validate_contrast_ratio,
    calculate_relative_luminance,
)

__all__ = ["apply_wcag_aa_formatting", "validate_contrast_ratio", "calculate_relative_luminance"]
