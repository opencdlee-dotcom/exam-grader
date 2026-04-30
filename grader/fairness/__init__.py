"""
Fairness Audit module for detecting grade disparities.

Provides:
- Section-level grade distribution analysis
- Disparate Impact (DI) calculation
- Demographic parity checks (future: with enrollment data)
"""

from .fairness_audit import (
    disparate_impact_ratio,
    analyze_section_grades,
    create_fairness_audit_sheet,
    generate_fairness_report,
)

__all__ = [
    "disparate_impact_ratio",
    "analyze_section_grades",
    "create_fairness_audit_sheet",
    "generate_fairness_report",
]
