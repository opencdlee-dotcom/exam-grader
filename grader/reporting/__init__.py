"""
Reporting module for automated grading session reports.

Provides:
- Daily grading summaries
- Question-level analysis
- Fairness audit summaries
- Confidence distribution breakdowns
- Actionable recommendations
- Interactive HTML analytics dashboard
- Confidence calibration reports
"""

from .daily_report import generate_daily_report, save_report_to_file
from .analytics_dashboard import generate_dashboard, collect_dashboard_data
from .calibration_report import (
    generate_calibration_report,
    collect_calibration_data,
    save_calibration_report,
)

__all__ = [
    "generate_daily_report",
    "save_report_to_file",
    "generate_dashboard",
    "collect_dashboard_data",
    "generate_calibration_report",
    "collect_calibration_data",
    "save_calibration_report",
]
