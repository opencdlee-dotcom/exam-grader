"""
Google Sheets grade report generation via gws CLI.
Creates a spreadsheet with grades + statistics, shares it with the configured email.
"""

import json
import logging
import statistics
import subprocess

from grader.config import GOOGLE_SHARE_EMAIL

logger = logging.getLogger(__name__)
from grader.report import extract_score


def _gws(args: list[str]) -> dict:
    """Run a gws CLI command and return parsed JSON output."""
    cmd = ["gws"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gws command failed: {' '.join(cmd)}\n{result.stderr}")
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def prompt_sheet_name() -> str:
    """Prompt user for exam number and course name, return formatted sheet title."""
    exam_num = input("Exam number (e.g. 3): ").strip()
    course_name = input("Course name (e.g. BIOL 281): ").strip()
    return f"Exam {exam_num} - {course_name}"


def create_grade_sheet(results: list[dict], sheet_name: str) -> str:
    """
    Create a Google Sheet with grade results and class statistics.

    Args:
        results: List of grade result dicts from batch grading
        sheet_name: Title for the spreadsheet (e.g. "Exam 3 - BIOL 281")

    Returns:
        URL of the created Google Sheet
    """
    # --- Extract scores ---
    rows = []
    scores = []
    total_points = None

    for result in results:
        name = result.get("student_file", "Unknown")
        if result.get("error"):
            rows.append([name, "ERROR", "", ""])
            continue

        score, total = extract_score(result["grade_text"])
        if score is not None:
            pct = round((score / total) * 100, 1) if total else 0
            rows.append([name, score, total, pct])
            scores.append(score)
            if total_points is None:
                total_points = total
        else:
            rows.append([name, "N/A", "", ""])

    # --- Compute statistics ---
    stats_rows = []
    if scores:
        sorted_scores = sorted(scores)
        q1 = statistics.quantiles(sorted_scores, n=4)[0] if len(sorted_scores) >= 2 else sorted_scores[0]
        q3 = statistics.quantiles(sorted_scores, n=4)[2] if len(sorted_scores) >= 2 else sorted_scores[-1]

        stat_values = [
            ("Highest", max(scores)),
            ("Lowest", min(scores)),
            ("Average", round(statistics.mean(scores), 2)),
            ("Median", round(statistics.median(scores), 2)),
            ("Upper Quartile (Q3)", round(q3, 2)),
            ("Lower Quartile (Q1)", round(q1, 2)),
        ]
        for label, value in stat_values:
            pct = round((value / total_points) * 100, 1) if total_points else ""
            stats_rows.append([label, value, pct])

    # --- Step 1: Create spreadsheet with two sheets ---
    create_body = {
        "properties": {"title": sheet_name},
        "sheets": [
            {"properties": {"title": "Grades", "sheetId": 0}},
            {"properties": {"title": "Statistics", "sheetId": 1}},
        ],
    }
    resp = _gws([
        "sheets", "spreadsheets", "create",
        "--json", json.dumps(create_body),
    ])
    spreadsheet_id = resp["spreadsheetId"]
    spreadsheet_url = resp["spreadsheetUrl"]
    logger.info("Created Google Sheet: %s", sheet_name)

    # --- Step 2: Write grades data ---
    grade_data = [["Student", "Raw Score", "Total Points", "Percentage"]] + rows
    _gws([
        "sheets", "spreadsheets", "values", "update",
        "--params", json.dumps({
            "spreadsheetId": spreadsheet_id,
            "range": "Grades!A1",
            "valueInputOption": "USER_ENTERED",
        }),
        "--json", json.dumps({
            "values": grade_data,
        }),
    ])

    # --- Step 3: Write statistics data ---
    stats_header = [f"Class Statistics — {sheet_name}"]
    stats_subheader = [f"Total Points Possible: {total_points or 'N/A'}"]
    stats_data = (
        [stats_header, stats_subheader, []]
        + [["Statistic", "Value", "Percentage"]]
        + stats_rows
    )
    _gws([
        "sheets", "spreadsheets", "values", "update",
        "--params", json.dumps({
            "spreadsheetId": spreadsheet_id,
            "range": "Statistics!A1",
            "valueInputOption": "USER_ENTERED",
        }),
        "--json", json.dumps({
            "values": stats_data,
        }),
    ])

    # --- Step 4: Format headers (bold + colored) ---
    format_requests = {
        "requests": [
            # Bold + blue background on Grades header row
            {
                "repeatCell": {
                    "range": {"sheetId": 0, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.267, "green": 0.447, "blue": 0.769},
                            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            },
            # Bold + blue background on Statistics header row (row 4 = index 3)
            {
                "repeatCell": {
                    "range": {"sheetId": 1, "startRowIndex": 3, "endRowIndex": 4},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.267, "green": 0.447, "blue": 0.769},
                            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            },
            # Bold the title on Statistics sheet
            {
                "repeatCell": {
                    "range": {"sheetId": 1, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True, "fontSize": 14},
                        }
                    },
                    "fields": "userEnteredFormat(textFormat)",
                }
            },
            # Auto-resize columns on Grades sheet
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": 0, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 4}
                }
            },
            # Auto-resize columns on Statistics sheet
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": 1, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 3}
                }
            },
        ]
    }
    _gws([
        "sheets", "spreadsheets", "batchUpdate",
        "--params", json.dumps({"spreadsheetId": spreadsheet_id}),
        "--json", json.dumps(format_requests),
    ])

    # --- Step 5: Share with configured email ---
    _gws([
        "drive", "permissions", "create",
        "--params", json.dumps({"fileId": spreadsheet_id}),
        "--json", json.dumps({
            "type": "user",
            "role": "writer",
            "emailAddress": GOOGLE_SHARE_EMAIL,
        }),
    ])
    logger.info("Shared with %s", GOOGLE_SHARE_EMAIL)

    return spreadsheet_url
