"""
Pull student submissions from Canvas LMS.
Downloads PDF attachments for grading.
"""

import logging
import os
import requests
from canvas.client import CanvasClient

logger = logging.getLogger(__name__)


def get_submissions(client: CanvasClient, course_id: int, assignment_id: int) -> list:
    """Fetch all submissions for an assignment."""
    return client.get_paginated(
        f"courses/{course_id}/assignments/{assignment_id}/submissions",
        params={"per_page": 50, "include[]": "user"}
    )


def download_submissions(
    client: CanvasClient,
    course_id: int,
    assignment_id: int,
    output_dir: str,
    skip_graded: bool = True,
) -> list[dict]:
    """
    Download all PDF submissions for an assignment.

    Args:
        client: Canvas API client
        course_id: Canvas course ID
        assignment_id: Canvas assignment ID
        output_dir: Directory to save downloaded PDFs
        skip_graded: Skip submissions that already have a grade

    Returns:
        List of dicts with student_name, student_id, pdf_path, submission_id
    """
    os.makedirs(output_dir, exist_ok=True)
    submissions = get_submissions(client, course_id, assignment_id)
    downloaded = []

    for sub in submissions:
        # Skip if no attachments
        attachments = sub.get("attachments", [])
        if not attachments:
            continue

        # Skip already graded
        if skip_graded and sub.get("grade") is not None:
            student_name = sub.get("user", {}).get("name", f"ID_{sub['user_id']}")
            logger.info("Skipping %s — already graded: %s", student_name, sub["grade"])
            continue

        # Download PDF attachments
        student_name = sub.get("user", {}).get("name", f"ID_{sub['user_id']}")
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in student_name)
        user_id = sub["user_id"]

        for att in attachments:
            if att.get("content-type", "").startswith("application/pdf") or att["filename"].lower().endswith(".pdf"):
                pdf_path = os.path.join(output_dir, f"{safe_name}_{user_id}_{att['filename']}")

                logger.info("Downloading: %s — %s", student_name, att["filename"])
                resp = requests.get(att["url"], stream=True)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Validate that downloaded file is actually a PDF
                with open(pdf_path, "rb") as f:
                    magic = f.read(5)
                if magic != b"%PDF-":
                    logger.warning("%s from %s is not a valid PDF (magic: %r). Skipping.", att["filename"], student_name, magic)
                    os.remove(pdf_path)
                    continue

                downloaded.append({
                    "student_name": student_name,
                    "student_id": sub["user_id"],
                    "submission_id": sub["id"],
                    "pdf_path": pdf_path,
                })

    return downloaded
