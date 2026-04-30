"""
Canvas LMS API client.
Handles authentication and base request methods.
"""

import logging

import requests
from grader.config import CANVAS_API_URL, CANVAS_API_TOKEN

logger = logging.getLogger(__name__)


class CanvasClient:
    def __init__(self, api_url: str = None, api_token: str = None):
        self.api_url = (api_url or CANVAS_API_URL).rstrip("/")
        self.token = api_token or CANVAS_API_TOKEN

        if not self.token or self.token == "your-canvas-token-here":
            raise ValueError(
                "CANVAS_API_TOKEN not set. Add your token to .env file.\n"
                "Get one at: Canvas → Account → Settings → New Access Token"
            )

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

    def get(self, endpoint: str, params: dict = None, timeout: int = 30) -> requests.Response:
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp

    def put(self, endpoint: str, data: dict = None, timeout: int = 30) -> requests.Response:
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        resp = self.session.put(url, json=data, timeout=timeout)
        resp.raise_for_status()
        return resp

    def get_paginated(self, endpoint: str, params: dict = None) -> list:
        """Fetch all pages of a paginated Canvas API response."""
        results = []
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        while url:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                logger.warning("Expected JSON array from Canvas API, got %s. Stopping pagination.", type(data).__name__)
                break
            params = None  # Only use params on first request

            # Check for next page link
            links = resp.headers.get("Link", "")
            url = None
            for part in links.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip(" <>")

        return results

    def list_courses(self) -> list:
        """List active courses."""
        return self.get_paginated(
            "courses",
            params={"enrollment_state": "active", "per_page": 50}
        )

    def list_assignments(self, course_id: int) -> list:
        """List assignments for a course."""
        return self.get_paginated(
            f"courses/{course_id}/assignments",
            params={"per_page": 50}
        )

    def list_students(self, course_id: int) -> list[dict]:
        """Fetch enrolled students for a course.
        Returns list of dicts with 'id', 'name', 'sortable_name'."""
        return self.get_paginated(
            f"courses/{course_id}/users",
            params={"enrollment_type[]": "student", "per_page": 100}
        )

    def mute_assignment(self, course_id: int, assignment_id: int) -> dict:
        """Mute assignment to hide grades from students during grading."""
        resp = self.put(
            f"courses/{course_id}/assignments/{assignment_id}",
            data={"assignment": {"muted": True}},
        )
        return resp.json()

    def unmute_assignment(self, course_id: int, assignment_id: int) -> dict:
        """Unmute assignment to release grades to students."""
        resp = self.put(
            f"courses/{course_id}/assignments/{assignment_id}",
            data={"assignment": {"muted": False}},
        )
        return resp.json()
