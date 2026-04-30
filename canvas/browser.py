"""
Browser-based grade entry via Canvas Gradebook using Playwright.
Fallback for when API-based grade posting needs visual verification.

Adapted from LabNoteBookGrader's SpeedGraderBrowser pattern.
"""

import logging
import os

logger = logging.getLogger(__name__)


class BrowserGradeEntry:
    """
    Opens Canvas Gradebook in a headed Chromium browser and enters grades
    by navigating the Gradebook UI. Preserves login session via persistent
    browser context.
    """

    def __init__(self, canvas_base_url: str):
        self.canvas_base_url = canvas_base_url.rstrip("/")
        self.context = None
        self.page = None
        self._playwright = None
        self._dialog_handler_registered = False

    async def launch(self, headless: bool = False):
        """Launch a headed browser with persistent login session."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        # Use persistent context to preserve Canvas login across sessions
        user_data_dir = os.path.join(
            os.path.expanduser("~"), ".exam-grader", "browser-data"
        )
        os.makedirs(user_data_dir, exist_ok=True)

        self.context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir,
            headless=headless,
            viewport={"width": 1280, "height": 900},
        )

        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()

        self._register_dialog_handler()

    async def navigate_to_canvas(self) -> bool:
        """Navigate to Canvas and check if login is needed. Returns True if logged in."""
        await self.page.goto(self.canvas_base_url)
        await self.page.wait_for_load_state("networkidle")
        await self._hide_eesy_overlay()
        return "login" not in self.page.url.lower()

    async def navigate_to_gradebook(self, course_id: int):
        """Navigate to the Gradebook (Grades tab) for a course."""
        url = f"{self.canvas_base_url}/courses/{course_id}/gradebook"
        await self.page.goto(url)
        await self.page.wait_for_load_state("networkidle")
        await self._hide_eesy_overlay()

        # Check if login is needed
        if "login" in self.page.url.lower():
            print("Canvas is asking you to log in.")
            print("Please sign in in the browser window, then press Enter here.")
            input("Press Enter when logged in...")
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
            await self._hide_eesy_overlay()

        # Wait for gradebook to fully render
        await self.page.wait_for_timeout(3000)

    async def navigate_to_speedgrader(
        self, course_id: int, assignment_id: int, user_id: int
    ):
        """Navigate to SpeedGrader for a specific student."""
        url = (
            f"{self.canvas_base_url}/courses/{course_id}/gradebook/speed_grader"
            f"?assignment_id={assignment_id}&student_id={user_id}"
        )
        await self.page.goto(url)
        await self.page.wait_for_load_state("networkidle")
        if "login" in self.page.url.lower():
            print("Canvas is asking you to log in.")
            print("Please sign in in the browser window, then press Enter here.")
            input("Press Enter when logged in...")
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
        await self.page.wait_for_timeout(2000)
        await self._hide_eesy_overlay()

    def _register_dialog_handler(self):
        """Auto-accept Canvas native confirm dialogs used for comment deletion."""
        if self.page is None or self._dialog_handler_registered:
            return

        async def _accept_dialog(dialog):
            await dialog.accept()

        self.page.on("dialog", _accept_dialog)
        self._dialog_handler_registered = True

    async def _hide_eesy_overlay(self):
        """Hide the Eesy floating widget that intercepts clicks in SpeedGrader."""
        if not self.page:
            return
        await self.page.evaluate(
            """
            () => {
                document
                    .querySelectorAll('.eesy,.eesy-tab2-container,[id*="eesy"],[class*="eesy"]')
                    .forEach((el) => el.style.setProperty('display', 'none', 'important'));
            }
            """
        )

    def _rce_frame(self):
        """Return the Canvas rich-comment editor iframe if present."""
        for frame in self.page.frames:
            frame_name = (frame.name or "").lower()
            if "rce" in frame_name and "_ifr" in frame_name:
                return frame
        return None

    async def _delete_existing_comments(self):
        """Delete all existing comments so each student ends with exactly one comment."""
        delete_buttons = self.page.locator('[data-testid$="-delete-button"]')
        while await delete_buttons.count() > 0:
            button = delete_buttons.first
            await button.wait_for(state="visible", timeout=5000)
            await button.evaluate("(el) => el.click()")
            await self.page.wait_for_timeout(1500)
            await self._hide_eesy_overlay()

    async def _submit_comment(self, comment: str):
        """Submit a SpeedGrader comment through the RCE iframe using real keystrokes."""
        frame = self._rce_frame()
        if frame is None:
            raise RuntimeError("Could not find Canvas comment editor iframe.")

        editor_body = frame.locator("body").first
        await editor_body.wait_for(state="visible", timeout=10000)
        await editor_body.click()
        await self.page.keyboard.press("Meta+A")
        await self.page.keyboard.type(comment, delay=0)

        submit_btn = self.page.locator(
            '[data-testid="submit-comment-button"], #comment_submit_button'
        ).first
        await submit_btn.wait_for(state="visible", timeout=10000)
        await submit_btn.evaluate("(el) => el.click()")
        await self.page.wait_for_timeout(1500)

    async def _verify_single_comment(self, expect_comment: bool):
        """Reload and verify the final comment count for the student."""
        await self.page.reload()
        await self.page.wait_for_load_state("networkidle")
        await self.page.wait_for_timeout(2000)
        await self._hide_eesy_overlay()

        count = await self.page.locator('[data-testid$="-delete-button"]').count()
        expected = 1 if expect_comment else 0
        if count != expected:
            raise RuntimeError(
                f"Expected {expected} saved comment(s) after submission, found {count}."
            )

    async def enter_grade(self, student_name: str, score: float, comment: str = "") -> bool:
        """
        Enter a grade for a student in SpeedGrader.

        Assumes the page is already on the target student's SpeedGrader view.
        Deletes any prior comments, saves the grade, submits the new comment via
        the RCE iframe, and verifies the student ends with exactly one comment.

        Returns:
            True if grade was successfully entered, False otherwise.

        Raises:
            RuntimeError: If no grade input element could be found.
        """
        await self._hide_eesy_overlay()

        grade_input = self.page.locator(
            '#grading-box-extended, #grade_container input, input[data-testid="grade-input"]'
        ).first

        if not await grade_input.is_visible():
            raise RuntimeError(
                f"Could not find grade input element for student '{student_name}'. "
                "SpeedGrader input was not visible."
            )

        await grade_input.fill(str(score))
        await grade_input.press("Enter")
        await self.page.wait_for_timeout(1000)

        await self._delete_existing_comments()

        if comment:
            await self._submit_comment(comment)

        await self._verify_single_comment(expect_comment=bool(comment))
        return True

    async def screenshot(self, output_path: str):
        """Take a screenshot of the current browser view."""
        await self.page.screenshot(path=output_path)

    async def close(self):
        """Close the browser and clean up."""
        if self.context:
            await self.context.close()
        if self._playwright:
            await self._playwright.stop()
        self.context = None
        self.page = None
        self._playwright = None
