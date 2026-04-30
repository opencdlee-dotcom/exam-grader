from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SectionConfig:
    section: str
    course_id: str
    assignment_id: str
    exam_max: float
    free_response_questions: tuple[str, ...] = ()
    name_overrides: dict[str, str] | None = None


DEFAULT_CANVAS_BASE = "https://canvas.csun.edu"
DEFAULT_STORAGE_STATE = Path("/tmp/canvas_auth_state.json")
DEFAULT_PAYLOAD_PATH = Path("/tmp/canvas_posting_list_clean.json")


SECTION_CONFIGS: dict[str, SectionConfig] = {
    "T5": SectionConfig(
        section="T5",
        course_id="180404",
        assignment_id="2628357",
        exam_max=32.0,
        free_response_questions=("Q23",),
        name_overrides={
            "Abdzel Flores": "Abdiel Flores",
            "Angel Gonzalez": "Avigail Gonsalez",
            "Anna Baizephan": "Anna Barseghian",
            "Donahue Conan": "Daniela Garcia",
            "Justice Havel": "Justice Hayes",
            "Klelin Pineda-Galvez": "Klelia Pineda-Galvez",
            "Michelle Chaving": "Michelle Chang",
            "Sheldon Robintic": "Sheldon Rosenthal",
            "Valeri Monelle": "Valerie Mancilla",
            "Xander Bugus": "Xander Reyes",
        },
    ),
    "W2": SectionConfig(
        section="W2",
        course_id="180296",
        assignment_id="2605690",
        exam_max=32.0,
        free_response_questions=("Q23",),
    ),
    "W5": SectionConfig(
        section="W5",
        course_id="194606",
        assignment_id="2627767",
        exam_max=32.0,
        free_response_questions=("Q23",),
    ),
}
