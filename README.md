# exam-grader

Multi-modal exam grading suite. Extracts answer keys from teacher copies, grades student exams with ensemble providers (Claude + GPT + Gemini), audits for bias and rubric drift, and posts to Canvas.

Bundles three packages:
- `grader/` — core grading pipeline
- `grading_intelligence/` — ensemble orchestration, prompt library, math grader, structured output
- `canvas/` — Canvas API + Speedgrader posting via Playwright

## Install

```bash
pip install git+https://github.com/opencdlee-dotcom/exam-grader.git
```

For Canvas posting:

```bash
pip install "exam-grader[canvas] @ git+https://github.com/opencdlee-dotcom/exam-grader.git"
playwright install chromium
```

For all optional providers:

```bash
pip install "exam-grader[all] @ git+https://github.com/opencdlee-dotcom/exam-grader.git"
```

## Usage

```python
from grader.pipeline import grade_submission, grade_exam_submission
from grader.exam_key import extract_answer_key
from grading_intelligence.ensemble_grader import EnsembleGrader, EnsembleConfig
from canvas.client import CanvasClient
```

## Source of truth

This is the canonical home for the exam grader. `professor-os` and any other consumer pulls from here via pip.
