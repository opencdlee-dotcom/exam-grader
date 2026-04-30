"""
Instance-based exam grading prompt builder.
Constructs the prompt that tells Claude how to score student exams
against an extracted answer key, with shift detection.
"""

from grader.handwriting import get_grading_handwriting_rules, ANTI_HALLUCINATION_PROTOCOL
from grader.language_detector import detect_language, is_non_english
from grader.concept_map import LANGUAGE_GRADING_INSTRUCTIONS


def format_exam_grading_prompt(extracted_key: dict, free_response_rubric: dict = None,
                               expected_name: str = None) -> str:
    """
    Build a grading prompt for instance-based exam scoring.
    Supports mixed exams with both objective and free response questions.

    Args:
        extracted_key: Dict from exam_key.extract_answer_key() with
                       'questions' list and 'total_points'
        free_response_rubric: Optional free response rubric dict from
                              free_response.load_rubric() or parse_rubric_text()
        expected_name: Optional known student name (from Canvas/filename) for
                       cross-validation against what's written on the exam

    Returns:
        Formatted prompt string for Claude vision grading
    """
    fr_points = free_response_rubric["total_points"] if free_response_rubric else 0
    objective_points = extracted_key["total_points"]
    total = objective_points + fr_points

    lines = [
        "You are grading a student's handwritten exam/test against a pre-extracted answer key.",
        "",
        "=== SCORING RULES ===",
        "- INSTANCE-BASED scoring: each expected answer term = 1 point",
        f"- Total points possible: {total}",
        "- Match answers by MEANING, not exact wording",
        "- Accept synonyms, abbreviations, alternate spellings, and scientifically equivalent terms",
        "- If a student's answer is correct and equivalent to the expected answer, award the point",
        "- Partial credit: if a question expects 3 terms and the student provides 2 correct ones, award 2 points",
        "- Misspellings: award the point if the intended term is clearly recognizable",
        "",
        "=== ANSWER TOLERANCE (BENEFIT OF THE DOUBT) ===",
        "Your default posture is to AWARD the point. Only withhold when the answer is clearly wrong.",
        "",
        "1. EQUIVALENT EXPRESSIONS: Award the point for any scientifically/factually equivalent expression.",
        "   Examples: 'Na+' = 'sodium ion' = 'sodium' (when context makes ion implicit)",
        "   'H2O' = 'water' = 'dihydrogen monoxide'",
        "   'increase' = 'goes up' = 'rises' = 'gets higher'",
        "",
        "2. HANDWRITING INTERPRETATION: When handwriting is ambiguous, attempt ALL reasonable",
        "   readings before marking wrong. If ANY plausible reading matches the expected answer,",
        "   award the point.",
        "   - Can read clearly on first pass: award the point (confidence: HIGH)",
        "   - Hesitate between two or more readings: award the point BUT mark confidence as LOW",
        "     for human review. Report ALL plausible readings you considered (e.g., 'Read as B,",
        "     but could also be D or P'). If any plausible reading matches, award the point.",
        "   - Cannot determine any plausible reading after maximum effort: mark as ILLEGIBLE —",
        "     do not award or penalize. Report what partial strokes/characters you can see.",
        "",
        get_grading_handwriting_rules(domain="biology"),
        "",
        ANTI_HALLUCINATION_PROTOCOL,
        "",
        "3. LETTER vs WORD ANSWERS: For multiple-choice questions, accept EITHER the letter",
        "   (A, B, C, D) OR the actual word/phrase of the correct answer. Both are equally valid.",
        "   Examples: If the correct answer is 'B) mitosis', accept 'B', 'b', 'mitosis', or 'B mitosis'.",
        "   If the student writes just the letter and it matches the correct choice, award the point.",
        "   If the student writes just the word/phrase and it matches, award the point.",
        "",
        "4. FORMAT VARIATIONS: These NEVER affect scoring:",
        "   - Units: 'mL' = 'ml' = 'ML', '1.5 mL' = '1.5mL' = '1.5 ml'",
        "   - Capitalization: 'dna' = 'DNA' = 'Dna'",
        "   - Plurals: 'mitochondria' = 'mitochondrion'",
        "   - Articles/prepositions: 'the cell membrane' = 'cell membrane'",
        "   - Punctuation, hyphens, spaces: 'wild-type' = 'wild type' = 'wildtype'",
        "   - Chemical notation variants: 'NaCl' = 'sodium chloride'",
        "",
        "5. ORDER INDEPENDENCE: If a question expects multiple terms and the student provides",
        "   all correct terms in ANY order, award full points. Order only matters if the question",
        "   explicitly asks for a sequence or ranking.",
        "",
        "6. PARTIAL KNOWLEDGE: If a student's answer demonstrates understanding of the concept",
        "   but uses imprecise language, award the point. Only penalize if the answer is",
        "   fundamentally wrong or demonstrates a misconception.",
        "   The answer must be SPECIFIC ENOUGH to distinguish the correct term from other plausible answers.",
        "   ACRONYMS: Accept standard domain acronyms if unambiguous (ER = endoplasmic reticulum,",
        "   PCR = polymerase chain reaction, ATP = adenosine triphosphate). If an acronym could",
        "   refer to multiple terms in context, the full term or a more specific variant is required.",
        "   Example: Key says 'exocytosis'. Student writes 'vesicle fusion with membrane' = CORRECT (specific).",
        "   Example: Key says 'endoplasmic reticulum'. Student writes 'cell organelle' = INCORRECT (too vague).",
        "   Example: Key says 'endoplasmic reticulum'. Student writes 'rough ER' = CORRECT (specific variant).",
        "   Example: Key says 'endoplasmic reticulum'. Student writes 'ER' = CORRECT (standard abbreviation).",
        "   Example: Key says 'mitosis'. Student writes 'cell growth' = INCORRECT (misconception).",
        "",
        "7. NUMERIC TOLERANCE: For numeric answers:",
        "   - Decimal answers: accept if within rounding of the last significant digit",
        "     (e.g., '1.53' and '1.5' both acceptable for a 2-sig-fig answer)",
        "   - Accept '1.5' = '1.50' = '~1.5'",
        "   - Whole numbers: exact match required UNLESS the answer is a temperature,",
        "     concentration, or measurement where ±1 unit is within reasonable experimental",
        "     or rounding tolerance.",
        "",
        "   TEMPERATURE-SPECIFIC TOLERANCE:",
        "   - For PCR/biology temperatures: accept ±1°C of the expected value.",
        "     94°C and 95°C are BOTH acceptable for denaturation temperature.",
        "     54°C and 55°C are BOTH acceptable for annealing temperature.",
        "     71°C and 72°C are BOTH acceptable for extension temperature.",
        "   - For boiling/freezing points: accept ±1°C (99-100°C, -1 to 0°C).",
        "   - For body temperature: accept 36.5-37.5°C or 98-99°F.",
        "   - When in doubt about whether a numeric answer is 'close enough',",
        "     award the point and mark confidence as LOW for human review.",
        "",
        "8. CROSSED-OUT ANSWERS: Students often cross out, scribble over, or erase answers and",
        "   rewrite. ALWAYS use the student's FINAL answer — the one that is NOT crossed out.",
        "   - If an answer is crossed out with a new answer beside/below it, grade the new one only",
        "   - If ALL attempts are crossed out with no replacement, treat as unanswered",
        "   - If two answers coexist for the same question with no clear crossing-out, mark",
        "     confidence as LOW, flag as AMBIGUOUS, and award points for whichever answer is",
        "     correct (benefit of the student). Explain both answers you see.",
        "   - NEVER penalize a student for changing their mind — only the final answer matters",
        "",
        "9. DIAGRAMS AND DRAWINGS: If a student draws a labeled diagram, flowchart, or visual",
        "   instead of writing a text answer, award PARTIAL credit if the drawing demonstrates",
        "   understanding of the concept. A diagram alone does not deserve full credit — the student",
        "   should provide the actual term or text answer. However, if the diagram clearly shows they",
        "   know the concept (e.g., drawing the process of mitosis with labeled phases instead of",
        "   writing 'mitosis'), award at least half credit for that term.",
        "",
        "10. PARAGRAPH-STYLE ANSWERS: If a question expects multiple distinct terms (e.g., 'list 3",
        "   types of cell division') and the student writes a paragraph instead of a numbered list,",
        "   extract each correct term from the prose and credit it. Do not penalize for format.",
        "   Example: Key expects 'mitosis, meiosis, binary fission'. Student writes 'Cells can divide",
        "   through mitosis or meiosis, and bacteria use binary fission.' = award all 3 points.",
        "",
        "11. ESL / NON-NATIVE PHRASING: Accept grammatically imperfect but scientifically correct",
        "    answers. Focus on whether the student demonstrates understanding of the concept, not",
        "    whether their English is polished.",
        "    Example: 'the thing in cell that make energy' for 'mitochondria' = INCORRECT (doesn't name it).",
        "    Example: 'mitocondria is where energy is make' for 'mitochondria' = CORRECT (term is there,",
        "    grammar doesn't matter).",
        "    Example: 'diffusion mean molecule go from high to low' for 'diffusion' = CORRECT (concept shown).",
        "",
        "=== SHIFT DETECTION (CRITICAL) ===",
        "Students sometimes spread a multi-part answer across multiple question lines.",
        "",
        "Example of shifting:",
        "  Answer key Q1 expects: mitosis, meiosis, binary fission (3 answers)",
        "  Student writes:",
        "    Q1 line: mitosis",
        "    Q2 line: meiosis",
        "    Q3 line: binary fission",
        "  This pushes Q2's actual answer to Q4's line, Q3's to Q5's line, etc.",
        "",
        "DETECTION PROCEDURE:",
        "1. First, read ALL student answers exactly as written (by question number/line)",
        "2. Check if the content of answers appears SHIFTED — correct terms appearing",
        "   in sequential wrong question slots",
        "3. If a shift is detected, report EXACTLY what you see: the original placement",
        "   AND your proposed realignment. Award points based on the REALIGNED answers.",
        "4. Mark ALL realigned questions as confidence: LOW — a human must verify",
        "   that the realignment is correct before the grade is final.",
        "5. A student should NOT be penalized for putting each instance on its own line",
        "6. Only realign if the shift pattern is CONSISTENT: at least 80% of subsequent",
        "   answers must be displaced by the same offset (e.g., all shifted down by 1).",
        "   If fewer than 80% match the same offset, flag for HUMAN REVIEW without",
        "   automatic realignment. Report the raw readings and your shift hypothesis.",
        "",
        "Signs of shifting:",
        "- A multi-answer question only has 1 answer, but the next question(s) contain",
        "  terms that belong to the previous question",
        "- Answers for later questions appear offset by a consistent number of lines",
        "- The content matches the key but is displaced by N positions",
        "",
        "=== ANSWER HUNTING (CRITICAL) ===",
        "Students do NOT always write answers on the designated answer line.",
        "Before scoring ANY question as blank or missing:",
        "",
        "1. Scan ALL submitted pages — front AND back of every page",
        "2. Check margins, spaces between questions, and any continuation areas",
        "3. Look for cues: 'see back', 'continued on back', arrows, circled content on other pages",
        "4. If a student wrote the answer ANYWHERE in their submission, credit it ONLY if it is",
        "   EXPLICITLY attributable to a specific question by one of these means:",
        "   - The student wrote a question number next to it (e.g., 'Q5: mitosis')",
        "   - There is an arrow or 'see back' reference pointing from the question to the answer",
        "   - The answer is written in the continuation space directly below the question",
        "   If the answer is found in an ambiguous location (random margin note, back of page",
        "   with no question reference), mark confidence as LOW and explain where you found it.",
        "5. Only mark a question as unanswered if you have checked every page and found nothing",
        "",
        "This is SEPARATE from shift detection. Shift detection handles sequential displacement.",
        "Answer hunting handles answers placed in arbitrary locations (backs of pages, margins, etc.).",
        "",
        "=== ANTI-BIAS PROTOCOL ===",
        "When scoring each question:",
        "1. First determine WHAT the student wrote (content extraction)",
        "2. Then determine if it matches the expected answer (scoring)",
        "3. NEVER let handwriting neatness, organization, or apparent effort influence scoring.",
        "   A messy but correct answer gets full credit. A neat but wrong answer gets zero.",
        "",
        "=== FAINT CONTENT PROTOCOL ===",
        "Some pages may have faded ink, light pencil, or poor scan quality that makes content",
        "hard to see. This is NOT the same as a blank answer. Follow these rules:",
        "",
        "1. NEVER score a question as 0 just because the writing is faint. Faint ≠ absent.",
        "2. If you can see ANY marks in an answer region — even barely visible ones — attempt",
        "   to read them. Adjust your contrast perception and look for stroke patterns.",
        "3. If you can partially read faint content (e.g., you can make out 3 of 5 characters",
        "   in a DNA complement strand), report what you CAN read and mark confidence as LOW.",
        "4. For multi-part answers (like complement strands, sequences, lists): if part of the",
        "   answer is clear and part is faint, award points for the clear parts. Do NOT zero",
        "   out the entire answer because one section is hard to read.",
        "5. If a page is so faint that you cannot read ANY content with confidence, note:",
        "   'PAGE FAINT — content appears present but is too faded to read reliably.'",
        "   Score as 0 but flag for MANDATORY human review.",
        "6. In SCAN/FORMAT NOTES, always report which pages/questions had faint content.",
        "",
        "=== PARTIAL CREDIT UNCERTAINTY ===",
        "When awarding partial credit on free response or multi-part questions:",
        "",
        "1. If the rubric has EXPLICIT criteria levels that clearly match the student's answer,",
        "   mark confidence as HIGH.",
        "2. If you are making a JUDGMENT CALL — the student's answer falls between two rubric",
        "   levels, or the rubric doesn't precisely cover what the student wrote — mark",
        "   confidence as LOW and explain your reasoning.",
        "3. For partial credit estimates (e.g., 0.5 or 1.5 on a 3-point question), ALWAYS",
        "   explain: 'Awarded X points because [specific reasoning]. This is a judgment call",
        "   — human review recommended.'",
        "4. Never silently award partial credit without justification. Every non-integer score",
        "   or score that doesn't match a named rubric level must be explained.",
        "",
        "=== STUDENT NAME VERIFICATION ===",
        "Before grading, identify the student's name from their exam submission.",
        "",
        "1. LOCATE THE NAME: Check the top of page 1 for a name field (often labeled",
        "   'Name:', 'Student:', 'Student Name:', or a printed/handwritten name at the top).",
        "   Also check page headers, footprints, and the top of subsequent pages.",
        "2. READING PROTOCOL: Apply the same handwriting interpretation rules as answers.",
        "   - Read first name and last name separately",
        "   - For ambiguous letters, try ALL plausible readings",
        "   - Common confusion pairs in names: l/I, m/n, a/o, e/c, u/v, r/n",
        "3. REPORT FORMAT: Report the name EXACTLY as you read it from the exam.",
        "   - If the name is clearly legible: report it with confidence HIGH",
        "   - If parts are ambiguous: report your best reading with confidence LOW",
        "     and note which characters are uncertain (e.g., 'Maria Gonzalez [confidence: LOW,",
        "     first letter of last name could be C or G]')",
        "   - If no name is found anywhere: report 'NOT FOUND' with confidence HIGH",
        "4. ALTERNATE READINGS: If the name is not 100% clear, also report alternate readings",
        "   in the STUDENT NAME field. Format: 'Best Reading (or: Alt1, Alt2)'.",
        "   Example: 'Moria Gonzalez (or: Maria Gonzalez, Moira Gonzalez) [confidence: LOW]'",
        "   This helps downstream matching when the name doesn't exactly match a roster.",
        "",
    ]

    if expected_name:
        lines.extend([
            f"5. CROSS-VALIDATION: The expected student for this exam is '{expected_name}'.",
            "   - If the name on the exam matches or is a plausible handwriting variant of the",
            "     expected name, report the expected name spelling and confidence HIGH.",
            "   - If the name on the exam is CLEARLY DIFFERENT from the expected name (different",
            "     person entirely, not just a handwriting misread), flag as NAME MISMATCH.",
            "     Report BOTH names: what you read and what was expected.",
            "   - A mismatch may mean: wrong exam in the file, student submitted another's work,",
            "     or the LMS record is wrong. This MUST be flagged for human review.",
            "",
        ])

    lines.extend([
        "",
        "=== SCORE GUARDRAILS ===",
        f"- The total score MUST NOT exceed {total}. Cap at full credit even if extra correct answers found.",
        "- If NO student work is visible on any page (blank pages, wrong document, or only printed",
        "  questions with no student writing), return GRADE: 0 / " + str(total) + " with a note explaining",
        "  that no student work was found. Do NOT guess or infer answers from ambiguous marks.",
        "  NOTE: Programmatic blank-page detection runs before you see the images. If you still",
        "  see no student writing, this confirms the submission is blank — score 0 with confidence HIGH.",
        "",
        "=== TWO-PHASE READING PROTOCOL (MANDATORY) ===",
        "For EVERY question, you MUST perform two separate cognitive steps:",
        "",
        "STEP 1 — RAW READ: Report exactly what you see written in the answer region.",
        "  - For single characters (letters, digits): state the character AND list all",
        "    plausible alternate readings based on the handwriting strokes.",
        "    Example: raw_reading='D', alternative_readings=['B','P']",
        "  - For words/phrases: transcribe verbatim before applying any interpretation.",
        "    Example: raw_reading='conjagation' (misspelled but readable)",
        "  - For sequences (DNA, numbers): transcribe each character individually.",
        "  - For blank/illegible: report 'BLANK' or 'ILLEGIBLE' with any partial strokes visible.",
        "  - Do NOT apply scoring rules, synonyms, or tolerance in this step.",
        "",
        "STEP 2 — SCORE: Apply the answer key, tolerance rules, and rubric to your Step 1 reading.",
        "  - Record your final interpretation in student_answer.",
        "  - Record points in points_earned.",
        "  - If an alternative_reading from Step 1 would match the key but your primary",
        "    reading does not, set reading_confidence='LOW' and explain in reasoning.",
        "",
        "Report BOTH steps in your output: raw_reading (Step 1) and student_answer (Step 2).",
        "This separation is critical — it allows auditing whether a wrong score came from",
        "misreading handwriting vs. the student genuinely answering incorrectly.",
        "",
        "=== ANSWER KEY ===",
        "NOTE: '(also accept: ...)' lists are ADDITIONAL known-good answers, not the ONLY acceptable",
        "alternatives. Any scientifically equivalent expression should also be accepted per the",
        "tolerance rules above, even if not explicitly listed here.",
    ])

    for q in extracted_key["questions"]:
        alternatives = q.get("alternatives", {})
        answer_parts = []
        for ans in q["answers"]:
            alts = alternatives.get(ans, [])
            if alts:
                answer_parts.append(f"{ans} (also accept: {', '.join(alts)})")
            else:
                answer_parts.append(ans)
        answers_str = ", ".join(answer_parts) if answer_parts else "(no answer expected)"
        tolerance_note = ""
        if q.get("tolerance"):
            tolerance_note = f" [tolerance: {q['tolerance']}]"
        lines.append(
            f"Q{q['number']} ({q['points']} pt{'s' if q['points'] != 1 else ''}): {answers_str}{tolerance_note}"
        )

    # Append free response rubric if provided
    if free_response_rubric:
        from grader.free_response import format_free_response_prompt
        lines.append(format_free_response_prompt(free_response_rubric))

    lines.extend([
        "",
        "=== OUTPUT FORMAT ===",
        "Respond in EXACTLY this format:",
        "",
        "STUDENT NAME: [Full name as read from exam] [confidence: HIGH/LOW]",
    ])

    if expected_name:
        lines.append("NAME MISMATCH: [Yes/No — only if name on exam differs from expected student]")

    lines.extend([
        "",
        "SHIFT DETECTED: [Yes/No]",
        "SHIFT DETAILS: [If yes, describe the shift pattern — e.g., 'Student spread Q1 answers "
        "across Q1-Q3, shifting all subsequent answers down by 2'. If no, write 'N/A']",
        "",
        f"GRADE: [score] / {total}",
    ])

    if free_response_rubric:
        lines.append(f"  (Objective: [obj_score]/{objective_points} + Free Response: [fr_score]/{fr_points})")

    lines.extend([
        "",
        "QUESTION BREAKDOWN:",
    ])

    for q in extracted_key["questions"]:
        lines.append(
            f"Q{q['number']} ({q['points']} pt{'s' if q['points'] != 1 else ''}): "
            f"[earned]/[{q['points']}] [confidence: HIGH/LOW] — [list which terms were found correct, which were missing or wrong]"
        )

    if free_response_rubric:
        lines.extend([
            "",
            "FREE RESPONSE BREAKDOWN:",
        ])
        for section in free_response_rubric["sections"]:
            lines.append(
                f"FR{section['number']} ({section['points']} pt{'s' if section['points'] != 1 else ''}): "
                f"[earned]/[{section['points']}] [confidence: HIGH/LOW] — [FULL deduction explanation: "
                f"what rubric required → what student wrote → what was missing/wrong → which rubric level matched]"
            )

    lines.extend([
        "",
        "CONFIDENCE KEY: HIGH = confident in reading and scoring. LOW = handwriting ambiguous,",
        "answer borderline, or scoring required significant interpretation. Flag LOW items for",
        "human review.",
        "",
        "COMMENTS:",
        "[Brief feedback: notable correct answers, common errors, and any shift compensation applied]",
        "",
        "SCAN/FORMAT NOTES (if applicable):",
        "[Any issues with handwriting legibility, scan quality, or unusual formatting]",
    ])

    return "\n".join(lines)


def inject_pre_read_answers(prompt: str, pre_read: dict[int, str | None]) -> str:
    """
    Inject high-resolution crop-based readings into the grading prompt.

    The pre-read answers act as a strong prior from 300 DPI cropped images.
    Claude should use these as primary readings, only overriding if the
    full page image clearly contradicts.

    Args:
        prompt: The existing grading prompt from format_exam_grading_prompt()
        pre_read: {question_number: answer_string} from answer_sheet_reader

    Returns:
        Modified prompt with pre-read answers injected before OUTPUT FORMAT
    """
    if not pre_read:
        return prompt

    strong_lines = []
    weak_lines = []

    for q_num in sorted(pre_read.keys()):
        ans = pre_read[q_num]
        if ans is None:
            continue

        is_strong = False
        if isinstance(ans, dict):
            primary = ans.get("primary") or "?"
            alt = ans.get("alternate")
            conf = str(ans.get("confidence", "low")).upper()
            if alt:
                label = f"{primary} (or {alt}?) [confidence: {conf}]"
            else:
                label = f"{primary} [confidence: {conf}]"
            is_strong = conf == "HIGH" and not alt
        else:
            label = str(ans)
            is_strong = True

        target = strong_lines if is_strong else weak_lines
        target.append(f"Q{q_num}: {label}")

    lines = [
        "",
        "=== PRE-READ ANSWERS (from high-resolution 300 DPI crops) ===",
        "A prior crop-based reading pass extracted the answers below from individually",
        "cropped regions before full-page grading.",
        "",
    ]

    if strong_lines:
        lines.extend([
            "Strong crop evidence:",
            "Use these as a strong prior unless the full-page image clearly contradicts them.",
            *strong_lines,
            "",
        ])

    if weak_lines:
        lines.extend([
            "Low-confidence crop evidence:",
            "These are NOT reliable answer priors. Use them only as attention guides for",
            "questions that deserve extra scrutiny. If your full-page read remains uncertain,",
            "mark confidence as LOW and report the competing readings you considered.",
            *weak_lines,
            "",
        ])

    lines.append("")

    # Insert before OUTPUT FORMAT section
    insert_marker = "=== OUTPUT FORMAT ==="
    if insert_marker in prompt:
        return prompt.replace(insert_marker, "\n".join(lines) + "\n" + insert_marker)

    # Fallback: append to end
    return prompt + "\n" + "\n".join(lines)


def inject_language_instructions(prompt: str, student_answer: str) -> str:
    """Inject multilingual grading instructions if student answered in a non-English language."""
    lang = detect_language(student_answer)
    if lang == "en":
        return prompt
    instruction = LANGUAGE_GRADING_INSTRUCTIONS.get(lang, "")
    if not instruction:
        return prompt
    # Inject after the first paragraph / before the rubric section
    return instruction + "\n\n" + prompt
