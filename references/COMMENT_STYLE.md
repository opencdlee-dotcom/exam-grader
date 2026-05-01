# Comment Style — Canonical Reference

This is how the professor writes comments to students. Every LLM-generated
comment in the hub (lab notebooks, exams, free response) must sound like it
came from her, not from a generic grader.

The phrases below are pulled verbatim from
`hub/notebook_grader/rubric.py::COMMENT_BANK_VARIANTS` — they are the source
of truth. When in doubt, mimic them.

---

## 1. Voice

- **First person, conversational, instructional.** Not a bot, not a TA report.
  The student is reading a note from their teacher.
- **Kind first, corrective second.** Open with something genuine before any
  deduction lands.
- **No hedging or apologizing.** Deductions are stated plainly with the point
  amount in parens; no "unfortunately", no "I had to", no "sadly".
- **No academic-speak.** Don't say "demonstrates insufficient depth" — say
  "your conclusion was too short".

## 2. Three-part structure

Every comment is assembled in this order:

1. **Opener** — one short positive sentence keyed to the strongest element of
   the submission (table / graph / diagram / data / notes). If a specific
   compliment is available, use that instead.
2. **Deductions** — each one is its own sentence, cited concretely, with the
   point amount in parentheses at the end.
3. **Close** — for a perfect paper, a single warm sign-off.

If there are no deductions, skip step 2 and use only the opener + close.

## 3. Opener phrasings (verbatim)

Pick one and rotate; never start every comment the same way.

- "Nice work on your table." / "Your data table was well-organized."
- "Nice work on your graph." / "Your graph looked great."
- "Your diagram was clear and well-labeled."
- "Your data collection was thorough." / "Good job recording your data."
- "Your notebook was well-organized." / "Your notes were thorough."

When a specific compliment is available (e.g. "Your trendline fit was
excellent and your R² was 0.998"), prefer it over the generic opener — but
keep the same warm, one-sentence shape.

## 4. Deduction phrasings (verbatim)

These are the lines to use or echo. **Always end with the point amount in
parentheses, e.g. `(-5)`.** Never use the form `-5 pts` or `Deduction: 5`.

- "Missing whale stamp. (-5)"
- "Missing table of contents. (-1)"
- "Missing purpose of experiment. (-5)"
- "Missing M&M. (-5)" / "Missing Materials & Methods section. Please include your procedure in your own words. (-5)"
- "Missing results. (-5)"
- "Missing graph with trendline/R². (-5)"
- "Label graph axes. (-1)" / "Please label your graph axes. (-1)"
- "Include units on axes. (-1)"
- "Show step-by-step work. (-2.5)"
- "Write out equations before plugging in. (-2.5)"
- "Math errors, followed your mistake through. (-2)"
- "Missing required values. (-2)"

## 5. Close phrasings (perfect score)

- "Great job! Keep up the excellent work."
- "Excellent work, keep it up!"
- "Perfect score, well done!"
- "Outstanding notebook, keep up the great work."

## 6. Signature moves

These are the user's *recognizable* phrases. Use them whenever the situation
fits — they are part of how she sounds.

- **Following the mistake through.** When math is wrong but the method is
  right: *"There were math errors in your calculations. I followed your
  mistake through so I only took off 2 points. (-2)"* This is first-person,
  and it tells the student her thought process. Always pair this phrasing
  with `(-2)`, not the full deduction.
- **Asking before scolding.** First-time format issues use a warning, not a
  deduction: *"PLEASE MAKE SURE THE FILE IS IN PROPER ORIENTATION AND PLEASE
  USE CAMSCANNER OR ADOBE SCANNER. POINTS WILL BE DEDUCTED NEXT TIME."*
- **Resubmit instead of zero.** Blank or near-blank submissions get a chance:
  *"Please resubmit. This is a blank document. I will allow you to resubmit.
  If this happens again it will be graded as a 0."*
- **Concrete checklists for missing analysis.** When the conclusion is gone
  or skeletal, lay out the four things it should contain: (1) what you
  observed and whether results matched your hypothesis, (2) a scientific
  explanation, (3) specific references to data values or graph trends,
  (4) sources of error and how to improve. This list shape is canonical —
  reuse it.

## 7. Thin-but-present analysis: feedback without a deduction

When the conclusion is short but real, do **not** deduct — just coach:

> "Good start on your conclusion! To make it stronger, try to reference
> specific numbers or trends from your data/graph, and explain the
> scientific reasoning behind why your results turned out the way they did.
> Adding a sentence on sources of error or how you could improve the
> experiment would also help."

This is the THIN_ANALYSIS pattern. It is positive, names exactly what's
missing, and leaves the points alone. Use it whenever the student tried but
fell short of the bar.

## 8. Variant rotation

The bank has 2-3 phrasings for most keys. The grader rotates them at random
so two students never get word-identical comments. When generating new
comments, vary your phrasing too — don't lock into a single template.

## 9. What to avoid

- **Canned-sounding hedges.** No "unfortunately", "I'm afraid", "I regret to
  say", "please note that".
- **Pure negativity.** Never list deductions without an opener or
  acknowledgement.
- **Vague feedback.** "Could be improved" is meaningless. Name the specific
  missing thing (e.g., "include a trendline equation", "label the y-axis").
- **Inflating point values in prose.** If the deduction is 2 points, the
  comment says `(-2)` — don't say "this cost you significant credit" or
  round up.
- **Lecturing.** Two short sentences beats one long one. The student is busy.
- **AI tells.** No em dashes used as drama beats, no "it's not just X — it's
  Y", no "let's dive in", no "indeed", no "essentially".

## 10. Lab notebook vs exam comments

The two contexts use the same voice but different structures.

**Lab notebooks** are the long-form case: opener → multiple deductions →
close. The deduction list can run 3-5 items because labs have more
categories. Each deduction names the section (M&M, results, graph,
calculations, analysis).

**Exams** are tighter. The COMMENTS field on an exam is one short paragraph
that:
- Names what the student did well ("good handling of the dilution math",
  "clear DNA complement strands"),
- Notes 1-2 common errors as patterns rather than a per-question list (the
  per-question breakdown lives in QUESTION BREAKDOWN, not COMMENTS),
- Mentions any shift compensation that was applied so the student knows.

For free response, deductions go inside FREE RESPONSE BREAKDOWN with the
explicit chain *what rubric required → what student wrote → what was
missing/wrong → which rubric level matched*. Keep it factual; the COMMENTS
field stays warm and high-level.

## 11. Quick checklist before emitting a comment

- [ ] Starts with a positive opener (or a specific compliment).
- [ ] Each deduction is concrete, names the section, ends in `(-N)`.
- [ ] No canned hedges, no academic-speak, no AI tells.
- [ ] If math errors: include "I followed your mistake through" phrasing
      and cap at `(-2)`.
- [ ] If analysis is thin-but-present: feedback only, no deduction.
- [ ] If perfect: opener + one warm close, that's it.
- [ ] Sounds like a teacher, not a rubric.
