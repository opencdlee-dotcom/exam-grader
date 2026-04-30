"""
Constraint Propagation for Matching Sections — Sudoku-Style Disambiguation.

In matching sections, each answer letter is used exactly once. This constraint
means resolving one ambiguity can cascade to resolve others — for free, with
zero image re-reads.

Example:
    Key: Q9=L, Q10=M, Q11=P, Q12=K, Q13=O, Q14=H, Q15=G, Q16=B, Q17=D, Q18=I
    Student answers (with ambiguity):
        Q9=L (confident), Q10=M (confident), Q11=P (confident),
        Q12=K (confident), Q13=? (B or D), Q14=H (confident),
        Q15=G (confident), Q16=B (confident), Q17=? (B or D), Q18=I (confident)

    After constraint propagation:
        Q16=B is confident → B is consumed → Q13 and Q17 cannot be B
        Q13: was B/D → must be D (not in key for Q13, but B eliminated)
        Q17: was B/D → must be D ✓ (matches key)
        Resolved 1 of 2 ambiguities. Q13=D is still wrong (key=O), but we
        correctly determined it's D not B.

Integration:
    Insert into Disambiguation Cascade between Step 3 (glyph fingerprint)
    and Step 4 (multi-read vote). If constraint propagation resolves the
    ambiguity, skip the remaining (expensive) cascade steps.

Usage:
    from grader.constraint_solver import propagate_matching_constraints

    result = propagate_matching_constraints(
        written={"Q9": "L", "Q10": "M", ...},
        ambiguous={"Q13": ["B", "D"], "Q17": ["B", "D"]},
        key={"Q9": "L", "Q10": "M", ..., "Q18": "I"},
    )
    # result.resolved = {"Q17": "D"}
    # result.still_ambiguous = {"Q13": ["D"]}  # B eliminated, but only D left
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Resolution:
    """A single resolved ambiguity."""
    question: str
    resolved_letter: str
    eliminated: list[str]
    reason: str
    matches_key: bool


@dataclass
class PropagationResult:
    """Result of constraint propagation across a matching section."""
    resolved: dict[str, str]            # {Q13: "D"} — fully resolved
    still_ambiguous: dict[str, list[str]]  # {Q17: ["B", "D"]} — narrowed but not resolved
    resolutions: list[Resolution]       # audit trail of each resolution step
    iterations: int                     # how many propagation passes were needed
    letters_used: set[str]              # all confidently assigned letters
    letters_available: set[str]         # letters not yet assigned

    @property
    def any_resolved(self) -> bool:
        return len(self.resolved) > 0

    @property
    def fully_resolved(self) -> bool:
        return len(self.still_ambiguous) == 0


def propagate_matching_constraints(
    written: dict[str, str],
    ambiguous: dict[str, list[str]],
    key: dict[str, str],
    *,
    all_answer_letters: Optional[set[str]] = None,
    max_iterations: int = 20,
) -> PropagationResult:
    """
    Resolve ambiguous matching answers using constraint propagation.

    Each answer letter in a matching section is used exactly once. If a letter
    is confidently assigned to one question, it's eliminated from all other
    questions' candidate sets.

    Args:
        written: Confident answers. {Q9: "L", Q10: "M", ...}
                 Only include answers where the letter is NOT ambiguous.
        ambiguous: Ambiguous answers with candidate lists.
                   {Q13: ["B", "D"], Q17: ["B", "D"]}
        key: The answer key for the matching section.
             {Q9: "L", Q10: "M", ..., Q18: "I"}
        all_answer_letters: The complete set of valid answer letters.
                            If None, derived from key values.
        max_iterations: Safety limit on propagation passes.

    Returns:
        PropagationResult with resolved answers, remaining ambiguities,
        and an audit trail of each resolution step.
    """
    if all_answer_letters is None:
        all_answer_letters = {v.upper() for v in key.values()}

    # Initialize state
    confident = {q: letter.upper() for q, letter in written.items()}
    candidates = {q: [c.upper() for c in cands] for q, cands in ambiguous.items()}
    resolutions: list[Resolution] = []

    letters_used = set(confident.values())
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        changed = False

        # Step 1: Eliminate used letters from all candidate sets
        for q, cands in list(candidates.items()):
            original = list(cands)
            filtered = [c for c in cands if c not in letters_used]

            if len(filtered) < len(original):
                eliminated = [c for c in original if c not in filtered]
                candidates[q] = filtered
                changed = True

                # If exactly one candidate remains, it's resolved
                if len(filtered) == 1:
                    resolved_letter = filtered[0]
                    confident[q] = resolved_letter
                    letters_used.add(resolved_letter)
                    del candidates[q]
                    matches = resolved_letter == key.get(q, "").upper()
                    resolutions.append(Resolution(
                        question=q,
                        resolved_letter=resolved_letter,
                        eliminated=eliminated,
                        reason=f"Only candidate remaining after eliminating {', '.join(eliminated)} (used elsewhere)",
                        matches_key=matches,
                    ))
                elif len(filtered) == 0:
                    # All candidates eliminated — this shouldn't happen with
                    # correct input, but handle gracefully
                    del candidates[q]
                    resolutions.append(Resolution(
                        question=q,
                        resolved_letter="?",
                        eliminated=original,
                        reason=f"All candidates eliminated — possible input error",
                        matches_key=False,
                    ))

        # Step 2: Naked singles — if a letter appears in only one candidate set,
        # that question must have that letter (even if the set has other candidates)
        unresolved_qs = list(candidates.keys())
        if unresolved_qs:
            # Find letters that are still available (not in confident)
            remaining_letters = all_answer_letters - letters_used
            for letter in remaining_letters:
                # Which unresolved questions could have this letter?
                possible_qs = [q for q in unresolved_qs if letter in candidates.get(q, [])]
                if len(possible_qs) == 1:
                    # This letter can only go in one place
                    q = possible_qs[0]
                    eliminated = [c for c in candidates[q] if c != letter]
                    confident[q] = letter
                    letters_used.add(letter)
                    del candidates[q]
                    matches = letter == key.get(q, "").upper()
                    resolutions.append(Resolution(
                        question=q,
                        resolved_letter=letter,
                        eliminated=eliminated,
                        reason=f"'{letter}' can only appear at {q} (naked single)",
                        matches_key=matches,
                    ))
                    changed = True

        if not changed:
            break

    letters_available = all_answer_letters - letters_used

    return PropagationResult(
        resolved={r.question: r.resolved_letter for r in resolutions if r.resolved_letter != "?"},
        still_ambiguous=dict(candidates),
        resolutions=resolutions,
        iterations=iteration,
        letters_used=letters_used,
        letters_available=letters_available,
    )


def build_matching_state_from_log(
    matching_log: str,
    key: dict[str, str],
    confidence_threshold: float = 0.80,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Parse a matching log into confident + ambiguous answer sets.

    Answers marked with checkmarks/crossmarks are confident.
    This function is a convenience bridge between matching_audit.py's log format
    and the constraint solver's input format.

    Args:
        matching_log: Log string like "Q9=L✓ Q10=M✓ Q11=J✗(P) → 3/4"
        key: Answer key {Q9: "L", Q10: "M", ...}
        confidence_threshold: Not used for log-based parsing (all logged = confident)

    Returns:
        (confident, ambiguous) tuple for propagate_matching_constraints()
    """
    import re
    confident = {}
    # Parse Q9=L from the log — all logged answers are confident reads
    for m in re.finditer(r'(Q\d+)=([A-Za-z]+)[✓✗]', matching_log):
        q = m.group(1)
        letter = m.group(2).upper()
        confident[q] = letter

    # Any key questions NOT in the log are unaccounted — not ambiguous per se
    # but missing. We don't create ambiguous entries for them since we don't
    # have candidate lists. The completeness gate handles missing questions.

    return confident, {}


def solve_from_log(
    matching_log: str,
    key: dict[str, str],
    ambiguous_overrides: Optional[dict[str, list[str]]] = None,
) -> PropagationResult:
    """
    Convenience: parse log, apply overrides for known ambiguities, and solve.

    Args:
        matching_log: Matching section log string
        key: Answer key
        ambiguous_overrides: Manually specified ambiguities from the grading agent
                             e.g. {"Q13": ["B", "D"]} when the agent flagged Q13

    Returns:
        PropagationResult
    """
    confident, _ = build_matching_state_from_log(matching_log, key)
    ambiguous = ambiguous_overrides or {}

    # Remove any ambiguous questions from confident (override takes precedence)
    for q in ambiguous:
        confident.pop(q, None)

    return propagate_matching_constraints(confident, ambiguous, key)


def format_propagation_report(result: PropagationResult, key: dict[str, str]) -> str:
    """Format propagation result as a human-readable report."""
    lines = ["CONSTRAINT PROPAGATION REPORT", "=" * 50]

    if not result.resolutions:
        lines.append("No ambiguities to resolve (all answers were confident).")
        return "\n".join(lines)

    lines.append(f"Iterations: {result.iterations}")
    lines.append(f"Resolved: {len(result.resolved)} questions")
    lines.append(f"Still ambiguous: {len(result.still_ambiguous)} questions")
    lines.append("")

    for r in result.resolutions:
        correct_mark = " (CORRECT)" if r.matches_key else f" (key={key.get(r.question, '?')})"
        lines.append(f"  {r.question} → {r.resolved_letter}{correct_mark}")
        lines.append(f"    Eliminated: {', '.join(r.eliminated)}")
        lines.append(f"    Reason: {r.reason}")
        lines.append("")

    if result.still_ambiguous:
        lines.append("Unresolved:")
        for q, cands in result.still_ambiguous.items():
            lines.append(f"  {q}: candidates = {', '.join(cands)}")

    if result.letters_available:
        lines.append(f"\nUnused letters: {', '.join(sorted(result.letters_available))}")

    lines.append("=" * 50)
    return "\n".join(lines)
