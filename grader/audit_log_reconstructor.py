"""
Audit Log Reconstructor — generates structured per-section breakdown logs
from stored q_scores JSON and validates arithmetic closure.

HIGH confidence is only granted when the reconstructed section sums
match the stored total exactly. Any mismatch → SCORE_AUDIT flag.

Inspired by financial audit reconstruction: artifacts (q_scores) independently
corroborate every line of the log, and closure is the proof of correctness.
"""
import json
import glob
from typing import Optional


# ── Schema definitions ─────────────────────────────────────────────────────────

SCHEMAS = {
    "T5": {
        "mc":      list(range(1, 9)),     # Q1-Q8
        "match":   list(range(9, 19)),    # Q9-Q18
        "fill":    list(range(19, 23)),   # Q19-Q22
        "fr_key":  "fr_score",
        "dna_key": "q24_dna",
        "codon_keys": ["q25a", "q25b", "q25c"],
        "calc_key": "q26_calc",
        "total_pts": 32,
        "match_key": {9:'L',10:'M',11:'P',12:'K',13:'O',14:'H',15:'G',16:'B',17:'D',18:'I'},
        "fill_key":  {19:'Denaturation',20:'Conjugation',21:'DNA',22:'Faster'},
    },
    "W2": {
        "mc":      list(range(1, 10)),    # Q1-Q9
        "match":   list(range(10, 19)),   # Q10-Q18
        "fill":    list(range(20, 24)),   # Q20-Q23 (Q19 blank)
        "fr_key":  "fr_score",
        "dna_key": "q25_dna",
        "codon_keys": ["q26a", "q26b", "q26c"],
        "calc_key": "q27_calc",
        "total_pts": 32,
        "match_key": {10:'K',11:'H',12:'B',13:'O',14:'I',15:'G',16:'M',17:'L',18:'P'},
        "fill_key":  {20:'Taq polymerase',21:'95',22:'Primers',23:None},
    },
    "W5": {
        "mc":      list(range(1, 9)),     # Q1-Q8
        "match":   list(range(9, 19)),    # Q9-Q18
        "fill":    list(range(19, 23)),   # Q19-Q22
        "fr_key":  "fr_score",
        "dna_key": "q24_dna",
        "codon_keys": ["q25a", "q25b", "q25c"],
        "calc_key": "q26_calc",
        "total_pts": 32,
        "match_key": {9:'M',10:'K',11:'P',12:'I',13:'O',14:'H',15:'G',16:'L',17:'B',18:'D'},
        "fill_key":  {19:'Denaturation',20:'Conjugation',21:'DNA',22:'Faster'},
    },
}


def _check_mark(score: float, correct: Optional[str], given: Optional[str] = None) -> str:
    return "✓" if score == 1 else f"✗({correct or '?'})"


def reconstruct_audit_log(student: dict, schema: dict) -> dict:
    """
    Generate structured audit log lines from stored q_scores.
    Returns dict with keys: log (str), confidence (HIGH/MEDIUM),
    computed_total (float), stored_total (float), closure_ok (bool).
    """
    qs = student.get("q_scores", {})
    section = student.get("section", "")

    # ── MC ─────────────────────────────────────────────────────────────────────
    mc_qs = schema["mc"]
    mc_score = sum(qs.get(f"Q{q}", 0) for q in mc_qs)

    # ── MATCHING ───────────────────────────────────────────────────────────────
    match_qs = schema["match"]
    match_key = schema["match_key"]
    match_parts = []
    match_score = 0.0
    for q in match_qs:
        v = qs.get(f"Q{q}", 0)
        match_score += v
        correct = match_key.get(q, "?")
        mark = _check_mark(v, correct)
        match_parts.append(f"Q{q}={correct}{mark}")
    match_line = f"MATCHING: {' '.join(match_parts)} → {int(match_score)}/{len(match_qs)}"

    # ── FILL-IN ────────────────────────────────────────────────────────────────
    fill_qs = schema["fill"]
    fill_key = schema["fill_key"]
    fill_score = sum(qs.get(f"Q{q}", 0) for q in fill_qs)
    fill_parts = [f"Q{q}={'✓' if qs.get(f'Q{q}',0)==1 else '✗'}" for q in fill_qs]
    fill_line = f"FILL-IN: {' '.join(fill_parts)} → {fill_score}/{len(fill_qs)}"

    # ── FR ─────────────────────────────────────────────────────────────────────
    fr_score = student.get(schema["fr_key"], 0) or 0
    fr_line = f"FR: {fr_score}/5"

    # ── DNA ────────────────────────────────────────────────────────────────────
    dna_score = student.get(schema["dna_key"], 0) or 0

    # ── CODONS ─────────────────────────────────────────────────────────────────
    codon_score = sum(student.get(k, 0) or 0 for k in schema["codon_keys"])

    # ── CALC ───────────────────────────────────────────────────────────────────
    calc_score = student.get(schema["calc_key"], 0) or 0

    # ── TOTAL line ─────────────────────────────────────────────────────────────
    computed = mc_score + match_score + fill_score + fr_score + dna_score + codon_score + calc_score
    stored   = student.get("total", 0) or 0
    closure  = abs(computed - stored) < 0.01

    total_line = (
        f"TOTAL: MC({mc_score})+MATCH({int(match_score)})+FILL({fill_score})"
        f"+FR({fr_score})+DNA({dna_score})+CODON({codon_score})+CALC({calc_score})"
        f" = {computed}/{schema['total_pts']}"
        f"  {'✓ CLOSED' if closure else f'⚠ MISMATCH stored={stored}'}"
    )

    full_log = "\n".join([match_line, fill_line, fr_line, total_line])

    confidence = "HIGH" if closure else "MEDIUM"
    if not closure:
        flags = student.get("flags", [])
        if isinstance(flags, list) and "SCORE_AUDIT" not in flags:
            flags.append("SCORE_AUDIT")
            student["flags"] = flags

    return {
        "log": full_log,
        "confidence": confidence,
        "computed_total": computed,
        "stored_total": stored,
        "closure_ok": closure,
    }


def batch_reconstruct(section: str) -> None:
    """Run on all JSON files for a section, write audit_log and confidence back."""
    if section == "T5":
        pattern = "/Users/user/Desktop/ex/ex2T5/grades_T5_b*.json"
    elif section == "W2":
        pattern = "/Users/user/Desktop/ex/ex2w2/grades_W2_b*.json"
    else:
        pattern = "/Users/user/Desktop/ex/ex2w5/grades_W5_b*.json"

    schema = SCHEMAS[section]
    files = sorted(f for f in glob.glob(pattern) if "pre_recheck" not in f)

    results = {"HIGH": [], "MEDIUM": [], "SCORE_AUDIT": []}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        students = data if isinstance(data, list) else data.get("students", [])
        changed = False
        for s in students:
            out = reconstruct_audit_log(s, schema)
            s["audit_log"] = out["log"]
            s["confidence"] = out["confidence"]
            results[out["confidence"]].append(s["name"])
            if not out["closure_ok"]:
                results["SCORE_AUDIT"].append(
                    f"{s['name']}: computed={out['computed_total']} stored={out['stored_total']}"
                )
            changed = True
        if changed:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=2)

    print(f"\n{section} Audit Log Reconstruction:")
    print(f"  HIGH  ({len(results['HIGH'])}): arithmetic closed, logs generated")
    print(f"  MEDIUM({len(results['MEDIUM'])}): arithmetic mismatch → SCORE_AUDIT")
    if results["SCORE_AUDIT"]:
        for s in results["SCORE_AUDIT"]:
            print(f"    ⚠  {s}")
    return results
