"""
Interactive HTML Analytics Dashboard — generated after every batch.

Produces a standalone HTML file with embedded Plotly.js charts showing:
- Score distribution histogram
- Question quality scatter (DI vs item difficulty)
- Confidence pie chart with estimated review time
- Section fairness bar chart
- At-risk student table

No server needed — opens directly in a browser.
"""

import json
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional


# ── Colour helpers ────────────────────────────────────────────────────────────

def get_question_color(di: float, p_value: float) -> str:
    """
    Return a hex colour for a question based on its psychometric properties.

    Green  (#2ecc71): DI >= 0.20 AND 0.30 <= p_value <= 0.70  — good question
    Yellow (#f39c12): DI >= 0.20 but p_value outside ideal range  — easy/hard but discriminating
    Red    (#e74c3c): DI < 0.20  — poor discrimination regardless of difficulty
    """
    if di < 0.20:
        return "#e74c3c"
    if 0.30 <= p_value <= 0.70:
        return "#2ecc71"
    return "#f39c12"


# ── Data collection ───────────────────────────────────────────────────────────

def collect_dashboard_data(results: List[dict], exam_name: str) -> dict:
    """
    Extract and aggregate grading data needed for the HTML dashboard.

    Args:
        results: List of result dicts from grading students.  Each dict may
                 contain any combination of:
                   score, total_possible, total, name, student_file,
                   confidence_level, section, structured_result (GradingResult)
        exam_name: Human-readable exam name, e.g. "BIOL 107 Exam 2"

    Returns:
        Aggregated dashboard data dict.
    """
    scores: List[float] = []
    max_score: float = 0.0
    students: List[dict] = []
    confidence_counts: dict = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # Per-question accumulators: q_num → {scores: [], totals: [], low_conf: int}
    q_accum: dict = {}
    # Per-section accumulators: section → {scores: [], di_values: []}
    sec_accum: dict = {}

    for r in results:
        if r.get("error"):
            continue

        # ── Score ────────────────────────────────────────────────────────────
        score = float(r.get("score", 0) or 0)
        scores.append(score)

        # Infer max score
        candidate_max = float(
            r.get("total_possible")
            or r.get("total")
            or (r.get("structured_result") and r["structured_result"].total_possible if r.get("structured_result") else None)
            or max_score
            or 0
        )
        if candidate_max > max_score:
            max_score = candidate_max

        # ── Confidence ───────────────────────────────────────────────────────
        conf = (r.get("confidence_level") or "").upper()
        if conf not in confidence_counts:
            conf = "MEDIUM"  # default if missing
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        # ── Student name ─────────────────────────────────────────────────────
        name = (
            r.get("name")
            or r.get("student_name")
            or (r.get("structured_result") and r["structured_result"].student_name if r.get("structured_result") else None)
            or r.get("student_file", "Unknown")
        )

        # ── Section ──────────────────────────────────────────────────────────
        section = r.get("section", "")
        if not section and r.get("structured_result"):
            section = ""

        pct = (score / max_score * 100) if max_score > 0 else 0.0

        students.append({
            "name": str(name),
            "score": score,
            "percentage": round(pct, 1),
            "confidence": conf or "MEDIUM",
            "section": section,
        })

        # Accumulate section stats
        if section:
            if section not in sec_accum:
                sec_accum[section] = {"scores": [], "di_values": []}
            sec_accum[section]["scores"].append(score)

        # ── Per-question stats (from structured_result) ───────────────────
        sr = r.get("structured_result")
        if sr and hasattr(sr, "question_results"):
            for q in sr.question_results:
                qn = q.question_number
                if qn not in q_accum:
                    q_accum[qn] = {"scores": [], "totals": [], "low_conf": 0}
                q_accum[qn]["scores"].append(q.points_earned)
                q_accum[qn]["totals"].append(q.points_possible)
                if hasattr(q, "confidence") and str(q.confidence).upper() in ("LOW", "ConfidenceLevel.LOW"):
                    q_accum[qn]["low_conf"] += 1

    # ── Build question_stats ──────────────────────────────────────────────────
    # Import analytics helpers if available
    question_stats: dict = {}
    try:
        from grader.analytics import (
            calculate_discrimination_index,
            calculate_item_difficulty,
            flag_problematic_questions,
        )
        problematic = flag_problematic_questions(results)
        all_q_nums = set()
        for r in results:
            if r.get("error"):
                continue
            sr = r.get("structured_result")
            if sr and hasattr(sr, "question_results"):
                for q in sr.question_results:
                    all_q_nums.add(q.question_number)

        for qn in sorted(all_q_nums):
            try:
                di = calculate_discrimination_index(results, qn)
                p_val = calculate_item_difficulty(results, qn)
            except Exception:
                di, p_val = 0.0, 0.5
            avg_score = 0.0
            if qn in q_accum and q_accum[qn]["scores"]:
                avg_score = statistics.mean(q_accum[qn]["scores"])
            flag = ""
            if qn in problematic:
                flag = ", ".join(problematic[qn].get("flags", []))
            question_stats[qn] = {
                "di": round(di, 3),
                "p_value": round(p_val, 3),
                "avg_score": round(avg_score, 2),
                "flag": flag,
            }
    except Exception:
        # Fall back to per-question accumulators without DI
        for qn, acc in sorted(q_accum.items()):
            s_list = acc["scores"]
            t_list = acc["totals"]
            avg_score = statistics.mean(s_list) if s_list else 0.0
            # Simple p-value: mean(earned) / mean(possible)
            avg_poss = statistics.mean(t_list) if t_list else 1.0
            p_val = (avg_score / avg_poss) if avg_poss > 0 else 0.0
            question_stats[qn] = {
                "di": 0.0,
                "p_value": round(p_val, 3),
                "avg_score": round(avg_score, 2),
                "flag": "",
            }

    # ── Section stats ─────────────────────────────────────────────────────────
    section_stats: dict = {}
    for sec, acc in sec_accum.items():
        s_list = acc["scores"]
        mean_s = statistics.mean(s_list) if s_list else 0.0
        std_s = statistics.stdev(s_list) if len(s_list) > 1 else 0.0
        # DI ratio: fraction of questions with DI >= 0.20 (approximate per-section)
        good_qs = sum(1 for qs in question_stats.values() if qs["di"] >= 0.20)
        di_ratio = good_qs / len(question_stats) if question_stats else 0.0
        section_stats[sec] = {
            "mean": round(mean_s, 2),
            "std": round(std_s, 2),
            "count": len(s_list),
            "di_ratio": round(di_ratio, 3),
        }

    # ── At-risk students (< 60%) ──────────────────────────────────────────────
    at_risk = [s for s in students if s["percentage"] < 60.0]
    at_risk.sort(key=lambda s: s["score"])

    return {
        "exam_name": exam_name,
        "scores": scores,
        "max_score": max_score,
        "students": students,
        "question_stats": question_stats,
        "confidence_counts": confidence_counts,
        "section_stats": section_stats,
        "at_risk": at_risk,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── Histogram helper ──────────────────────────────────────────────────────────

def _build_score_histogram_data(scores: List[float], max_score: float) -> dict:
    """
    Bin scores into 10 percentage-based buckets (0-10%, 10-20%, ..., 90-100%).

    Args:
        scores: Raw scores earned.
        max_score: Maximum possible score (used to compute percentages).

    Returns:
        Plotly-compatible data dict with x labels, y counts, and marker colors.
    """
    labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
    counts = [0] * 10
    colors = []

    for s in scores:
        pct = (s / max_score * 100) if max_score > 0 else 0.0
        bucket = min(int(pct // 10), 9)
        counts[bucket] += 1

    for i in range(10):
        midpoint = i * 10 + 5
        if midpoint < 50:
            colors.append("#e74c3c")   # red
        elif midpoint < 70:
            colors.append("#f39c12")   # yellow
        else:
            colors.append("#2ecc71")   # green

    return {"x": labels, "y": counts, "colors": colors}


# ── CSS ───────────────────────────────────────────────────────────────────────

_EMBEDDED_CSS = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: Helvetica, Arial, sans-serif;
        background: #f4f6f9;
        color: #333;
        padding: 20px;
    }
    h1 {
        font-size: 1.6em;
        margin-bottom: 6px;
        color: #2c3e50;
    }
    .meta {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 20px;
    }
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
    }
    .card {
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        padding: 20px;
        border-radius: 8px;
        margin: 10px;
        min-height: 300px;
    }
    .card h2 {
        font-size: 1.1em;
        margin-bottom: 12px;
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88em;
    }
    th {
        background: #2c3e50;
        color: #fff;
        padding: 8px 10px;
        text-align: left;
    }
    td {
        padding: 7px 10px;
        border-bottom: 1px solid #ecf0f1;
    }
    tr:nth-child(even) td { background: #f9f9f9; }
    tr.danger td { background: #ffeeee !important; }
    tr.warning td { background: #fff8e1 !important; }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .badge-high   { background: #d5f5e3; color: #1e8449; }
    .badge-medium { background: #fef9e7; color: #b7770d; }
    .badge-low    { background: #fadbd8; color: #c0392b; }
    @media print {
        body { background: #fff; padding: 0; }
        .card { box-shadow: none; border: 1px solid #ccc; }
        .grid-2 { display: block; }
    }
    @media (max-width: 768px) {
        .grid-2 { grid-template-columns: 1fr; }
    }
"""


# ── At-risk table builder ─────────────────────────────────────────────────────

def _build_at_risk_table(at_risk: List[dict]) -> str:
    """Build the HTML at-risk student table (no Plotly)."""
    if not at_risk:
        return "<p style='color:#27ae60;font-weight:bold;'>No at-risk students — all students above 60%.</p>"

    rows = []
    for s in at_risk:
        css_class = "danger" if s["percentage"] < 50 else "warning"
        conf_lower = s["confidence"].lower()
        badge_class = f"badge-{conf_lower}"
        rows.append(
            f'<tr class="{css_class}">'
            f'<td>{s["name"]}</td>'
            f'<td>{s["score"]:.1f}</td>'
            f'<td>{s["percentage"]:.0f}%</td>'
            f'<td><span class="badge {badge_class}">{s["confidence"]}</span></td>'
            f'<td>{s["section"] or "—"}</td>'
            f"</tr>"
        )

    return (
        "<table>"
        "<thead><tr><th>Student</th><th>Score</th><th>%</th><th>Confidence</th><th>Section</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ── Dashboard generator ───────────────────────────────────────────────────────

def generate_dashboard(data: dict, output_path: str) -> str:
    """
    Generate a standalone HTML analytics dashboard and write it to disk.

    Args:
        data: Aggregated dashboard data from collect_dashboard_data().
        output_path: Absolute or relative path where the HTML file is written.

    Returns:
        The path that was written.
    """
    exam_name = data.get("exam_name", "Exam")
    scores = data.get("scores", [])
    max_score = data.get("max_score", 0.0)
    students = data.get("students", [])
    question_stats = data.get("question_stats", {})
    confidence_counts = data.get("confidence_counts", {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    section_stats = data.get("section_stats", {})
    at_risk = data.get("at_risk", [])
    generated_at = data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

    n = len(scores)
    mean_score = statistics.mean(scores) if scores else 0.0
    mean_pct = (mean_score / max_score * 100) if max_score > 0 else 0.0

    # ── Chart 1 — Score distribution histogram ────────────────────────────────
    hist = _build_score_histogram_data(scores, max_score)
    hist_data = json.dumps([{
        "type": "bar",
        "x": hist["x"],
        "y": hist["y"],
        "marker": {"color": hist["colors"]},
        "hovertemplate": "%{x}: %{y} students<extra></extra>",
    }])
    hist_layout = json.dumps({
        "title": "",
        "xaxis": {"title": "Score Range"},
        "yaxis": {"title": "Students"},
        "margin": {"t": 10, "l": 50, "r": 20, "b": 60},
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
    })

    # ── Chart 2 — Question quality scatter ────────────────────────────────────
    q_nums = sorted(question_stats.keys())
    q_x = [question_stats[q]["p_value"] for q in q_nums]      # item difficulty
    q_y = [question_stats[q]["di"] for q in q_nums]            # discrimination index
    q_text = [f"Q{q}" for q in q_nums]
    q_colors = [get_question_color(question_stats[q]["di"], question_stats[q]["p_value"]) for q in q_nums]
    q_hover = [
        f"Q{q}: DI={question_stats[q]['di']:.2f}, p={question_stats[q]['p_value']:.2f}"
        + (f", {question_stats[q]['flag']}" if question_stats[q].get("flag") else "")
        for q in q_nums
    ]

    scatter_data = json.dumps([{
        "type": "scatter",
        "mode": "markers+text",
        "x": q_x,
        "y": q_y,
        "text": q_text,
        "customdata": q_hover,
        "hovertemplate": "%{customdata}<extra></extra>",
        "textposition": "top center",
        "marker": {"color": q_colors, "size": 12, "line": {"color": "#fff", "width": 1}},
    }])
    scatter_layout = json.dumps({
        "title": "",
        "xaxis": {"title": "Item Difficulty (p-value)", "range": [0, 1]},
        "yaxis": {"title": "Discrimination Index", "range": [-0.5, 1.0]},
        "margin": {"t": 10, "l": 60, "r": 20, "b": 60},
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
        "shapes": [
            # Vertical lines at p=0.30 and p=0.70 (ideal difficulty zone)
            {"type": "line", "x0": 0.30, "x1": 0.30, "y0": -0.5, "y1": 1.0,
             "line": {"color": "#95a5a6", "dash": "dash", "width": 1}},
            {"type": "line", "x0": 0.70, "x1": 0.70, "y0": -0.5, "y1": 1.0,
             "line": {"color": "#95a5a6", "dash": "dash", "width": 1}},
            # Horizontal line at DI=0.20 (minimum acceptable discrimination)
            {"type": "line", "x0": 0, "x1": 1, "y0": 0.20, "y1": 0.20,
             "line": {"color": "#e67e22", "dash": "dot", "width": 1.5}},
        ],
    })

    # ── Chart 3 — Confidence pie ──────────────────────────────────────────────
    high_c = confidence_counts.get("HIGH", 0)
    med_c = confidence_counts.get("MEDIUM", 0)
    low_c = confidence_counts.get("LOW", 0)
    review_min = low_c * 5 + med_c * 2
    pie_data = json.dumps([{
        "type": "pie",
        "labels": ["HIGH", "MEDIUM", "LOW"],
        "values": [high_c, med_c, low_c],
        "marker": {"colors": ["#2ecc71", "#f39c12", "#e74c3c"]},
        "textinfo": "label+percent",
        "hovertemplate": "%{label}: %{value} students<extra></extra>",
    }])
    pie_layout = json.dumps({
        "title": "",
        "margin": {"t": 10, "l": 20, "r": 20, "b": 10},
        "paper_bgcolor": "#fff",
        "legend": {"orientation": "h", "y": -0.1},
        "annotations": [{
            "text": f"~{review_min} min<br>review",
            "x": 0.5, "y": 0.5,
            "font": {"size": 13},
            "showarrow": False,
        }],
    })

    # ── Chart 4 — Section fairness bars ──────────────────────────────────────
    sec_names = list(section_stats.keys())
    sec_means = [section_stats[s]["mean"] for s in sec_names]
    sec_stds = [section_stats[s]["std"] for s in sec_names]
    sec_colors = []
    for s in sec_names:
        ratio = section_stats[s]["di_ratio"]
        if ratio >= 0.70:
            sec_colors.append("#2ecc71")
        elif ratio >= 0.40:
            sec_colors.append("#f39c12")
        else:
            sec_colors.append("#e74c3c")

    if not sec_names:
        # Provide a placeholder so Plotly doesn't crash
        sec_names, sec_means, sec_stds, sec_colors = ["(no sections)"], [0], [0], ["#95a5a6"]

    bars_data = json.dumps([{
        "type": "bar",
        "x": sec_names,
        "y": sec_means,
        "error_y": {"type": "data", "array": sec_stds, "visible": True},
        "marker": {"color": sec_colors},
        "hovertemplate": "%{x}: %{y:.1f} ± %{error_y.array:.1f}<extra></extra>",
    }])
    bars_layout = json.dumps({
        "title": "",
        "xaxis": {"title": "Section"},
        "yaxis": {"title": "Mean Score", "range": [0, max(max_score * 1.05, 1)]},
        "margin": {"t": 10, "l": 60, "r": 20, "b": 60},
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
    })

    # ── At-risk table HTML ────────────────────────────────────────────────────
    at_risk_html = _build_at_risk_table(at_risk)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grading Dashboard &mdash; {exam_name}</title>
  <style>
{_EMBEDDED_CSS}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
  <h1>Grading Dashboard &mdash; {exam_name}</h1>
  <p class="meta">Generated: {generated_at} &nbsp;|&nbsp; {n} students &nbsp;|&nbsp; Mean: {mean_score:.1f} ({mean_pct:.0f}%)</p>

  <div class="grid-2">
    <div class="card">
      <h2>Score Distribution</h2>
      <div id="score-dist" style="height:280px;"></div>
    </div>
    <div class="card">
      <h2>Grading Confidence</h2>
      <div id="confidence-pie" style="height:280px;"></div>
    </div>
  </div>

  <div class="card">
    <h2>Question Quality (DI vs Item Difficulty)</h2>
    <p style="font-size:0.8em;color:#888;margin-bottom:8px;">
      Green = good (DI&nbsp;&ge;&nbsp;0.20, 0.30&nbsp;&le;&nbsp;p&nbsp;&le;&nbsp;0.70)&nbsp;&nbsp;
      Yellow = borderline&nbsp;&nbsp;
      Red = poor discrimination&nbsp;&nbsp;
      Dashed lines = ideal difficulty range&nbsp;&nbsp;
      Dotted line = DI&nbsp;=&nbsp;0.20 threshold
    </p>
    <div id="question-quality" style="height:340px;"></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Grade Distribution by Section</h2>
      <div id="fairness-bars" style="height:280px;"></div>
    </div>
    <div class="card">
      <h2>At-Risk Students (&lt;60%)</h2>
      <p style="font-size:0.8em;color:#888;margin-bottom:10px;">
        Red rows = below 50%. Sorted by score ascending.
      </p>
      {at_risk_html}
    </div>
  </div>

  <script>
    var histData   = {hist_data};
    var histLayout = {hist_layout};
    Plotly.newPlot('score-dist', histData, histLayout, {{responsive: true}});

    var scatterData   = {scatter_data};
    var scatterLayout = {scatter_layout};
    Plotly.newPlot('question-quality', scatterData, scatterLayout, {{responsive: true}});

    var pieData   = {pie_data};
    var pieLayout = {pie_layout};
    Plotly.newPlot('confidence-pie', pieData, pieLayout, {{responsive: true}});

    var barsData   = {bars_data};
    var barsLayout = {bars_layout};
    Plotly.newPlot('fairness-bars', barsData, barsLayout, {{responsive: true}});
  </script>
</body>
</html>"""

    # ── Write file ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return output_path
