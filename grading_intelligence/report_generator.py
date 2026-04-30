"""
Batch Analytics HTML Dashboard Generator.

Generates a standalone, self-contained HTML report with embedded Plotly.js
charts for university-grade batch grading analytics. No npm or build step
required — just a single HTML file with a CDN link to Plotly.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from typing import Optional

from grading_intelligence.analytics import (
    compute_item_analysis,
    compute_class_analytics,
    flag_question_quality,
    compute_bloom_coverage,
)
from grading_intelligence.structured_output import GradingResult, BloomLevel


def generate_analytics_html(
    results: list[GradingResult],
    output_path: Optional[str] = None,
) -> str:
    """Generate a standalone HTML analytics dashboard from batch grading results.

    Args:
        results: List of GradingResult objects from a graded batch.
        output_path: If provided, write the HTML file to this path.

    Returns:
        The complete HTML as a string.
    """
    if not results:
        html = _wrap_html("No Results", "<p>No grading results to analyze.</p>")
        if output_path:
            _write(output_path, html)
        return html

    # Compute analytics
    analytics = compute_class_analytics(results, "Batch Grading Report")
    item_analyses = analytics.item_analyses
    quality_flags = analytics.quality_flags
    bloom_coverage = compute_bloom_coverage(item_analyses)

    percentages = [r.percentage for r in results]
    num_students = len(results)

    # Summary stats
    mean_val = analytics.mean_score
    median_val = analytics.median_score
    std_val = analytics.std_dev
    min_val = analytics.min_score
    max_val = analytics.max_score

    # Confidence distribution
    high_count = 0
    low_count = 0
    for r in results:
        for q in r.question_results:
            if q.confidence.value == "HIGH":
                high_count += 1
            else:
                low_count += 1
        for s in r.section_results:
            if s.confidence.value == "HIGH":
                high_count += 1
            else:
                low_count += 1

    # At-risk students (below 60%)
    at_risk = []
    for r in results:
        if r.percentage < 60:
            at_risk.append({
                "file": r.student_file,
                "score": r.total_score,
                "total": r.total_possible,
                "pct": r.percentage,
            })
    at_risk.sort(key=lambda x: x["pct"])

    # Build chart data
    # 1. Score distribution histogram
    hist_data = json.dumps(percentages)

    # 2. Question difficulty scatter
    scatter_x = []
    scatter_y = []
    scatter_size = []
    scatter_color = []
    scatter_text = []
    bloom_color_map = {
        "remember": "#74b9ff",
        "understand": "#55efc4",
        "apply": "#ffeaa7",
        "analyze": "#fdcb6e",
        "evaluate": "#e17055",
        "create": "#d63031",
    }
    for item in item_analyses:
        scatter_x.append(item.difficulty_index)
        scatter_y.append(item.discrimination_index)
        scatter_size.append(max(item.mean_score * 3, 10))
        bloom = item.bloom_level.value if item.bloom_level else "remember"
        scatter_color.append(bloom_color_map.get(bloom, "#636e72"))
        scatter_text.append(
            f"Q{item.question_number}<br>"
            f"Difficulty: {item.difficulty_index:.2f}<br>"
            f"Discrimination: {item.discrimination_index:.2f}<br>"
            f"Bloom: {bloom.capitalize()}<br>"
            f"Mean: {item.mean_score:.1f}"
        )

    # 3. Bloom's coverage
    bloom_labels = [k.capitalize() for k in bloom_coverage.keys()]
    bloom_values = list(bloom_coverage.values())

    # 4. Quality flags data
    flag_cards_html = ""
    if quality_flags:
        for flag in quality_flags:
            severity_color = {
                "critical": "#e17055",
                "warning": "#fdcb6e",
                "info": "#74b9ff",
            }.get(flag.severity, "#636e72")
            severity_icon = {
                "critical": "!!",
                "warning": "!",
                "info": "i",
            }.get(flag.severity, "?")
            flag_cards_html += f"""
            <div class="flag-card" style="border-left:4px solid {severity_color};">
              <div class="flag-header">
                <span class="flag-icon" style="background:{severity_color};">{severity_icon}</span>
                <strong>Q{flag.question_number}</strong>
                <span class="flag-type">{flag.flag_type.replace('_', ' ').title()}</span>
                <span class="flag-severity" style="color:{severity_color};">{flag.severity.upper()}</span>
              </div>
              <div class="flag-detail">{flag.detail}</div>
            </div>"""
    else:
        flag_cards_html = '<div class="no-flags">All questions passed quality checks.</div>'

    # 5. At-risk table rows
    at_risk_rows = ""
    if at_risk:
        for s in at_risk:
            trend = "&#9660;" if s["pct"] < 50 else "&#9654;"
            trend_color = "#e17055" if s["pct"] < 50 else "#fdcb6e"
            at_risk_rows += f"""
            <tr>
              <td>{_esc(s['file'])}</td>
              <td>{s['score']:.1f} / {s['total']:.1f}</td>
              <td style="font-weight:700;color:{trend_color};">{s['pct']:.1f}%</td>
              <td style="color:{trend_color};">{trend}</td>
            </tr>"""
    else:
        at_risk_rows = '<tr><td colspan="4" style="text-align:center;color:#636e72;">No at-risk students detected.</td></tr>'

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""
    <!-- Summary Stats Bar -->
    <div class="stats-bar">
      <div class="stat-card">
        <div class="stat-label">Students</div>
        <div class="stat-value">{num_students}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Mean</div>
        <div class="stat-value">{mean_val:.1f}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Median</div>
        <div class="stat-value">{median_val:.1f}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Std Dev</div>
        <div class="stat-value">{std_val:.1f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Min</div>
        <div class="stat-value">{min_val:.1f}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Max</div>
        <div class="stat-value">{max_val:.1f}%</div>
      </div>
    </div>

    <!-- Charts Row 1 -->
    <div class="chart-row">
      <div class="chart-card wide">
        <h3>Score Distribution</h3>
        <div id="histogram"></div>
      </div>
      <div class="chart-card">
        <h3>Confidence Distribution</h3>
        <div id="confidence-pie"></div>
      </div>
    </div>

    <!-- Charts Row 2 -->
    <div class="chart-row">
      <div class="chart-card wide">
        <h3>Question Analysis</h3>
        <p class="chart-subtitle">Difficulty vs. Discrimination &mdash; size = mean score, color = Bloom's level</p>
        <div id="scatter"></div>
      </div>
      <div class="chart-card">
        <h3>Bloom's Coverage</h3>
        <div id="bloom-pie"></div>
      </div>
    </div>

    <!-- Quality Flags -->
    <div class="section-card">
      <h3>Question Quality Flags</h3>
      <div class="flags-container">
        {flag_cards_html}
      </div>
    </div>

    <!-- At-Risk Students -->
    <div class="section-card">
      <h3>At-Risk Students <span class="badge">&lt; 60%</span></h3>
      <table class="data-table">
        <thead>
          <tr><th>Student</th><th>Score</th><th>Percentage</th><th>Trend</th></tr>
        </thead>
        <tbody>
          {at_risk_rows}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Generated by Grading Intelligence Platform &mdash; {generated_at}
    </div>

    <script>
      // Score Distribution Histogram
      Plotly.newPlot('histogram', [{{
        x: {hist_data},
        type: 'histogram',
        nbinsx: 20,
        marker: {{ color: '#6c5ce7', line: {{ color: '#fff', width: 1 }} }},
        hovertemplate: '%{{x:.1f}}%<br>Count: %{{y}}<extra></extra>'
      }}], {{
        xaxis: {{ title: 'Score (%)', range: [0, 105] }},
        yaxis: {{ title: 'Number of Students' }},
        margin: {{ t: 20, b: 50, l: 50, r: 20 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#fafafa',
        font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }},
        bargap: 0.05
      }}, {{ responsive: true, displayModeBar: false }});

      // Confidence Pie
      Plotly.newPlot('confidence-pie', [{{
        labels: ['HIGH', 'LOW'],
        values: [{high_count}, {low_count}],
        type: 'pie',
        hole: 0.5,
        marker: {{ colors: ['#00b894', '#fdcb6e'] }},
        textinfo: 'label+percent',
        hovertemplate: '%{{label}}: %{{value}} questions<extra></extra>'
      }}], {{
        margin: {{ t: 20, b: 20, l: 20, r: 20 }},
        paper_bgcolor: 'transparent',
        font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }},
        showlegend: false
      }}, {{ responsive: true, displayModeBar: false }});

      // Question Scatter
      Plotly.newPlot('scatter', [{{
        x: {json.dumps(scatter_x)},
        y: {json.dumps(scatter_y)},
        mode: 'markers',
        marker: {{
          size: {json.dumps(scatter_size)},
          color: {json.dumps(scatter_color)},
          line: {{ color: '#fff', width: 1 }},
          opacity: 0.85
        }},
        text: {json.dumps(scatter_text)},
        hovertemplate: '%{{text}}<extra></extra>',
        type: 'scatter'
      }}], {{
        xaxis: {{ title: 'Difficulty Index (0=hard, 1=easy)', range: [-0.05, 1.05] }},
        yaxis: {{ title: 'Discrimination Index', range: [-1.05, 1.05] }},
        margin: {{ t: 20, b: 50, l: 60, r: 20 }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#fafafa',
        font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }},
        shapes: [
          {{ type: 'line', x0: 0, x1: 1, y0: 0.15, y1: 0.15, line: {{ color: '#e17055', dash: 'dash', width: 1 }} }},
          {{ type: 'rect', x0: 0.95, x1: 1.05, y0: -1.05, y1: 1.05, fillcolor: 'rgba(116,185,255,0.08)', line: {{ width: 0 }} }},
          {{ type: 'rect', x0: -0.05, x1: 0.10, y0: -1.05, y1: 1.05, fillcolor: 'rgba(225,112,85,0.08)', line: {{ width: 0 }} }}
        ],
        annotations: [
          {{ x: 0.98, y: -0.9, text: 'Too Easy', showarrow: false, font: {{ size: 10, color: '#74b9ff' }} }},
          {{ x: 0.05, y: -0.9, text: 'Too Hard', showarrow: false, font: {{ size: 10, color: '#e17055' }} }},
          {{ x: 0.5, y: 0.17, text: 'Poor discrimination threshold', showarrow: false, font: {{ size: 9, color: '#e17055' }} }}
        ]
      }}, {{ responsive: true, displayModeBar: false }});

      // Bloom's Pie
      Plotly.newPlot('bloom-pie', [{{
        labels: {json.dumps(bloom_labels)},
        values: {json.dumps(bloom_values)},
        type: 'pie',
        hole: 0.45,
        marker: {{ colors: ['#74b9ff', '#55efc4', '#ffeaa7', '#fdcb6e', '#e17055', '#d63031'] }},
        textinfo: 'label+value',
        hovertemplate: '%{{label}}: %{{value}} questions (%{{percent}})<extra></extra>'
      }}], {{
        margin: {{ t: 20, b: 20, l: 20, r: 20 }},
        paper_bgcolor: 'transparent',
        font: {{ family: '-apple-system, BlinkMacSystemFont, sans-serif' }},
        showlegend: false
      }}, {{ responsive: true, displayModeBar: false }});
    </script>
    """

    html = _wrap_html("Batch Analytics Report", body)
    if output_path:
        _write(output_path, html)
    return html


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """HTML-escape a string."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write(path: str, content: str) -> None:
    """Write content to a file, creating directories as needed."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _wrap_html(title: str, body: str) -> str:
    """Wrap body content in a complete HTML document with Plotly CDN and styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)} — Grading Intelligence</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #f5f6fa;
    --header-bg: #1a1e2e;
    --accent: #6c5ce7;
    --card-bg: #ffffff;
    --border: #e2e5ed;
    --text: #2d3436;
    --text-muted: #636e72;
    --green: #00b894;
    --yellow: #fdcb6e;
    --red: #e17055;
    --radius: 8px;
    --shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: var(--header-bg); color:#fff;
    padding: 24px 40px;
    display:flex; align-items:center; justify-content:space-between;
  }}
  .header h1 {{ font-size:22px; font-weight:600; }}
  .header h1 span {{ color: var(--accent); }}
  .header .subtitle {{ font-size:13px; color:#b2bec3; }}

  .container {{ max-width:1400px; margin:0 auto; padding:24px 32px 48px; }}

  /* Stats bar */
  .stats-bar {{
    display:grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap:16px;
    margin-bottom:24px;
  }}
  .stat-card {{
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding:20px;
    text-align:center;
  }}
  .stat-label {{ font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }}
  .stat-value {{ font-size:28px; font-weight:800; color:var(--accent); }}

  /* Chart layout */
  .chart-row {{
    display:grid;
    grid-template-columns: 2fr 1fr;
    gap:20px;
    margin-bottom:20px;
  }}
  @media (max-width:900px) {{
    .chart-row {{ grid-template-columns:1fr; }}
  }}
  .chart-card {{
    background:var(--card-bg);
    border-radius:var(--radius);
    box-shadow:var(--shadow);
    padding:20px;
  }}
  .chart-card h3 {{ font-size:15px; font-weight:600; margin-bottom:4px; }}
  .chart-subtitle {{ font-size:12px; color:var(--text-muted); margin-bottom:8px; }}

  /* Section cards */
  .section-card {{
    background:var(--card-bg);
    border-radius:var(--radius);
    box-shadow:var(--shadow);
    padding:24px;
    margin-bottom:20px;
  }}
  .section-card h3 {{ font-size:16px; font-weight:600; margin-bottom:16px; }}
  .badge {{
    font-size:11px; font-weight:600;
    background:#f8d7da; color:#721c24;
    padding:2px 8px; border-radius:10px; vertical-align:middle;
  }}

  /* Flags */
  .flags-container {{ display:flex; flex-direction:column; gap:10px; }}
  .flag-card {{
    padding:12px 16px;
    background:#fafafa;
    border-radius:6px;
  }}
  .flag-header {{
    display:flex; align-items:center; gap:10px;
    font-size:14px; margin-bottom:4px;
  }}
  .flag-icon {{
    width:22px; height:22px; border-radius:50%; color:#fff;
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:700; flex-shrink:0;
  }}
  .flag-type {{ color:var(--text-muted); font-size:13px; }}
  .flag-severity {{ margin-left:auto; font-size:11px; font-weight:700; }}
  .flag-detail {{ font-size:13px; color:var(--text-muted); padding-left:32px; }}
  .no-flags {{ color:var(--green); font-weight:600; font-size:14px; padding:8px 0; }}

  /* Data table */
  .data-table {{
    width:100%;
    border-collapse:collapse;
    font-size:14px;
  }}
  .data-table th {{
    text-align:left;
    font-size:12px;
    color:var(--text-muted);
    text-transform:uppercase;
    letter-spacing:0.5px;
    padding:10px 16px;
    border-bottom:2px solid var(--border);
  }}
  .data-table td {{
    padding:10px 16px;
    border-bottom:1px solid var(--border);
  }}
  .data-table tr:hover td {{ background:#f8f9ff; }}

  /* Footer */
  .footer {{
    text-align:center;
    font-size:12px;
    color:var(--text-muted);
    padding:24px 0;
  }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1><span>AI</span> Grading Analytics</h1>
    <div class="subtitle">{_esc(title)}</div>
  </div>
</div>
<div class="container">
{body}
</div>
</body>
</html>"""
