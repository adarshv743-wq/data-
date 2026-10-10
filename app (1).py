"""
DAG-EDA prototype - SINGLE FILE VERSION (Render-ready)
=======================================================

This one file contains: Flask routes, the analysis engine, the SQLite schema,
all HTML templates and the CSS. No templates/ or static/ folders are needed.

-------------------------------------------------------------------------
DEPLOY ON RENDER (no other file required)
-------------------------------------------------------------------------
1. Put this file in a GitHub/GitLab repo, named  app.py  (repo root).
2. Render dashboard -> New -> Web Service -> connect the repo.
3. Settings:
     Runtime        : Python 3
     Build Command  : pip install Flask pandas networkx gunicorn
     Start Command  : gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
     Health Check   : /health
4. Environment variables:
     DAG_EDA_SECRET = <any long random string>
     PYTHON_VERSION = 3.11.9          (recommended)
     DATA_DIR       = /var/data       (ONLY if you attach a persistent Disk
                                       mounted at /var/data - paid plans)
5. Deploy.

NOTE: On Render's free plan the filesystem is ephemeral. Without a Disk the app
stores the SQLite DB and uploads in /tmp, so history is lost on every
redeploy/restart. Attach a Disk (mount path /var/data, set DATA_DIR) to keep it.

Optional: you may instead commit a requirements.txt containing
    Flask>=3.0,<4.0
    pandas>=2.0,<3.0
    networkx>=3.0,<4.0
    gunicorn>=21.2
and use "pip install -r requirements.txt" as the Build Command.

-------------------------------------------------------------------------
RUN LOCALLY
-------------------------------------------------------------------------
    pip install Flask pandas networkx gunicorn
    python app.py            -> http://127.0.0.1:5000
"""
import json
import math
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, url_for)
from jinja2 import ChoiceLoader, DictLoader
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ON_RENDER = bool(os.environ.get("RENDER"))

# Where the DB and uploads live. Use a persistent disk on Render via DATA_DIR.
DATA_DIR = os.environ.get("DATA_DIR") or ("/tmp/dag_eda" if ON_RENDER else BASE_DIR)
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "dag_eda.db")
ALLOWED_EXTENSIONS = {"csv"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("DAG_EDA_SECRET", "replace-this-development-secret")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    summary_json TEXT,
    analyses_json TEXT,
    intent_graph_json TEXT,
    knowledge_graph_json TEXT
);

CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    task TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_history_session_id ON analysis_history(session_id);
"""


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------
def _safe_round(value):
    try:
        if pd.isna(value) or not math.isfinite(float(value)):
            return None
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def dataframe_summary(df):
    columns = []
    for col in df.columns:
        s = df[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "sample": [str(v) for v in s.dropna().head(3).tolist()],
        }
        if pd.api.types.is_numeric_dtype(s):
            item["mean"] = _safe_round(s.mean())
            item["min"] = _safe_round(s.min())
            item["max"] = _safe_round(s.max())
        columns.append(item)
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": columns,
        "numeric_columns": [str(c) for c in df.select_dtypes(include="number").columns],
        "categorical_columns": [str(c) for c in df.select_dtypes(exclude="number").columns],
    }


def _find_column(columns, keywords):
    for col in columns:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(col).lower())
        if any(k in normalized for k in keywords):
            return str(col)
    return None


def analyze_dataset(df, question):
    """Create explainable candidate analyses from a question and CSV schema."""
    numeric = [str(c) for c in df.select_dtypes(include="number").columns]
    categorical = [str(c) for c in df.select_dtypes(exclude="number").columns]
    cols = [str(c) for c in df.columns]
    q = question.lower()
    analyses = []

    target = _find_column(cols, ["sales", "revenue", "profit", "price", "score",
                                 "mark", "grade", "target", "amount"])
    if target is None and numeric:
        target = numeric[0]

    if numeric:
        analyses.append({
            "id": "numeric_overview",
            "title": "Explore numeric distributions",
            "reason": "Numeric columns can reveal typical values, spread, and unusual observations.",
            "method": "Descriptive statistics and histograms",
            "columns": numeric[:6],
            "priority": 1,
        })

    if target and target in numeric and len(numeric) > 1:
        related = []
        for col in numeric:
            if col == target:
                continue
            corr = df[[target, col]].corr(numeric_only=True).iloc[0, 1]
            if pd.notna(corr):
                related.append((col, float(corr)))
        related.sort(key=lambda x: abs(x[1]), reverse=True)
        analyses.append({
            "id": "correlation",
            "title": f"What relates to {target}?",
            "reason": "Correlation can highlight numeric variables that move together; it does not prove causation.",
            "method": "Pearson correlation",
            "target": target,
            "columns": [target] + [c for c, _ in related[:5]],
            "correlations": [{"column": c, "correlation": round(v, 3)} for c, v in related[:8]],
            "priority": 1 if any(w in q for w in ["why", "factor", "affect", "relate", "increase", "decrease"]) else 2,
        })

    if categorical and numeric:
        cat = _find_column(categorical, ["region", "category", "product", "department",
                                         "class", "group", "city", "type"])
        cat = cat or categorical[0]
        num = target if (target in numeric) else numeric[0]
        groups = []
        try:
            grouped = df.groupby(cat, dropna=False)[num].agg(["mean", "count"]).reset_index().head(15)
            groups = [{
                "category": str(row[cat]),
                "mean": _safe_round(row["mean"]),
                "count": int(row["count"]),
            } for _, row in grouped.iterrows()]
        except Exception:
            groups = []
        analyses.append({
            "id": "group_comparison",
            "title": f"Compare {num} across {cat}",
            "reason": "Group comparisons can show whether the selected measure differs between categories.",
            "method": "Group means and counts",
            "target": num,
            "group_by": cat,
            "columns": [cat, num],
            "groups": groups,
            "priority": 1 if any(w in q for w in ["compare", "category", "region", "group", "which"]) else 2,
        })

    missing = []
    for col in cols:
        count = int(df[col].isna().sum())
        if count:
            missing.append({"column": col, "missing": count,
                            "percent": round(count / max(len(df), 1) * 100, 2)})
    analyses.append({
        "id": "data_quality",
        "title": "Check data quality",
        "reason": "Missing values and unusual column cardinality can affect later conclusions.",
        "method": "Missing-value and uniqueness scan",
        "columns": cols,
        "missing": missing,
        "priority": 1 if missing else 3,
    })

    analyses.sort(key=lambda a: a.get("priority", 3))
    return analyses


def build_graphs(question, df, analyses):
    """Return JSON-friendly node/edge dictionaries for the intent and knowledge graphs."""
    intent_nodes = [{"id": "root", "label": question, "kind": "question"}]
    intent_edges = []
    for idx, analysis in enumerate(analyses, start=1):
        node_id = f"task_{idx}"
        intent_nodes.append({"id": node_id, "label": analysis["title"],
                             "kind": "task", "analysis_id": analysis["id"]})
        intent_edges.append({"source": "root", "target": node_id})
        detail_id = f"detail_{idx}"
        intent_nodes.append({"id": detail_id, "label": analysis["method"],
                             "kind": "method", "analysis_id": analysis["id"]})
        intent_edges.append({"source": node_id, "target": detail_id})

    knowledge_nodes = []
    knowledge_edges = []
    concept_map = {
        "sales": ["sales", "revenue", "turnover", "amount"],
        "pricing": ["price", "cost", "discount"],
        "customer demand": ["units", "quantity", "orders", "demand"],
        "performance": ["profit", "score", "mark", "grade", "rating"],
        "time": ["date", "time", "month", "year", "day"],
        "group/category": ["region", "category", "product", "department", "class", "type", "city"],
    }
    existing = set()
    for concept, terms in concept_map.items():
        concept_id = "concept_" + concept.replace("/", "_").replace(" ", "_")
        matched = []
        for col in df.columns:
            normalized = re.sub(r"[^a-z0-9]+", " ", str(col).lower())
            if any(term in normalized for term in terms):
                matched.append(str(col))
        knowledge_nodes.append({"id": concept_id, "label": concept.title(),
                                "kind": "concept", "matched": bool(matched)})
        for col in matched:
            var_id = "var_" + re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_")
            if var_id not in existing:
                existing.add(var_id)
                knowledge_nodes.append({"id": var_id, "label": col, "kind": "variable",
                                        "dtype": str(df[col].dtype)})
            knowledge_edges.append({"source": concept_id, "target": var_id, "label": "measured by"})

    for col in df.columns:
        var_id = "var_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_")
        if var_id not in existing:
            existing.add(var_id)
            knowledge_nodes.append({"id": var_id, "label": str(col), "kind": "variable",
                                    "dtype": str(df[col].dtype)})
    return ({"nodes": intent_nodes, "edges": intent_edges},
            {"nodes": knowledge_nodes, "edges": knowledge_edges})


# ---------------------------------------------------------------------------
# Embedded CSS
# ---------------------------------------------------------------------------
STYLE_CSS = r""":root { --ink:#182638; --muted:#617185; --line:#dce3eb; --paper:#f5f7fa; --white:#fff; --blue:#244f78; --blue2:#e8f0f8; --green:#dcefe8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
a { color:var(--blue); }
.topbar { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:16px max(24px, calc((100vw - 1180px)/2)); background:#fff; border-bottom:1px solid var(--line); }
.brand { font-weight:800; font-size:1.2rem; text-decoration:none; color:var(--ink); }
.brand span { display:block; font-size:.72rem; font-weight:500; color:var(--muted); letter-spacing:.04em; }
nav { display:flex; gap:20px; } nav a { text-decoration:none; font-weight:600; }
.container { max-width:1180px; margin:0 auto; padding:28px 22px 60px; }
.hero { padding:34px 0 24px; max-width:850px; } .hero.compact { padding-top:12px; }
.eyebrow { color:#507394; text-transform:uppercase; letter-spacing:.11em; font-size:.75rem; font-weight:800; margin:0 0 8px; }
h1 { font-size:clamp(2rem,4vw,3.4rem); line-height:1.08; letter-spacing:-.04em; margin:0 0 16px; }
h2 { font-size:1.45rem; margin:0 0 14px; } h3 { font-size:1.05rem; margin:8px 0; }
.lead { font-size:1.12rem; color:var(--muted); max-width:760px; margin:0; }
.panel { background:var(--white); border:1px solid var(--line); border-radius:16px; padding:26px; margin:18px 0; box-shadow:0 5px 22px rgba(25,43,64,.035); }
.upload-form { display:grid; gap:10px; max-width:780px; } label { font-weight:700; font-size:.92rem; }
textarea,input[type=file],select { width:100%; border:1px solid #c9d4e0; border-radius:9px; padding:12px; font:inherit; background:white; color:var(--ink); }
textarea:focus,input:focus,select:focus { outline:3px solid #d9e8f8; border-color:#6d94b8; }
button { cursor:pointer; border:0; background:var(--blue); color:white; border-radius:9px; padding:12px 17px; font:inherit; font-weight:750; width:max-content; }
button:hover { filter:brightness(1.08); } button:disabled { opacity:.65; cursor:default; }
button.secondary { background:var(--blue2); color:var(--blue); padding:9px 12px; font-size:.88rem; }
.hint,.muted { color:var(--muted); font-size:.9rem; } .hint { margin:0 0 8px; }
.notice { border-left:3px solid #d4aa5a; padding:10px 14px; background:#fff8e9; color:#6d5628; margin:22px 0 0; }
.feature-grid,.charts-grid,.analysis-cards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.feature-grid article,.analysis-card,.chart-card { border:1px solid var(--line); border-radius:12px; padding:17px; background:#fff; min-width:0; }
.feature-grid p,.analysis-card p { color:var(--muted); margin:8px 0 12px; }
.recent-list { list-style:none; padding:0; margin:0; } .recent-list li { display:flex; justify-content:space-between; gap:20px; border-top:1px solid var(--line); padding:12px 0; } .recent-list span { color:var(--muted); font-size:.85rem; white-space:nowrap; }
.flash-list { margin:0 0 16px; } .flash { padding:12px 14px; border-radius:8px; background:#fff0ef; border:1px solid #f0c4bf; color:#8b3028; }
.stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:14px 0 22px; }
.stat { background:#fff; border:1px solid var(--line); border-radius:12px; padding:18px; } .stat span { display:block; color:var(--muted); font-size:.9rem; } .stat strong { font-size:1.8rem; }
.section-heading { display:flex; justify-content:space-between; align-items:center; gap:15px; } .section-heading h2 { margin-bottom:0; }
.tag { display:inline-block; padding:4px 8px; background:var(--blue2); color:var(--blue); border-radius:6px; font-size:.75rem; font-weight:750; }
.priority { font-size:.78rem; color:var(--muted); } .card-top { display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
.graph { height:350px; border:1px solid var(--line); border-radius:12px; margin:18px 0; background:#fbfcfe; }
.table-wrap { overflow:auto; margin-top:16px; } table { width:100%; border-collapse:collapse; font-size:.9rem; } th,td { text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); white-space:nowrap; } th { background:#f7f9fc; color:#506176; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
.chart-card canvas { max-height:260px; margin-top:18px; } .chart-card label { display:block; margin-top:12px; } .chart-card select { margin-top:5px; }
footer { text-align:center; padding:24px; color:var(--muted); font-size:.85rem; border-top:1px solid var(--line); }
@media(max-width:850px) { .feature-grid,.charts-grid,.analysis-cards { grid-template-columns:1fr 1fr; } .stats-grid { grid-template-columns:repeat(2,1fr); } .graph { height:300px; } }
@media(max-width:580px) { .topbar { align-items:flex-start; flex-direction:column; } .container { padding:20px 14px 40px; } .panel { padding:18px; } .feature-grid,.charts-grid,.analysis-cards { grid-template-columns:1fr; } .section-heading { align-items:flex-start; flex-direction:column; } .recent-list li { flex-direction:column; gap:2px; } }
"""

# ---------------------------------------------------------------------------
# Embedded HTML templates
# ---------------------------------------------------------------------------
BASE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}DAG-EDA{% endblock %}</title>
  <link rel="stylesheet" href="{{ url_for('style_css') }}">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.29.2/dist/cytoscape.min.js"></script>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{{ url_for('index') }}">DAG-EDA <span>Research prototype</span></a>
    <nav><a href="{{ url_for('index') }}">New analysis</a><a href="{{ url_for('history') }}">History</a></nav>
  </header>
  <main class="container">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}<div class="flash-list">{% for category, message in messages %}<div class="flash {{ category }}">{{ message }}</div>{% endfor %}</div>{% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
  </main>
  <footer>Educational prototype &middot; Validate findings before making decisions.</footer>
</body>
</html>
"""

INDEX_HTML = r"""{% extends "base.html" %}
{% block title %}DAG-EDA · Start{% endblock %}
{% block content %}
<section class="hero">
  <p class="eyebrow">AI-assisted exploratory data analysis</p>
  <h1>Turn a broad question into a clear analysis path.</h1>
  <p class="lead">Upload a CSV, ask a question, and explore candidate analyses, dataset concepts, and interactive visualizations.</p>
</section>
<section class="panel">
  <h2>Start an analysis</h2>
  <form method="post" enctype="multipart/form-data" class="upload-form">
    <label for="question">Your analytical question</label>
    <textarea id="question" name="question" rows="3" placeholder="Example: What factors are associated with lower sales?" required></textarea>
    <label for="dataset">Dataset (CSV)</label>
    <input id="dataset" type="file" name="dataset" accept=".csv" required>
    <p class="hint">Use a CSV with column headers (max {{ max_mb }} MB). Files are processed by the Flask app.</p>
    <button type="submit">Analyze dataset</button>
  </form>
</section>
<section class="panel">
  <h2>What this prototype does</h2>
  <div class="feature-grid">
    <article><h3>Intent graph</h3><p>Breaks the question into candidate analysis tasks and methods.</p></article>
    <article><h3>Knowledge graph</h3><p>Maps common domain concepts to matching dataset columns.</p></article>
    <article><h3>Data dashboard</h3><p>Shows column quality, summary statistics, and charts.</p></article>
  </div>
  <p class="notice">The suggestion engine is a transparent rule-based baseline, not a full LLM. It does not infer causation from correlation.</p>
</section>
{% if recent %}
<section class="panel">
  <h2>Recent sessions</h2>
  <ul class="recent-list">
  {% for item in recent %}
    <li><a href="{{ url_for('dashboard', session_id=item.id) }}">{{ item.question }}</a><span>{{ item.created_at }}</span></li>
  {% endfor %}
  </ul>
</section>
{% endif %}
{% endblock %}
"""

HISTORY_HTML = r"""{% extends "base.html" %}
{% block title %}DAG-EDA · History{% endblock %}
{% block content %}
<section class="hero compact"><p class="eyebrow">Saved work</p><h1>Analysis history</h1><p class="lead">Open a previous session to revisit its question, graphs, and candidate analyses.</p></section>
<section class="panel">
{% if rows %}
<div class="table-wrap"><table><thead><tr><th>Question</th><th>Dataset</th><th>Rows</th><th>Columns</th><th>Created</th><th></th></tr></thead><tbody>
{% for r in rows %}<tr><td>{{ r.question }}</td><td>{{ r.filename }}</td><td>{{ r.row_count }}</td><td>{{ r.column_count }}</td><td>{{ r.created_at }}</td><td><a href="{{ url_for('dashboard', session_id=r.id) }}">Open</a></td></tr>{% endfor %}
</tbody></table></div>
{% else %}<p>No saved sessions yet. Upload a CSV to get started.</p>{% endif %}
</section>
{% endblock %}
"""

DASHBOARD_HTML = r"""{% extends "base.html" %}
{% block title %}DAG-EDA · Dashboard{% endblock %}
{% block content %}
<section class="hero compact">
  <p class="eyebrow">Analysis session #{{ session.id }}</p>
  <h1>{{ session.question }}</h1>
  <p class="lead">{{ session.filename }} · {{ session.row_count }} rows · {{ session.column_count }} columns</p>
</section>

<section class="stats-grid">
  <div class="stat"><span>Rows</span><strong>{{ session.summary.rows }}</strong></div>
  <div class="stat"><span>Columns</span><strong>{{ session.summary.columns_count }}</strong></div>
  <div class="stat"><span>Numeric fields</span><strong>{{ session.summary.numeric_columns|length }}</strong></div>
  <div class="stat"><span>Other fields</span><strong>{{ session.summary.categorical_columns|length }}</strong></div>
</section>

<section class="panel">
  <div class="section-heading"><div><p class="eyebrow">01 · Intent graph</p><h2>Possible analysis paths</h2></div><span class="tag">Explore the branches</span></div>
  <p class="muted">Select a task below to view the suggested method and save it to the session history.</p>
  <div id="intent-graph" class="graph"></div>
  <div class="analysis-cards">
    {% for a in session.analyses %}
    <article class="analysis-card">
      <div class="card-top"><span class="tag">{{ a.method }}</span><span class="priority">Priority {{ a.priority }}</span></div>
      <h3>{{ a.title }}</h3><p>{{ a.reason }}</p>
      <button class="secondary save-task" data-task="{{ a.title }}" data-result='{{ a|tojson|forceescape }}'>Save explored path</button>
    </article>
    {% endfor %}
  </div>
</section>

<section class="panel">
  <div class="section-heading"><div><p class="eyebrow">02 · Knowledge graph</p><h2>Concepts linked to dataset variables</h2></div></div>
  <p class="muted">These links are inferred from column names using simple keyword matching. Review them before relying on them.</p>
  <div id="knowledge-graph" class="graph"></div>
  <div class="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing values</th><th>Unique values</th><th>Sample values</th></tr></thead><tbody>
  {% for c in session.summary.columns %}
    <tr><td>{{ c.name }}</td><td>{{ c.dtype }}</td><td>{{ c.missing }} ({{ c.missing_pct }}%)</td><td>{{ c.unique }}</td><td>{{ c.sample|join(", ") }}</td></tr>
  {% endfor %}
  </tbody></table></div>
</section>

<section class="panel">
  <p class="eyebrow">03 · Interactive dashboard</p><h2>Explore the data</h2>
  <div class="charts-grid">
    <article class="chart-card"><h3>Numeric distributions</h3><label for="numeric-select">Choose a numeric column</label><select id="numeric-select"></select><canvas id="histogram"></canvas></article>
    <article class="chart-card"><h3>Correlation overview</h3><canvas id="correlation-chart"></canvas><p class="hint">Correlation indicates association, not causation.</p></article>
    <article class="chart-card"><h3>Missing values</h3><canvas id="missing-chart"></canvas></article>
  </div>
</section>

<script>
const intentGraph = {{ session.intent_graph|tojson }};
const knowledgeGraph = {{ session.knowledge_graph|tojson }};
const summary = {{ session.summary|tojson }};
const analyses = {{ session.analyses|tojson }};
const sessionId = {{ session.id }};

function renderGraph(containerId, graph, isKnowledge) {
  const container = document.getElementById(containerId);
  const elements = [];
  graph.nodes.forEach(n => elements.push({ data: { id: n.id, label: n.label, kind: n.kind } }));
  graph.edges.forEach((e, i) => elements.push({ data: { id: 'edge' + i, source: e.source, target: e.target, label: e.label || '' } }));
  if (typeof cytoscape === 'undefined') {
    container.innerHTML = '<p>Graph library could not load. Use the analysis cards and table below.</p>';
    return;
  }
  cytoscape({
    container, elements,
    style: [
      { selector: 'node', style: { 'label': 'data(label)', 'font-size': 11, 'text-wrap': 'wrap', 'text-max-width': 110, 'text-valign': 'center', 'text-halign': 'center', 'background-color': '#dce7f5', 'color': '#18324b', 'width': 'label', 'height': 'label', 'padding': '12px', 'shape': 'round-rectangle', 'border-width': 1, 'border-color': '#8ba7c7' } },
      { selector: 'node[kind="question"]', style: { 'background-color': '#1e456b', 'color': '#ffffff' } },
      { selector: 'node[kind="concept"]', style: { 'background-color': '#dcefe8', 'border-color': '#7cae9c' } },
      { selector: 'node[kind="variable"]', style: { 'background-color': '#fff0d8', 'border-color': '#d9b779' } },
      { selector: 'node[kind="method"]', style: { 'background-color': '#eee5fa', 'border-color': '#b7a1d8' } },
      { selector: 'edge', style: { 'width': 1.5, 'line-color': '#91a2b6', 'target-arrow-color': '#91a2b6', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'label': 'data(label)', 'font-size': 9, 'color': '#667085' } }
    ],
    layout: { name: 'breadthfirst', directed: true, padding: 30, spacingFactor: isKnowledge ? 1.1 : 1.25, animate: false }
  });
}
renderGraph('intent-graph', intentGraph, false);
renderGraph('knowledge-graph', knowledgeGraph, true);

const numeric = summary.numeric_columns || [];
const select = document.getElementById('numeric-select');
numeric.forEach(col => { const o = document.createElement('option'); o.value = col; o.textContent = col; select.appendChild(o); });

async function loadChartData() {
  const dist = analyses.find(a => a.id === 'numeric_overview');
  if (numeric.length && dist) {
    const response = await fetch(`/api/session/${sessionId}/chart-data`);
    if (response.ok) {
      const payload = await response.json();
      const drawHistogram = (column) => {
        const item = payload.histograms[column] || { labels: [], values: [] };
        if (window.histChart) window.histChart.destroy();
        window.histChart = new Chart(document.getElementById('histogram'), {
          type: 'bar', data: { labels: item.labels, datasets: [{ label: column, data: item.values }] },
          options: { responsive: true, plugins: { legend: { display: false } } }
        });
      };
      drawHistogram(numeric[0]);
      select.addEventListener('change', () => drawHistogram(select.value));
      if (payload.correlations.labels.length) {
        new Chart(document.getElementById('correlation-chart'), {
          type: 'bar', data: { labels: payload.correlations.labels, datasets: [{ label: 'Correlation', data: payload.correlations.values }] },
          options: { indexAxis: 'y', scales: { x: { min: -1, max: 1 } } }
        });
      }
      new Chart(document.getElementById('missing-chart'), {
        type: 'bar', data: { labels: payload.missing.labels, datasets: [{ label: 'Missing values', data: payload.missing.values }] },
        options: { plugins: { legend: { display: false } } }
      });
    }
  } else {
    document.getElementById('histogram').replaceWith(document.createTextNode('No numeric columns available for numeric charts.'));
    document.getElementById('correlation-chart').replaceWith(document.createTextNode('No numeric columns available for correlations.'));
  }
}
loadChartData();

document.querySelectorAll('.save-task').forEach(button => {
  button.addEventListener('click', async () => {
    let result = {};
    try { result = JSON.parse(button.dataset.result); } catch(e) {}
    const response = await fetch(`/api/session/${sessionId}/task`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: button.dataset.task, result })
    });
    button.textContent = response.ok ? 'Saved' : 'Could not save';
    button.disabled = true;
  });
});
</script>
{% endblock %}
"""

# Register embedded templates so {% extends "base.html" %} keeps working.
app.jinja_loader = ChoiceLoader([
    DictLoader({
        "base.html": BASE_HTML,
        "index.html": INDEX_HTML,
        "history.html": HISTORY_HTML,
        "dashboard.html": DASHBOARD_HTML,
    }),
    app.jinja_loader,
])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_csv(path):
    return pd.read_csv(path)


@app.route("/style.css")
def style_css():
    resp = Response(STYLE_CSS, mimetype="text/css")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.errorhandler(413)
def too_large(_e):
    flash(f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        uploaded = request.files.get("dataset")
        if not question:
            flash("Please enter a question you want to investigate.", "error")
            return redirect(url_for("index"))
        if not uploaded or not uploaded.filename:
            flash("Please upload a CSV dataset.", "error")
            return redirect(url_for("index"))
        if not allowed_file(uploaded.filename):
            flash("This prototype accepts CSV files only.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded.filename) or "dataset.csv"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stored_name = f"{stamp}_{filename}"
        path = os.path.join(UPLOAD_DIR, stored_name)
        uploaded.save(path)

        try:
            df = load_csv(path)
            if df.empty or len(df.columns) == 0:
                raise ValueError("The uploaded CSV has no data.")
            if len(df) > 100000:
                raise ValueError("For this demo, please use a CSV with 100,000 rows or fewer.")
            summary = dataframe_summary(df)
            analyses = analyze_dataset(df, question)
            intent_graph, knowledge_graph = build_graphs(question, df, analyses)
            with get_db() as conn:
                cur = conn.execute("""
                    INSERT INTO sessions
                    (filename, question, created_at, row_count, column_count, summary_json,
                     analyses_json, intent_graph_json, knowledge_graph_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stored_name, question, datetime.now().isoformat(timespec="seconds"),
                    int(df.shape[0]), int(df.shape[1]),
                    json.dumps(summary), json.dumps(analyses),
                    json.dumps(intent_graph), json.dumps(knowledge_graph),
                ))
                session_id = cur.lastrowid
            return redirect(url_for("dashboard", session_id=session_id))
        except Exception as exc:
            if os.path.exists(path):
                os.remove(path)
            flash(f"Could not analyze this dataset: {exc}", "error")
            return redirect(url_for("index"))

    with get_db() as conn:
        recent = conn.execute(
            "SELECT id, filename, question, created_at FROM sessions ORDER BY id DESC LIMIT 5"
        ).fetchall()
    return render_template("index.html", recent=recent, max_mb=MAX_UPLOAD_MB)


@app.route("/dashboard/<int:session_id>")
def dashboard(session_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        flash("Analysis session not found.", "error")
        return redirect(url_for("index"))

    data = dict(row)
    data["summary"] = json.loads(data["summary_json"])
    data["analyses"] = json.loads(data["analyses_json"])
    data["intent_graph"] = json.loads(data["intent_graph_json"])
    data["knowledge_graph"] = json.loads(data["knowledge_graph_json"])
    return render_template("dashboard.html", session=data)


@app.route("/history")
def history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename, question, created_at, row_count, column_count "
            "FROM sessions ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/api/session/<int:session_id>/task", methods=["POST"])
def save_task(session_id):
    payload = request.get_json(silent=True) or {}
    task = str(payload.get("task", "")).strip()
    result = payload.get("result", {})
    if not task:
        return jsonify({"error": "Task is required"}), 400
    with get_db() as conn:
        exists = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if exists is None:
            return jsonify({"error": "Session not found"}), 404
        conn.execute(
            "INSERT INTO analysis_history (session_id, task, result_json, created_at) VALUES (?, ?, ?, ?)",
            (session_id, task, json.dumps(result), datetime.now().isoformat(timespec="seconds")),
        )
    return jsonify({"saved": True})


@app.route("/api/session/<int:session_id>/chart-data")
def chart_data(session_id):
    with get_db() as conn:
        row = conn.execute("SELECT filename FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Session not found"}), 404
    path = os.path.join(UPLOAD_DIR, row["filename"])
    if not os.path.exists(path):
        return jsonify({"error": "Uploaded CSV is no longer available"}), 404
    try:
        df = load_csv(path)
        histograms = {}
        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if series.empty:
                histograms[str(col)] = {"labels": [], "values": []}
                continue
            counts = pd.cut(series, bins=min(10, max(1, series.nunique())), duplicates="drop")
            freq = counts.value_counts(sort=False)
            histograms[str(col)] = {
                "labels": [str(interval) for interval in freq.index],
                "values": [int(v) for v in freq.values],
            }
        numeric = df.select_dtypes(include="number")
        correlations = {"labels": [], "values": []}
        if numeric.shape[1] > 1:
            target = numeric.columns[0]
            corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").dropna()
            corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
            correlations = {"labels": [str(c) for c in corr.index],
                            "values": [round(float(v), 3) for v in corr.values]}
        missing = df.isna().sum().sort_values(ascending=False).head(15)
        return jsonify({
            "histograms": histograms,
            "correlations": correlations,
            "missing": {"labels": [str(c) for c in missing.index],
                        "values": [int(v) for v in missing.values]},
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "DAG-EDA prototype"})


# Create tables at import time so gunicorn workers have a ready database.
init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0" if ON_RENDER else "127.0.0.1", port=port, debug=debug)
