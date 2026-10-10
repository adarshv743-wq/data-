import os
import json
import sqlite3
from datetime import datetime

import pandas as pd
import networkx as nx
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from analysis_engine import analyze_dataset, build_graphs, dataframe_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "dag_eda.db")
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get("DAG_EDA_SECRET", "replace-this-development-secret")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
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
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_csv(path):
    # Keep a reasonable file size limit in the upload route; this reads CSV data.
    return pd.read_csv(path)


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

        filename = secure_filename(uploaded.filename)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    json.dumps(intent_graph), json.dumps(knowledge_graph)
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
    return render_template("index.html", recent=recent)


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
            (session_id, task, json.dumps(result), datetime.now().isoformat(timespec="seconds"))
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
            counts, edges = pd.cut(series, bins=min(10, max(1, series.nunique())), retbins=True, duplicates="drop")
            freq = counts.value_counts(sort=False)
            histograms[str(col)] = {
                "labels": [str(interval) for interval in freq.index],
                "values": [int(v) for v in freq.values]
            }
        numeric = df.select_dtypes(include="number")
        correlations = {"labels": [], "values": []}
        if numeric.shape[1] > 1:
            target = numeric.columns[0]
            corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").dropna()
            corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
            correlations = {"labels": [str(c) for c in corr.index], "values": [round(float(v), 3) for v in corr.values]}
        missing = df.isna().sum().sort_values(ascending=False).head(15)
        return jsonify({
            "histograms": histograms,
            "correlations": correlations,
            "missing": {"labels": [str(c) for c in missing.index], "values": [int(v) for v in missing.values]}
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "DAG-EDA prototype"})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
