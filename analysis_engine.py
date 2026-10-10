"""Lightweight data-analysis and graph-building engine for the DAG-EDA prototype.

This is a transparent heuristic baseline, not a trained LLM or a reproduction
of the original research system. It uses dataset metadata, numeric summaries,
and correlations to suggest useful analysis paths.
"""
import math
import re
import networkx as nx
import pandas as pd


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
            "sample": [str(v) for v in s.dropna().head(3).tolist()]
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


def _safe_round(value):
    try:
        if pd.isna(value) or not math.isfinite(float(value)):
            return None
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


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

    target = _find_column(cols, ["sales", "revenue", "profit", "price", "score", "mark", "grade", "target", "amount"])
    if target is None and numeric:
        target = numeric[0]

    if numeric:
        analyses.append({
            "id": "numeric_overview",
            "title": "Explore numeric distributions",
            "reason": "Numeric columns can reveal typical values, spread, and unusual observations.",
            "method": "Descriptive statistics and histograms",
            "columns": numeric[:6],
            "priority": 1
        })

    if target and len(numeric) > 1:
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
            "priority": 1 if any(w in q for w in ["why", "factor", "affect", "relate", "increase", "decrease"]) else 2
        })

    if categorical and numeric:
        cat = _find_column(categorical, ["region", "category", "product", "department", "class", "group", "city", "type"])
        cat = cat or categorical[0]
        num = target or numeric[0]
        groups = []
        try:
            grouped = df.groupby(cat, dropna=False)[num].agg(["mean", "count"]).reset_index().head(15)
            groups = [{
                "category": str(row[cat]),
                "mean": _safe_round(row["mean"]),
                "count": int(row["count"])
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
            "priority": 1 if any(w in q for w in ["compare", "category", "region", "group", "which"]) else 2
        })

    missing = []
    for col in cols:
        count = int(df[col].isna().sum())
        if count:
            missing.append({"column": col, "missing": count, "percent": round(count / max(len(df), 1) * 100, 2)})
    analyses.append({
        "id": "data_quality",
        "title": "Check data quality",
        "reason": "Missing values and unusual column cardinality can affect later conclusions.",
        "method": "Missing-value and uniqueness scan",
        "columns": cols,
        "missing": missing,
        "priority": 1 if missing else 3
    })

    analyses.sort(key=lambda a: a.get("priority", 3))
    return analyses


def build_graphs(question, df, analyses):
    """Return JSON-friendly node/edge dictionaries for the intent and knowledge graphs."""
    intent_nodes = [{"id": "root", "label": question, "kind": "question"}]
    intent_edges = []
    for idx, analysis in enumerate(analyses, start=1):
        node_id = f"task_{idx}"
        intent_nodes.append({
            "id": node_id,
            "label": analysis["title"],
            "kind": "task",
            "analysis_id": analysis["id"]
        })
        intent_edges.append({"source": "root", "target": node_id})
        # Branch from a concrete task into a suggested next step.
        detail_id = f"detail_{idx}"
        intent_nodes.append({
            "id": detail_id,
            "label": analysis["method"],
            "kind": "method",
            "analysis_id": analysis["id"]
        })
        intent_edges.append({"source": node_id, "target": detail_id})

    knowledge_nodes = []
    knowledge_edges = []
    concept_map = {
        "sales": ["sales", "revenue", "turnover", "amount"],
        "pricing": ["price", "cost", "discount"],
        "customer demand": ["units", "quantity", "orders", "demand"],
        "performance": ["profit", "score", "mark", "grade", "rating"],
        "time": ["date", "time", "month", "year", "day"],
        "group/category": ["region", "category", "product", "department", "class", "type", "city"]
    }
    for concept, terms in concept_map.items():
        concept_id = "concept_" + concept.replace("/", "_").replace(" ", "_")
        matched = []
        for col in df.columns:
            normalized = re.sub(r"[^a-z0-9]+", " ", str(col).lower())
            if any(term in normalized for term in terms):
                matched.append(str(col))
        knowledge_nodes.append({
            "id": concept_id,
            "label": concept.title(),
            "kind": "concept",
            "matched": bool(matched)
        })
        for col in matched:
            var_id = "var_" + re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_")
            if not any(n["id"] == var_id for n in knowledge_nodes):
                knowledge_nodes.append({
                    "id": var_id,
                    "label": col,
                    "kind": "variable",
                    "dtype": str(df[col].dtype)
                })
            knowledge_edges.append({"source": concept_id, "target": var_id, "label": "measured by"})

    # Add all remaining variables so users can see unmapped fields too.
    for col in df.columns:
        var_id = "var_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_")
        if not any(n["id"] == var_id for n in knowledge_nodes):
            knowledge_nodes.append({
                "id": var_id,
                "label": str(col),
                "kind": "variable",
                "dtype": str(df[col].dtype)
            })
    return (
        {"nodes": intent_nodes, "edges": intent_edges},
        {"nodes": knowledge_nodes, "edges": knowledge_edges}
    )
