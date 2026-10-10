-- SQLite schema for the DAG-EDA prototype.
-- The Flask application also creates these tables automatically at startup.

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
