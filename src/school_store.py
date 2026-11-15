"""Storage for ONE school's own data (one installation = one school).

Everything lives in a single SQLite file: school_data/school.db. Imported files are stored as raw records;
features for the AI models are computed from them (see features.py).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "school_data"
DB = DATA_DIR / "school.db"
DOCS_DIR = DATA_DIR / "docs"

DEFAULT_PROFILE = {
    "school_name": "", "area": "", "school_type": "Private", "ghanaian_language": "Twi",
    "academic_year": "2026/2027",
    "term_dates": {"1": ["2026-09-08", "2026-12-18"], "2": ["2027-01-12", "2027-03-25"],
                   "3": ["2027-04-20", "2027-07-23"]},
    "ca_weight": 30, "office_phone": "", "office_email": "",
}

# Unique key of each table: a re-import of the same record replaces the old one
KEYS = {
    "students": ["student_id"],
    "attendance": ["date", "student_id", "subject"],
    "assessments": ["student_id", "term", "subject", "assessment_type", "assessment_name"],
    "question_marks": ["student_id", "term", "subject", "assessment_name", "question_no"],
    "engagement": ["student_id", "term", "subject"],
    "discipline": ["date", "student_id", "incident"],
    "interventions": ["date", "student_id", "intervention_type"],
    "question_bank": ["subject", "question"],
}


def _conn():
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS import_log (at TEXT, table_name TEXT, file TEXT, rows_added INTEGER,"
                " rows_replaced INTEGER, rows_skipped INTEGER)")
    return con


def get_profile() -> dict:
    with _conn() as con:
        row = con.execute("SELECT value FROM meta WHERE key='profile'").fetchone()
    return {**DEFAULT_PROFILE, **json.loads(row[0])} if row else dict(DEFAULT_PROFILE)


def save_profile(profile: dict) -> None:
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('profile', ?)", (json.dumps(profile),))


def is_configured() -> bool:
    return bool(get_profile().get("school_name"))


def read(table: str) -> pd.DataFrame:
    with _conn() as con:
        exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        return pd.read_sql(f"SELECT * FROM {table}", con) if exists else pd.DataFrame()


def upsert(table: str, df: pd.DataFrame, file_name: str = "", skipped: int = 0) -> tuple[int, int]:
    """Add rows; rows with the same key as existing ones replace them. Returns (added, replaced)."""
    old = read(table)
    key = KEYS[table]
    df = df.copy()
    for k in key:
        df[k] = df[k].astype(str)
    if old.empty:
        merged, replaced = df, 0
    else:
        for k in key:
            old[k] = old[k].astype(str)
        old_keys = set(map(tuple, old[key].to_numpy()))
        replaced = sum(tuple(r) in old_keys for r in df[key].to_numpy())
        merged = pd.concat([old, df], ignore_index=True).drop_duplicates(key, keep="last")
    with _conn() as con:
        merged.to_sql(table, con, if_exists="replace", index=False)
        con.execute("INSERT INTO import_log VALUES (datetime('now','localtime'), ?, ?, ?, ?, ?)",
                    (table, file_name, len(df) - replaced, replaced, skipped))
    return len(df) - replaced, replaced


def clear(table: str | None = None) -> None:
    with _conn() as con:
        tables = [table] if table else list(KEYS)
        for t in tables:
            con.execute(f"DROP TABLE IF EXISTS {t}")


def import_log() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM import_log ORDER BY at DESC", con)


def has_data() -> bool:
    return not read("students").empty
