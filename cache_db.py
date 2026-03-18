"""Persistent SQLite cache for run metadata, params, and metric histories."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

_db_path: Optional[str] = None
_local = threading.local()

DEFAULT_CACHE_DIR = os.path.expanduser("~/.mlflow_compare_ui")


def init_db(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Create the cache directory and initialise the SQLite schema."""
    global _db_path
    os.makedirs(cache_dir, exist_ok=True)
    _db_path = os.path.join(cache_dir, "cache.db")
    conn = _get_conn()
    conn.executescript(_SCHEMA)
    conn.commit()
    log.info("SQLite cache initialised at %s", _db_path)


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread connection (SQLite connections aren't thread-safe)."""
    if _db_path is None:
        raise RuntimeError("cache_db.init_db() has not been called")
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    experiment_id  TEXT NOT NULL,
    run_name       TEXT,
    start_time     TEXT,
    start_time_ms  INTEGER,
    status         TEXT,
    tags           TEXT,
    tags_list      TEXT,
    metric_keys    TEXT,
    fetched_at     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(experiment_id, start_time_ms);

CREATE TABLE IF NOT EXISTS params (
    run_id  TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS metric_history (
    run_id      TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    step        INTEGER NOT NULL,
    value       REAL,
    timestamp   INTEGER,
    PRIMARY KEY (run_id, metric_name, step)
);

CREATE INDEX IF NOT EXISTS idx_mh_run_metric ON metric_history(run_id, metric_name);

CREATE TABLE IF NOT EXISTS fetch_status (
    run_id      TEXT NOT NULL,
    data_type   TEXT NOT NULL,
    fetched_at  INTEGER,
    PRIMARY KEY (run_id, data_type)
);
"""

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def get_cached_runs(experiment_id: str) -> list[dict[str, Any]]:
    """Return all cached run records for an experiment."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM runs WHERE experiment_id = ? ORDER BY start_time_ms DESC",
        (experiment_id,),
    ).fetchall()
    return [_row_to_run(r) for r in rows]


def get_max_start_time(experiment_id: str) -> Optional[int]:
    """Return the most recent start_time_ms for cached runs, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT MAX(start_time_ms) FROM runs WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    return row[0] if row and row[0] is not None else None



def upsert_runs(runs: list[dict[str, Any]], experiment_id: str) -> None:
    """Insert or replace run metadata rows."""
    if not runs:
        return
    conn = _get_conn()
    now = int(time.time())
    conn.executemany(
        """INSERT OR REPLACE INTO runs
           (run_id, experiment_id, run_name, start_time, start_time_ms,
            status, tags, tags_list, metric_keys, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["run_id"],
                experiment_id,
                r["run_name"],
                r["start_time"],
                r["start_time_ms"],
                r["status"],
                json.dumps(r.get("tags", {})),
                json.dumps(r.get("tags_list", [])),
                json.dumps(r.get("metric_keys", [])),
                now,
            )
            for r in runs
        ],
    )
    conn.commit()


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row from the runs table to the dict the app expects."""
    return {
        "run_id": row["run_id"],
        "run_name": row["run_name"],
        "start_time": row["start_time"],
        "start_time_ms": row["start_time_ms"],
        "status": row["status"],
        "tags": json.loads(row["tags"]) if row["tags"] else {},
        "tags_list": json.loads(row["tags_list"]) if row["tags_list"] else [],
        "metric_keys": json.loads(row["metric_keys"]) if row["metric_keys"] else [],
    }


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

def get_cached_params(run_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return {run_id: {key: value}} for runs whose params have been fetched."""
    if not run_ids:
        return {}
    conn = _get_conn()
    result: dict[str, dict[str, str]] = {}
    fetched_ids = _get_fetched_ids(run_ids, "params")
    if not fetched_ids:
        return result
    placeholders = ",".join("?" for _ in fetched_ids)
    rows = conn.execute(
        f"SELECT run_id, key, value FROM params WHERE run_id IN ({placeholders})",
        fetched_ids,
    ).fetchall()
    for r in rows:
        result.setdefault(r["run_id"], {})[r["key"]] = r["value"]
    for rid in fetched_ids:
        result.setdefault(rid, {})
    return result


def upsert_params(run_id: str, params: dict[str, str]) -> None:
    """Store params for a run and mark it as fetched."""
    conn = _get_conn()
    now = int(time.time())
    conn.executemany(
        "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
        [(run_id, k, v) for k, v in params.items()],
    )
    conn.execute(
        "INSERT OR REPLACE INTO fetch_status (run_id, data_type, fetched_at) VALUES (?, 'params', ?)",
        (run_id, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Metric history
# ---------------------------------------------------------------------------

def get_cached_metric_history(
    run_id: str, metric_name: str
) -> Optional[dict[str, Any]]:
    """Return cached metric history dict, or None if not fetched yet."""
    if not _is_fetched(run_id, metric_name):
        return None
    conn = _get_conn()
    rows = conn.execute(
        "SELECT step, value, timestamp FROM metric_history "
        "WHERE run_id = ? AND metric_name = ? ORDER BY step",
        (run_id, metric_name),
    ).fetchall()
    return {
        "run_id": run_id,
        "metric": metric_name,
        "steps": [r["step"] for r in rows],
        "values": [r["value"] for r in rows],
        "timestamps": [r["timestamp"] for r in rows],
    }


def upsert_metric_history(
    run_id: str,
    metric_name: str,
    steps: list[int],
    values: list[float],
    timestamps: list[int],
) -> None:
    """Store metric history rows and mark as fetched."""
    conn = _get_conn()
    now = int(time.time())
    conn.executemany(
        "INSERT OR REPLACE INTO metric_history "
        "(run_id, metric_name, step, value, timestamp) VALUES (?, ?, ?, ?, ?)",
        [(run_id, metric_name, s, v, t) for s, v, t in zip(steps, values, timestamps)],
    )
    conn.execute(
        "INSERT OR REPLACE INTO fetch_status (run_id, data_type, fetched_at) VALUES (?, ?, ?)",
        (run_id, metric_name, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fetch status helpers
# ---------------------------------------------------------------------------

def _is_fetched(run_id: str, data_type: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM fetch_status WHERE run_id = ? AND data_type = ?",
        (run_id, data_type),
    ).fetchone()
    return row is not None


def _get_fetched_ids(run_ids: list[str], data_type: str) -> list[str]:
    """Return the subset of run_ids that have been fetched for data_type."""
    if not run_ids:
        return []
    conn = _get_conn()
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT run_id FROM fetch_status WHERE data_type = ? AND run_id IN ({placeholders})",
        [data_type] + list(run_ids),
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clear_cache(experiment_id: Optional[str] = None) -> None:
    """Wipe cached data. If experiment_id given, only clear that experiment's runs."""
    conn = _get_conn()
    if experiment_id:
        run_ids_rows = conn.execute(
            "SELECT run_id FROM runs WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
        run_ids = [r[0] for r in run_ids_rows]
        if run_ids:
            ph = ",".join("?" for _ in run_ids)
            conn.execute(f"DELETE FROM params WHERE run_id IN ({ph})", run_ids)
            conn.execute(f"DELETE FROM metric_history WHERE run_id IN ({ph})", run_ids)
            conn.execute(f"DELETE FROM fetch_status WHERE run_id IN ({ph})", run_ids)
        conn.execute("DELETE FROM runs WHERE experiment_id = ?", (experiment_id,))
    else:
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM params")
        conn.execute("DELETE FROM metric_history")
        conn.execute("DELETE FROM fetch_status")
    conn.commit()
