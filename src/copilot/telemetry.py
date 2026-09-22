"""What every LLM call cost and whether it could be trusted.

One row per generated work order. This table is what the Operations tab reads, and it
is deliberately separate from the work order logic: the deferred human rating rubric
and LLM judge both attach here, not to the generator.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from copilot import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    engine_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    model TEXT NOT NULL,
    latency_s REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    grounded INTEGER NOT NULL,
    failures TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""


def connect() -> sqlite3.Connection:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.TELEMETRY_DB)
    conn.execute(SCHEMA)
    return conn


def record(
    engine_id: int,
    cycle: int,
    model: str,
    latency_s: float,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cost_usd: float,
    grounded: bool,
    failures: list[str],
    payload: str,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO work_orders (created_at, engine_id, cycle, model, latency_s, input_tokens,"
            " output_tokens, cache_read_tokens, cost_usd, grounded, failures, payload)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                engine_id,
                cycle,
                model,
                latency_s,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cost_usd,
                int(grounded),
                json.dumps(failures),
                payload,
            ),
        )


def read_all() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM work_orders ORDER BY id DESC", conn)


def summary_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Latency, cost and groundedness per model, which is the model comparison."""
    return frame.groupby("model").agg(
        calls=("id", "count"),
        p50_latency=("latency_s", lambda s: s.quantile(0.5)),
        p95_latency=("latency_s", lambda s: s.quantile(0.95)),
        mean_cost=("cost_usd", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        cache_reads=("cache_read_tokens", "mean"),
        grounded=("grounded", "mean"),
    )
