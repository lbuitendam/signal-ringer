# storage.py
from __future__ import annotations
import os, json, sqlite3
from contextlib import closing
from typing import Any, Dict, Iterable, Optional, List
import pandas as pd

DATA_DIR = os.path.join("data")
DB_PATH = os.path.join(DATA_DIR, "signal_ringer.db")
os.makedirs(DATA_DIR, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    # check_same_thread False so Streamlit threads can share the connection.
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db() -> None:
    with closing(get_conn()) as con, closing(con.cursor()) as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts(
            id TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL,
            confidence REAL,
            rr REAL,
            msg TEXT,
            meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders(
            id TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            qty REAL,
            strategy TEXT,
            reason TEXT,
            meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades(
            id TEXT PRIMARY KEY,
            open_ts TEXT,
            close_ts TEXT,
            symbol TEXT,
            side TEXT,
            entry REAL,
            exit REAL,
            pnl REAL,
            r REAL,
            strategy TEXT,
            notes TEXT,
            meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS journal(
            id TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            type TEXT NOT NULL,         -- emotion | plan | setup | note
            text TEXT,
            tags TEXT,
            link_id TEXT,
            meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS backtests(
            id TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL,
            symbol TEXT,
            timeframe TEXT,
            strategy TEXT,
            metrics TEXT,
            params TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist(
            symbol TEXT PRIMARY KEY,
            timeframe TEXT,
            enabled INTEGER,
            datasource TEXT,
            meta TEXT
        )""")
        con.commit()

# ---------- Alerts ----------
def upsert_alerts(rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with closing(get_conn()) as con, closing(con.cursor()) as cur:
        cur.executemany("""
            INSERT OR REPLACE INTO alerts
            (id, ts_utc, symbol, timeframe, strategy, side, price, confidence, rr, msg, meta)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, [
            (
                r["id"],
                r.get("time") or r.get("ts_utc"),
                r["symbol"],
                r.get("tf") or r.get("timeframe"),
                r["strategy"],
                r["side"],
                r.get("price"),
                r.get("confidence"),
                r.get("rr"),
                r.get("msg"),
                json.dumps(r.get("meta", {})),
            )
            for r in rows
        ])
        con.commit()
        return cur.rowcount

def fetch_alerts(filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    filters = filters or {}
    where, args = [], []
    if "symbols" in filters and filters["symbols"]:
        qmarks = ",".join("?" for _ in filters["symbols"])
        where.append(f"symbol IN ({qmarks})")
        args.extend(filters["symbols"])
    if "strategies" in filters and filters["strategies"]:
        qmarks = ",".join("?" for _ in filters["strategies"])
        where.append(f"strategy IN ({qmarks})")
        args.extend(filters["strategies"])
    if "time_min" in filters and filters["time_min"]:
        where.append("ts_utc >= ?"); args.append(filters["time_min"])
    if "time_max" in filters and filters["time_max"]:
        where.append("ts_utc <= ?"); args.append(filters["time_max"])
    sql = "SELECT * FROM alerts"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts_utc DESC"
    with closing(get_conn()) as con:
        return pd.read_sql_query(sql, con, params=args)

# ---------- Journal ----------
def insert_journal(row: Dict[str, Any]) -> None:
    with closing(get_conn()) as con, closing(con.cursor()) as cur:
        cur.execute("""
            INSERT OR REPLACE INTO journal
            (id, ts_utc, type, text, tags, link_id, meta)
            VALUES(?,?,?,?,?,?,?)
        """, (
            row["id"],
            row["ts_utc"],
            row["type"],
            row.get("text",""),
            row.get("tags",""),
            row.get("link_id",""),
            json.dumps(row.get("meta", {})),
        ))
        con.commit()

def fetch_journal(kind: Optional[str] = None) -> pd.DataFrame:
    with closing(get_conn()) as con:
        if kind:
            return pd.read_sql_query(
                "SELECT * FROM journal WHERE type=? ORDER BY ts_utc DESC", con, params=[kind]
            )
        return pd.read_sql_query("SELECT * FROM journal ORDER BY ts_utc DESC", con)

# ---------- Generic export ----------
def export_csv(table: str, path: str) -> str:
    with closing(get_conn()) as con:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
