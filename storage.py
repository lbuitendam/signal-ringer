# storage.py
from __future__ import annotations

import os
import json
import hashlib
import sqlite3
from contextlib import closing
from typing import Iterable, Dict, Any, Optional, List

import pandas as pd

DB_PATH = os.path.join("data", "signal_ringer.db")
os.makedirs("data", exist_ok=True)

def db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with closing(db()) as con, closing(con.cursor()) as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts(
          id TEXT PRIMARY KEY,
          ts_utc TEXT,
          symbol TEXT,
          timeframe TEXT,
          strategy TEXT,
          side TEXT,
          price REAL,
          confidence REAL,
          rr REAL,
          msg TEXT,
          meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders(
          id TEXT PRIMARY KEY,
          ts_utc TEXT, symbol TEXT, side TEXT,
          entry REAL, sl REAL, tp1 REAL, tp2 REAL, qty REAL,
          strategy TEXT, reason TEXT, meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades(
          id TEXT PRIMARY KEY,
          open_ts TEXT, close_ts TEXT,
          symbol TEXT, side TEXT,
          entry REAL, exit REAL,
          pnl REAL, r REAL,
          strategy TEXT, notes TEXT, meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS journal(
          id TEXT PRIMARY KEY,
          ts_utc TEXT,
          type TEXT, -- emotion|plan|setup|note
          text TEXT,
          tags TEXT,
          link_id TEXT,
          meta TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS backtests(
          id TEXT PRIMARY KEY,
          ts_utc TEXT, symbol TEXT, timeframe TEXT,
          strategy TEXT, metrics TEXT, params TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist(
          symbol TEXT PRIMARY KEY,
          timeframe TEXT, enabled INT,
          datasource TEXT, meta TEXT
        )""")
        con.commit()

def alert_id(symbol: str, timeframe: str, t_iso: str, name: str, side: str) -> str:
    key = f"{symbol}|{timeframe}|{t_iso}|{name}|{side}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

def upsert_alerts(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with closing(db()) as con, closing(con.cursor()) as cur:
        cur.executemany(
            "INSERT OR REPLACE INTO alerts(id, ts_utc, symbol, timeframe, strategy, side, price, confidence, rr, msg, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["id"],
                    r["time"],
                    r["symbol"],
                    r["tf"],
                    r["strategy"],
                    r["side"],
                    float(r.get("price", 0.0)),
                    float(r.get("confidence", 0.0)),
                    (r.get("rr") if r.get("rr") is not None else None),
                    r.get("msg", ""),
                    json.dumps(r.get("meta", {})),
                )
                for r in rows
            ],
        )
        con.commit()

def fetch_alerts(symbols: Optional[List[str]] = None,
                 strategies: Optional[List[str]] = None,
                 start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    q = "SELECT id, ts_utc, symbol, timeframe, strategy, side, price, confidence, rr, msg, meta FROM alerts"
    conds = []
    args: List[Any] = []
    if symbols:
        conds.append("symbol IN (%s)" % ",".join("?" * len(symbols)))
        args += symbols
    if strategies:
        conds.append("strategy IN (%s)" % ",".join("?" * len(strategies)))
        args += strategies
    if start:
        conds.append("ts_utc >= ?"); args.append(start)
    if end:
        conds.append("ts_utc <= ?"); args.append(end)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY ts_utc DESC"
    with closing(db()) as con:
        df = pd.read_sql_query(q, con, params=args)
    return df

def export_csv(table: str, path: str) -> str:
    with closing(db()) as con:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def insert_journal(row: Dict[str, Any]) -> None:
    with closing(db()) as con, closing(con.cursor()) as cur:
        cur.execute(
            "INSERT OR REPLACE INTO journal(id, ts_utc, type, text, tags, link_id, meta) VALUES(?,?,?,?,?,?,?)",
            (
                row["id"], row["ts_utc"], row["type"],
                row.get("text", ""), row.get("tags", ""), row.get("link_id", ""),
                json.dumps(row.get("meta", {})),
            ),
        )
        con.commit()

def fetch_journal(jtype: Optional[str] = None) -> pd.DataFrame:
    q = "SELECT * FROM journal"
    args: List[Any] = []
    if jtype:
        q += " WHERE type = ?"; args.append(jtype)
    q += " ORDER BY ts_utc DESC"
    with closing(db()) as con:
        return pd.read_sql_query(q, con, params=args)
