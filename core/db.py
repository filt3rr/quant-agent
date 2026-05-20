"""
core/db.py — SQLite persistence layer (WAL mode, async-safe)

Replaces JSON file storage for signal records and portfolio positions.
Uses WAL journal mode for concurrent reads, asyncio.to_thread() so DB
calls never block the event loop.

Public API:
  await init_db()                      — create tables, run JSON migration
  await execute(sql, params)           — INSERT / UPDATE / DELETE
  await executemany(sql, param_list)   — bulk INSERT / UPDATE
  await fetchall(sql, params)          — SELECT → list[dict]
  await fetchone(sql, params)          — SELECT → dict | None

Migration: if legacy storage/pnl.json or storage/portfolio.json exist
they are imported on first run, then renamed to .json.bak.
"""
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional

from core.logger import get_logger
from config.settings import SYS

log = get_logger("db")

DB_PATH = SYS.STORAGE_DIR / "quant_agent.db"
_SCHEMA_VERSION = 1


# ── Low-level connection helpers (synchronous, run in thread) ──────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_records (
            signal_id    TEXT PRIMARY KEY,
            symbol       TEXT NOT NULL,
            signal_type  TEXT NOT NULL,
            conviction   REAL NOT NULL DEFAULT 0,
            entry_price  REAL NOT NULL DEFAULT 0,
            target_price REAL,
            stop_loss    REAL,
            reason       TEXT DEFAULT '',
            agent        TEXT DEFAULT '',
            ts           REAL NOT NULL DEFAULT 0,
            outcome      TEXT DEFAULT 'open',
            exit_price   REAL DEFAULT 0,
            exit_ts      REAL DEFAULT 0,
            pnl_pct      REAL DEFAULT 0,
            checked_1h   INTEGER DEFAULT 0,
            checked_4h   INTEGER DEFAULT 0,
            checked_1d   INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_sr_symbol  ON signal_records(symbol);
        CREATE INDEX IF NOT EXISTS idx_sr_ts      ON signal_records(ts);
        CREATE INDEX IF NOT EXISTS idx_sr_outcome ON signal_records(outcome);

        CREATE TABLE IF NOT EXISTS positions (
            pos_id           TEXT PRIMARY KEY,
            symbol           TEXT NOT NULL,
            signal_type      TEXT DEFAULT '',
            entry_price      REAL DEFAULT 0,
            shares           REAL DEFAULT 0,
            notional         REAL DEFAULT 0,
            stop_loss        REAL DEFAULT 0,
            target_price     REAL DEFAULT 0,
            conviction       REAL DEFAULT 0,
            agent            TEXT DEFAULT '',
            sector           TEXT DEFAULT '',
            opened_ts        REAL DEFAULT 0,
            closed_ts        REAL DEFAULT 0,
            close_price      REAL DEFAULT 0,
            close_reason     TEXT DEFAULT '',
            pnl_usd          REAL DEFAULT 0,
            pnl_pct          REAL DEFAULT 0,
            status           TEXT DEFAULT 'open',
            current_price    REAL DEFAULT 0,
            unrealized_pnl   REAL DEFAULT 0,
            unrealized_pct   REAL DEFAULT 0,
            risk_1r          REAL DEFAULT 0,
            breakeven_set    INTEGER DEFAULT 0,
            trail_activated  INTEGER DEFAULT 0,
            near_earnings    INTEGER DEFAULT 0,
            earnings_exit_ts REAL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);

        CREATE TABLE IF NOT EXISTS db_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT ''
        );
    """)
    conn.commit()


def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM db_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO db_meta(key,value) VALUES(?,?)", (key, value)
    )
    conn.commit()


# ── JSON migrations ────────────────────────────────────────────────────────

def _migrate_pnl_json(conn: sqlite3.Connection):
    pnl_file = SYS.STORAGE_DIR / "pnl.json"
    if not pnl_file.exists():
        return
    try:
        data = json.loads(pnl_file.read_text(encoding="utf-8"))
        rows = 0
        for sig_id, r in data.items():
            conn.execute(
                """INSERT OR IGNORE INTO signal_records
                   (signal_id,symbol,signal_type,conviction,entry_price,
                    target_price,stop_loss,reason,agent,ts,
                    outcome,exit_price,exit_ts,pnl_pct,
                    checked_1h,checked_4h,checked_1d)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r.get("signal_id", sig_id),
                    r.get("symbol", ""),
                    r.get("signal_type", ""),
                    r.get("conviction", 0.0),
                    r.get("entry_price", 0.0),
                    r.get("target_price"),
                    r.get("stop_loss"),
                    r.get("reason", ""),
                    r.get("agent", ""),
                    r.get("ts", 0.0),
                    r.get("outcome", "open"),
                    r.get("exit_price", 0.0),
                    r.get("exit_ts", 0.0),
                    r.get("pnl_pct", 0.0),
                    int(bool(r.get("checked_1h", False))),
                    int(bool(r.get("checked_4h", False))),
                    int(bool(r.get("checked_1d", False))),
                ),
            )
            rows += 1
        conn.commit()
        pnl_file.rename(pnl_file.with_suffix(".json.bak"))
        log.info(f"DB: migrated {rows} signal records from pnl.json")
    except Exception as e:
        log.warning(f"DB: pnl.json migration error: {e}")


def _migrate_portfolio_json(conn: sqlite3.Connection):
    pf_file = SYS.STORAGE_DIR / "portfolio.json"
    if not pf_file.exists():
        return
    try:
        data = json.loads(pf_file.read_text(encoding="utf-8"))
        cash = data.get("cash", 25_000.0)
        _meta_set(conn, "portfolio_cash", str(cash))
        rows = 0
        for pos_id, p in data.get("positions", {}).items():
            conn.execute(
                """INSERT OR IGNORE INTO positions
                   (pos_id,symbol,signal_type,entry_price,shares,notional,
                    stop_loss,target_price,conviction,agent,sector,
                    opened_ts,closed_ts,close_price,close_reason,
                    pnl_usd,pnl_pct,status,current_price,
                    unrealized_pnl,unrealized_pct,risk_1r,
                    breakeven_set,trail_activated,near_earnings,earnings_exit_ts)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p.get("pos_id", pos_id),
                    p.get("symbol", ""),
                    p.get("signal_type", ""),
                    p.get("entry_price", 0.0),
                    p.get("shares", 0.0),
                    p.get("notional", 0.0),
                    p.get("stop_loss", 0.0),
                    p.get("target_price", 0.0),
                    p.get("conviction", 0.0),
                    p.get("agent", ""),
                    p.get("sector", ""),
                    p.get("opened_ts", 0.0),
                    p.get("closed_ts", 0.0),
                    p.get("close_price", 0.0),
                    p.get("close_reason", ""),
                    p.get("pnl_usd", 0.0),
                    p.get("pnl_pct", 0.0),
                    p.get("status", "open"),
                    p.get("current_price", 0.0),
                    p.get("unrealized_pnl", 0.0),
                    p.get("unrealized_pct", 0.0),
                    p.get("risk_1r", 0.0),
                    int(bool(p.get("breakeven_set", False))),
                    int(bool(p.get("trail_activated", False))),
                    int(bool(p.get("near_earnings", False))),
                    p.get("earnings_exit_ts", 0.0),
                ),
            )
            rows += 1
        conn.commit()
        pf_file.rename(pf_file.with_suffix(".json.bak"))
        log.info(f"DB: migrated {rows} positions from portfolio.json")
    except Exception as e:
        log.warning(f"DB: portfolio.json migration error: {e}")


# ── Synchronous worker functions (called via asyncio.to_thread) ────────────

def _init_db_sync():
    conn = _connect()
    try:
        _create_tables(conn)
        version = int(_meta_get(conn, "schema_version", "0"))
        if version < _SCHEMA_VERSION:
            _migrate_pnl_json(conn)
            _migrate_portfolio_json(conn)
            _meta_set(conn, "schema_version", str(_SCHEMA_VERSION))
            log.info(f"DB ready (schema v{_SCHEMA_VERSION}): {DB_PATH.name}")
        else:
            log.info(f"DB loaded (schema v{version}): {DB_PATH.name}")
    finally:
        conn.close()


def _execute_sync(sql: str, params: tuple):
    conn = _connect()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _executemany_sync(sql: str, param_list: list):
    conn = _connect()
    try:
        conn.executemany(sql, param_list)
        conn.commit()
    finally:
        conn.close()


def _fetchall_sync(sql: str, params: tuple) -> List[dict]:
    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _fetchone_sync(sql: str, params: tuple) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Async public API ───────────────────────────────────────────────────────

async def init_db():
    """Create tables and run one-time JSON migration. Call once at startup."""
    await asyncio.to_thread(_init_db_sync)


async def execute(sql: str, params: tuple = ()):
    await asyncio.to_thread(_execute_sync, sql, params)


async def executemany(sql: str, param_list: list):
    await asyncio.to_thread(_executemany_sync, sql, param_list)


async def fetchall(sql: str, params: tuple = ()) -> List[dict]:
    return await asyncio.to_thread(_fetchall_sync, sql, params)


async def fetchone(sql: str, params: tuple = ()) -> Optional[dict]:
    return await asyncio.to_thread(_fetchone_sync, sql, params)
