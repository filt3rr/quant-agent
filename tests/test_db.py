"""
tests/test_db.py -- SQLite persistence layer unit tests

Run with: pytest tests/test_db.py -v
Tests use a temporary DB file so they never touch the real storage.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for each test."""
    import core.db as db_module
    fake_db = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", fake_db)
    monkeypatch.setattr("config.settings.SYS.STORAGE_DIR", tmp_path)
    return fake_db


class TestDbInit:
    def test_init_creates_tables(self, tmp_db):
        """init_db() should create signal_records, positions, db_meta tables."""
        import core.db as db
        asyncio.get_event_loop().run_until_complete(db.init_db())
        import sqlite3
        conn = sqlite3.connect(str(tmp_db))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "signal_records" in tables
        assert "positions" in tables
        assert "db_meta" in tables

    def test_init_sets_schema_version(self, tmp_db):
        """After init_db(), db_meta should have schema_version=1."""
        import core.db as db
        asyncio.get_event_loop().run_until_complete(db.init_db())
        row = asyncio.get_event_loop().run_until_complete(
            db.fetchone("SELECT value FROM db_meta WHERE key='schema_version'")
        )
        assert row is not None
        assert row["value"] == "1"

    def test_init_idempotent(self, tmp_db):
        """Calling init_db() twice should not raise or duplicate data."""
        import core.db as db
        loop = asyncio.get_event_loop()
        loop.run_until_complete(db.init_db())
        loop.run_until_complete(db.init_db())  # second call must be safe
        row = loop.run_until_complete(
            db.fetchone("SELECT value FROM db_meta WHERE key='schema_version'")
        )
        assert row["value"] == "1"

    def test_wal_mode_enabled(self, tmp_db):
        """DB should be in WAL journal mode after init."""
        import core.db as db, sqlite3
        asyncio.get_event_loop().run_until_complete(db.init_db())
        conn = sqlite3.connect(str(tmp_db))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


class TestSignalRecordsCRUD:
    def _setup(self, tmp_db):
        import core.db as db
        asyncio.get_event_loop().run_until_complete(db.init_db())
        return db

    def test_insert_and_fetch_signal(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        ts = time.time()
        loop.run_until_complete(db.execute(
            """INSERT INTO signal_records
               (signal_id,symbol,signal_type,conviction,entry_price,ts,outcome)
               VALUES(?,?,?,?,?,?,?)""",
            ("TEST_1", "AAPL", "BUY", 0.85, 150.0, ts, "open"),
        ))
        rows = loop.run_until_complete(
            db.fetchall("SELECT * FROM signal_records WHERE symbol='AAPL'")
        )
        assert len(rows) == 1
        assert rows[0]["signal_id"] == "TEST_1"
        assert rows[0]["conviction"] == pytest.approx(0.85)

    def test_insert_ignore_duplicate(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        sql = ("INSERT OR IGNORE INTO signal_records "
               "(signal_id,symbol,signal_type,conviction,entry_price,ts) "
               "VALUES(?,?,?,?,?,?)")
        params = ("DUP_1", "TSLA", "BUY", 0.7, 200.0, time.time())
        loop.run_until_complete(db.execute(sql, params))
        loop.run_until_complete(db.execute(sql, params))  # duplicate
        rows = loop.run_until_complete(
            db.fetchall("SELECT * FROM signal_records WHERE signal_id='DUP_1'")
        )
        assert len(rows) == 1

    def test_update_outcome(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(db.execute(
            "INSERT INTO signal_records(signal_id,symbol,signal_type,conviction,entry_price,ts) "
            "VALUES(?,?,?,?,?,?)",
            ("UPD_1", "NVDA", "BUY", 0.9, 500.0, time.time()),
        ))
        loop.run_until_complete(db.execute(
            "UPDATE signal_records SET outcome='win', pnl_pct=5.2 WHERE signal_id=?",
            ("UPD_1",),
        ))
        row = loop.run_until_complete(
            db.fetchone("SELECT outcome, pnl_pct FROM signal_records WHERE signal_id='UPD_1'")
        )
        assert row["outcome"] == "win"
        assert row["pnl_pct"] == pytest.approx(5.2)

    def test_bulk_insert(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        records = [
            (f"BULK_{i}", f"SYM{i}", "BUY", 0.7, 100.0 + i, time.time())
            for i in range(10)
        ]
        loop.run_until_complete(db.executemany(
            "INSERT OR IGNORE INTO signal_records"
            "(signal_id,symbol,signal_type,conviction,entry_price,ts) "
            "VALUES(?,?,?,?,?,?)",
            records,
        ))
        count = loop.run_until_complete(
            db.fetchone("SELECT COUNT(*) as cnt FROM signal_records")
        )
        assert count["cnt"] == 10


class TestPositionsCRUD:
    def _setup(self, tmp_db):
        import core.db as db
        asyncio.get_event_loop().run_until_complete(db.init_db())
        return db

    def test_insert_and_fetch_position(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(db.execute(
            "INSERT INTO positions(pos_id,symbol,status,entry_price,shares,notional,opened_ts) "
            "VALUES(?,?,?,?,?,?,?)",
            ("POS_1", "AAPL", "open", 150.0, 10.0, 1500.0, time.time()),
        ))
        row = loop.run_until_complete(
            db.fetchone("SELECT * FROM positions WHERE pos_id='POS_1'")
        )
        assert row["symbol"] == "AAPL"
        assert row["status"] == "open"
        assert row["shares"] == pytest.approx(10.0)

    def test_close_position(self, tmp_db):
        db = self._setup(tmp_db)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(db.execute(
            "INSERT INTO positions(pos_id,symbol,status,entry_price,shares,notional,opened_ts) "
            "VALUES(?,?,?,?,?,?,?)",
            ("POS_2", "TSLA", "open", 200.0, 5.0, 1000.0, time.time()),
        ))
        loop.run_until_complete(db.execute(
            "UPDATE positions SET status='closed', close_price=?, pnl_pct=? WHERE pos_id=?",
            (210.0, 5.0, "POS_2"),
        ))
        row = loop.run_until_complete(
            db.fetchone("SELECT status, close_price FROM positions WHERE pos_id='POS_2'")
        )
        assert row["status"] == "closed"
        assert row["close_price"] == pytest.approx(210.0)


class TestJsonMigration:
    def test_pnl_json_migration(self, tmp_db, tmp_path, monkeypatch):
        """Existing pnl.json is imported and renamed to pnl.json.bak."""
        import core.db as db_module

        # Patch storage dir to tmp_path
        monkeypatch.setattr(db_module, "DB_PATH", tmp_db)
        monkeypatch.setattr("config.settings.SYS.STORAGE_DIR", tmp_path)

        # Create a fake pnl.json
        pnl_data = {
            "AAPL_1000": {
                "signal_id": "AAPL_1000",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "conviction": 0.8,
                "entry_price": 150.0,
                "target_price": 165.0,
                "stop_loss": 140.0,
                "reason": "test",
                "agent": "test_agent",
                "ts": 1000.0,
                "outcome": "win",
                "exit_price": 160.0,
                "exit_ts": 2000.0,
                "pnl_pct": 6.67,
                "checked_1h": True,
                "checked_4h": True,
                "checked_1d": True,
            }
        }
        (tmp_path / "pnl.json").write_text(json.dumps(pnl_data))

        asyncio.get_event_loop().run_until_complete(db_module.init_db())

        # Record should be in DB
        rows = asyncio.get_event_loop().run_until_complete(
            db_module.fetchall("SELECT * FROM signal_records")
        )
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["outcome"] == "win"

        # JSON should be renamed
        assert not (tmp_path / "pnl.json").exists()
        assert (tmp_path / "pnl.json.bak").exists()

    def test_portfolio_json_migration(self, tmp_db, tmp_path, monkeypatch):
        """Existing portfolio.json is imported and renamed to portfolio.json.bak."""
        import core.db as db_module

        monkeypatch.setattr(db_module, "DB_PATH", tmp_db)
        monkeypatch.setattr("config.settings.SYS.STORAGE_DIR", tmp_path)

        portfolio_data = {
            "cash": 22500.0,
            "positions": {
                "POS_A": {
                    "pos_id": "POS_A",
                    "symbol": "NVDA",
                    "signal_type": "BUY",
                    "entry_price": 500.0,
                    "shares": 5.0,
                    "notional": 2500.0,
                    "stop_loss": 475.0,
                    "target_price": 550.0,
                    "conviction": 0.85,
                    "agent": "tier2",
                    "sector": "Technology",
                    "opened_ts": 1000.0,
                    "closed_ts": 0.0,
                    "close_price": 0.0,
                    "close_reason": "",
                    "pnl_usd": 0.0,
                    "pnl_pct": 0.0,
                    "status": "open",
                    "current_price": 510.0,
                    "unrealized_pnl": 50.0,
                    "unrealized_pct": 2.0,
                    "risk_1r": 25.0,
                    "breakeven_set": False,
                    "trail_activated": False,
                    "near_earnings": False,
                    "earnings_exit_ts": 0.0,
                }
            },
        }
        (tmp_path / "portfolio.json").write_text(json.dumps(portfolio_data))

        asyncio.get_event_loop().run_until_complete(db_module.init_db())

        rows = asyncio.get_event_loop().run_until_complete(
            db_module.fetchall("SELECT * FROM positions")
        )
        assert len(rows) == 1
        assert rows[0]["symbol"] == "NVDA"

        cash_row = asyncio.get_event_loop().run_until_complete(
            db_module.fetchone("SELECT value FROM db_meta WHERE key='portfolio_cash'")
        )
        assert cash_row is not None
        assert float(cash_row["value"]) == pytest.approx(22500.0)

        assert not (tmp_path / "portfolio.json").exists()
        assert (tmp_path / "portfolio.json.bak").exists()
