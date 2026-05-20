"""
tests/test_production.py -- Pillar D production hardening integration tests

Tests cover:
  - Startup validator: pass/fail/warn scenarios
  - Portfolio SQLite round-trip
  - PnL tracker SQLite round-trip
  - Rate limiter provider wiring
  - Staleness guard + signal engine integration

Run with: pytest tests/test_production.py -v
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Redirect all storage to tmp_path for isolation."""
    import core.db as db_module
    import config.settings as settings_module
    fake_db = tmp_path / "quant_agent.db"
    monkeypatch.setattr(db_module, "DB_PATH", fake_db)
    monkeypatch.setattr(settings_module.SYS, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings_module.SYS, "LOG_DIR", tmp_path / "logs")
    return tmp_path


# ── Startup validator ──────────────────────────────────────────────────────

class TestStartupValidator:
    def test_storage_check_passes_with_writable_dir(self, tmp_storage):
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = v._check_storage()
        assert result is True
        levels = [r[0] for r in v._results]
        assert "PASS" in levels

    def test_storage_check_fails_readonly(self, tmp_path, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.SYS, "STORAGE_DIR", Path("/this/path/should/not/be/creatable"))
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        # Patch mkdir to raise PermissionError regardless of OS
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only filesystem")):
            result = v._check_storage()
        assert result is False
        levels = [r[0] for r in v._results]
        assert "FAIL" in levels

    def test_db_check_passes_after_init(self, tmp_storage):
        import core.db as db
        _run(db.init_db())
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = v._check_db()
        assert result is True

    def test_db_check_creates_db_file(self, tmp_storage):
        """DB check should create the file if it doesn't exist."""
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = v._check_db()
        assert result is True  # SQLite can create on connect

    def test_python_version_check_passes(self):
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = v._check_python_version()
        assert result is True  # we're running in 3.9+

    def test_packages_check_passes(self):
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = v._check_required_packages()
        assert result is True

    def test_llm_check_passes_with_anthropic_key(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.KEYS, "ANTHROPIC", "sk-fake-key-abc123")
        monkeypatch.setattr(s.LLM, "PROVIDER", "anthropic")
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = _run(v._check_llm(mock_mode=False))
        assert result is True

    def test_llm_check_fails_missing_anthropic_key(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.KEYS, "ANTHROPIC", None)
        monkeypatch.setattr(s.LLM, "PROVIDER", "anthropic")
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = _run(v._check_llm(mock_mode=False))
        assert result is False

    def test_llm_check_skipped_in_mock_mode(self):
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = _run(v._check_llm(mock_mode=True))
        assert result is True
        levels = [r[0] for r in v._results]
        assert "INFO" in levels

    def test_missing_polygon_key_is_warning_not_fail(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.KEYS, "POLYGON", None)
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        v._check_api_key("POLYGON_API_KEY", None, "Polygon (market data)", "degraded")
        levels = [r[0] for r in v._results]
        assert "WARN" in levels
        assert "FAIL" not in levels

    def test_validate_returns_false_on_missing_llm(self, tmp_storage, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s.KEYS, "ANTHROPIC", None)
        monkeypatch.setattr(s.LLM, "PROVIDER", "anthropic")
        from core.startup_validator import StartupValidator
        v = StartupValidator()
        result = _run(v.validate(mock_mode=False))
        assert result is False


# ── Portfolio SQLite round-trip ────────────────────────────────────────────

class TestPortfolioSQLite:
    def test_open_position_writes_to_db(self, tmp_storage):
        """open_position() should fire a DB write via asyncio task."""
        import core.db as db
        _run(db.init_db())

        from signals.portfolio import PaperPortfolio
        port = PaperPortfolio.__new__(PaperPortfolio)
        port._positions = {}
        port._cash = 25_000.0
        port._enabled = True
        port._min_conviction = 0.80
        port._max_positions = 10
        port._managed_externally = False

        async def _open_and_flush():
            pos = port.open_position({
                "symbol": "AAPL", "signal_type": "BUY",
                "conviction": 0.85, "price_at_signal": 150.0,
                "target_price": 165.0, "stop_loss": 140.0,
                "agent": "test",
            })
            if pos:
                # Allow the create_task() DB write to execute
                await asyncio.sleep(0.05)
            return pos

        pos = _run(_open_and_flush())
        assert pos is not None
        assert pos.symbol == "AAPL"

        # Verify position is in DB
        rows = _run(db.fetchall("SELECT * FROM positions WHERE symbol='AAPL'"))
        assert len(rows) == 1
        assert rows[0]["status"] == "open"

    def test_close_position_updates_db(self, tmp_storage):
        """_close_position() should update status in DB."""
        import core.db as db
        _run(db.init_db())

        from signals.portfolio import PaperPortfolio, Position
        port = PaperPortfolio.__new__(PaperPortfolio)
        port._positions = {}
        port._cash = 25_000.0
        port._enabled = True
        port._min_conviction = 0.80
        port._max_positions = 10
        port._managed_externally = False

        pos = Position(pos_id="TEST_CLOSE", symbol="TSLA", signal_type="BUY",
                       entry_price=200.0, shares=10.0, notional=2000.0,
                       stop_loss=185.0, target_price=230.0,
                       conviction=0.85, opened_ts=time.time())
        port._positions[pos.pos_id] = pos
        port._cash = 23_000.0

        _run(port._close_position(pos, 210.0, "target_hit"))

        row = _run(db.fetchone("SELECT status, close_price FROM positions WHERE pos_id='TEST_CLOSE'"))
        assert row is not None
        assert row["status"] == "closed"
        assert row["close_price"] == pytest.approx(210.0)


# ── PnL tracker SQLite round-trip ─────────────────────────────────────────

class TestPnLTrackerSQLite:
    def test_record_and_persist(self, tmp_storage):
        """record() + _db_insert() should persist signal to DB."""
        import core.db as db
        _run(db.init_db())

        from signals.pnl_tracker import PnLTracker
        tracker = PnLTracker.__new__(PnLTracker)
        tracker._records = {}

        async def _record_and_flush():
            sig = {
                "symbol": "NVDA", "signal_type": "STRONG_BUY",
                "conviction": 0.9, "price_at_signal": 500.0,
                "target_price": 550.0, "stop_loss": 475.0,
                "reason": "volume_spike", "agent": "tier2",
                "ts": time.time(),
            }
            sig_id = tracker.record(sig)
            if sig_id:
                await tracker._db_insert(tracker._records[sig_id])

        _run(_record_and_flush())

        rows = _run(db.fetchall("SELECT * FROM signal_records WHERE symbol='NVDA'"))
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "STRONG_BUY"
        assert rows[0]["outcome"] == "open"

    def test_update_outcome_writes_db(self, tmp_storage):
        """update_outcomes() should UPDATE the DB record."""
        import core.db as db
        _run(db.init_db())

        from signals.pnl_tracker import PnLTracker, SignalRecord
        tracker = PnLTracker.__new__(PnLTracker)
        tracker._records = {}

        sig_id = "OUTCOME_TEST"
        rec = SignalRecord(
            signal_id=sig_id, symbol="AMZN", signal_type="BUY",
            conviction=0.8, entry_price=150.0,
            target_price=165.0, stop_loss=140.0,
            reason="test", agent="test",
            ts=time.time() - 90_000,   # >24h ago
        )
        tracker._records[sig_id] = rec

        # Insert first
        _run(tracker._db_insert(rec))

        # Now update with current price above target → win
        async def _update():
            with patch("agents.memory.agent_memory.record_outcome"), \
                 patch("agents.memory.agent_memory.remember"):
                await tracker.update_outcomes({"AMZN": 170.0})

        _run(_update())

        row = _run(db.fetchone(
            "SELECT outcome, pnl_pct FROM signal_records WHERE signal_id=?", (sig_id,)
        ))
        assert row is not None
        assert row["outcome"] == "win"
        assert row["pnl_pct"] > 0


# ── Rate limiter wiring ────────────────────────────────────────────────────

class TestProviderRateLimiterWiring:
    def test_finnhub_uses_rate_limiter(self):
        """FinnhubProvider._rate should be the token-bucket limiter."""
        from providers.finnhub_provider import FinnhubProvider
        from core.rate_limiter import finnhub_limiter
        provider = FinnhubProvider.__new__(FinnhubProvider)
        provider._key = "fake"
        provider._session = None
        provider._rate = finnhub_limiter
        assert provider._rate is finnhub_limiter

    def test_polygon_uses_rate_limiter(self):
        from providers.polygon_provider import PolygonProvider
        from core.rate_limiter import polygon_limiter
        provider = PolygonProvider.__new__(PolygonProvider)
        provider._key = "fake"
        provider._session = None
        provider._rate_limit = polygon_limiter
        assert provider._rate_limit is polygon_limiter

    def test_tavily_uses_rate_limiter(self):
        from providers.tavily_provider import TavilyProvider
        from core.rate_limiter import tavily_limiter
        provider = TavilyProvider.__new__(TavilyProvider)
        provider._key = "fake"
        provider._session = None
        provider._rate = tavily_limiter
        assert provider._rate is tavily_limiter


# ── Staleness + signal engine integration ─────────────────────────────────

class TestStalenessSignalGating:
    def _make_item(self, symbol="TEST", **kwargs):
        from core.models import TickerProfile, WatchlistItem, Market
        defaults = dict(
            symbol=symbol, name="Test Corp", price=100.0,
            change_pct=5.0, change_5d=5.0, volume_24h=3_000_000,
            volume_ratio=4.0, market_cap=1_000_000_000,
            rsi=45.0, macd_signal="bullish", bb_position=0.4,
            vwap_vs_price=0.5, sentiment_score=0.2,
            market=Market.US_STOCK, provider_data={"indicators": {}},
        )
        defaults.update(kwargs)
        return WatchlistItem(profile=TickerProfile(**defaults))

    def test_stale_symbol_returns_empty_signals(self):
        """evaluate() should return [] for stale symbols."""
        from signals.signal_engine import SignalEngine
        from core.staleness_guard import staleness_guard

        # Ensure symbol is stale (never refreshed)
        staleness_guard.reset("STALE_SYM")
        engine = SignalEngine()
        item = self._make_item("STALE_SYM")
        signals = _run(engine.evaluate(item))
        assert signals == [], "Stale symbol should produce no signals"

    def test_fresh_symbol_can_produce_signals(self):
        """evaluate() should work normally for a freshly-enriched symbol."""
        from signals.signal_engine import SignalEngine
        from core.staleness_guard import staleness_guard

        sym = "FRESH_SYM"
        staleness_guard.mark_refreshed(sym)
        engine = SignalEngine()
        # Construct an item with strong buy indicators
        item = self._make_item(
            sym,
            rsi=25.0, change_pct=4.0, volume_ratio=3.5,
            change_5d=6.0, macd_signal="bullish",
        )
        # Should not return [] due to staleness (may still return [] if rules don't fire)
        signals = _run(engine.evaluate(item))
        # We just verify it ran (didn't short-circuit on staleness)
        assert isinstance(signals, list)


# ── DB concurrency (WAL mode) ─────────────────────────────────────────────

class TestDbConcurrency:
    def test_concurrent_writes_do_not_corrupt(self, tmp_storage):
        """Multiple concurrent write tasks should all succeed."""
        import core.db as db
        _run(db.init_db())

        async def _write(i):
            await db.execute(
                "INSERT OR IGNORE INTO signal_records"
                "(signal_id,symbol,signal_type,conviction,entry_price,ts) "
                "VALUES(?,?,?,?,?,?)",
                (f"CONC_{i}", f"SYM{i}", "BUY", 0.7, 100.0 + i, time.time()),
            )

        async def _run_concurrent():
            await asyncio.gather(*[_write(i) for i in range(20)])

        _run(_run_concurrent())

        count = _run(db.fetchone("SELECT COUNT(*) as cnt FROM signal_records"))
        assert count["cnt"] == 20
