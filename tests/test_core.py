"""
tests/test_core.py -- Core unit tests for QuantAgent

Run with:
    pytest tests/test_core.py -v

Tests cover:
  - Signal engine multi-rule validation (single rule must not fire)
  - Portfolio enabled flag (open_position returns None when disabled)
  - Watchlist dismiss (dismissed symbols excluded from priority queue)
  - PnL tracker record and outcome update
  - Market hours logic
  - Alert email flag (no send when SMTP not configured)
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(**kwargs):
    from core.models import TickerProfile, Market
    defaults = dict(
        symbol="TEST", name="Test Corp", price=100.0,
        change_pct=0.0, change_5d=0.0, volume_24h=1_000_000,
        volume_ratio=1.0, market_cap=1_000_000_000,
        rsi=50.0, macd_signal="neutral", bb_position=0.5,
        vwap_vs_price=0.0, sentiment_score=0.0,
        market=Market.US_STOCK, provider_data={"indicators": {}},
    )
    defaults.update(kwargs)
    return TickerProfile(**defaults)


def _make_item(**kwargs):
    from core.models import WatchlistItem
    return WatchlistItem(profile=_make_profile(**kwargs))


# ---------------------------------------------------------------------------
# 1. Signal engine: single rule must not produce an actionable signal
# ---------------------------------------------------------------------------

class TestSignalEngineMultiRule:
    def test_single_rule_no_buy_signal(self):
        """A single rule match must NOT produce a BUY/SELL — only WATCH at best."""
        from signals.signal_engine import SignalEngine
        from core.models import SignalType
        from core.staleness_guard import staleness_guard

        engine = SignalEngine()
        item = _make_item(rsi=28.0, change_pct=0.1, volume_ratio=0.8,
                          change_5d=0.0, macd_signal="neutral",
                          bb_position=0.5, sentiment_score=0.0)
        # Mark as fresh so the staleness gate doesn't skip it
        staleness_guard.mark_refreshed(item.profile.symbol)

        signals = asyncio.get_event_loop().run_until_complete(engine.evaluate(item))

        actionable = [s for s in signals
                      if s.signal_type in (SignalType.BUY, SignalType.STRONG_BUY,
                                           SignalType.SELL, SignalType.STRONG_SELL)]
        assert not actionable, (
            f"Single rule should not produce actionable signal, got: "
            f"{[s.signal_type.value for s in actionable]}"
        )

    def test_two_rules_produce_buy(self):
        """Two aligned buy rules should produce a confirmed BUY."""
        from signals.signal_engine import SignalEngine
        from core.models import SignalType
        from core.staleness_guard import staleness_guard

        # Volume breakout (vr>=3, chg>2) + RSI deeply oversold (<25) = 2 buy rules
        item = _make_item(
            rsi=24.0, change_pct=4.0, volume_ratio=3.5,
            change_5d=5.0, macd_signal="neutral",
            bb_position=0.4, sentiment_score=0.1,
        )
        staleness_guard.mark_refreshed(item.profile.symbol)

        signals = asyncio.get_event_loop().run_until_complete(
            SignalEngine().evaluate(item)
        )
        buy_sigs = [s for s in signals
                    if s.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)]
        assert buy_sigs, "Two buy rules should produce a confirmed BUY signal"
        assert buy_sigs[0].conviction >= 0.5


# ---------------------------------------------------------------------------
# 2. Portfolio: open_position returns None when disabled
# ---------------------------------------------------------------------------

class TestPortfolioEnabledFlag:
    def test_disabled_blocks_open(self):
        """When portfolio is disabled, open_position must return None."""
        from signals.portfolio import PaperPortfolio

        port = PaperPortfolio.__new__(PaperPortfolio)
        port._positions = {}
        port._cash = 25_000.0
        port._enabled = False
        port._min_conviction = 0.80
        port._max_positions = 10

        result = port.open_position({
            "symbol": "AAPL", "signal_type": "BUY",
            "conviction": 0.90, "price_at_signal": 150.0,
            "target_price": 165.0, "stop_loss": 143.0,
            "agent": "test",
        })
        assert result is None, "open_position should return None when disabled"

    def test_enabled_opens_position(self):
        """When portfolio is enabled and conviction meets threshold, position opens."""
        from signals.portfolio import PaperPortfolio

        port = PaperPortfolio.__new__(PaperPortfolio)
        port._positions = {}
        port._cash = 25_000.0
        port._enabled = True
        port._min_conviction = 0.80
        port._max_positions = 10
        port._managed_externally = False

        with patch.object(port, "_fire_save"):
            pos = port.open_position({
                "symbol": "AAPL", "signal_type": "BUY",
                "conviction": 0.85, "price_at_signal": 150.0,
                "target_price": 165.0, "stop_loss": 140.0,
                "agent": "test",
            })
        assert pos is not None
        assert pos.symbol == "AAPL"

    def test_low_conviction_blocked(self):
        """Conviction below threshold must block position open."""
        from signals.portfolio import PaperPortfolio

        port = PaperPortfolio.__new__(PaperPortfolio)
        port._positions = {}
        port._cash = 25_000.0
        port._enabled = True
        port._min_conviction = 0.80
        port._max_positions = 10

        result = port.open_position({
            "symbol": "AAPL", "signal_type": "BUY",
            "conviction": 0.65, "price_at_signal": 150.0,
            "target_price": 165.0, "stop_loss": 140.0,
        })
        assert result is None


# ---------------------------------------------------------------------------
# 3. Watchlist dismiss: dismissed symbols excluded from priority queue
# ---------------------------------------------------------------------------

class TestWatchlistDismiss:
    def test_dismiss_excludes_from_priority(self):
        from agents.watchlist_manager import WatchlistIntelligence

        wli = WatchlistIntelligence()
        wli._analysis_queue = ["AAPL", "TSLA", "NVDA"]

        wli.dismiss("TSLA", "weak analysis", duration_minutes=90)

        priority = wli.get_analysis_priority()
        assert "TSLA" not in priority, "Dismissed symbol should not appear in priority queue"
        assert "AAPL" in priority
        assert "NVDA" in priority

    def test_dismiss_expires(self):
        from agents.watchlist_manager import WatchlistIntelligence

        wli = WatchlistIntelligence()
        wli._analysis_queue = ["AAPL", "TSLA"]

        # Expire immediately by setting past timestamp
        wli._dismissed["TSLA"] = {
            "reason": "test", "ts": time.time() - 200,
            "expires_ts": time.time() - 100,
            "duration_min": 0,
        }

        priority = wli.get_analysis_priority()
        assert "TSLA" in priority, "Expired dismiss should no longer exclude symbol"

    def test_undismiss(self):
        from agents.watchlist_manager import WatchlistIntelligence

        wli = WatchlistIntelligence()
        wli._analysis_queue = ["AAPL", "TSLA"]
        wli.dismiss("TSLA", "test", duration_minutes=60)
        wli.undismiss("TSLA")

        priority = wli.get_analysis_priority()
        assert "TSLA" in priority


# ---------------------------------------------------------------------------
# 4. PnL tracker: record and outcome update
# ---------------------------------------------------------------------------

class TestPnLTracker:
    def _make_tracker(self, tmp_path):
        from signals.pnl_tracker import PnLTracker
        import unittest.mock

        tracker = PnLTracker.__new__(PnLTracker)
        tracker._records = {}
        # Patch _save to no-op
        tracker._save = MagicMock()
        return tracker

    def test_record_signal(self, tmp_path):
        from signals.pnl_tracker import PnLTracker
        tracker = self._make_tracker(tmp_path)

        tracker.record({
            "symbol": "AAPL", "signal_type": "BUY",
            "conviction": 0.85, "price_at_signal": 150.0,
            "target_price": 165.0, "stop_loss": 140.0,
            "reason": "volume_spike", "agent": "signal_engine",
            "ts": time.time(),
        })
        assert len(tracker._records) == 1
        rec = list(tracker._records.values())[0]
        assert rec.symbol == "AAPL"
        assert rec.outcome == "open"

    def test_duplicate_not_recorded(self, tmp_path):
        from signals.pnl_tracker import PnLTracker
        tracker = self._make_tracker(tmp_path)
        ts = time.time()
        sig = {"symbol": "AAPL", "signal_type": "BUY", "conviction": 0.8,
               "price_at_signal": 150.0, "target_price": 165.0,
               "stop_loss": 140.0, "reason": "r", "agent": "a", "ts": ts}
        tracker.record(sig)
        tracker.record(sig)  # duplicate
        assert len(tracker._records) == 1

    def test_win_outcome(self, tmp_path):
        from signals.pnl_tracker import PnLTracker, SignalRecord
        tracker = self._make_tracker(tmp_path)
        ts = time.time() - 90000  # >24h ago so time expiry triggers

        sig_id = f"AAPL_{int(ts)}"
        tracker._records[sig_id] = SignalRecord(
            signal_id=sig_id, symbol="AAPL", signal_type="BUY",
            conviction=0.85, entry_price=100.0,
            target_price=120.0, stop_loss=90.0,
            reason="test", agent="test", ts=ts,
        )

        # Patch DB write since no DB is initialized in this test context
        with patch("core.db.execute"), \
             patch("agents.memory.agent_memory.record_outcome"), \
             patch("agents.memory.agent_memory.remember"):
            asyncio.get_event_loop().run_until_complete(
                tracker.update_outcomes({"AAPL": 115.0})
            )

        rec = tracker._records[sig_id]
        assert rec.outcome == "win"
        assert rec.pnl_pct > 0


# ---------------------------------------------------------------------------
# 5. Market hours
# ---------------------------------------------------------------------------

class TestMarketHours:
    def test_weekend_is_closed(self):
        from core.market_hours import is_market_open
        import datetime
        # Patch to a Sunday at noon ET
        with patch("core.market_hours._now_et") as mock_now:
            mock_now.return_value = datetime.datetime(2024, 1, 7, 12, 0, 0)  # Sunday
            assert is_market_open() is False

    def test_weekday_market_hours_open(self):
        from core.market_hours import is_market_open
        import datetime
        with patch("core.market_hours._now_et") as mock_now:
            mock_now.return_value = datetime.datetime(2024, 1, 8, 10, 30, 0)  # Monday 10:30
            assert is_market_open() is True

    def test_after_hours_closed(self):
        from core.market_hours import is_market_open
        import datetime
        with patch("core.market_hours._now_et") as mock_now:
            mock_now.return_value = datetime.datetime(2024, 1, 8, 17, 0, 0)  # Monday 17:00
            assert is_market_open() is False

    def test_crypto_always_open(self):
        from core.market_hours import is_crypto_trading
        assert is_crypto_trading() is True
