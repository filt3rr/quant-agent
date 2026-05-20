"""
tests/test_staleness.py -- StalenessGuard unit tests

Run with: pytest tests/test_staleness.py -v
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStalenessGuard:
    def _make(self):
        from core.staleness_guard import StalenessGuard
        return StalenessGuard()

    # ── mark_refreshed / is_stale ────────────────────────────────────────

    def test_never_refreshed_is_stale(self):
        """A symbol that was never refreshed should always be stale."""
        sg = self._make()
        assert sg.is_stale("AAPL", ttl_seconds=300) is True

    def test_just_refreshed_is_fresh(self):
        """A symbol refreshed right now should not be stale."""
        sg = self._make()
        sg.mark_refreshed("AAPL")
        assert sg.is_stale("AAPL", ttl_seconds=300) is False

    def test_stale_after_ttl(self):
        """A symbol refreshed more than ttl seconds ago is stale."""
        sg = self._make()
        sg._refreshed["TSLA"] = time.time() - 400   # 400s ago
        assert sg.is_stale("TSLA", ttl_seconds=300) is True

    def test_fresh_within_ttl(self):
        """A symbol refreshed less than ttl seconds ago is fresh."""
        sg = self._make()
        sg._refreshed["NVDA"] = time.time() - 100
        assert sg.is_stale("NVDA", ttl_seconds=300) is False

    # ── age_seconds ──────────────────────────────────────────────────────

    def test_age_infinity_when_never_refreshed(self):
        sg = self._make()
        age = sg.age_seconds("UNKNOWN")
        assert age == float("inf")

    def test_age_approximate(self):
        sg = self._make()
        sg._refreshed["MSFT"] = time.time() - 60
        age = sg.age_seconds("MSFT")
        assert 58 <= age <= 65

    # ── reset ────────────────────────────────────────────────────────────

    def test_reset_marks_stale(self):
        sg = self._make()
        sg.mark_refreshed("GOOG")
        assert sg.is_stale("GOOG", ttl_seconds=300) is False
        sg.reset("GOOG")
        assert sg.is_stale("GOOG", ttl_seconds=300) is True

    def test_reset_nonexistent_no_error(self):
        sg = self._make()
        sg.reset("DOESNOTEXIST")   # should not raise

    # ── TTL per symbol type ──────────────────────────────────────────────

    def test_crypto_ttl_is_10min(self):
        sg = self._make()
        ttl = sg.ttl_for("BTC")
        assert ttl == 600    # 10 min

    def test_crypto_ttl_usd_suffix(self):
        sg = self._make()
        assert sg.ttl_for("SOL-USD") == 600
        assert sg.ttl_for("ETHUSDT") == 600

    def test_stock_ttl_depends_on_market_hours(self):
        sg = self._make()
        # During market hours → 5 min
        with patch("core.staleness_guard._is_market_hours", return_value=True):
            assert sg.ttl_for("AAPL") == 300
        # After hours → 30 min
        with patch("core.staleness_guard._is_market_hours", return_value=False):
            assert sg.ttl_for("AAPL") == 1800

    # ── get_stale_symbols ────────────────────────────────────────────────

    def test_get_stale_symbols_empty_watchlist(self):
        sg = self._make()
        result = sg.get_stale_symbols([])
        assert result == []

    def test_get_stale_symbols_all_fresh(self):
        sg = self._make()
        for sym in ["A", "B", "C"]:
            sg.mark_refreshed(sym)
        stale = sg.get_stale_symbols(["A", "B", "C"], ttl_seconds=300)
        assert stale == []

    def test_get_stale_symbols_mixed(self):
        sg = self._make()
        sg.mark_refreshed("FRESH")
        sg._refreshed["OLD"] = time.time() - 400
        stale = sg.get_stale_symbols(["FRESH", "OLD", "NEVER"], ttl_seconds=300)
        assert "FRESH" not in stale
        assert "OLD" in stale
        assert "NEVER" in stale

    # ── get_stats ────────────────────────────────────────────────────────

    def test_get_stats_returns_correct_counts(self):
        sg = self._make()
        sg.mark_refreshed("X")
        sg._refreshed["Y"] = time.time() - 9999
        # Pass a fake watchlist dict
        stats = sg.get_stats({"X": None, "Y": None, "Z": None})
        assert stats["total"] == 3
        # X is fresh, Y and Z are stale
        assert stats["stale"] >= 2

    def test_signals_blocked_counter(self):
        sg = self._make()
        # is_stale() on a symbol that was refreshed recently but TTL=0 → blocks
        sg._refreshed["BLK"] = time.time() - 400
        initial_blocks = sg._stale_signals_blocked
        sg.is_stale("BLK", ttl_seconds=300)
        assert sg._stale_signals_blocked == initial_blocks + 1


class TestMarketHoursHelper:
    def test_is_market_hours_returns_bool(self):
        from core.staleness_guard import _is_market_hours
        result = _is_market_hours()
        assert isinstance(result, bool)

    def test_is_market_hours_weekday_open(self):
        import core.staleness_guard as sg_mod
        with patch.object(sg_mod, "_is_market_hours", return_value=True):
            assert sg_mod._is_market_hours() is True

    def test_is_market_hours_weekend(self):
        import core.staleness_guard as sg_mod
        with patch.object(sg_mod, "_is_market_hours", return_value=False):
            assert sg_mod._is_market_hours() is False
