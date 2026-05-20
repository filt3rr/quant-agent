"""
core/staleness_guard.py — Per-symbol data freshness tracking

Tracks when each symbol's market data was last successfully enriched.
During market hours stale data = >5 min old; after hours = >30 min old.
The signal engine checks staleness before generating signals so a hung
provider never causes stale prices to trigger real-money orders.

Usage:
    from core.staleness_guard import staleness_guard

    # In scanner after successful enrichment:
    staleness_guard.mark_refreshed("AAPL")

    # In signal engine before evaluating:
    if staleness_guard.is_stale("AAPL"):
        return []

    # Dashboard / heartbeat:
    info = staleness_guard.get_stats(scanner.watchlist)
"""
import time
from typing import Dict, Optional

from core.logger import get_logger

log = get_logger("staleness")

_MARKET_HOURS_TTL  = 300    # 5 min  — during US equity session
_AFTER_HOURS_TTL   = 1_800  # 30 min — pre/post/overnight
_CRYPTO_TTL        = 600    # 10 min — crypto trades 24/7


def _is_market_hours() -> bool:
    """True between 09:30 – 16:00 ET on weekdays (approximate)."""
    try:
        import datetime, zoneinfo
        now = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        t = now.time()
        return datetime.time(9, 30) <= t < datetime.time(16, 0)
    except Exception:
        import datetime
        utc = datetime.datetime.utcnow()
        et = utc.replace(hour=(utc.hour - 4) % 24)
        if et.weekday() >= 5:
            return False
        return 9 <= et.hour < 16


class StalenessGuard:
    def __init__(self):
        self._refreshed: Dict[str, float] = {}   # symbol → last refresh unix ts
        self._stale_signals_blocked = 0

    def mark_refreshed(self, symbol: str):
        """Call after a successful provider enrichment for this symbol."""
        self._refreshed[symbol] = time.time()

    def ttl_for(self, symbol: str) -> float:
        """Return the freshness TTL in seconds appropriate for this symbol."""
        sym = symbol.upper()
        is_crypto = any(
            sym.endswith(x) for x in ("-USD", "/USD", "USDT", "BTC")
        ) or sym in {"BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"}
        if is_crypto:
            return _CRYPTO_TTL
        return _MARKET_HOURS_TTL if _is_market_hours() else _AFTER_HOURS_TTL

    def is_stale(self, symbol: str, ttl_seconds: Optional[float] = None) -> bool:
        """Return True if the symbol's data is older than ttl_seconds."""
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_for(symbol)
        last = self._refreshed.get(symbol, 0.0)
        stale = (time.time() - last) > ttl
        if stale and last > 0:
            self._stale_signals_blocked += 1
        return stale

    def age_seconds(self, symbol: str) -> float:
        """Return seconds since last refresh (or infinity if never refreshed)."""
        last = self._refreshed.get(symbol, 0.0)
        return time.time() - last if last > 0 else float("inf")

    def get_stale_symbols(self, symbols, ttl_seconds: Optional[float] = None) -> list:
        """Return the subset of symbols that are currently stale."""
        return [s for s in symbols if self.is_stale(s, ttl_seconds)]

    def get_stats(self, watchlist: Optional[dict] = None) -> dict:
        """Return staleness summary for the heartbeat / dashboard."""
        now = time.time()
        all_syms = list(watchlist.keys()) if watchlist else list(self._refreshed.keys())
        total = len(all_syms)
        stale_count = sum(1 for s in all_syms if self.is_stale(s))
        oldest_sym = ""
        oldest_age = 0.0
        if all_syms:
            sym = max(all_syms, key=lambda s: now - self._refreshed.get(s, 0))
            oldest_age = round(now - self._refreshed.get(sym, now), 1)
            oldest_sym = sym
        return {
            "total":                 total,
            "stale":                 stale_count,
            "fresh":                 total - stale_count,
            "oldest_symbol":         oldest_sym,
            "oldest_age_s":          oldest_age,
            "signals_blocked_total": self._stale_signals_blocked,
        }

    def reset(self, symbol: str):
        """Force a symbol to appear stale (e.g. after a provider error)."""
        self._refreshed.pop(symbol, None)


staleness_guard = StalenessGuard()
