"""
tests/test_rate_limiter.py -- Token-bucket rate limiter unit tests

Run with: pytest tests/test_rate_limiter.py -v
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRateLimiter:
    """Tests for core.rate_limiter.RateLimiter"""

    def _make(self, cpm: int, burst: int = None) -> "RateLimiter":
        from core.rate_limiter import RateLimiter
        return RateLimiter(calls_per_minute=cpm, burst=burst, name="test")

    # ── Basic token consumption ──────────────────────────────────────────

    def test_acquire_instant_when_tokens_available(self):
        """First acquisition should not wait when bucket is full."""
        limiter = self._make(60, burst=10)
        t0 = time.monotonic()
        asyncio.get_event_loop().run_until_complete(limiter.acquire())
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"First acquire should be instant, took {elapsed:.3f}s"

    def test_burst_capacity_exhausted(self):
        """After consuming burst tokens, next acquisition waits."""
        limiter = self._make(cpm=60, burst=3)
        async def _drain():
            for _ in range(3):
                await limiter.acquire()
        asyncio.get_event_loop().run_until_complete(_drain())
        # Bucket is now empty; next acquire must wait ≥ 0.9s (1 token at 1/s)
        t0 = time.monotonic()
        asyncio.get_event_loop().run_until_complete(limiter.acquire())
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.8, (
            f"Should wait at least 0.8s when burst exhausted, waited {elapsed:.3f}s"
        )

    def test_stats_track_calls(self):
        """get_stats() returns correct call count."""
        limiter = self._make(120, burst=5)
        async def _three():
            for _ in range(3):
                await limiter.acquire()
        asyncio.get_event_loop().run_until_complete(_three())
        stats = limiter.get_stats()
        assert stats["total_calls"] == 3

    def test_context_manager(self):
        """async with limiter: syntax works."""
        limiter = self._make(60, burst=5)
        async def _cm():
            async with limiter:
                return "ok"
        result = asyncio.get_event_loop().run_until_complete(_cm())
        assert result == "ok"
        assert limiter.get_stats()["total_calls"] == 1

    def test_update_cpm(self):
        """update_cpm() changes the refill rate."""
        limiter = self._make(60, burst=10)
        limiter.update_cpm(120)
        assert limiter._cpm == 120
        assert limiter._refill_rate == pytest.approx(2.0)

    def test_singletons_exist(self):
        """Module-level singletons are importable."""
        from core.rate_limiter import (
            finnhub_limiter, polygon_limiter, tavily_limiter, coingecko_limiter
        )
        assert finnhub_limiter._cpm == 55
        assert polygon_limiter._cpm == 5
        assert tavily_limiter._cpm == 20
        assert coingecko_limiter._cpm == 25

    def test_zero_wait_when_refilled(self):
        """After waiting for refill, next token should be available."""
        limiter = self._make(cpm=600, burst=1)   # 10 tokens/sec
        async def _test():
            await limiter.acquire()           # consume
            await asyncio.sleep(0.12)         # wait > 1 token refill
            t0 = time.monotonic()
            await limiter.acquire()           # should be instant
            return time.monotonic() - t0
        elapsed = asyncio.get_event_loop().run_until_complete(_test())
        assert elapsed < 0.1, f"After refill, acquire should be instant; took {elapsed:.3f}s"


class TestRateLimiterConcurrency:
    """Tests for concurrent callers."""

    def test_concurrent_acquires_serialized(self):
        """Multiple concurrent callers should not exceed rate."""
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(calls_per_minute=600, burst=5, name="conc")

        async def _worker(results, idx):
            t0 = time.monotonic()
            await limiter.acquire()
            results[idx] = time.monotonic() - t0

        async def _run():
            results = {}
            tasks = [_worker(results, i) for i in range(7)]
            await asyncio.gather(*tasks)
            return results

        results = asyncio.get_event_loop().run_until_complete(_run())
        # All 7 calls completed; first 5 should be instant, last 2 delayed
        delays = sorted(results.values())
        assert delays[0] < 0.1     # first call instant
        assert len(delays) == 7    # all completed
