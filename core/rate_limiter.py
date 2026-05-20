"""
core/rate_limiter.py — Token-bucket rate limiter for API providers

Enforces a real calls-per-minute ceiling (not just concurrency).
Each provider gets its own singleton limiter configured to its tier limits.

Usage:
    from core.rate_limiter import finnhub_limiter

    async with finnhub_limiter:
        resp = await session.get(url)

    # or explicit acquire:
    await finnhub_limiter.acquire()
    resp = await session.get(url)

Rate limits (conservative, free-tier safe):
  Finnhub   — 60  req/min  (free), burst = 10
  Polygon   — 5   req/min  (free Starter), burst = 5
  Tavily    — 20  req/min  (free), burst = 5
  CoinGecko — 30  req/min  (public), burst = 10
"""
import asyncio
import time
from typing import Optional

from core.logger import get_logger

log = get_logger("rate_limiter")


class RateLimiter:
    """
    Token-bucket rate limiter.
    Refills at `calls_per_minute / 60` tokens per second.
    `burst` is the maximum tokens that can accumulate (default = calls_per_minute).
    """

    def __init__(self, calls_per_minute: int, burst: Optional[int] = None, name: str = ""):
        self._cpm = max(1, calls_per_minute)
        self._burst = burst or calls_per_minute
        self._tokens = float(self._burst)
        self._refill_rate = self._cpm / 60.0   # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._name = name or f"limiter({calls_per_minute}cpm)"
        self._total_waits = 0
        self._total_calls = 0

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1):
        """Block until `tokens` tokens are available, then consume them."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_calls += 1
                    return
                wait = (tokens - self._tokens) / self._refill_rate

            self._total_waits += 1
            log.debug(f"{self._name}: rate-limited, sleeping {wait:.2f}s")
            await asyncio.sleep(wait)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *_):
        pass

    def get_stats(self) -> dict:
        return {
            "name":        self._name,
            "cpm":         self._cpm,
            "tokens":      round(self._tokens, 2),
            "total_calls": self._total_calls,
            "total_waits": self._total_waits,
        }

    def update_cpm(self, calls_per_minute: int):
        """Dynamically adjust the rate (e.g. when user upgrades API tier)."""
        self._cpm = max(1, calls_per_minute)
        self._burst = calls_per_minute
        self._refill_rate = self._cpm / 60.0
        log.info(f"{self._name}: rate updated to {calls_per_minute} req/min")


# ── Per-provider singletons ────────────────────────────────────────────────
# Set conservatively; operators can call limiter.update_cpm() at runtime.

finnhub_limiter   = RateLimiter(calls_per_minute=55,  burst=10,  name="finnhub")
polygon_limiter   = RateLimiter(calls_per_minute=5,   burst=5,   name="polygon")
tavily_limiter    = RateLimiter(calls_per_minute=20,  burst=5,   name="tavily")
coingecko_limiter = RateLimiter(calls_per_minute=25,  burst=10,  name="coingecko")
