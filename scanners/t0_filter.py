"""
scanners/t0_filter.py -- Multi-timeframe (T0) pre-filter

Runs BEFORE Tier-1 LLM analysis.  Pure rule-based, no LLM calls.
Fetches 1-hour OHLCV bars (up to 30d) via the existing fetch_ohlcv_yf
helper and evaluates three alignment checks:

  h4_aligned        : 4h trend (resampled from 1h) is above 20-bar EMA
                      and last 4h candle closed higher than prior (bullish)
                      or below prior (bearish).
  h1_vwap_confirmed : 1h close is on the correct side of the rolling 24-bar
                      VWAP (above for bullish, below for bearish).
  volume_expanding  : last 3 completed 1h bars show non-decreasing volume
                      (each ≥ 90% of prior bar to allow minor variation).

A T0Result bundles all three flags and a score 0-3.
Results are cached for 20 minutes so repeat T1 calls for the same symbol
skip the network hit.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from core.models import Market
from core.logger import get_logger

log = get_logger("t0_filter")

_CACHE_TTL = 1_200   # 20 minutes
_cache: Dict[str, tuple] = {}   # symbol -> (T0Result, ts)


@dataclass
class T0Result:
    symbol:             str
    h4_aligned:         Optional[bool] = None
    h1_vwap_confirmed:  Optional[bool] = None
    volume_expanding:   Optional[bool] = None
    score:              int  = 0
    error:              str  = ""

    def to_dict(self) -> dict:
        return {
            "h4_aligned":        self.h4_aligned,
            "h1_vwap_confirmed": self.h1_vwap_confirmed,
            "volume_expanding":  self.volume_expanding,
            "t0_score":          self.score,
        }


async def run_t0_filter(
    symbol: str,
    market: Market,
    bias: str = "bullish",   # "bullish" | "bearish"
) -> T0Result:
    """
    Returns a T0Result with alignment flags stamped on success.
    Never raises — returns a result with error set on failure.
    """
    cache_key = f"{symbol}:{bias}"
    now = time.time()
    if cache_key in _cache:
        cached, ts = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            return cached

    result = T0Result(symbol=symbol)
    try:
        from scanners.technicals import fetch_ohlcv_yf

        # 30 days of 1h bars (~195 bars for US stocks, 720 for crypto)
        h1_df = await fetch_ohlcv_yf(symbol, period="30d", interval="1h")
        if h1_df is None or len(h1_df) < 24:
            result.error = "insufficient_data"
            _cache[cache_key] = (result, now)
            return result

        import pandas as pd
        import numpy as np

        h1_df = h1_df.copy()
        # Ensure DatetimeIndex for resampling
        if not isinstance(h1_df.index, pd.DatetimeIndex):
            result.error = "non_datetime_index"
            _cache[cache_key] = (result, now)
            return result

        score = 0

        # ── CHECK 1: h4_aligned ───────────────────────────────────────────
        try:
            h4_df = h1_df.resample("4h").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["close"])

            if len(h4_df) >= 20:
                close4 = h4_df["close"].astype(float)
                ema20  = close4.ewm(span=20, adjust=False).mean()
                last_c = float(close4.iloc[-1])
                prev_c = float(close4.iloc[-2])
                last_e = float(ema20.iloc[-1])

                if bias == "bearish":
                    aligned = last_c < last_e and last_c < prev_c
                else:
                    aligned = last_c > last_e and last_c >= prev_c * 0.998  # tiny tolerance

                result.h4_aligned = aligned
                if aligned:
                    score += 1
        except Exception as e:
            log.debug(f"T0 h4 check [{symbol}]: {e}")

        # ── CHECK 2: h1_vwap_confirmed ────────────────────────────────────
        try:
            recent = h1_df.tail(24)  # approximate intraday VWAP window
            close1 = recent["close"].astype(float)
            high1  = recent["high"].astype(float)
            low1   = recent["low"].astype(float)
            vol1   = recent["volume"].astype(float)

            tp   = (high1 + low1 + close1) / 3
            tvol = float(vol1.sum())
            vwap = float((tp * vol1).sum() / tvol) if tvol > 0 else float(close1.mean())
            last_c1 = float(close1.iloc[-1])

            if bias == "bearish":
                confirmed = last_c1 < vwap
            else:
                confirmed = last_c1 > vwap

            result.h1_vwap_confirmed = confirmed
            if confirmed:
                score += 1
        except Exception as e:
            log.debug(f"T0 VWAP check [{symbol}]: {e}")

        # ── CHECK 3: volume_expanding ─────────────────────────────────────
        try:
            vol = h1_df["volume"].astype(float)
            if len(vol) >= 4:
                # last 3 completed bars (skip the current incomplete bar)
                v = [float(vol.iloc[-4]), float(vol.iloc[-3]), float(vol.iloc[-2])]
                expanding = (v[1] >= v[0] * 0.90) and (v[2] >= v[1] * 0.90)
                result.volume_expanding = expanding
                if expanding:
                    score += 1
        except Exception as e:
            log.debug(f"T0 volume check [{symbol}]: {e}")

        result.score = score
        _cache[cache_key] = (result, now)

        log.debug(
            f"T0 [{symbol}] bias={bias} score={score}/3 "
            f"h4={result.h4_aligned} vwap={result.h1_vwap_confirmed} vol={result.volume_expanding}"
        )
        return result

    except Exception as e:
        result.error = str(e)[:80]
        log.debug(f"T0 filter error [{symbol}]: {e}")
        _cache[cache_key] = (result, now)
        return result
