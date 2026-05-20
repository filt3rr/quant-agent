"""
scanners/technicals.py -- Technical indicators (Windows-safe, crypto-aware, noise-free)
"""
import asyncio
import logging
import time
import warnings
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from core.logger import get_logger

log = get_logger("technicals")

# Suppress ALL yfinance/pandas noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for lib in ("yfinance", "peewee", "urllib3", "requests", "asyncio"):
    logging.getLogger(lib).setLevel(logging.CRITICAL)

# Crypto symbols that use -USD suffix in yfinance
# Curated list -- tokens NOT on this list won't get -USD appended
_CRYPTO_SYMS = {
    "BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","DOT","MATIC",
    "LINK","UNI","LTC","BCH","ATOM","XLM","NEAR","ALGO","ICP","FIL",
    "ETC","APT","ARB","OP","AAVE","MKR","CRV","COMP","SNX","SUSHI",
    "PEPE","SHIB","FLOKI","WIF","BONK","MEME","TON","TRX","EOS",
    "XMR","ZEC","DASH","NEO","VET","HBAR","GRT","IMX","MANA","SAND",
    "AXS","ENJ","GALA","THETA","FTM","ONE","KAVA","ZIL","CHZ","BAT",
    "USDT","USDC","BUSD","DAI","TUSD","USDP","FDUSD",
    "JUP","ENA","TAO","HYPE","PENGU","ONDO","VIRTUAL",
    "SUI","SEI","TIA","PYTH","JTO","RNDR","INJ","BLUR",
    "DYDX","LDO","RPL","ANKR","OCEAN","FET","AGIX",
    "PAXG","WBTC","STETH","RETH",
}

# Symbols known to not exist in yfinance -- skip immediately, no network call
_YF_SKIP = {
    "USD1","ASTER","FIGR_HELOC","TRUMP","HYPE","PENGU",
    "VIRTUAL","ONDO","TAO","UNI","PEPE",  # often not on Yahoo
}

# Cache: symbol -> (indicators, timestamp)
_cache: Dict[str, tuple] = {}
_CACHE_TTL = 300           # 5 minutes
_fetch_count = 0           # prune trigger counter


def _prune_cache():
    """Remove expired entries to prevent unbounded dict growth."""
    cutoff = time.time() - _CACHE_TTL
    stale = [k for k, (_, ts) in list(_cache.items()) if ts < cutoff]
    for k in stale:
        _cache.pop(k, None)


def _yf_symbol(symbol: str) -> Optional[str]:
    """Convert scanner symbol to yfinance symbol. Returns None to skip."""
    sym = symbol.upper().strip()
    if sym in _YF_SKIP:
        return None
    if "-" in sym or "=" in sym:
        return sym
    if sym in _CRYPTO_SYMS:
        return f"{sym}-USD"
    return sym


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def compute_indicators(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 14:
        return {}

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    result = {}

    # RSI (14)
    try:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        result["rsi"] = _safe_float((100 - (100 / (1 + rs))).iloc[-1])
    except Exception:
        pass

    # MACD
    try:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        sig    = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - sig
        prev   = hist.iloc[-2] if len(hist) > 1 else 0
        result["macd"]       = _safe_float(macd.iloc[-1])
        result["macd_hist"]  = _safe_float(hist.iloc[-1])
        result["macd_cross"] = (
            "bullish" if hist.iloc[-1] > 0 and prev <= 0 else
            "bearish" if hist.iloc[-1] < 0 and prev >= 0 else
            ("bullish" if hist.iloc[-1] > 0 else "bearish")
        )
    except Exception:
        pass

    # Bollinger Bands (20, 2)
    try:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        rng   = upper.iloc[-1] - lower.iloc[-1]
        result["bb_position"] = _safe_float(
            (close.iloc[-1] - lower.iloc[-1]) / rng if rng > 0 else 0.5
        )
        result["bb_squeeze"]  = bool(_safe_float(std20.iloc[-1] / sma20.iloc[-1]) < 0.02)
        result["bb_upper"]    = _safe_float(upper.iloc[-1])
        result["bb_lower"]    = _safe_float(lower.iloc[-1])
    except Exception:
        pass

    # VWAP -- use last 5 bars only (avoids multi-month drift on daily data)
    try:
        window    = min(5, len(close))
        h5, l5, c5, v5 = high[-window:], low[-window:], close[-window:], vol[-window:]
        typical   = (h5 + l5 + c5) / 3
        vwap_val  = (typical * v5).sum() / v5.sum() if v5.sum() > 0 else close.iloc[-1]
        cur       = close.iloc[-1]
        vwap_diff = (cur - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0
        # Clamp to sane range -- if >15% off, data is suspect
        if abs(vwap_diff) <= 15:
            result["vwap"]          = _safe_float(vwap_val)
            result["vwap_vs_price"] = _safe_float(vwap_diff)
        else:
            result["vwap"]          = _safe_float(cur)
            result["vwap_vs_price"] = 0.0
    except Exception:
        pass

    # ATR (14)
    try:
        tr  = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        result["atr"]     = _safe_float(atr.iloc[-1])
        result["atr_pct"] = _safe_float(
            atr.iloc[-1] / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        )
    except Exception:
        pass

    # Volume
    try:
        avg_vol = vol.rolling(20).mean().iloc[-1]
        result["volume_ratio"] = _safe_float(vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0)
        result["avg_volume"]   = _safe_float(avg_vol)
    except Exception:
        pass

    # Momentum ROC
    try:
        for n in [5, 10, 20]:
            if len(close) > n:
                roc = (close - close.shift(n)) / close.shift(n) * 100
                result[f"roc{n}"] = _safe_float(roc.iloc[-1])
    except Exception:
        pass

    # Stochastic (14, 3)
    try:
        low14   = low.rolling(14).min()
        high14  = high.rolling(14).max()
        stoch_k = (close - low14) / (high14 - low14) * 100
        result["stoch_k"] = _safe_float(stoch_k.iloc[-1])
        result["stoch_d"] = _safe_float(stoch_k.rolling(3).mean().iloc[-1])
    except Exception:
        pass

    # EMAs
    try:
        for p in [9, 21, 50]:
            if len(close) >= p:
                result[f"ema{p}"] = _safe_float(
                    close.ewm(span=p, adjust=False).mean().iloc[-1]
                )
    except Exception:
        pass

    return result


async def fetch_ohlcv_yf(symbol: str, period: str = "60d", interval: str = "1d") -> Optional[pd.DataFrame]:
    global _fetch_count
    _fetch_count += 1
    if _fetch_count % 20 == 0:
        _prune_cache()

    yf_sym = _yf_symbol(symbol)
    if yf_sym is None:
        return None  # Skip known-bad symbols instantly

    # Check cache — key includes period+interval so different callers don't collide
    _cache_key = f"{yf_sym}:{period}:{interval}"
    cached = _cache.get(_cache_key)
    if cached:
        df, ts = cached
        if time.time() - ts < _CACHE_TTL:
            return df

    try:
        import yfinance as yf

        loop = asyncio.get_event_loop()

        def _fetch():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = yf.Ticker(yf_sym)
                # Suppress yfinance's own stderr output
                import io, sys
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    h = t.history(period=period, interval=interval)
                finally:
                    sys.stderr = old_stderr
                return h

        hist = await loop.run_in_executor(None, _fetch)
        if hist is None or hist.empty:
            _cache[_cache_key] = (None, time.time())
            return None
        hist.columns = [c.lower() for c in hist.columns]
        needed = [c for c in ["open", "high", "low", "close", "volume"] if c in hist.columns]
        if "close" not in needed:
            return None
        df = hist[needed]
        _cache[_cache_key] = (df, time.time())
        return df
    except Exception as e:
        log.debug(f"yfinance [{yf_sym}]: {type(e).__name__}")
        return None


async def get_technicals(symbol: str) -> Dict:
    df, df_1h = await asyncio.gather(
        fetch_ohlcv_yf(symbol),
        fetch_ohlcv_yf(symbol, period="5d", interval="1h"),
        return_exceptions=True,
    )
    if isinstance(df, Exception) or df is None or (hasattr(df, "empty") and df.empty):
        df = None
    if isinstance(df_1h, Exception):
        df_1h = None

    if df is None:
        return {}
    result = compute_indicators(df)
    result["latest_close"] = _safe_float(df["close"].iloc[-1])
    if len(df) > 1:
        result["prev_close"] = _safe_float(df["close"].iloc[-2])
        # True overnight gap: today's open vs previous close (not intraday move)
        if "open" in df.columns:
            prev_c = _safe_float(df["close"].iloc[-2])
            today_o = _safe_float(df["open"].iloc[-1])
            if prev_c > 0:
                result["gap_pct"] = _safe_float((today_o - prev_c) / prev_c * 100)
    if len(df) >= 5:
        c0, c5 = df["close"].iloc[-1], df["close"].iloc[-5]
        result["change_5d_pct"] = _safe_float((c0 - c5) / c5 * 100) if c5 else 0

    # Multi-timeframe trend: compare avg of last 4h vs preceding 4h using 1h bars
    result["mtf_trend"] = "neutral"
    try:
        if df_1h is not None and not df_1h.empty and len(df_1h) >= 8:
            closes = df_1h["close"].tail(8).values.astype(float)
            first_half  = closes[:4].mean()
            second_half = closes[4:].mean()
            if first_half > 0:
                chg = (second_half - first_half) / first_half
                if chg > 0.005:
                    result["mtf_trend"] = "up"
                elif chg < -0.005:
                    result["mtf_trend"] = "down"
    except Exception:
        pass

    return result


def score_technicals(indicators: Dict) -> Tuple[float, str]:
    if not indicators:
        return 0.0, "No data"

    score = 50.0
    factors = []

    rsi = indicators.get("rsi", 50)
    if 5 < rsi < 95:  # 0 or 100 = insufficient data, ignore
        if rsi < 30:
            score += 15; factors.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 45:
            score += 7;  factors.append(f"RSI bullish ({rsi:.0f})")
        elif rsi > 70:
            score -= 12; factors.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 55:
            score += 5

    cross = indicators.get("macd_cross", "")
    if cross == "bullish":
        score += 10; factors.append("MACD bullish")
    elif cross == "bearish":
        score -= 10; factors.append("MACD bearish")

    bb = indicators.get("bb_position", 0.5)
    if bb > 0:
        if bb < 0.2:
            score += 8;  factors.append(f"Near BB lower ({bb:.2f})")
        elif bb > 0.8:
            score -= 8;  factors.append(f"Near BB upper ({bb:.2f})")

    vr = indicators.get("volume_ratio", 1.0)
    if vr > 2.5:
        score += 10; factors.append(f"Vol spike {vr:.1f}x")
    elif vr > 1.5:
        score += 5;  factors.append(f"Elevated vol {vr:.1f}x")

    vwap = indicators.get("vwap_vs_price", 0)
    if vwap > 1:
        score += 5;  factors.append(f"+{vwap:.1f}% above VWAP")
    elif vwap < -2:
        score -= 5

    roc5 = indicators.get("roc5", 0)
    if roc5 > 3:
        score += 7;  factors.append(f"5d +{roc5:.1f}%")
    elif roc5 < -3:
        score -= 7

    stoch = indicators.get("stoch_k", 50)
    if stoch > 0:
        if stoch < 20:
            score += 5; factors.append(f"Stoch OS ({stoch:.0f})")
        elif stoch > 80:
            score -= 5

    score = max(0.0, min(100.0, score))
    return score, (" | ".join(factors) if factors else "Neutral")
