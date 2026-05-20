"""
agents/backtest.py -- Walk-forward signal backtest engine

Replays the rule-based signal engine against historical daily OHLCV data
to estimate strategy edge before committing real or paper capital.

Design:
  - Fetches up to 365d of daily bars via yfinance (free, no API key needed)
  - Slides a rolling indicator window forward bar by bar (no lookahead)
  - Runs the same 9 synchronous signal rules used live
  - Tracks virtual positions: open on signal, close on stop/target/time (5d)
  - Returns per-trade log + summary statistics

Usage (via /api/backtest endpoint):
  POST /api/backtest  {"symbol": "AAPL", "days": 180}
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from core.models import Market, SignalType, SignalReason, TickerProfile

log = get_logger("backtest")

# Minimum bars needed to compute all indicators (26-period EMA + 20 BB + warmup)
_MIN_WARMUP = 30
# ATR risk multipliers (mirrors portfolio.py signal engine)
_ATR_TARGET = 3.0
_ATR_STOP   = 1.5


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BtTrade:
    bar_in:      int
    date_in:     str
    signal:      str     # BUY / STRONG_BUY
    reason:      str
    entry:       float
    stop:        float
    target:      float
    conviction:  float
    # Filled on close
    bar_out:     int   = -1
    date_out:    str   = ""
    exit_price:  float = 0.0
    exit_reason: str   = ""     # stop_hit | target_hit | time_exit
    pnl_pct:     float = 0.0
    outcome:     str   = "open"  # win | loss | scratch | open


@dataclass
class BtResult:
    symbol:      str
    period_days: int
    bars_tested: int
    trades:      List[BtTrade]  = field(default_factory=list)
    # Summary stats (filled after simulation)
    total_trades:    int   = 0
    wins:            int   = 0
    losses:          int   = 0
    win_rate:        float = 0.0
    avg_gain_pct:    float = 0.0
    avg_loss_pct:    float = 0.0
    expectancy_pct:  float = 0.0
    max_drawdown_pct:float = 0.0
    total_return_pct:float = 0.0
    sharpe:          float = 0.0
    profit_factor:   float = 0.0
    elapsed_ms:      int   = 0
    error:           str   = ""


# ──────────────────────────────────────────────────────────────────────────────

def _compute_bar_indicators(close: np.ndarray, high: np.ndarray,
                             low: np.ndarray, vol: np.ndarray,
                             i: int) -> Dict:
    """Compute all indicators at bar i using only data up to and including bar i."""
    c = close[:i + 1]
    h = high[:i + 1]
    l = low[:i + 1]
    v = vol[:i + 1]
    n = len(c)
    ind: Dict = {}

    if n < 14:
        return ind

    # RSI-14
    try:
        delta = np.diff(c)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.convolve(gain, np.ones(14) / 14, mode="valid")[-1]
        avg_l = np.convolve(loss, np.ones(14) / 14, mode="valid")[-1]
        rs    = avg_g / avg_l if avg_l > 0 else 100.0
        ind["rsi"] = round(100 - 100 / (1 + rs), 2)
    except Exception:
        ind["rsi"] = 50.0

    # MACD (12/26/9)
    try:
        def _ema(arr, span):
            alpha = 2 / (span + 1)
            out = np.zeros(len(arr))
            out[0] = arr[0]
            for j in range(1, len(arr)):
                out[j] = alpha * arr[j] + (1 - alpha) * out[j - 1]
            return out

        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        macd  = ema12 - ema26
        sig9  = _ema(macd, 9)
        hist  = macd - sig9
        ind["macd_hist"]  = round(float(hist[-1]), 4)
        ind["macd_cross"] = (
            "bullish" if hist[-1] > 0 and (len(hist) < 2 or hist[-2] <= 0) else
            "bearish" if hist[-1] < 0 and (len(hist) < 2 or hist[-2] >= 0) else
            ("bullish" if hist[-1] > 0 else "bearish")
        )
    except Exception:
        ind["macd_hist"] = 0.0
        ind["macd_cross"] = "neutral"

    # Bollinger Bands (20)
    if n >= 20:
        try:
            sma  = np.mean(c[-20:])
            std  = np.std(c[-20:], ddof=1)
            upper = sma + 2 * std
            lower = sma - 2 * std
            rng   = upper - lower
            ind["bb_position"] = round((c[-1] - lower) / rng, 3) if rng > 0 else 0.5
            ind["bb_upper"] = round(float(upper), 4)
            ind["bb_lower"] = round(float(lower), 4)
            ind["bb_squeeze"] = bool(std / sma < 0.02) if sma > 0 else False
            ind["bb_width"] = round(rng / sma, 4) if sma > 0 else 0.0
        except Exception:
            ind["bb_position"] = 0.5

    # ATR-14
    try:
        tr_vals = []
        for j in range(max(1, n - 14), n):
            tr_vals.append(max(h[j] - l[j],
                               abs(h[j] - c[j - 1]),
                               abs(l[j] - c[j - 1])))
        ind["atr"] = round(float(np.mean(tr_vals)), 4) if tr_vals else c[-1] * 0.02
    except Exception:
        ind["atr"] = c[-1] * 0.02

    # Volume ratio (current vs 20-bar avg)
    try:
        avg_vol = np.mean(v[-20:]) if n >= 20 else np.mean(v)
        ind["volume_ratio"] = round(float(v[-1] / avg_vol), 2) if avg_vol > 0 else 1.0
    except Exception:
        ind["volume_ratio"] = 1.0

    # VWAP (5-bar)
    try:
        w = min(5, n)
        typical = (h[-w:] + l[-w:] + c[-w:]) / 3
        tv = v[-w:]
        vwap = float((typical * tv).sum() / tv.sum()) if tv.sum() > 0 else float(c[-1])
        vwap_diff = (float(c[-1]) - vwap) / vwap * 100 if vwap > 0 else 0.0
        ind["vwap"] = round(vwap, 4)
        ind["vwap_vs_price"] = round(vwap_diff, 2)
    except Exception:
        ind["vwap_vs_price"] = 0.0

    return ind


def _build_profile(sym: str, price: float, change_pct: float, ind: Dict) -> TickerProfile:
    """Build a minimal TickerProfile from bar data + computed indicators."""
    p = TickerProfile(
        symbol=sym,
        name=sym,
        market=Market.US_STOCK,
        price=price,
        change_pct=change_pct,
        rsi=ind.get("rsi", 50.0),
        macd_signal=ind.get("macd_cross", "neutral"),
        bb_position=ind.get("bb_position", 0.5),
        vwap_vs_price=ind.get("vwap_vs_price", 0.0),
        volume_ratio=ind.get("volume_ratio", 1.0),
        momentum_score=0.0,
        provider_data={"indicators": ind},
    )
    return p


def _run_rules(profile: TickerProfile) -> List[Tuple]:
    """
    Run the subset of rules that are purely indicator-based (no live data needed).
    Returns list of (SignalType, conviction, reason, summary) tuples.
    """
    results = []
    ind = profile.provider_data.get("indicators", {})

    # Volume breakout
    vr  = profile.volume_ratio
    chg = profile.change_pct
    if vr >= 3.0 and chg > 2.0:
        conv = min(0.90, 0.5 + (vr - 3) * 0.1 + chg * 0.02)
        results.append((
            SignalType.STRONG_BUY if chg > 5 else SignalType.BUY,
            conv, "VOLUME_SPIKE",
            f"Vol {vr:.1f}x + {chg:+.1f}%"
        ))
    elif vr >= 2.0 and chg < -3.0:
        results.append((SignalType.SELL, 0.60, "VOLUME_SPIKE",
                        f"Vol {vr:.1f}x + {chg:.1f}% dump"))

    # RSI reversal
    rsi = profile.rsi
    if 0 < rsi < 25 and chg > -1:
        results.append((SignalType.BUY, 0.70, "TECHNICAL",
                        f"RSI deeply oversold {rsi:.0f}"))
    elif rsi > 75 and chg < 0.5 and rsi <= 100:
        results.append((SignalType.SELL, 0.65, "TECHNICAL",
                        f"RSI overbought {rsi:.0f}"))

    # MACD cross
    macd_cross = ind.get("macd_cross", "")
    macd_hist  = ind.get("macd_hist", 0)
    if macd_cross == "bullish" and macd_hist > 0:
        results.append((SignalType.BUY, 0.65, "TECHNICAL", "MACD bullish cross"))
    elif macd_cross == "bearish" and macd_hist < 0:
        results.append((SignalType.SELL, 0.60, "TECHNICAL", "MACD bearish cross"))

    # Bollinger squeeze breakout
    bb_pos = ind.get("bb_position", 0.5)
    bb_sq  = ind.get("bb_squeeze", False)
    if bb_sq and chg > 1.5 and bb_pos > 0.7:
        results.append((SignalType.BUY, 0.72, "BREAKOUT", "BB squeeze breakout bullish"))
    elif bb_sq and chg < -1.5 and bb_pos < 0.3:
        results.append((SignalType.SELL, 0.70, "BREAKOUT", "BB squeeze breakout bearish"))

    # Mean reversion
    if rsi < 35 and chg > -0.5 and bb_pos < 0.25:
        results.append((SignalType.BUY, 0.65, "TECHNICAL",
                        f"Mean reversion: RSI {rsi:.0f} + BB {bb_pos:.0%}"))

    # VWAP reclaim
    vwap_diff = profile.vwap_vs_price
    if vwap_diff > 0.5 and chg > 0.8:
        results.append((SignalType.BUY, 0.62, "TECHNICAL", "VWAP reclaim"))
    elif vwap_diff < -0.5 and chg < -0.8:
        results.append((SignalType.SELL, 0.60, "TECHNICAL", "VWAP breakdown"))

    # Gap up
    if chg > 5.0:
        results.append((SignalType.BUY, min(0.80, 0.55 + chg * 0.02), "MOMENTUM",
                        f"Gap up {chg:.1f}%"))

    return results


def _evaluate_bar(sym: str, price: float, change_pct: float,
                  ind: Dict) -> Optional[Tuple]:
    """
    Evaluate one bar: run all rules, require >=2 in agreement.
    Returns (SignalType, conviction, reason, summary) or None.
    """
    profile = _build_profile(sym, price, change_pct, ind)
    raw = _run_rules(profile)

    buys  = [r for r in raw if r[0] in (SignalType.BUY, SignalType.STRONG_BUY)]
    sells = [r for r in raw if r[0] in (SignalType.SELL, SignalType.STRONG_SELL)]

    if len(buys) >= 2:
        conviction = round(min(0.92, sum(r[1] for r in buys) / len(buys) + 0.05), 3)
        reason     = " + ".join(set(r[2] for r in buys[:3]))
        summary    = "; ".join(r[3] for r in buys[:2])
        sig_type   = SignalType.STRONG_BUY if conviction > 0.75 else SignalType.BUY
        return (sig_type, conviction, reason, summary)

    if len(sells) >= 2:
        conviction = round(min(0.88, sum(r[1] for r in sells) / len(sells)), 3)
        reason     = " + ".join(set(r[2] for r in sells[:3]))
        summary    = "; ".join(r[3] for r in sells[:2])
        return (SignalType.SELL, conviction, reason, summary)

    return None


# ──────────────────────────────────────────────────────────────────────────────

async def run_backtest(symbol: str, days: int = 90,
                       account_size: float = 25_000.0,
                       risk_pct: float = 0.01) -> BtResult:
    """
    Full walk-forward backtest for one symbol.
    Returns a BtResult with per-trade log and summary statistics.
    """
    t0 = time.time()
    result = BtResult(symbol=symbol.upper(), period_days=days, bars_tested=0)

    # ── 1. Fetch historical OHLCV ─────────────────────────────────────
    try:
        from scanners.technicals import fetch_ohlcv_yf
        period_str = f"{min(days + 60, 365)}d"  # extra bars for indicator warmup
        df = await fetch_ohlcv_yf(symbol, period=period_str, interval="1d")
        if df is None or len(df) < _MIN_WARMUP + 5:
            result.error = f"Insufficient data for {symbol} (need {_MIN_WARMUP + 5}+ bars)"
            return result
    except Exception as e:
        result.error = f"Data fetch failed: {e}"
        return result

    # Trim to requested period (keeping warmup rows before the window)
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    dates = [str(d)[:10] for d in df.index]
    n_bars = len(close)

    # Determine the start bar (skip warmup)
    start_bar = max(_MIN_WARMUP, n_bars - days)

    # ── 2. Walk-forward simulation ────────────────────────────────────
    open_trade: Optional[BtTrade] = None
    trades: List[BtTrade] = []

    for i in range(start_bar, n_bars):
        price      = close[i]
        prev_close = close[i - 1] if i > 0 else price
        change_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        ind        = _compute_bar_indicators(close, high, low, vol, i)
        atr        = ind.get("atr", price * 0.02)

        # ── Check open position for exit ─────────────────────────────
        if open_trade is not None:
            bars_held = i - open_trade.bar_in
            is_long   = "BUY" in open_trade.signal

            exit_reason = None
            if is_long and price <= open_trade.stop:
                exit_reason = "stop_hit"
            elif is_long and price >= open_trade.target:
                exit_reason = "target_hit"
            elif bars_held >= 5:
                exit_reason = "time_exit"

            if exit_reason:
                open_trade.bar_out    = i
                open_trade.date_out   = dates[i]
                open_trade.exit_price = round(price, 4)
                open_trade.exit_reason = exit_reason
                open_trade.pnl_pct    = round(
                    (price - open_trade.entry) / open_trade.entry * 100
                    if is_long else
                    (open_trade.entry - price) / open_trade.entry * 100,
                    3
                )
                open_trade.outcome = (
                    "win" if open_trade.pnl_pct > 0.5
                    else "loss" if open_trade.pnl_pct < -0.5
                    else "scratch"
                )
                trades.append(open_trade)
                open_trade = None

        # ── Look for new signal (only if flat) ───────────────────────
        if open_trade is None:
            sig = _evaluate_bar(symbol, price, change_pct, ind)
            if sig is not None:
                sig_type, conviction, reason, summary = sig
                if "BUY" in sig_type.value:
                    target = round(price + atr * _ATR_TARGET, 4)
                    stop   = round(price - atr * _ATR_STOP,   4)
                else:
                    target = round(price - atr * _ATR_TARGET, 4)
                    stop   = round(price + atr * _ATR_STOP,   4)

                open_trade = BtTrade(
                    bar_in=i, date_in=dates[i],
                    signal=sig_type.value,
                    reason=reason, entry=round(price, 4),
                    stop=stop, target=target,
                    conviction=conviction,
                )

    # Close any still-open position at last bar
    if open_trade is not None:
        last_price = close[-1]
        open_trade.bar_out    = n_bars - 1
        open_trade.date_out   = dates[-1]
        open_trade.exit_price = round(last_price, 4)
        open_trade.exit_reason = "end_of_data"
        open_trade.pnl_pct    = round(
            (last_price - open_trade.entry) / open_trade.entry * 100, 3
        )
        open_trade.outcome = (
            "win" if open_trade.pnl_pct > 0.5
            else "loss" if open_trade.pnl_pct < -0.5
            else "scratch"
        )
        trades.append(open_trade)

    result.bars_tested = n_bars - start_bar
    result.trades      = trades

    # ── 3. Compute summary stats ──────────────────────────────────────
    closed  = [t for t in trades if t.outcome != "open"]
    wins    = [t for t in closed if t.outcome == "win"]
    losses  = [t for t in closed if t.outcome == "loss"]

    result.total_trades = len(closed)
    result.wins         = len(wins)
    result.losses       = len(losses)
    result.win_rate     = round(len(wins) / len(closed) * 100, 1) if closed else 0.0

    avg_gain = sum(t.pnl_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
    result.avg_gain_pct = round(avg_gain, 3)
    result.avg_loss_pct = round(avg_loss, 3)
    result.expectancy_pct = round(
        (result.win_rate / 100) * avg_gain + (1 - result.win_rate / 100) * avg_loss, 3
    )

    # Gross profit / loss for profit factor
    gross_win  = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    result.profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    # Equity curve for drawdown + Sharpe
    equity = 1.0
    equity_curve = [1.0]
    for t in closed:
        equity *= (1 + t.pnl_pct / 100)
        equity_curve.append(equity)

    result.total_return_pct = round((equity - 1.0) * 100, 2)

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown_pct = round(max_dd * 100, 2)

    # Sharpe (annualized, assuming 252 trading days, risk-free = 2%)
    if len(closed) >= 3:
        returns = [t.pnl_pct / 100 for t in closed]
        mean_r  = float(np.mean(returns))
        std_r   = float(np.std(returns, ddof=1))
        rf_per_trade = 0.02 / 252
        result.sharpe = round(
            (mean_r - rf_per_trade) / std_r * np.sqrt(252) if std_r > 0 else 0.0, 2
        )

    result.elapsed_ms = int((time.time() - t0) * 1000)
    log.info(
        f"Backtest [{symbol}] {days}d: {result.total_trades} trades "
        f"WR={result.win_rate}% E={result.expectancy_pct:+.2f}% "
        f"DD={result.max_drawdown_pct:.1f}% ({result.elapsed_ms}ms)"
    )
    return result
