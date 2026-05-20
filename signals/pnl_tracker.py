"""
signals/pnl_tracker.py -- Signal outcome tracking and P&L scoring

Records every signal emitted, checks outcomes after 1h/4h/1D, computes
win rate, avg gain/loss, and per-rule performance.

Persistence: SQLite via core.db (WAL mode).
On first run, existing storage/pnl.json is migrated automatically by
core.db.init_db() and renamed to pnl.json.bak.
"""
import asyncio
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.bus import bus
from core.logger import get_logger
from config.settings import SYS
import core.db as db

log = get_logger("pnl_tracker")
_ARCHIVE_DAYS = 90


@dataclass
class SignalRecord:
    signal_id: str
    symbol: str
    signal_type: str
    conviction: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    reason: str
    agent: str
    ts: float
    outcome: str = "open"
    exit_price: float = 0.0
    exit_ts: float = 0.0
    pnl_pct: float = 0.0
    checked_1h: bool = False
    checked_4h: bool = False
    checked_1d: bool = False


def _row_to_record(r: dict) -> SignalRecord:
    return SignalRecord(
        signal_id   = r["signal_id"],
        symbol      = r["symbol"],
        signal_type = r["signal_type"],
        conviction  = r.get("conviction", 0.0),
        entry_price = r.get("entry_price", 0.0),
        target_price= r.get("target_price"),
        stop_loss   = r.get("stop_loss"),
        reason      = r.get("reason", ""),
        agent       = r.get("agent", ""),
        ts          = r.get("ts", 0.0),
        outcome     = r.get("outcome", "open"),
        exit_price  = r.get("exit_price", 0.0),
        exit_ts     = r.get("exit_ts", 0.0),
        pnl_pct     = r.get("pnl_pct", 0.0),
        checked_1h  = bool(r.get("checked_1h", 0)),
        checked_4h  = bool(r.get("checked_4h", 0)),
        checked_1d  = bool(r.get("checked_1d", 0)),
    )


_INSERT_SQL = """
    INSERT OR IGNORE INTO signal_records
    (signal_id,symbol,signal_type,conviction,entry_price,target_price,
     stop_loss,reason,agent,ts,outcome,exit_price,exit_ts,pnl_pct,
     checked_1h,checked_4h,checked_1d)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_UPDATE_OUTCOME_SQL = """
    UPDATE signal_records
    SET outcome=?, exit_price=?, exit_ts=?, pnl_pct=?,
        checked_1h=?, checked_4h=?, checked_1d=?
    WHERE signal_id=?
"""


def _rec_params(r: SignalRecord) -> tuple:
    return (
        r.signal_id, r.symbol, r.signal_type, r.conviction, r.entry_price,
        r.target_price, r.stop_loss, r.reason, r.agent, r.ts,
        r.outcome, r.exit_price, r.exit_ts, r.pnl_pct,
        int(r.checked_1h), int(r.checked_4h), int(r.checked_1d),
    )


_SPY_CACHE: Dict[str, object] = {"df": None, "ts": 0.0}
_SPY_CACHE_TTL = 3_600  # refresh SPY benchmark at most once per hour


class PnLTracker:
    def __init__(self):
        self._records: Dict[str, SignalRecord] = {}
        self._try_sync_load()

    # ── Best-effort sync load at import time ──────────────────────────────

    def _try_sync_load(self):
        """Synchronous load from DB if it already exists (e.g. on restart)."""
        try:
            import sqlite3
            if not db.DB_PATH.exists():
                return
            conn = sqlite3.connect(str(db.DB_PATH), timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - _ARCHIVE_DAYS * 86_400
            rows = conn.execute(
                "SELECT * FROM signal_records WHERE ts > ? ORDER BY ts", (cutoff,)
            ).fetchall()
            conn.close()
            for r in rows:
                rec = _row_to_record(dict(r))
                self._records[rec.signal_id] = rec
            if self._records:
                log.info(f"P&L tracker loaded {len(self._records)} records from DB")
        except Exception as e:
            log.debug(f"P&L sync load: {e}")

    # ── Async load (called from start() after init_db() has run) ─────────

    async def _load_from_db(self):
        cutoff = time.time() - _ARCHIVE_DAYS * 86_400
        rows = await db.fetchall(
            "SELECT * FROM signal_records WHERE ts > ? ORDER BY ts", (cutoff,)
        )
        self._records.clear()
        for r in rows:
            rec = _row_to_record(r)
            self._records[rec.signal_id] = rec
        log.info(f"P&L tracker loaded {len(self._records)} records from DB")

    # ── Rotation ──────────────────────────────────────────────────────────

    def _rotate(self):
        """Evict closed records older than _ARCHIVE_DAYS from the in-memory cache."""
        cutoff = time.time() - _ARCHIVE_DAYS * 86_400
        stale = [k for k, v in self._records.items()
                 if v.outcome != "open" and v.ts < cutoff]
        for k in stale:
            del self._records[k]

    async def _db_rotate(self):
        """Delete records beyond retention window from SQLite."""
        cutoff = time.time() - _ARCHIVE_DAYS * 86_400
        await db.execute(
            "DELETE FROM signal_records WHERE outcome != 'open' AND ts < ?", (cutoff,)
        )
        if self._records:
            log.info(f"P&L rotation: DB cleanup, {len(self._records)} active records")

    # ── Public record API ─────────────────────────────────────────────────

    def record(self, signal: dict) -> Optional[str]:
        """
        Add signal to in-memory dict. Returns signal_id if new, None if duplicate.
        Caller should await _db_insert() for persistence.
        """
        sig_id = f"{signal['symbol']}_{int(signal.get('ts', time.time()))}"
        if sig_id in self._records:
            return None
        rec = SignalRecord(
            signal_id   = sig_id,
            symbol      = signal.get("symbol", ""),
            signal_type = signal.get("signal_type", ""),
            conviction  = signal.get("conviction", 0),
            entry_price = signal.get("price_at_signal", 0),
            target_price= signal.get("target_price"),
            stop_loss   = signal.get("stop_loss"),
            reason      = signal.get("reason", ""),
            agent       = signal.get("agent", ""),
            ts          = signal.get("ts", time.time()),
        )
        self._records[sig_id] = rec
        return sig_id

    async def _db_insert(self, rec: SignalRecord):
        await db.execute(_INSERT_SQL, _rec_params(rec))

    async def update_outcomes(self, current_prices: Dict[str, float]):
        """Check open signals and mark outcomes based on current prices."""
        now = time.time()
        for sig_id, rec in list(self._records.items()):
            if rec.outcome != "open":
                continue
            cur = current_prices.get(rec.symbol)
            if not cur or rec.entry_price <= 0:
                continue

            age_h  = (now - rec.ts) / 3600
            is_buy = "BUY" in rec.signal_type
            pnl    = (cur - rec.entry_price) / rec.entry_price * 100
            if not is_buy:
                pnl = -pnl

            if rec.stop_loss and rec.target_price:
                if is_buy:
                    if cur <= rec.stop_loss:
                        rec.outcome = "loss"
                    elif cur >= rec.target_price:
                        rec.outcome = "win"
                else:
                    if cur >= rec.stop_loss:
                        rec.outcome = "loss"
                    elif cur <= rec.target_price:
                        rec.outcome = "win"

            if age_h >= 24 and rec.outcome == "open":
                rec.outcome = "win" if pnl > 0.5 else "loss" if pnl < -0.5 else "scratch"

            if rec.outcome != "open":
                rec.exit_price = cur
                rec.exit_ts    = now
                rec.pnl_pct    = round(pnl, 2)
                log.info(
                    f"P&L: {rec.symbol} {rec.signal_type} → {rec.outcome} ({pnl:+.2f}%)"
                )
                await db.execute(_UPDATE_OUTCOME_SQL, (
                    rec.outcome, rec.exit_price, rec.exit_ts, rec.pnl_pct,
                    int(rec.checked_1h), int(rec.checked_4h), int(rec.checked_1d),
                    sig_id,
                ))
                try:
                    from agents.self_improvement import self_improvement as _si
                    _si.layer_store.record_outcome(
                        rec.symbol, rec.signal_type, rec.outcome == "win"
                    )
                except Exception as e:
                    log.debug(f"Self-improvement outcome [{rec.symbol}]: {e}")
                try:
                    from agents.memory import agent_memory
                    mem_outcome = "correct" if rec.outcome == "win" else "wrong"
                    agent_memory.record_outcome(
                        rec.symbol, rec.signal_type, mem_outcome, rec.pnl_pct
                    )
                    agent_memory.remember(
                        rec.symbol, "signal_outcome",
                        f"Signal {rec.signal_type} on {rec.symbol} via {rec.agent} "
                        f"{'won' if mem_outcome=='correct' else 'lost'} "
                        f"{rec.pnl_pct:+.2f}% (reason: {rec.reason})",
                        tags=[mem_outcome, rec.signal_type, rec.agent],
                        confidence=min(0.95, abs(rec.pnl_pct) / 10),
                        related_signal=rec.signal_type,
                    )
                except Exception as me:
                    log.debug(f"Memory update [{rec.symbol}]: {me}")

    # ── Stats (read from _records cache) ─────────────────────────────────

    def get_stats(self) -> dict:
        closed = [r for r in self._records.values() if r.outcome != "open"]
        open_  = [r for r in self._records.values() if r.outcome == "open"]
        wins   = [r for r in closed if r.outcome == "win"]
        losses = [r for r in closed if r.outcome == "loss"]
        total  = len(closed)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        avg_gain = sum(r.pnl_pct for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0
        by_reason: Dict[str, dict] = {}
        for r in closed:
            key = r.reason
            if key not in by_reason:
                by_reason[key] = {"wins": 0, "losses": 0, "pnl": 0}
            by_reason[key]["wins" if r.outcome == "win" else "losses"] += 1
            by_reason[key]["pnl"] += r.pnl_pct
        by_agent: Dict[str, dict] = {}
        for r in closed:
            key = r.agent
            if key not in by_agent:
                by_agent[key] = {"wins": 0, "losses": 0}
            by_agent[key]["wins" if r.outcome == "win" else "losses"] += 1
        return {
            "total_signals": len(self._records),
            "open":           len(open_),
            "closed":         total,
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(win_rate, 1),
            "avg_gain_pct":   round(avg_gain, 2),
            "avg_loss_pct":   round(avg_loss, 2),
            "expectancy":     round((win_rate/100 * avg_gain) + ((1 - win_rate/100) * avg_loss), 2),
            "by_reason":      by_reason,
            "by_agent":       by_agent,
            "recent": [
                {"symbol": r.symbol, "type": r.signal_type,
                 "outcome": r.outcome, "pnl": r.pnl_pct,
                 "entry": r.entry_price, "exit": r.exit_price,
                 "agent": r.agent, "ts": r.ts}
                for r in sorted(closed, key=lambda x: x.exit_ts, reverse=True)[:30]
            ],
            "open_signals": [
                {"symbol": r.symbol, "type": r.signal_type,
                 "entry": r.entry_price, "conviction": r.conviction,
                 "target": r.target_price, "stop": r.stop_loss,
                 "age_h": round((time.time() - r.ts) / 3600, 1)}
                for r in open_[:20]
            ],
        }

    def get_advanced_stats(self) -> dict:
        closed = [r for r in self._records.values() if r.outcome != "open"]
        if not closed:
            return {"error": "No closed signals yet"}
        closed_sorted = sorted(closed, key=lambda x: x.exit_ts)
        returns = [r.pnl_pct / 100 for r in closed_sorted]
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity_pct = [round((e - 1) * 100, 2) for e in equity]
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        if len(returns) > 1:
            mean_r = statistics.mean(returns)
            std_r  = statistics.stdev(returns) if len(returns) > 1 else 0.01
            daily_rf = 0.02 / 252
            sharpe = ((mean_r - daily_rf) / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            sharpe = 0
        max_win_streak = max_loss_streak = cur_streak = 0
        cur_type = None
        for r in closed_sorted:
            if r.outcome == "win":
                cur_streak = cur_streak + 1 if cur_type == "win" else 1
                cur_type = "win"
                max_win_streak = max(max_win_streak, cur_streak)
            elif r.outcome == "loss":
                cur_streak = cur_streak + 1 if cur_type == "loss" else 1
                cur_type = "loss"
                max_loss_streak = max(max_loss_streak, cur_streak)
        by_agent: dict = {}
        for r in closed:
            a = r.agent
            if a not in by_agent:
                by_agent[a] = {"wins": 0, "losses": 0, "total_pnl": 0}
            if r.outcome == "win":
                by_agent[a]["wins"] += 1
            elif r.outcome == "loss":
                by_agent[a]["losses"] += 1
            by_agent[a]["total_pnl"] = round(by_agent[a]["total_pnl"] + r.pnl_pct, 2)
        for a in by_agent:
            total = by_agent[a]["wins"] + by_agent[a]["losses"]
            by_agent[a]["win_rate"] = round(
                by_agent[a]["wins"] / total * 100, 1) if total > 0 else 0
        monthly: dict = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for r in closed_sorted:
            month = time.strftime("%Y-%m", time.localtime(r.exit_ts)) if r.exit_ts else "unknown"
            monthly[month]["pnl"] = round(monthly[month]["pnl"] + r.pnl_pct, 2)
            monthly[month]["trades"] += 1
            if r.outcome == "win":
                monthly[month]["wins"] += 1
        wins   = [r for r in closed if r.outcome == "win"]
        losses = [r for r in closed if r.outcome == "loss"]
        total  = len(closed)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        avg_gain = sum(r.pnl_pct for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0
        loss_sum = sum(r.pnl_pct for r in losses)
        profit_factor = (sum(r.pnl_pct for r in wins) / abs(loss_sum)) if losses and loss_sum != 0 else 999
        return {
            "total_signals":    len(self._records),
            "closed":           total,
            "open":             len([r for r in self._records.values() if r.outcome == "open"]),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(win_rate, 1),
            "avg_gain_pct":     round(avg_gain, 2),
            "avg_loss_pct":     round(avg_loss, 2),
            "expectancy":       round((win_rate/100*avg_gain)+((1-win_rate/100)*avg_loss), 2),
            "sharpe_ratio":     round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "profit_factor":    round(min(profit_factor, 999), 2),
            "total_return_pct": round((equity[-1] - 1) * 100, 2),
            "max_win_streak":   max_win_streak,
            "max_loss_streak":  max_loss_streak,
            "equity_curve":     equity_pct[-50:],
            "by_agent":         by_agent,
            "monthly":          dict(sorted(monthly.items())),
            "recent": [
                {"symbol": r.symbol, "type": r.signal_type,
                 "outcome": r.outcome, "pnl": r.pnl_pct,
                 "entry": r.entry_price, "exit": r.exit_price,
                 "agent": r.agent, "ts": r.ts, "exit_ts": r.exit_ts}
                for r in reversed(closed_sorted[-30:])
            ],
            "open_signals": [
                {"symbol": r.symbol, "type": r.signal_type,
                 "entry": r.entry_price, "conviction": r.conviction,
                 "target": r.target_price, "stop": r.stop_loss,
                 "age_h": round((time.time() - r.ts) / 3600, 1)}
                for r in list(self._records.values()) if r.outcome == "open"
            ][:20],
        }

    def get_analytics(self) -> dict:
        """
        Full Pillar C analytics payload:
          equity_curve, attribution, daily_pnl, rolling, dd_periods
        """
        closed = sorted(
            [r for r in self._records.values() if r.outcome not in ("open",)],
            key=lambda x: x.exit_ts or x.ts,
        )
        now = time.time()

        # Equity curve
        equity = 1.0
        equity_curve = []
        for r in closed:
            equity *= (1 + r.pnl_pct / 100)
            equity_curve.append({
                "ts":      r.exit_ts or r.ts,
                "date":    time.strftime("%Y-%m-%d", time.localtime(r.exit_ts or r.ts)),
                "value":   round((equity - 1) * 100, 2),
                "pnl":     r.pnl_pct,
                "outcome": r.outcome,
                "symbol":  r.symbol,
            })

        # Drawdown periods
        peak = 1.0
        dd_start = None
        dd_start_date = ""
        min_in_dd = 1.0
        cur_equity_val = 1.0
        dd_periods = []
        for entry in equity_curve:
            cur_equity_val = 1 + entry["value"] / 100
            if cur_equity_val >= peak:
                if dd_start is not None and peak > 0:
                    depth = round((peak - min_in_dd) / peak * 100, 2)
                    if depth >= 1.0:
                        dd_periods.append({
                            "start": dd_start_date,
                            "end":   entry["date"],
                            "depth_pct": depth,
                        })
                    dd_start = None
                peak = cur_equity_val
            else:
                if dd_start is None:
                    dd_start = True
                    dd_start_date = entry["date"]
                    min_in_dd = cur_equity_val
                else:
                    min_in_dd = min(min_in_dd, cur_equity_val)

        # Attribution heatmap
        windows = {"7d": 7 * 86_400, "30d": 30 * 86_400, "90d": 90 * 86_400}
        attribution: dict = {}
        for wname, wsecs in windows.items():
            cutoff = now - wsecs
            for r in closed:
                if (r.exit_ts or r.ts) < cutoff:
                    continue
                key = (r.reason or "unknown").replace("SignalReason.", "")
                if key not in attribution:
                    attribution[key] = {}
                if wname not in attribution[key]:
                    attribution[key][wname] = {"wins": 0, "losses": 0, "trades": 0, "pnl": 0.0}
                d = attribution[key][wname]
                d["trades"] += 1
                d["pnl"] = round(d["pnl"] + r.pnl_pct, 2)
                if r.outcome == "win":
                    d["wins"] += 1
                elif r.outcome in ("loss", "scratch"):
                    d["losses"] += 1
        for key in attribution:
            for wname in attribution[key]:
                d = attribution[key][wname]
                t = d["wins"] + d["losses"]
                d["win_rate"] = round(d["wins"] / t * 100, 1) if t > 0 else 0

        # Daily P&L
        daily_pnl: dict = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
        for r in closed:
            ts = r.exit_ts or r.ts
            if not ts:
                continue
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            daily_pnl[day]["pnl"]    = round(daily_pnl[day]["pnl"] + r.pnl_pct, 2)
            daily_pnl[day]["trades"] += 1
            if r.outcome == "win":
                daily_pnl[day]["wins"] += 1
        daily = dict(daily_pnl)

        # Rolling stats
        def _rolling(records):
            if not records:
                return {"trades": 0, "win_rate": 0, "sharpe": 0, "expectancy": 0, "profit_factor": 0}
            wins   = [r for r in records if r.outcome == "win"]
            losses = [r for r in records if r.outcome in ("loss", "scratch")]
            total  = len(records)
            wr     = len(wins) / total * 100 if total else 0
            ag     = sum(r.pnl_pct for r in wins)   / len(wins)   if wins   else 0
            al     = sum(r.pnl_pct for r in losses) / len(losses) if losses else 0
            gw     = sum(r.pnl_pct for r in wins)
            gl     = abs(sum(r.pnl_pct for r in losses))
            pf     = gw / gl if gl > 0 else 999.0
            rets   = [r.pnl_pct / 100 for r in records]
            sharpe = sortino = 0.0
            if len(rets) > 1:
                mr  = statistics.mean(rets)
                sr  = statistics.stdev(rets) or 0.001
                sharpe = round((mr - 0.02 / 252) / sr * math.sqrt(252), 2)
                neg = [r for r in rets if r < 0]
                dsr = statistics.stdev(neg) if len(neg) > 1 else sr
                sortino = round((mr - 0.02 / 252) / dsr * math.sqrt(252), 2)
            avg_hold_h = 0.0
            held = [r for r in records if r.exit_ts and r.ts]
            if held:
                avg_hold_h = round(sum((r.exit_ts - r.ts) / 3600 for r in held) / len(held), 1)
            return {
                "trades":        total,
                "win_rate":      round(wr, 1),
                "sharpe":        round(sharpe, 2),
                "sortino":       round(sortino, 2),
                "avg_hold_h":    avg_hold_h,
                "expectancy":    round((wr/100 * ag) + ((1 - wr/100) * al), 2),
                "profit_factor": round(min(pf, 99.0), 2),
            }

        rolling = {}
        for wname, wsecs in windows.items():
            cutoff = now - wsecs
            rolling[wname] = _rolling([r for r in closed if (r.exit_ts or r.ts) > cutoff])

        # Streaks
        max_win_streak = max_loss_streak = cur_streak = 0
        cur_type = None
        for r in closed:
            if r.outcome == "win":
                cur_streak = cur_streak + 1 if cur_type == "win" else 1
                cur_type = "win"
                max_win_streak = max(max_win_streak, cur_streak)
            elif r.outcome in ("loss", "scratch"):
                cur_streak = cur_streak + 1 if cur_type == "loss" else 1
                cur_type = "loss"
                max_loss_streak = max(max_loss_streak, cur_streak)

        # SPY benchmark curve (normalized to same start date as our equity curve)
        # Cached for 1 hour so repeated analytics calls don't hammer yfinance.
        spy_curve = []
        if equity_curve:
            try:
                import yfinance as yf
                import warnings as _w
                start_date = equity_curve[0]["date"]
                now_t = time.time()
                spy_df = _SPY_CACHE["df"]
                if spy_df is None or (now_t - _SPY_CACHE["ts"]) > _SPY_CACHE_TTL:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        spy_df = yf.Ticker("SPY").history(period="2y", interval="1d")
                    if spy_df is not None and not spy_df.empty:
                        spy_df.columns = [c.lower() for c in spy_df.columns]
                    _SPY_CACHE["df"] = spy_df
                    _SPY_CACHE["ts"] = now_t
                if spy_df is not None and not spy_df.empty:
                    filtered = spy_df[spy_df.index.strftime("%Y-%m-%d") >= start_date]
                    if len(filtered) >= 2:
                        spy_base = float(filtered["close"].iloc[0])
                        for idx, row in filtered.iterrows():
                            spy_curve.append({
                                "date":  idx.strftime("%Y-%m-%d"),
                                "value": round((float(row["close"]) - spy_base) / spy_base * 100, 2),
                            })
            except Exception:
                pass

        return {
            "equity_curve":    equity_curve[-500:],
            "spy_curve":       spy_curve[-500:],
            "dd_periods":      dd_periods[-20:],
            "attribution":     attribution,
            "daily_pnl":       daily,
            "rolling":         rolling,
            "max_win_streak":  max_win_streak,
            "max_loss_streak": max_loss_streak,
        }

    # ── Async event loop ──────────────────────────────────────────────────

    async def start(self):
        await self._load_from_db()
        q = await bus.subscribe("signal")
        log.info("P&L tracker started (SQLite backend)")
        check_interval  = 900
        rotate_interval = 86_400
        last_check  = 0
        last_rotate = time.time()

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)
                if event.topic == "signal":
                    sig_id = self.record(event.data)
                    if sig_id:
                        asyncio.create_task(
                            self._db_insert(self._records[sig_id])
                        )
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.debug(f"P&L tracker error: {e}")

            now = time.time()
            if now - last_check > check_interval:
                try:
                    from scanners.market_scanner import scanner
                    prices = {sym: item.profile.price for sym, item in scanner.watchlist.items()}
                    if prices:
                        await self.update_outcomes(prices)
                except Exception as e:
                    log.debug(f"P&L outcome check error: {e}")
                last_check = now

            if now - last_rotate > rotate_interval:
                self._rotate()
                await self._db_rotate()
                last_rotate = now


pnl_tracker = PnLTracker()
