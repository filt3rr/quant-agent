"""
main.py -- QuantAgent v9 Orchestrator

Analysis pipeline (v9):
  Tier 1 (Quick Screen): configurable parallel workers, 1 LLM call each
  Tier 2 (Deep Dive):    sequential, 3 LLM calls, only for Tier-1 high-conv results
  Both tiers feed the same signal engine, portfolio, and self-improvement loop.

Config is live-adjustable from the dashboard Controls tab.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import KEYS, SYS, MARKET, LLM
from core.bus import emit
from core.db import init_db
from core.logger import get_logger
from core.market_hours import is_market_open, market_status
from core.startup_validator import startup_validator
from core.staleness_guard import staleness_guard
from providers.registry import registry
from scanners.market_scanner import scanner
from signals.signal_engine import signal_engine
from signals.pnl_tracker import pnl_tracker
from signals.alerts import alerts_manager
from signals.portfolio import paper_portfolio
from signals.execution_engine import execution_engine
from signals.macro_monitor import macro_monitor
from agents.digest import digest_agent
from signals.webhooks import webhook_manager
from agents.memory import agent_memory
from agents.watchlist_manager import wl_intelligence
from agents.self_improvement import self_improvement
from agents.analysis_config import analysis_config
from agents.analysis_queue import analysis_queue
from dashboard.server import app, state_updater

import uvicorn

log = get_logger("main")
MOCK_MODE = os.environ.get("QUANT_MOCK_MODE", "0") == "1"
_shutdown = asyncio.Event()


# ── Scanner ────────────────────────────────────────────────────────────────

async def scanner_loop():
    log.info("Scanner loop starting...")
    await scanner.start()


# ── Signal engine ──────────────────────────────────────────────────────────

async def signal_loop():
    log.info("Signal engine loop starting...")
    await asyncio.sleep(30)
    while not _shutdown.is_set():
        try:
            status = market_status()
            if not status["us_stocks"] and not status["crypto"]:
                await asyncio.sleep(60)
                continue
            if scanner.watchlist:
                wl = scanner.get_watchlist()
                if not status["us_stocks"]:
                    from core.models import Market as Mkt
                    wl = [i for i in wl if i.profile.market == Mkt.CRYPTO]
                await signal_engine.run_on_watchlist(wl[:50])
        except Exception as e:
            log.error(f"Signal loop error: {e}")
        await asyncio.sleep(SYS.SCAN_INTERVAL)


# ── Tier-1 queue feeder (picks candidates and enqueues them) ───────────────

async def tier1_feeder_loop():
    """Continuously feeds high-quality tickers into the Tier-1 analysis queue."""
    log.info("Tier-1 feeder loop starting...")
    recently_analyzed: dict = {}   # symbol → last ts
    await asyncio.sleep(65)        # let scanner populate first

    while not _shutdown.is_set():
        try:
            cfg = analysis_config.get()
            if not cfg.tier1_enabled and not cfg.tier2_enabled:
                await asyncio.sleep(30)
                continue

            if not scanner.watchlist:
                await asyncio.sleep(15)
                continue

            status = market_status()
            wl = scanner.get_watchlist()

            # Market-hours gate
            if not status["us_stocks"]:
                from core.models import Market as Mkt
                wl = [i for i in wl if i.profile.market == Mkt.CRYPTO]
                if not wl:
                    await asyncio.sleep(300)
                    continue

            now = time.time()
            cooldown = cfg.analysis_cooldown

            # Build candidate list
            priority = wl_intelligence.get_analysis_priority()
            if priority:
                priority_set = set(priority[:30])
                candidates = [i for i in wl
                              if i.profile.symbol in priority_set
                              and i.profile.composite_score >= cfg.min_composite_score]
            else:
                candidates = [i for i in wl[:30]
                              if i.profile.composite_score >= cfg.min_composite_score]

            # Filter cooldown, already-queued, and cold sectors
            _cold_sectors: set = set()
            try:
                _cold_sectors = set(self_improvement.get_cold_sectors())
            except Exception:
                pass
            candidates = [
                c for c in candidates
                if (now - recently_analyzed.get(c.profile.symbol, 0)) > cooldown
                and not analysis_queue.is_queued_or_active(c.profile.symbol)
                and (c.profile.sector or "") not in _cold_sectors
            ]

            # If no fresh candidates, re-analyze open portfolio positions
            if not candidates and paper_portfolio.open_positions:
                portfolio_syms = {p.symbol for p in paper_portfolio.open_positions}
                candidates = [i for i in wl
                              if i.profile.symbol in portfolio_syms
                              and not analysis_queue.is_queued_or_active(i.profile.symbol)][:3]

            if not candidates:
                await asyncio.sleep(30)
                continue

            # Opening-bell boost: first 60 minutes of the US session
            mkt = market_status()
            minutes_open = mkt.get("minutes_since_open", 999)
            is_opening_window = status["us_stocks"] and 0 <= minutes_open < 60
            if is_opening_window:
                has_signal = {i.profile.symbol for i in wl if i.latest_signal is not None}
                candidates = sorted(
                    candidates,
                    key=lambda c: (c.profile.symbol in has_signal, c.profile.composite_score),
                    reverse=True,
                )
            batch_size = min(cfg.tickers_per_cycle * 2, 15) if is_opening_window else cfg.tickers_per_cycle

            # Enqueue up to tickers_per_cycle (or doubled during opening window)
            batch = candidates[:batch_size]

            # ── T0 pre-filter: stamp intraday alignment flags on each profile ──
            try:
                from scanners.t0_filter import run_t0_filter
                t0_tasks = []
                for item in batch:
                    bias = "bearish" if (item.latest_signal and
                                         "SELL" in getattr(item.latest_signal, "signal_type", {}).value
                                         ) else "bullish"
                    t0_tasks.append(run_t0_filter(item.profile.symbol, item.profile.market, bias))
                t0_results = await asyncio.gather(*t0_tasks, return_exceptions=True)
                for item, t0 in zip(batch, t0_results):
                    if not isinstance(t0, Exception) and not t0.error:
                        item.profile.h4_aligned        = t0.h4_aligned
                        item.profile.h1_vwap_confirmed = t0.h1_vwap_confirmed
                        item.profile.volume_expanding  = t0.volume_expanding
                        item.profile.t0_score          = t0.score
            except Exception as _t0e:
                log.debug(f"T0 filter batch error: {_t0e}")

            enqueued = 0
            for item in batch:
                sym = item.profile.symbol
                if await analysis_queue.enqueue_tier1(sym):
                    recently_analyzed[sym] = now
                    enqueued += 1

            if enqueued:
                log.info(f"Feeder: queued {enqueued} tickers for Tier-1")
                await emit("agent.activity", {
                    "agent_id": "feeder", "agent_type": "master",
                    "symbol": batch[0].profile.symbol if batch else "–",
                    "action": "planning",
                    "message": (
                        f"Queued {enqueued} tickers for analysis: "
                        f"{', '.join(i.profile.symbol for i in batch[:enqueued])} | "
                        f"Market: {'open' if status['us_stocks'] else 'closed'} | "
                        f"T1-workers: {cfg.tier1_workers} | "
                        f"T2-threshold: {cfg.tier2_threshold:.0%}"
                    ),
                }, "main")

        except Exception as e:
            log.error(f"Feeder loop error: {e}")

        await asyncio.sleep(60)


# ── Tier-1 workers (parallel) ──────────────────────────────────────────────

async def tier1_worker(worker_id: int):
    """Pull from Tier-1 queue and run quick screen."""
    from agents.deep_analysis import orchestrator
    log.info(f"Tier-1 worker {worker_id} ready")

    while not _shutdown.is_set():
        entry = None
        try:
            entry = await asyncio.wait_for(
                analysis_queue.get_tier1(), timeout=10
            )
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            log.debug(f"T1 worker {worker_id} queue error: {e}")
            await asyncio.sleep(2)
            continue

        sym = entry.symbol
        cfg = analysis_config.get()
        log.info(f"[T1-W{worker_id}] Analyzing {sym}")

        try:
            # Get watchlist item
            item = scanner.watchlist.get(sym)
            if not item:
                analysis_queue.fail(entry, "Not in watchlist")
                analysis_queue.done_tier1(entry)
                continue

            result = await asyncio.wait_for(
                orchestrator.analyze_tier1(item),
                timeout=cfg.tier1_timeout + 30
            )

            signal = result.get("signal", "HOLD")
            conviction = result.get("conviction", 0.0)
            analysis_queue.complete(entry, signal=signal, conviction=conviction)

            # Emit AI signal to dashboard/dedup/PnL tracker
            try:
                await signal_engine.emit_ai_signal(item, signal, conviction, tier=1)
            except Exception:
                pass

            # Route through execution engine for order coordination
            try:
                if signal not in ("HOLD", "WATCH") and conviction >= cfg.tier2_threshold:
                    await execution_engine.register_t1_signal(
                        sym, signal, conviction, item.profile.price
                    )
            except Exception:
                pass

            # Auto-promote to Tier 2 if conviction high enough
            if (cfg.tier2_enabled
                    and conviction >= cfg.tier2_threshold
                    and signal not in ("HOLD",)):
                log.info(f"[T1-W{worker_id}] {sym} promoted to Tier-2 "
                         f"(conv={conviction:.0%} >= {cfg.tier2_threshold:.0%})")
                await analysis_queue.enqueue_tier2(sym, source="auto-promote")

            # Watchlist intelligence: dismiss weak signals
            try:
                if signal in ("SELL", "STRONG_SELL") and conviction > 0.55:
                    wl_intelligence.dismiss(sym, f"T1: {signal} ({conviction:.0%})", 90)
                elif signal in ("BUY", "STRONG_BUY") and conviction >= 0.65:
                    log.info(f"[T1] {sym} promising: {signal} ({conviction:.0%})")
            except Exception:
                pass

        except asyncio.TimeoutError:
            log.error(f"[T1-W{worker_id}] {sym} timed out")
            if entry:
                analysis_queue.fail(entry, f"Timeout after {cfg.tier1_timeout}s")
        except Exception as e:
            log.error(f"[T1-W{worker_id}] {sym} error: {type(e).__name__}: {e}")
            if entry:
                analysis_queue.fail(entry, f"{type(e).__name__}: {str(e)[:80]}")
        finally:
            try:
                analysis_queue.done_tier1(entry)
            except Exception:
                pass


async def tier1_worker_pool():
    """Spawn and manage the Tier-1 worker pool, respecting config changes."""
    log.info("Tier-1 worker pool starting...")
    workers = []
    last_worker_count = 0

    while not _shutdown.is_set():
        cfg = analysis_config.get()
        n = cfg.tier1_workers if cfg.tier1_enabled else 0

        if n != last_worker_count:
            # Cancel existing workers
            for t in workers:
                t.cancel()
            workers.clear()
            # Spawn new workers
            for i in range(n):
                t = asyncio.create_task(
                    tier1_worker(i + 1), name=f"t1_worker_{i+1}"
                )
                workers.append(t)
            log.info(f"Tier-1 worker pool: {n} workers active")
            last_worker_count = n

        await asyncio.sleep(15)

    for t in workers:
        t.cancel()


# ── Tier-2 workers (pool, mirroring tier1_worker_pool) ────────────────────

async def tier2_worker(worker_id: int):
    """One Tier-2 worker in the pool — pulls from T2 queue and runs deep dive."""
    from agents.deep_analysis import orchestrator
    log.info(f"Tier-2 worker {worker_id} ready")

    while not _shutdown.is_set():
        entry = None
        try:
            entry = await asyncio.wait_for(
                analysis_queue.get_tier2(), timeout=10
            )
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            log.debug(f"T2 worker {worker_id} queue error: {e}")
            await asyncio.sleep(2)
            continue

        sym = entry.symbol
        cfg = analysis_config.get()

        if not cfg.tier2_enabled:
            analysis_queue.fail(entry, "Tier-2 disabled")
            analysis_queue.done_tier2(entry)
            continue

        log.info(f"[T2-W{worker_id}] Deep dive: {sym}")
        try:
            item = scanner.watchlist.get(sym)
            if not item:
                analysis_queue.fail(entry, "Not in watchlist")
                analysis_queue.done_tier2(entry)
                continue

            result = await asyncio.wait_for(
                orchestrator.analyze_tier2(item),
                timeout=cfg.tier2_timeout + 60
            )

            signal = result.get("signal", "HOLD")
            conviction = result.get("conviction", 0.0)
            analysis_queue.complete(entry, signal=signal, conviction=conviction)

            try:
                await signal_engine.emit_ai_signal(item, signal, conviction, tier=2)
            except Exception:
                pass

            try:
                await execution_engine.confirm_t2_signal(
                    sym, signal, conviction, item.profile.price
                )
            except Exception:
                pass

            try:
                if signal in ("SELL", "STRONG_SELL") and conviction > 0.55:
                    wl_intelligence.dismiss(sym, f"T2: {signal} ({conviction:.0%})", 120)
                elif signal in ("BUY", "STRONG_BUY") and conviction >= 0.65:
                    log.info(f"[T2-W{worker_id}] {sym} confirmed: {signal} ({conviction:.0%})")
            except Exception:
                pass

        except asyncio.TimeoutError:
            log.error(f"[T2-W{worker_id}] {sym} timed out after {cfg.tier2_timeout}s")
            if entry:
                analysis_queue.fail(entry, f"Timeout after {cfg.tier2_timeout}s")
        except Exception as e:
            log.error(f"[T2-W{worker_id}] {sym} error: {type(e).__name__}: {e}")
            if entry:
                analysis_queue.fail(entry, f"{type(e).__name__}: {str(e)[:80]}")
        finally:
            try:
                analysis_queue.done_tier2(entry)
            except Exception:
                pass

        await asyncio.sleep(2)


async def tier2_worker_pool():
    """Spawn and manage the Tier-2 worker pool, respecting config changes."""
    log.info("Tier-2 worker pool starting...")
    workers = []
    last_worker_count = 0

    while not _shutdown.is_set():
        cfg = analysis_config.get()
        n = cfg.tier2_workers if cfg.tier2_enabled else 0

        if n != last_worker_count:
            for t in workers:
                t.cancel()
            workers.clear()
            for i in range(n):
                t = asyncio.create_task(
                    tier2_worker(i + 1), name=f"t2_worker_{i+1}"
                )
                workers.append(t)
            log.info(f"Tier-2 worker pool: {n} workers active")
            last_worker_count = n

        await asyncio.sleep(15)

    for t in workers:
        t.cancel()


# ── Autonomous narrator ────────────────────────────────────────────────────

async def narrator_loop():
    """Every 5 minutes emit a plain-English summary of what the agent is seeing.
    Rule-based (no LLM), so it doesn't compete for LM Studio slots."""
    log.info("Narrator loop starting...")
    await asyncio.sleep(180)   # let scanner + workers warm up

    while not _shutdown.is_set():
        try:
            await _emit_narrator_comment()
        except Exception as e:
            log.debug(f"Narrator error: {e}")
        await asyncio.sleep(300)   # every 5 minutes


async def _emit_narrator_comment():
    wl = scanner.get_watchlist()
    if not wl:
        return

    status = market_status()
    session = ("US market open" if status["us_stocks"]
               else "after-hours / crypto" if status["crypto"] else "markets closed")

    parts: list[str] = []

    # Top tickers
    top = sorted(wl, key=lambda x: x.profile.composite_score, reverse=True)[:3]
    if top:
        top_str = ", ".join(
            f"{i.profile.symbol}({i.profile.composite_score:.0f})"
            for i in top
        )
        parts.append(f"Top watchlist: {top_str}.")

    # Recent signals
    buys  = [i for i in wl if i.latest_signal and
             "BUY" in getattr(i.latest_signal.signal_type, "value", "")][:3]
    sells = [i for i in wl if i.latest_signal and
             "SELL" in getattr(i.latest_signal.signal_type, "value", "")][:2]
    if buys:
        parts.append(
            "Active buy signals: "
            + ", ".join(f"{i.profile.symbol}@{i.latest_signal.conviction:.0%}" for i in buys)
            + "."
        )
    if sells:
        parts.append(
            "Sell signals: "
            + ", ".join(f"{i.profile.symbol}" for i in sells)
            + "."
        )

    # Hot/cold sectors
    try:
        hot  = self_improvement.get_hot_sectors()[:2]
        cold = self_improvement.get_cold_sectors()[:2]
        if hot:
            parts.append(f"Hot sectors: {', '.join(hot)}.")
        if cold:
            parts.append(f"Cold (suppressed): {', '.join(cold)}.")
    except Exception:
        pass

    # Open positions
    try:
        positions = paper_portfolio.open_positions
        if positions:
            pos_parts = []
            for p in positions[:3]:
                try:
                    pnl = getattr(p, "unrealized_pnl_pct", None) or getattr(p, "pnl_pct", 0)
                    pos_parts.append(f"{p.symbol} {'+' if pnl>=0 else ''}{pnl:.1f}%")
                except Exception:
                    pos_parts.append(p.symbol)
            if pos_parts:
                parts.append(f"Open positions: {', '.join(pos_parts)}.")
    except Exception:
        pass

    # Pipeline status
    try:
        qst = analysis_queue.get_status()
        t1q = qst.get("tier1_queue", 0)
        t2q = qst.get("tier2_queue", 0)
        active = len(qst.get("active", []))
        if t1q or t2q or active:
            parts.append(f"Pipeline: {t1q} T1 / {t2q} T2 queued, {active} active.")
    except Exception:
        pass

    # Macro
    try:
        spy = macro_monitor._spy_chg
        vix = macro_monitor._vix
        if spy is not None:
            macro_str = f"SPY {'+' if spy>=0 else ''}{spy:.1f}%"
            if vix:
                macro_str += f" VIX {vix:.0f}"
            parts.append(f"Macro: {macro_str}.")
    except Exception:
        pass

    if not parts:
        return

    message = f"[Narrator] {session.capitalize()}. " + " ".join(parts)

    await emit("agent.activity", {
        "agent_id": "narrator",
        "agent_type": "narrator",
        "symbol": "—",
        "action": "reasoning",
        "message": message,
        "ts": time.time(),
    }, "narrator")


# ── Feature A: Breakout Alert → Auto Deep Dive ────────────────────────────

async def breakout_monitor_loop():
    """Detect T0 score breakouts and auto-promote tickers to Tier-2."""
    _t0_history: dict = {}   # sym → deque of recent t0 scores
    from collections import deque
    await asyncio.sleep(120)

    while not _shutdown.is_set():
        try:
            wl = scanner.get_watchlist()
            cfg = analysis_config.get()
            for item in wl:
                sym   = item.profile.symbol
                score = getattr(item.profile, "t0_score", None)
                if score is None:
                    continue

                hist = _t0_history.setdefault(sym, deque(maxlen=5))
                prev = hist[-1] if hist else score
                hist.append(score)

                # Breakout: previously ≤1, now ≥2, with an active BUY signal
                if prev <= 1 and score >= 2:
                    has_buy = (
                        item.latest_signal and
                        "BUY" in getattr(item.latest_signal.signal_type, "value", "")
                    )
                    already = analysis_queue.is_queued_or_active(sym)
                    if has_buy and not already and cfg.tier2_enabled:
                        await analysis_queue.enqueue_tier2(sym, source="breakout_alert")
                        sig_val = item.latest_signal.signal_type.value
                        conv    = item.latest_signal.conviction
                        await emit("agent.activity", {
                            "agent_id":   "breakout_monitor",
                            "agent_type": "narrator",
                            "symbol":     sym,
                            "action":     "planning",
                            "message": (
                                f"[Breakout Alert] {sym} T0 score jumped "
                                f"{prev}→{score}/3 with {sig_val} "
                                f"({conv:.0%} conv) — auto-queuing deep dive."
                            ),
                            "ts": time.time(),
                        }, "breakout_monitor")
                        log.info(f"[Breakout] {sym} T0 {prev}→{score}, enqueued T2")
        except Exception as e:
            log.debug(f"Breakout monitor error: {e}")
        await asyncio.sleep(120)


# ── Feature C: Position Guardian ──────────────────────────────────────────

async def position_guardian_loop():
    """Every 10 min monitor open positions and emit guardian commentary."""
    await asyncio.sleep(300)

    while not _shutdown.is_set():
        try:
            positions = paper_portfolio.open_positions
            if not positions:
                await asyncio.sleep(600)
                continue

            parts: list[str] = []
            urgent: list[str] = []

            for p in positions[:6]:
                current = p.current_price or p.entry_price
                upct    = p.unrealized_pct
                stop    = p.stop_loss
                hold_h  = (time.time() - p.opened_ts) / 3600
                sign    = "+" if upct >= 0 else ""
                entry   = f"{p.symbol} {sign}{upct:.1f}% ({hold_h:.0f}h)"

                # Warn when the price feed hasn't updated recently
                price_age = staleness_guard.age_seconds(p.symbol)
                if price_age > 1800:
                    entry += f" [STALE {price_age // 60:.0f}min — P&L unreliable]"

                if stop > 0 and current > 0:
                    stop_dist = abs(current - stop) / current * 100
                    if stop_dist < 2.0:
                        entry += f" ⚠ {stop_dist:.1f}% from stop"
                        urgent.append(p.symbol)

                parts.append(entry)

            msg = f"[Position Guardian] {len(positions)} open: " + " | ".join(parts) + "."
            if urgent:
                msg += f" NEAR STOP: {', '.join(urgent)} — monitor closely."

            await emit("agent.activity", {
                "agent_id":   "guardian",
                "agent_type": "narrator",
                "symbol":     positions[0].symbol,
                "action":     "reasoning",
                "message":    msg,
                "ts":         time.time(),
            }, "guardian")

        except Exception as e:
            log.debug(f"Position guardian error: {e}")
        await asyncio.sleep(600)


# ── Feature E: Regime-Gated Strategy ──────────────────────────────────────

# Regime adjustments are DELTAS applied to the user's configured baseline, not
# absolute values. This prevents repeated regime cycling from drifting the user's
# chosen thresholds away from their original intent.
_REGIME_DELTAS: dict = {
    "volatile":      {"tier2_threshold": +0.15, "tier1_workers": -1},
    "trending_up":   {"tier2_threshold": -0.05, "tier1_workers": +1},
    "trending_down": {"tier2_threshold": +0.10, "tier1_workers":  0},
    "sideways":      {"tier2_threshold": +0.05, "tier1_workers":  0},
}


async def regime_gate_loop():
    """Adjust analysis thresholds whenever the macro regime changes."""
    _last_regime: str = "unknown"
    await asyncio.sleep(90)

    # Capture user baseline once — regime adjustments are relative to this
    _cfg0           = analysis_config.get()
    _baseline_t2    = _cfg0.tier2_threshold
    _baseline_t1w   = _cfg0.tier1_workers

    while not _shutdown.is_set():
        try:
            regime = macro_monitor._last_regime
            if regime != _last_regime and regime in _REGIME_DELTAS:
                d    = _REGIME_DELTAS[regime]
                new_t2  = round(max(0.50, min(0.95, _baseline_t2  + d["tier2_threshold"])), 2)
                new_t1w = max(1, min(8, _baseline_t1w + d["tier1_workers"]))
                changes = {"tier2_threshold": new_t2, "tier1_workers": new_t1w}
                analysis_config.update(**changes)
                _last_regime = regime
                msg = (
                    f"[Regime Gate] Macro shifted to '{regime}'. "
                    f"T2 threshold {_baseline_t2:.0%}→{new_t2:.0%} "
                    f"({d['tier2_threshold']:+.0%} offset from baseline), "
                    f"T1 workers {_baseline_t1w}→{new_t1w}."
                )
                await emit("agent.activity", {
                    "agent_id":   "regime_gate",
                    "agent_type": "narrator",
                    "symbol":     "SPY",
                    "action":     "planning",
                    "message":    msg,
                    "ts":         time.time(),
                }, "regime_gate")
                log.info(f"Regime gate applied: {regime} → {changes}")
        except Exception as e:
            log.debug(f"Regime gate error: {e}")
        await asyncio.sleep(60)


# ── Feature F: Self-Narrated Learning Journal ──────────────────────────────

async def journal_loop():
    """Emit a plain-English learning journal entry every 3 hours."""
    await asyncio.sleep(3600)

    while not _shutdown.is_set():
        try:
            entry = self_improvement.generate_journal_entry()
            await emit("agent.activity", {
                "agent_id":   "journal",
                "agent_type": "narrator",
                "symbol":     "—",
                "action":     "reasoning",
                "message":    f"[Learning Journal] {entry}",
                "ts":         time.time(),
            }, "journal")
            log.info("Learning journal emitted")
        except Exception as e:
            log.debug(f"Journal loop error: {e}")
        await asyncio.sleep(10800)   # every 3 hours


# ── Feature G: Cross-Asset Correlation Monitor ─────────────────────────────

async def correlation_monitor_loop():
    """Every 15 min, flag tickers diverging significantly from their sector peers."""
    await asyncio.sleep(600)

    while not _shutdown.is_set():
        try:
            wl = scanner.get_watchlist()
            if not wl:
                await asyncio.sleep(900)
                continue

            # Group by sector
            sectors: dict = {}
            for item in wl:
                sec = item.profile.sector or "Unknown"
                sectors.setdefault(sec, []).append(item)

            divergences: list[tuple] = []
            for sec, items in sectors.items():
                if len(items) < 2:
                    continue
                # Use 1D price change (change_pct) — composite_score is a watchlist
                # ranking metric, not a short-term price proxy.
                changes = [(i.profile.symbol, i.profile.change_pct) for i in items]
                avg_chg = sum(c for _, c in changes) / len(changes)
                for sym, chg in changes:
                    delta = chg - avg_chg
                    if delta >= 3.0:
                        divergences.append((sym, sec, delta, "leading"))
                    elif delta <= -3.0:
                        divergences.append((sym, sec, delta, "lagging"))

            if divergences:
                divergences.sort(key=lambda x: abs(x[2]), reverse=True)
                parts = []
                for sym, sec, delta, role in divergences[:5]:
                    sign = "+" if delta >= 0 else ""
                    parts.append(f"{sym} ({sec}) {sign}{delta:.1f}% vs peers [{role}]")
                msg = "[Correlation Monitor] Sector divergences: " + "; ".join(parts) + "."
                await emit("agent.activity", {
                    "agent_id":   "correlation",
                    "agent_type": "narrator",
                    "symbol":     divergences[0][0],
                    "action":     "reasoning",
                    "message":    msg,
                    "ts":         time.time(),
                }, "correlation")

        except Exception as e:
            log.debug(f"Correlation monitor error: {e}")
        await asyncio.sleep(900)   # every 15 min


# ── Feature H: Dynamic Watchlist Curation ─────────────────────────────────

async def curation_loop():
    """Auto-curate watchlist every 6 hours via WatchlistIntelligence.auto_curate()."""
    await asyncio.sleep(3600)

    while not _shutdown.is_set():
        try:
            if scanner.watchlist:
                wl     = scanner.get_watchlist()
                report = wl_intelligence.auto_curate(wl)

                parts: list[str] = []
                if report["curated_removed"]:
                    syms = [r["symbol"] for r in report["curated_removed"]]
                    parts.append(f"Auto-dismissed (bearish streak): {', '.join(syms)}")
                if report["curated_flagged"]:
                    syms = [r["symbol"] for r in report["curated_flagged"]]
                    parts.append(f"Stale high-alerts: {', '.join(syms)}")
                if report["auto_undismissed"]:
                    parts.append(f"Re-activated: {', '.join(report['auto_undismissed'])}")

                if parts:
                    msg = "[Curation] " + "; ".join(parts) + "."
                    await emit("agent.activity", {
                        "agent_id":   "curation",
                        "agent_type": "narrator",
                        "symbol":     "—",
                        "action":     "planning",
                        "message":    msg,
                        "ts":         time.time(),
                    }, "curation")
                    log.info(f"Curation: {report}")
        except Exception as e:
            log.debug(f"Curation loop error: {e}")
        await asyncio.sleep(21600)   # every 6 hours


# ── Watchlist intelligence ─────────────────────────────────────────────────

async def watchlist_intelligence_loop():
    while not _shutdown.is_set():
        try:
            if scanner.watchlist:
                wl = scanner.get_watchlist()
                result = wl_intelligence.update(wl)
                if result["promoted"]:
                    for item in result["promoted"]:
                        await emit("alert.triggered", {
                            "alert_type": "high_alert_promotion",
                            "symbol": item["symbol"],
                            "score": item["score"],
                            "note": f"Score {item['score']:.0f} — HIGH ALERT tier",
                            "ts": time.time(),
                        }, "wl_intelligence")
                await emit("watchlist.intelligence", result, "wl_intelligence")
        except Exception as e:
            log.debug(f"WL intelligence: {e}")
        await asyncio.sleep(30)


# ── Heartbeat ──────────────────────────────────────────────────────────────

async def heartbeat_loop():
    from agents.llm_router import get_llm_stats
    while not _shutdown.is_set():
        llm  = get_llm_stats()
        mkt  = market_status()
        si   = self_improvement.get_params()
        cfg  = analysis_config.get()
        qs   = analysis_queue.get_stats()
        qst  = analysis_queue.get_status()
        await emit("heartbeat", {
            "scan_count":          scanner._scan_count,
            "watchlist_count":     len(scanner.watchlist),
            "scanner_paused":      scanner.paused,
            "high_alert":          wl_intelligence.get_high_alert(),
            "llm_provider":        LLM.PROVIDER,
            "llm_model":           LLM.display_name(),
            "llm_tok_per_sec":     llm.get("tok_per_sec", 0),
            "llm_calls":           llm.get("calls", 0),
            "llm_errors":          llm.get("errors", 0),
            "memory_entries":      agent_memory.get_stats().get("total_entries", 0),
            "market_open":         mkt["us_stocks"],
            "market_time_et":      mkt["time_et"],
            "si_update_count":     si.update_count,
            "si_regime":           si.current_regime,
            "si_signals_analyzed": si.total_signals_analyzed,
            # v9 pipeline stats
            "t1_queue":            qst["tier1_queue"],
            "t2_queue":            qst["tier2_queue"],
            "t1_active":           len([e for e in qst["active"] if e["tier"] == 1]),
            "t2_active":           len([e for e in qst["active"] if e["tier"] == 2]),
            "t1_done":             qs["tier1_done"],
            "t2_done":             qs["tier2_done"],
            "avg_t1_s":            qs["avg_tier1_s"],
            "avg_t2_s":            qs["avg_tier2_s"],
            "tier1_enabled":       cfg.tier1_enabled,
            "tier2_enabled":       cfg.tier2_enabled,
            "tier1_workers":       cfg.tier1_workers,
            "tier2_threshold":     cfg.tier2_threshold,
            "ts":                  time.time(),
            # Execution engine stats
            "exec_mode":           execution_engine._mode,
            "exec_paused":         execution_engine._paused,
            "exec_pending":        len(execution_engine._pending),
            "exec_orders_today":   len([o for o in execution_engine._orders
                                        if o.ts_submitted > time.time() - 86_400]),
            "exec_cb_tripped":     execution_engine._cb.tripped,
            # Macro regime
            "macro_regime":        macro_monitor._last_regime,
            "macro_spy_chg":       macro_monitor._spy_chg,
            "macro_vix":           macro_monitor._vix,
            # Staleness
            "staleness_stale":     staleness_guard.get_stats(scanner.watchlist).get("stale", 0),
            "staleness_fresh":     staleness_guard.get_stats(scanner.watchlist).get("fresh", 0),
        }, "main")
        await asyncio.sleep(10)


# ── Dashboard ──────────────────────────────────────────────────────────────

async def run_dashboard():
    config = uvicorn.Config(app, host="0.0.0.0", port=SYS.DASHBOARD_PORT,
                            log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    await server.serve()


# ── Startup ────────────────────────────────────────────────────────────────

async def startup():
    cfg = analysis_config.get()

    # ── Pre-flight validation ──────────────────────────────────────────────
    ok = await startup_validator.validate(mock_mode=MOCK_MODE)
    if not ok:
        log.error("Pre-flight validation failed — aborting startup")
        sys.exit(1)

    # ── SQLite init (creates tables + migrates JSON on first run) ──────────
    await init_db()

    log.info("=" * 62)
    log.info("  QUANT AGENT v9 -- Autonomous Market Intelligence")
    log.info("=" * 62)
    log.info("")
    if MOCK_MODE:
        log.info("  MOCK MODE -- simulated data")
    else:
        log.info("  Checking API connections...")
        await registry.startup_check()
    log.info("")
    log.info(f"  LLM Provider:     {LLM.PROVIDER} ({LLM.display_name()})")
    log.info(f"  Dashboard:        http://localhost:{SYS.DASHBOARD_PORT}")
    log.info(f"  Scan interval:    {SYS.SCAN_INTERVAL}s")
    log.info(f"  Memory:           {agent_memory.get_stats()['total_entries']} entries")
    log.info(f"  Pipeline:         Tier-1 × {cfg.tier1_workers} workers | "
             f"Tier-2 threshold {cfg.tier2_threshold:.0%}")
    log.info("")


async def main():
    await startup()
    tasks = [
        asyncio.create_task(state_updater(),               name="state_updater"),
        asyncio.create_task(heartbeat_loop(),              name="heartbeat"),
        asyncio.create_task(run_dashboard(),               name="dashboard"),
        asyncio.create_task(scanner_loop(),                name="scanner"),
        asyncio.create_task(signal_loop(),                 name="signals"),
        asyncio.create_task(tier1_feeder_loop(),           name="t1_feeder"),
        asyncio.create_task(tier1_worker_pool(),           name="t1_pool"),
        asyncio.create_task(tier2_worker_pool(),            name="t2_pool"),
        asyncio.create_task(wl_intelligence.start(),       name="wl_signal_listener"),
        asyncio.create_task(pnl_tracker.start(),           name="pnl_tracker"),
        asyncio.create_task(alerts_manager.start(),        name="alerts"),
        asyncio.create_task(paper_portfolio.start(),       name="portfolio"),
        asyncio.create_task(agent_memory.start(),          name="memory"),
        asyncio.create_task(watchlist_intelligence_loop(), name="wl_intel"),
        asyncio.create_task(narrator_loop(),                name="narrator"),
        asyncio.create_task(breakout_monitor_loop(),        name="breakout_monitor"),
        asyncio.create_task(position_guardian_loop(),       name="position_guardian"),
        asyncio.create_task(regime_gate_loop(),             name="regime_gate"),
        asyncio.create_task(journal_loop(),                 name="journal"),
        asyncio.create_task(correlation_monitor_loop(),     name="correlation_monitor"),
        asyncio.create_task(curation_loop(),                name="curation"),
        asyncio.create_task(self_improvement.start(),      name="self_improvement"),
        asyncio.create_task(execution_engine.start(),      name="execution"),
        asyncio.create_task(macro_monitor.start(),         name="macro_monitor"),
        asyncio.create_task(digest_agent.start(),          name="digest"),
        asyncio.create_task(webhook_manager.start(),       name="webhooks"),
    ]
    log.info("All systems running.")
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        _shutdown.set()
        for t in tasks:
            t.cancel()
        await registry.close_all()
        log.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye.")
