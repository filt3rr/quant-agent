"""
signals/execution_engine.py -- Live Execution Engine

Bridges signal generation → actual order placement.

Modes:
  paper  : routes through internal paper_portfolio (JSON simulation, existing behaviour)
  live   : routes through Alpaca Markets paper-trading API (real paper orders)

Features:
  - Signal coordination: T1 LLM signals create pending slots; T2 confirms or vetoes
    before capital is committed.  Slots expire after 8 min (configurable).
  - Circuit breakers: daily-loss limit, max-drawdown limit, position-count cap.
    When tripped all new order placement is blocked until manually reset.
  - Slippage tracking: records expected vs actual fill price on every order.
  - Broker reconciliation: every 5 min in live mode, compares system positions
    to Alpaca account state and logs mismatches.
  - Data-staleness gate: rejects signals whose quote is older than STALE_QUOTE_S.
  - Emergency kill switch: cancels all open orders + closes all positions.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.bus import emit
from core.logger import get_logger
from config.settings import SYS

log = get_logger("execution")

EXEC_CONFIG_FILE = SYS.STORAGE_DIR / "execution_config.json"
EXEC_LOG_FILE    = SYS.STORAGE_DIR / "execution_orders.json"

COORD_TIMEOUT_S  = 480   # 8 min — T1 slot expires if T2 doesn't confirm
STALE_QUOTE_S    = 60    # reject signals if quote age > this many seconds
RECONCILE_S      = 300   # broker reconciliation interval in live mode
T1_AUTO_EXEC_CONV = 0.85 # minimum T1 conviction to auto-execute after slot expiry


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerConfig:
    daily_loss_limit_pct:  float = 0.02   # pause when today's loss ≥ 2%
    max_drawdown_pct:      float = 0.05   # pause when drawdown from peak ≥ 5%
    max_open_positions:    int   = 10     # refuse new entries when at limit
    # Live state (updated continuously)
    tripped:               bool  = False
    trip_reason:           str   = ""
    trip_ts:               float = 0.0
    peak_equity:           float = 0.0
    day_start_equity:      float = 0.0
    day_start_ts:          float = 0.0


@dataclass
class PendingSlot:
    """A T1 signal waiting for T2 confirmation."""
    symbol:        str
    t1_signal:     str
    t1_conviction: float
    t1_price:      float
    t1_ts:         float
    expires_at:    float
    t2_signal:     str   = ""
    t2_conviction: float = 0.0
    t2_ts:         float = 0.0
    status:        str   = "pending"  # pending | confirmed | vetoed | expired


@dataclass
class OrderRecord:
    order_id:       str
    symbol:         str
    side:           str     # BUY | SELL
    qty:            float
    order_type:     str     # market | limit | simulated | kill_switch
    expected_price: float
    fill_price:     float
    slippage_pct:   float
    status:         str     # submitted | filled | cancelled | rejected | simulated
    ts_submitted:   float
    ts_filled:      float
    source:         str     # rule | t1 | t1_confirmed | t2_standalone | kill_switch | manual
    conviction:     float
    broker_id:      str     # Alpaca order ID (empty for paper)
    mode:           str     # paper | live


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Single point of order placement for the entire system.
    Instantiate once; call start() as an async background task.
    """

    def __init__(self):
        self._mode:          str  = "paper"   # "paper" | "live"
        self._coordination:  bool = True       # gate T1 signals on T2 confirmation
        self._paused:        bool = False      # True when circuit breaker tripped
        self._cb:   CircuitBreakerConfig = CircuitBreakerConfig()

        self._pending: Dict[str, PendingSlot] = {}   # symbol → slot
        self._orders:  List[OrderRecord]       = []   # recent orders (capped 200)

        self._last_reconcile: float = 0.0
        self._alpaca = None   # lazy-init AlpacaProvider instance

        self._load_config()
        self._load_orders()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_config(self):
        try:
            if EXEC_CONFIG_FILE.exists():
                d = json.loads(EXEC_CONFIG_FILE.read_text(encoding="utf-8"))
                self._mode        = d.get("mode", "paper")
                self._coordination = d.get("coordination", True)
                self._cb.daily_loss_limit_pct = d.get("daily_loss_limit_pct", 0.02)
                self._cb.max_drawdown_pct     = d.get("max_drawdown_pct", 0.05)
                self._cb.max_open_positions   = d.get("max_open_positions", 10)
                log.info(f"Execution config loaded: mode={self._mode} coord={self._coordination}")
        except Exception as e:
            log.debug(f"Execution config load: {e}")

    def _save_config(self):
        try:
            EXEC_CONFIG_FILE.write_text(json.dumps({
                "mode":                 self._mode,
                "coordination":         self._coordination,
                "daily_loss_limit_pct": self._cb.daily_loss_limit_pct,
                "max_drawdown_pct":     self._cb.max_drawdown_pct,
                "max_open_positions":   self._cb.max_open_positions,
            }, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug(f"Execution config save: {e}")

    def _load_orders(self):
        try:
            if EXEC_LOG_FILE.exists():
                data = json.loads(EXEC_LOG_FILE.read_text(encoding="utf-8"))
                for o in data.get("orders", [])[-200:]:
                    self._orders.append(OrderRecord(**o))
                log.info(f"Execution: loaded {len(self._orders)} historical orders")
        except Exception as e:
            log.debug(f"Execution order load: {e}")

    def _save_orders(self):
        try:
            EXEC_LOG_FILE.write_text(json.dumps({
                "orders": [asdict(o) for o in self._orders[-200:]]
            }, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug(f"Execution order save: {e}")

    # ── Alpaca lazy init ──────────────────────────────────────────────────────

    def _get_alpaca(self):
        if self._alpaca is None:
            from providers.alpaca_provider import AlpacaProvider
            self._alpaca = AlpacaProvider()
        return self._alpaca

    # ── Public configuration ──────────────────────────────────────────────────

    def set_mode(self, mode: str):
        """Switch between 'paper' (JSON sim) and 'live' (Alpaca API)."""
        if mode not in ("paper", "live"):
            raise ValueError(f"Unknown mode: {mode}")
        prev = self._mode
        self._mode = mode

        # When switching to live: mark paper_portfolio as externally managed
        # so it doesn't auto-open from the bus.
        from signals.portfolio import paper_portfolio
        paper_portfolio._managed_externally = (mode == "live" or self._coordination)

        self._save_config()
        log.info(f"Execution mode: {prev} → {mode}")

    def set_coordination(self, enabled: bool):
        self._coordination = bool(enabled)
        from signals.portfolio import paper_portfolio
        paper_portfolio._managed_externally = (self._mode == "live" or self._coordination)
        self._save_config()
        log.info(f"Signal coordination: {'ON' if enabled else 'OFF'}")

    def update_circuit_breaker(self, daily_loss_pct: float = None,
                                max_drawdown_pct: float = None,
                                max_positions: int = None):
        if daily_loss_pct is not None:
            self._cb.daily_loss_limit_pct = max(0.005, min(0.20, daily_loss_pct))
        if max_drawdown_pct is not None:
            self._cb.max_drawdown_pct = max(0.01, min(0.50, max_drawdown_pct))
        if max_positions is not None:
            self._cb.max_open_positions = max(1, min(50, int(max_positions)))
        self._save_config()

    def reset_circuit_breaker(self):
        self._paused = False
        self._cb.tripped    = False
        self._cb.trip_reason = ""
        self._cb.trip_ts     = 0.0
        log.info("Circuit breaker RESET — trading resumed")

    # ── Signal entry points ───────────────────────────────────────────────────

    async def register_t1_signal(self, symbol: str, signal: str,
                                  conviction: float, price: float):
        """
        Called by the Tier-1 worker when a high-conviction signal is produced.
        If coordination is enabled, creates a pending slot and waits for T2.
        If coordination is off (or T2 is disabled), executes immediately.
        """
        if signal in ("HOLD", "WATCH", ""):
            return
        if not self._can_open(symbol):
            return

        from agents.analysis_config import analysis_config
        coord_active = self._coordination and analysis_config.get().tier2_enabled

        if not coord_active:
            await self._execute_signal(symbol, signal, conviction, price, source="t1")
            return

        slot = PendingSlot(
            symbol=symbol,
            t1_signal=signal,
            t1_conviction=conviction,
            t1_price=price,
            t1_ts=time.time(),
            expires_at=time.time() + COORD_TIMEOUT_S,
        )
        self._pending[symbol] = slot
        log.info(f"[EXEC] T1 pending [{symbol}] {signal} {conviction:.0%} "
                 f"(expires in {COORD_TIMEOUT_S//60}min)")
        await emit("execution.pending", {
            "symbol": symbol, "signal": signal,
            "conviction": conviction, "expires_at": slot.expires_at
        }, "execution")

    async def confirm_t2_signal(self, symbol: str, signal: str,
                                 conviction: float, price: float):
        """
        Called by the Tier-2 worker after a deep dive.
        If a T1 pending slot exists: confirm, veto, or pass based on agreement.
        If no slot exists (T2 standalone): execute directly with conviction gate.
        """
        slot = self._pending.get(symbol)

        if slot is None:
            # Standalone T2 — execute if conviction is high enough
            if signal not in ("HOLD", "WATCH") and conviction >= 0.80:
                await self._execute_signal(symbol, signal, conviction, price,
                                           source="t2_standalone")
            return

        del self._pending[symbol]
        slot.t2_signal     = signal
        slot.t2_conviction = conviction
        slot.t2_ts         = time.time()

        t1_is_buy = "BUY" in slot.t1_signal
        t2_is_buy = "BUY" in signal
        t2_hold   = signal in ("HOLD", "WATCH")

        # T2 HOLD: only execute if T1 was very confident
        if t2_hold:
            if slot.t1_conviction >= T1_AUTO_EXEC_CONV:
                slot.status = "confirmed"
                log.info(f"[EXEC] [{symbol}] T2=HOLD but T1 conv {slot.t1_conviction:.0%} ≥ threshold → execute T1")
                await self._execute_signal(symbol, slot.t1_signal, slot.t1_conviction,
                                           slot.t1_price, source="t1_confirmed")
            else:
                slot.status = "vetoed"
                log.info(f"[EXEC] [{symbol}] T2=HOLD + T1 conv {slot.t1_conviction:.0%} < threshold → vetoed")
            return

        # T2 contradicts T1 with meaningful conviction → veto
        if t1_is_buy != t2_is_buy and conviction >= 0.65:
            slot.status = "vetoed"
            log.info(f"[EXEC] [{symbol}] T2 VETO (T1={slot.t1_signal} T2={signal} "
                     f"conv={conviction:.0%})")
            await emit("execution.vetoed", {
                "symbol": symbol, "t1": slot.t1_signal, "t2": signal,
                "t2_conviction": conviction
            }, "execution")
            return

        # T2 agrees (or low-conviction contradiction) → execute using T2 signal
        final_conv = max(slot.t1_conviction, conviction)
        slot.status = "confirmed"
        log.info(f"[EXEC] [{symbol}] T2 CONFIRMED {signal} {final_conv:.0%}")
        await self._execute_signal(symbol, signal, final_conv, price,
                                   source="t2_confirmed")

    # ── Core execution ────────────────────────────────────────────────────────

    async def _execute_signal(self, symbol: str, signal: str, conviction: float,
                               price: float, source: str):
        """Route a validated signal to paper_portfolio or Alpaca."""
        # Circuit breaker check
        ok, reason = self._check_circuit_breakers()
        if not ok:
            log.warning(f"[EXEC BLOCKED] [{symbol}] {reason}")
            return

        # Data-staleness gate
        try:
            from scanners.market_scanner import scanner
            item = scanner.watchlist.get(symbol)
            if item and (time.time() - item.profile.scan_ts) > STALE_QUOTE_S:
                age = time.time() - item.profile.scan_ts
                log.warning(f"[EXEC STALE] [{symbol}] quote is {age:.0f}s old — skipping")
                return
        except Exception:
            pass

        # Build a signal dict compatible with paper_portfolio.open_position()
        sig_dict = self._build_sig_dict(symbol, signal, conviction, price)

        if self._mode == "paper":
            await self._execute_paper(sig_dict, source)
        else:
            await self._execute_live(sig_dict, source)

    def _build_sig_dict(self, symbol: str, signal: str, conviction: float,
                         price: float) -> Dict:
        """Build a signal dict that portfolio.open_position() can consume."""
        from signals.signal_engine import _calc_target_stop
        from core.models import SignalType
        try:
            sig_type = SignalType(signal)
        except ValueError:
            sig_type = SignalType.BUY if "BUY" in signal else SignalType.SELL

        # ATR-based target/stop (best effort — defaults if no ATR available)
        target, stop = None, None
        try:
            from scanners.market_scanner import scanner
            item = scanner.watchlist.get(symbol)
            if item:
                ind = item.profile.provider_data.get("indicators", {})
                atr = ind.get("atr", price * 0.02)
                target, stop = _calc_target_stop(price, sig_type, atr)
        except Exception:
            pass

        if target is None:
            target = round(price * 1.06, 4)
        if stop is None:
            stop = round(price * 0.98, 4)

        return {
            "symbol":          symbol,
            "signal_type":     signal,
            "conviction":      conviction,
            "price_at_signal": price,
            "target_price":    target,
            "stop_loss":       stop,
            "agent":           "execution_engine",
            "ts":              time.time(),
        }

    async def _execute_paper(self, sig_dict: Dict, source: str):
        """Execute via internal paper_portfolio (JSON simulation)."""
        from signals.portfolio import paper_portfolio
        pos = paper_portfolio.open_position(sig_dict)
        if pos is None:
            return

        rec = OrderRecord(
            order_id       = str(uuid.uuid4())[:8],
            symbol         = sig_dict["symbol"],
            side           = "BUY" if "BUY" in sig_dict["signal_type"] else "SELL",
            qty            = pos.shares,
            order_type     = "simulated",
            expected_price = sig_dict["price_at_signal"],
            fill_price     = sig_dict["price_at_signal"],
            slippage_pct   = 0.0,
            status         = "filled",
            ts_submitted   = time.time(),
            ts_filled      = time.time(),
            source         = source,
            conviction     = sig_dict["conviction"],
            broker_id      = "",
            mode           = "paper",
        )
        self._append_order(rec)
        log.info(f"[PAPER ORDER] {rec.symbol} {rec.side} {rec.qty:.2f}sh "
                 f"@ ${rec.fill_price:.4f} (source={source})")
        await emit("execution.order", asdict(rec), "execution")

    async def _execute_live(self, sig_dict: Dict, source: str):
        """Execute via Alpaca Markets (paper or live account)."""
        symbol     = sig_dict["symbol"]
        signal     = sig_dict["signal_type"]
        conviction = sig_dict["conviction"]
        price      = sig_dict["price_at_signal"]
        stop       = sig_dict.get("stop_loss", price * 0.98)

        side = "buy" if "BUY" in signal else "sell"

        # Conviction determines order type
        if conviction >= 0.90:
            order_type  = "market"
            limit_price = None
        else:
            order_type  = "limit"
            spread_adj  = price * 0.0005   # 0.05% above ask for buys
            limit_price = round(
                price + spread_adj if side == "buy" else price - spread_adj, 2
            )

        # Size from portfolio risk logic
        from signals.portfolio import ACCOUNT_SIZE, RISK_PER_TRADE, paper_portfolio
        stop_dist = abs(price - stop) if stop and stop != price else price * 0.02
        risk_usd  = paper_portfolio.account_value * RISK_PER_TRADE
        qty       = max(1.0, round(risk_usd / stop_dist, 0))

        alpaca = self._get_alpaca()
        result = await alpaca.place_order(
            symbol=symbol, qty=qty, side=side,
            order_type=order_type, limit_price=limit_price, paper=True
        )

        error = result.get("_error", "") if result else "No response"
        fill_price = float(result.get("filled_avg_price") or price) if result else price
        slippage = round((fill_price - price) / price * 100, 4) if price else 0.0

        rec = OrderRecord(
            order_id       = str(uuid.uuid4())[:8],
            symbol         = symbol,
            side           = side.upper(),
            qty            = qty,
            order_type     = order_type,
            expected_price = price,
            fill_price     = fill_price,
            slippage_pct   = slippage,
            status         = result.get("status", "rejected") if result else "rejected",
            ts_submitted   = time.time(),
            ts_filled      = time.time() if fill_price > 0 else 0,
            source         = source,
            conviction     = conviction,
            broker_id      = result.get("order_id", "") if result else "",
            mode           = "live",
        )

        if error:
            rec.status = "rejected"
            log.error(f"[LIVE ORDER REJECTED] [{symbol}]: {error}")
        else:
            log.info(f"[LIVE ORDER] {symbol} {side.upper()} {qty}sh "
                     f"{order_type} expected=${price:.4f} "
                     f"fill=${fill_price:.4f} slip={slippage:+.3f}%")
            # Mirror in paper_portfolio for P&L tracking
            from signals.portfolio import paper_portfolio
            paper_portfolio.open_position(sig_dict)

        self._append_order(rec)
        await emit("execution.order", asdict(rec), "execution")

    def _append_order(self, rec: OrderRecord):
        self._orders.append(rec)
        self._orders = self._orders[-200:]
        self._save_orders()

    # ── Circuit breakers ──────────────────────────────────────────────────────

    def _check_circuit_breakers(self) -> Tuple[bool, str]:
        if self._paused:
            return False, f"TRIPPED: {self._cb.trip_reason}"

        from signals.portfolio import paper_portfolio

        # Position count
        open_count = len(paper_portfolio.open_positions)
        if open_count >= self._cb.max_open_positions:
            return False, f"Max positions ({self._cb.max_open_positions}) reached"

        # Get equity
        try:
            equity = paper_portfolio.account_value
        except Exception:
            return True, ""

        # Update peak
        if self._cb.peak_equity <= 0:
            self._cb.peak_equity = equity
        elif equity > self._cb.peak_equity:
            self._cb.peak_equity = equity

        # Init day start
        now = time.time()
        if self._cb.day_start_equity <= 0 or (now - self._cb.day_start_ts) > 86_400:
            self._cb.day_start_equity = equity
            self._cb.day_start_ts     = now

        # Daily loss limit
        if self._cb.day_start_equity > 0:
            daily_ret = (equity - self._cb.day_start_equity) / self._cb.day_start_equity
            if daily_ret <= -self._cb.daily_loss_limit_pct:
                reason = (f"Daily loss limit ({self._cb.daily_loss_limit_pct:.0%}) hit "
                          f"({daily_ret:.2%} today)")
                self._trip(reason)
                return False, reason

        # Max drawdown
        if self._cb.peak_equity > 0:
            drawdown = (equity - self._cb.peak_equity) / self._cb.peak_equity
            if drawdown <= -self._cb.max_drawdown_pct:
                reason = (f"Max drawdown ({self._cb.max_drawdown_pct:.0%}) hit "
                          f"({drawdown:.2%} from peak)")
                self._trip(reason)
                return False, reason

        return True, ""

    def _trip(self, reason: str):
        if not self._paused:
            self._paused           = True
            self._cb.tripped       = True
            self._cb.trip_reason   = reason
            self._cb.trip_ts       = time.time()
            log.error(f"⚡ CIRCUIT BREAKER TRIPPED: {reason}")
            asyncio.get_event_loop().create_task(
                emit("execution.circuit_breaker",
                     {"reason": reason, "ts": self._cb.trip_ts},
                     "execution")
            )

    def _can_open(self, symbol: str) -> bool:
        from signals.portfolio import paper_portfolio
        return (not any(p.symbol == symbol for p in paper_portfolio.open_positions)
                and not self._paused)

    # ── Kill switch ───────────────────────────────────────────────────────────

    async def emergency_close_all(self) -> Dict:
        """Cancel all pending orders and close every open position."""
        from signals.portfolio import paper_portfolio
        results = {"cancelled_orders": 0, "closed_positions": 0, "errors": []}

        if self._mode == "live":
            alpaca = self._get_alpaca()
            try:
                await alpaca.cancel_all_orders(paper=True)
                results["cancelled_orders"] = 1
                await alpaca.close_all_positions(paper=True)
            except Exception as e:
                results["errors"].append(str(e))

        # Regardless of mode: close all positions in paper_portfolio
        for pos in list(paper_portfolio.open_positions):
            price = pos.current_price or pos.entry_price
            await paper_portfolio._close_position(pos, price, "kill_switch")
            results["closed_positions"] += 1

        self._trip("Emergency kill switch activated")

        kill_rec = OrderRecord(
            order_id=f"KILL-{str(uuid.uuid4())[:6]}", symbol="ALL",
            side="SELL", qty=0, order_type="kill_switch",
            expected_price=0, fill_price=0, slippage_pct=0,
            status="filled", ts_submitted=time.time(), ts_filled=time.time(),
            source="kill_switch", conviction=1.0, broker_id="", mode=self._mode,
        )
        self._append_order(kill_rec)

        log.warning(f"☠ KILL SWITCH: closed {results['closed_positions']} positions")
        await emit("execution.kill_switch", results, "execution")
        return results

    # ── Broker reconciliation ─────────────────────────────────────────────────

    async def reconcile_with_broker(self) -> Dict:
        """
        Compare system positions to Alpaca account state.
        Returns a dict describing mismatches.
        Only meaningful in live mode (no-op in paper).
        """
        if self._mode != "live":
            return {"note": "reconciliation only runs in live mode", "mismatches": []}

        from signals.portfolio import paper_portfolio
        alpaca = self._get_alpaca()
        try:
            broker_pos = await alpaca.get_positions(paper=True)
        except Exception as e:
            return {"error": str(e), "mismatches": []}

        system_map = {p.symbol: p for p in paper_portfolio.open_positions}
        broker_map  = {p["symbol"]: p for p in broker_pos}

        mismatches = []

        for sym, pos in system_map.items():
            if sym not in broker_map:
                mismatches.append(
                    f"{sym}: system holds {pos.shares:.2f}sh, broker has none"
                )
            else:
                bq = broker_map[sym]["qty"]
                pct_diff = abs(bq - pos.shares) / max(pos.shares, 0.01)
                if pct_diff > 0.05:
                    mismatches.append(
                        f"{sym}: system={pos.shares:.2f}sh broker={bq:.2f}sh"
                        f" ({pct_diff:.0%} diff)"
                    )

        for sym in broker_map:
            if sym not in system_map:
                mismatches.append(
                    f"{sym}: broker holds {broker_map[sym]['qty']:.2f}sh, "
                    f"system has no record"
                )

        self._last_reconcile = time.time()

        if mismatches:
            log.warning(f"RECONCILIATION MISMATCH ({len(mismatches)}): "
                        f"{'; '.join(mismatches[:3])}")
        else:
            log.info(f"RECONCILIATION OK — {len(broker_pos)} broker positions match")

        await emit("execution.reconcile",
                   {"mismatches": mismatches, "ts": self._last_reconcile},
                   "execution")
        return {"mismatches": mismatches, "broker_positions": len(broker_pos),
                "system_positions": len(system_map)}

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        from signals.portfolio import paper_portfolio
        try:
            equity = paper_portfolio.account_value
        except Exception:
            equity = 0.0

        daily_pnl_pct = 0.0
        if self._cb.day_start_equity > 0:
            daily_pnl_pct = round(
                (equity - self._cb.day_start_equity) / self._cb.day_start_equity * 100, 3
            )

        drawdown_pct = 0.0
        if self._cb.peak_equity > 0:
            drawdown_pct = round(
                (equity - self._cb.peak_equity) / self._cb.peak_equity * 100, 3
            )

        live_orders = [o for o in self._orders if o.mode == "live" and o.slippage_pct != 0]
        avg_slip = round(
            sum(abs(o.slippage_pct) for o in live_orders) / len(live_orders), 4
        ) if live_orders else 0.0

        return {
            "mode":             self._mode,
            "coordination":     self._coordination,
            "paused":           self._paused,
            "circuit_breaker":  {
                "tripped":             self._cb.tripped,
                "reason":              self._cb.trip_reason,
                "trip_ts":             self._cb.trip_ts,
                "daily_loss_limit":    self._cb.daily_loss_limit_pct,
                "max_drawdown":        self._cb.max_drawdown_pct,
                "max_open_positions":  self._cb.max_open_positions,
                "peak_equity":         round(self._cb.peak_equity, 2),
                "day_start_equity":    round(self._cb.day_start_equity, 2),
                "daily_pnl_pct":       daily_pnl_pct,
                "drawdown_pct":        drawdown_pct,
            },
            "pending_slots":    [
                {
                    "symbol":        s.symbol,
                    "t1_signal":     s.t1_signal,
                    "t1_conviction": round(s.t1_conviction, 3),
                    "t1_ts":         s.t1_ts,
                    "expires_in_s":  max(0, round(s.expires_at - time.time())),
                    "status":        s.status,
                }
                for s in self._pending.values()
            ],
            "recent_orders":    [asdict(o) for o in reversed(self._orders[-30:])],
            "orders_today":     len([o for o in self._orders
                                     if o.ts_submitted > time.time() - 86_400]),
            "avg_slippage_pct": avg_slip,
            "last_reconcile":   self._last_reconcile,
        }

    # ── Background loop ───────────────────────────────────────────────────────

    async def start(self):
        """Background loop: expire pending slots, reconcile, reset daily CB."""
        from signals.portfolio import paper_portfolio

        # Mark portfolio as externally managed if needed
        paper_portfolio._managed_externally = (
            self._mode == "live" or self._coordination
        )

        log.info(f"Execution engine started — mode={self._mode} "
                 f"coordination={self._coordination}")

        while True:
            try:
                now = time.time()

                # ── Expire pending T1 slots ──
                for sym in list(self._pending.keys()):
                    slot = self._pending[sym]
                    if now > slot.expires_at:
                        if slot.t1_conviction >= T1_AUTO_EXEC_CONV:
                            log.info(f"[EXEC] [{sym}] T1 slot expired; high conv "
                                     f"{slot.t1_conviction:.0%} → auto-execute")
                            slot.status = "expired"
                            del self._pending[sym]
                            await self._execute_signal(
                                sym, slot.t1_signal, slot.t1_conviction,
                                slot.t1_price, source="t1_expired"
                            )
                        else:
                            log.info(f"[EXEC] [{sym}] T1 slot expired (conv "
                                     f"{slot.t1_conviction:.0%} < {T1_AUTO_EXEC_CONV:.0%}) "
                                     f"— discarded")
                            slot.status = "expired"
                            del self._pending[sym]

                # ── Broker reconcile ──
                if self._mode == "live" and (now - self._last_reconcile) > RECONCILE_S:
                    await self.reconcile_with_broker()

                # ── Daily CB reset at market open ──
                try:
                    from core.market_hours import market_status
                    mkt = market_status()
                    if mkt.get("us_stocks") and 0 <= mkt.get("minutes_since_open", -1) < 1:
                        eq = paper_portfolio.account_value
                        if abs(eq - self._cb.day_start_equity) / max(self._cb.day_start_equity, 1) > 0.001:
                            self._cb.day_start_equity = eq
                            self._cb.day_start_ts     = now
                            log.info(f"[CB] Daily reset — equity=${eq:,.2f}")
                except Exception:
                    pass

            except Exception as e:
                log.error(f"Execution engine loop error: {e}")

            await asyncio.sleep(30)


# Singleton
execution_engine = ExecutionEngine()
