"""
signals/portfolio.py -- Paper portfolio tracker

Tracks hypothetical positions based on signals:
  - Opens position when STRONG_BUY or BUY fires with >70% conviction
  - Closes when target hit, stop hit, SELL fires, or 5 days elapsed
  - Tracks running P&L, open positions, closed trades

Persistence: SQLite via core.db (WAL mode).
On first run, existing storage/portfolio.json is migrated automatically
by core.db.init_db() and renamed to portfolio.json.bak.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from core.bus import bus, emit
from core.logger import get_logger
from config.settings import SYS
import core.db as db

log = get_logger("portfolio")

ACCOUNT_SIZE   = 25_000.0
RISK_PER_TRADE = 0.01
MAX_POSITIONS  = 10
AUTO_TRADE_MIN_CONVICTION = 0.80


@dataclass
class Position:
    pos_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    signal_type: str = ""
    entry_price: float = 0.0
    shares: float = 0.0
    notional: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    conviction: float = 0.0
    agent: str = ""
    sector: str = ""
    opened_ts: float = field(default_factory=time.time)
    closed_ts: float = 0.0
    close_price: float = 0.0
    close_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pct: float = 0.0
    risk_1r: float = 0.0
    breakeven_set: bool = False
    trail_activated: bool = False
    near_earnings: bool = False
    earnings_exit_ts: float = 0.0


def _row_to_position(r: dict) -> Position:
    return Position(
        pos_id          = r["pos_id"],
        symbol          = r["symbol"],
        signal_type     = r.get("signal_type", ""),
        entry_price     = r.get("entry_price", 0.0),
        shares          = r.get("shares", 0.0),
        notional        = r.get("notional", 0.0),
        stop_loss       = r.get("stop_loss", 0.0),
        target_price    = r.get("target_price", 0.0),
        conviction      = r.get("conviction", 0.0),
        agent           = r.get("agent", ""),
        sector          = r.get("sector", ""),
        opened_ts       = r.get("opened_ts", 0.0),
        closed_ts       = r.get("closed_ts", 0.0),
        close_price     = r.get("close_price", 0.0),
        close_reason    = r.get("close_reason", ""),
        pnl_usd         = r.get("pnl_usd", 0.0),
        pnl_pct         = r.get("pnl_pct", 0.0),
        status          = r.get("status", "open"),
        current_price   = r.get("current_price", 0.0),
        unrealized_pnl  = r.get("unrealized_pnl", 0.0),
        unrealized_pct  = r.get("unrealized_pct", 0.0),
        risk_1r         = r.get("risk_1r", 0.0),
        breakeven_set   = bool(r.get("breakeven_set", 0)),
        trail_activated = bool(r.get("trail_activated", 0)),
        near_earnings   = bool(r.get("near_earnings", 0)),
        earnings_exit_ts= r.get("earnings_exit_ts", 0.0),
    )


_UPSERT_POS_SQL = """
    INSERT OR REPLACE INTO positions
    (pos_id,symbol,signal_type,entry_price,shares,notional,stop_loss,
     target_price,conviction,agent,sector,opened_ts,closed_ts,close_price,
     close_reason,pnl_usd,pnl_pct,status,current_price,unrealized_pnl,
     unrealized_pct,risk_1r,breakeven_set,trail_activated,near_earnings,
     earnings_exit_ts)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def _pos_params(p: Position) -> tuple:
    return (
        p.pos_id, p.symbol, p.signal_type, p.entry_price, p.shares, p.notional,
        p.stop_loss, p.target_price, p.conviction, p.agent, p.sector,
        p.opened_ts, p.closed_ts, p.close_price, p.close_reason,
        p.pnl_usd, p.pnl_pct, p.status, p.current_price,
        p.unrealized_pnl, p.unrealized_pct, p.risk_1r,
        int(p.breakeven_set), int(p.trail_activated),
        int(p.near_earnings), p.earnings_exit_ts,
    )


class PaperPortfolio:
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._cash = ACCOUNT_SIZE
        self._enabled = False
        self._min_conviction = AUTO_TRADE_MIN_CONVICTION
        self._max_positions = MAX_POSITIONS
        self._managed_externally = False
        self._try_sync_load()

    # ── Settings ──────────────────────────────────────────────────────────

    def get_settings(self) -> Dict:
        return {
            "enabled":         self._enabled,
            "min_conviction":  self._min_conviction,
            "max_positions":   self._max_positions,
            "account_size":    ACCOUNT_SIZE,
            "risk_per_trade":  RISK_PER_TRADE,
        }

    def update_settings(self, enabled: bool = None, min_conviction: float = None,
                        max_positions: int = None):
        if enabled is not None:
            self._enabled = bool(enabled)
            log.info(f"Paper trading {'ENABLED' if self._enabled else 'DISABLED'}")
        if min_conviction is not None:
            self._min_conviction = max(0.50, min(0.99, float(min_conviction)))
            log.info(f"Paper trading min conviction set to {self._min_conviction:.0%}")
        if max_positions is not None:
            self._max_positions = max(1, min(20, int(max_positions)))
            log.info(f"Paper trading max positions set to {self._max_positions}")

    # ── Persistence ───────────────────────────────────────────────────────

    def _try_sync_load(self):
        """Synchronous best-effort load from DB at import time (restart case)."""
        try:
            import sqlite3
            if not db.DB_PATH.exists():
                return
            conn = sqlite3.connect(str(db.DB_PATH), timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Load all positions (open + closed last 90 days)
            cutoff = time.time() - 90 * 86_400
            rows = conn.execute(
                "SELECT * FROM positions WHERE opened_ts > ? OR status='open'", (cutoff,)
            ).fetchall()
            meta = conn.execute(
                "SELECT value FROM db_meta WHERE key='portfolio_cash'"
            ).fetchone()
            conn.close()
            for r in rows:
                pos = _row_to_position(dict(r))
                self._positions[pos.pos_id] = pos
            if meta:
                self._cash = float(meta["value"])
            if self._positions:
                log.info(
                    f"Portfolio loaded {len(self._positions)} positions "
                    f"(cash=${self._cash:,.0f}) from DB"
                )
        except Exception as e:
            log.debug(f"Portfolio sync load: {e}")

    async def _load_from_db(self):
        """Full async reload from DB (called from start() after init_db)."""
        cutoff = time.time() - 90 * 86_400
        rows = await db.fetchall(
            "SELECT * FROM positions WHERE opened_ts > ? OR status='open'", (cutoff,)
        )
        self._positions.clear()
        for r in rows:
            pos = _row_to_position(r)
            self._positions[pos.pos_id] = pos
        meta = await db.fetchone("SELECT value FROM db_meta WHERE key='portfolio_cash'")
        if meta:
            self._cash = float(meta["value"])
        log.info(
            f"Portfolio loaded {len(self._positions)} positions "
            f"(cash=${self._cash:,.0f}) from DB"
        )

    async def _save_position(self, pos: Position):
        await db.execute(_UPSERT_POS_SQL, _pos_params(pos))

    async def _save_cash(self):
        await db.execute(
            "INSERT OR REPLACE INTO db_meta(key,value) VALUES('portfolio_cash',?)",
            (str(self._cash),),
        )

    def _fire_save(self, pos: Position):
        """Schedule async DB writes from sync context (open_position)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_position(pos))
            loop.create_task(self._save_cash())
        except RuntimeError:
            pass  # no running loop (test context)

    # ── Portfolio properties ──────────────────────────────────────────────

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.status == "open"]

    @property
    def closed_positions(self) -> List[Position]:
        return sorted(
            [p for p in self._positions.values() if p.status == "closed"],
            key=lambda x: x.closed_ts, reverse=True,
        )

    @property
    def account_value(self) -> float:
        mkt = sum(
            (p.current_price * p.shares if p.current_price > 0 else p.notional)
            for p in self.open_positions
        )
        return round(self._cash + mkt, 2)

    def can_open(self, symbol: str) -> bool:
        already_open = any(p.symbol == symbol for p in self.open_positions)
        return (not already_open
                and len(self.open_positions) < self._max_positions
                and self._cash > 100)

    # ── Open position ─────────────────────────────────────────────────────

    def open_position(self, signal: Dict) -> Optional[Position]:
        if not self._enabled:
            return None

        sym        = signal.get("symbol", "")
        sig_type   = signal.get("signal_type", "")
        conviction = signal.get("conviction", 0)
        price      = signal.get("price_at_signal", 0)
        target     = signal.get("target_price", 0)
        stop       = signal.get("stop_loss", 0)

        if not self.can_open(sym):
            return None
        if conviction < self._min_conviction:
            return None
        if price <= 0 or stop <= 0:
            return None
        if "BUY" not in sig_type:
            return None

        # ── Regime-aware risk scaling ──────────────────────────────────
        regime_risk_mult = 1.0
        regime = "unknown"
        try:
            from agents.self_improvement import self_improvement as _si
            params = _si.get_params()
            regime = params.current_regime
            regime_risk_mult = {
                "volatile":      0.50,
                "trending_down": 0.60,
                "sideways":      0.80,
                "trending_up":   1.00,
            }.get(regime, 1.0)
        except Exception:
            pass

        # ── Sector concentration check ─────────────────────────────────
        incoming_sector = ""
        sector_size_mult = 1.0
        try:
            from scanners.market_scanner import scanner
            wl_item = scanner.watchlist.get(sym)
            if wl_item:
                incoming_sector = wl_item.profile.sector or ""
        except Exception:
            pass

        if incoming_sector:
            same_sector = sum(1 for p in self.open_positions if p.sector == incoming_sector)
            if same_sector >= 3:
                log.info(f"SECTOR VETO [{sym}]: {same_sector} open in '{incoming_sector}'")
                return None
            elif same_sector == 2:
                sector_size_mult = 0.50
            elif same_sector == 1:
                sector_size_mult = 0.75

        # ── Pre-earnings check (uses cached data — no await needed) ───
        near_earnings = False
        earnings_exit_ts = 0.0
        earnings_size_mult = 1.0
        earnings_stop_mult = 1.0
        try:
            import datetime
            from signals.alerts import alerts_manager
            upcoming = alerts_manager.get_earnings(3)   # uses cached data
            for evt in upcoming:
                if evt.get("symbol", "").upper() == sym:
                    near_earnings = True
                    earnings_size_mult = 0.50
                    earnings_stop_mult = 2.0
                    try:
                        edate = datetime.date.fromisoformat(evt["date"])
                        earnings_exit_ts = (
                            datetime.datetime.combine(
                                edate + datetime.timedelta(days=1),
                                datetime.time(16, 0),
                            ).timestamp()
                        )
                    except Exception:
                        earnings_exit_ts = time.time() + 4 * 86_400
                    log.info(
                        f"PRE-EARNINGS [{sym}]: earnings {evt.get('date')} → "
                        f"50% size, 2× stop, exit +1d"
                    )
                    break
        except Exception:
            pass

        # ── Position sizing ────────────────────────────────────────────
        stop_dist     = abs(price - stop) * earnings_stop_mult
        if stop_dist <= 0:
            stop_dist = price * 0.02
        effective_risk = (RISK_PER_TRADE * regime_risk_mult
                          * sector_size_mult * earnings_size_mult)
        risk_usd  = self._cash * effective_risk
        shares    = risk_usd / stop_dist
        notional  = shares * price

        max_notional = (self._cash * 0.20 * regime_risk_mult
                        * sector_size_mult * earnings_size_mult)
        if notional > max_notional:
            shares   = max_notional / price
            notional = max_notional

        if notional > self._cash or notional < 10:
            return None

        if regime_risk_mult < 1.0:
            log.info(
                f"REGIME SIZING [{sym}]: {regime} → "
                f"{regime_risk_mult:.0%} risk (${risk_usd:.0f})"
            )

        pos = Position(
            symbol          = sym,
            signal_type     = sig_type,
            entry_price     = price,
            shares          = round(shares, 4),
            notional        = round(notional, 2),
            stop_loss       = round(stop, 4),
            target_price    = round(target, 4) if target else round(price * 1.06, 4),
            conviction      = conviction,
            agent           = signal.get("agent", ""),
            sector          = incoming_sector,
            risk_1r         = round(abs(price - stop), 4),
            current_price   = price,
            near_earnings   = near_earnings,
            earnings_exit_ts= earnings_exit_ts,
        )
        self._positions[pos.pos_id] = pos
        self._cash = round(self._cash - notional, 2)
        self._fire_save(pos)

        log.info(
            f"PAPER TRADE OPENED: {sym} {sig_type} {shares:.1f}sh @ ${price:.4f} "
            f"notional=${notional:.0f} stop=${stop:.4f} "
            f"regime={regime}({regime_risk_mult:.0%}) sector={incoming_sector or '?'}"
        )
        return pos

    # ── Update & close ────────────────────────────────────────────────────

    async def update_positions(self, prices: Dict[str, float]):
        """Update open positions: live P&L, trailing stops, and close checks."""
        changed_stops: List[Position] = []

        for pos in list(self.open_positions):
            price = prices.get(pos.symbol)
            if not price:
                continue

            is_long = "BUY" in pos.signal_type

            pos.current_price = price
            if is_long:
                pos.unrealized_pnl = round((price - pos.entry_price) * pos.shares, 2)
            else:
                pos.unrealized_pnl = round((pos.entry_price - price) * pos.shares, 2)
            pos.unrealized_pct = round(pos.unrealized_pnl / pos.notional * 100, 2)

            if is_long and pos.risk_1r > 0:
                r_multiple = (price - pos.entry_price) / pos.risk_1r
                if r_multiple >= 2.0 and not pos.trail_activated:
                    new_stop = round(price - pos.risk_1r, 4)
                    if new_stop > pos.stop_loss:
                        log.info(
                            f"TRAIL STOP [{pos.symbol}] +{r_multiple:.1f}R "
                            f"stop {pos.stop_loss:.4f} → {new_stop:.4f}"
                        )
                        pos.stop_loss = new_stop
                        pos.trail_activated = True
                        changed_stops.append(pos)
                elif r_multiple >= 2.0 and pos.trail_activated:
                    new_stop = round(price - pos.risk_1r, 4)
                    if new_stop > pos.stop_loss:
                        pos.stop_loss = new_stop
                        changed_stops.append(pos)
                elif r_multiple >= 1.0 and not pos.breakeven_set:
                    log.info(
                        f"BREAKEVEN STOP [{pos.symbol}] +1R → {pos.entry_price:.4f}"
                    )
                    pos.stop_loss = pos.entry_price
                    pos.breakeven_set = True
                    changed_stops.append(pos)

            close_reason = None
            if is_long and price <= pos.stop_loss:
                close_reason = "stop_hit"
            elif pos.target_price > 0 and is_long and price >= pos.target_price:
                close_reason = "target_hit"
            elif pos.earnings_exit_ts > 0 and time.time() >= pos.earnings_exit_ts:
                close_reason = "earnings_exit"
            elif (time.time() - pos.opened_ts) > 5 * 86_400:
                close_reason = "time_exit"

            if close_reason:
                await self._close_position(pos, price, close_reason)

        if changed_stops:
            await asyncio.gather(
                *[self._save_position(p) for p in changed_stops],
                return_exceptions=True,
            )

    async def close_on_sell_signal(self, signal: Dict):
        sym = signal.get("symbol", "")
        if "SELL" not in signal.get("signal_type", ""):
            return
        for pos in self.open_positions:
            if pos.symbol == sym:
                price = signal.get("price_at_signal", pos.entry_price)
                await self._close_position(pos, price, "sell_signal")

    async def _close_position(self, pos: Position, exit_price: float, reason: str):
        pos.status      = "closed"
        pos.closed_ts   = time.time()
        pos.close_price = exit_price
        pos.close_reason= reason
        proceeds    = pos.shares * exit_price
        pos.pnl_usd = round(proceeds - pos.notional, 2)
        pos.pnl_pct = round(pos.pnl_usd / pos.notional * 100, 2)
        self._cash  = round(self._cash + proceeds, 2)

        await self._save_position(pos)
        await self._save_cash()

        await emit("portfolio.closed", {
            "symbol":      pos.symbol,
            "signal_type": pos.signal_type,
            "entry":       pos.entry_price,
            "exit":        exit_price,
            "pnl_usd":     pos.pnl_usd,
            "pnl_pct":     pos.pnl_pct,
            "reason":      reason,
            "held_hours":  round((pos.closed_ts - pos.opened_ts) / 3600, 1),
        }, "portfolio")

        emoji = "WIN" if pos.pnl_usd > 0 else "LOSS"
        log.info(
            f"PAPER TRADE CLOSED [{emoji}]: {pos.symbol} @ ${exit_price:.4f} "
            f"PNL=${pos.pnl_usd:+.2f} ({pos.pnl_pct:+.2f}%) -- {reason}"
        )

    def get_summary(self) -> Dict:
        closed = self.closed_positions
        wins   = [p for p in closed if p.pnl_usd > 0]
        losses = [p for p in closed if p.pnl_usd <= 0]
        total_pnl = sum(p.pnl_usd for p in closed)
        win_rate  = len(wins) / len(closed) * 100 if closed else 0
        open_pos  = self.open_positions
        unrealized_pnl = round(sum(p.unrealized_pnl for p in open_pos), 2)
        return {
            "account_size":         ACCOUNT_SIZE,
            "account_value":        self.account_value,
            "cash":                 self._cash,
            "total_pnl_usd":        round(total_pnl, 2),
            "total_pnl_pct":        round(total_pnl / ACCOUNT_SIZE * 100, 2),
            "unrealized_pnl":       unrealized_pnl,
            "open_count":           len(open_pos),
            "closed_count":         len(closed),
            "win_rate":             round(win_rate, 1),
            "wins":                 len(wins),
            "losses":               len(losses),
            "open_positions":       [asdict(p) for p in open_pos],
            "recent_closed":        [asdict(p) for p in closed[:20]],
            "paper_trading_enabled":self._enabled,
        }

    async def start(self):
        await self._load_from_db()
        q = await bus.subscribe("signal", "watchlist.update")
        log.info("Paper portfolio started (account=${:,.0f})".format(ACCOUNT_SIZE))

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)

                if event.topic == "signal":
                    sig = event.data
                    if (not self._managed_externally
                            and self._enabled
                            and sig.get("conviction", 0) >= self._min_conviction):
                        pos = self.open_position(sig)
                        if pos:
                            await emit("portfolio.opened", asdict(pos), "portfolio")
                    await self.close_on_sell_signal(sig)

                elif event.topic == "watchlist.update":
                    items  = event.data.get("items", [])
                    prices = {i["profile"]["symbol"]: i["profile"]["price"] for i in items}
                    await self.update_positions(prices)

            except asyncio.TimeoutError:
                if self.open_positions:
                    try:
                        from scanners.market_scanner import scanner
                        prices = {
                            sym: item.profile.price
                            for sym, item in scanner.watchlist.items()
                            if any(p.symbol == sym for p in self.open_positions)
                        }
                        if prices:
                            await self.update_positions(prices)
                    except Exception as e:
                        log.debug(f"Portfolio proactive price check: {e}")
            except Exception as e:
                log.debug(f"Portfolio loop error: {e}")


paper_portfolio = PaperPortfolio()
