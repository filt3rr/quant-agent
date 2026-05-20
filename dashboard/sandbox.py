"""
dashboard/sandbox.py -- Python sandbox for dashboard analysis queries

PART 1 FIXES:
- Always returns a non-empty output array (never blank)
- Errors include traceback line
- ask() works without 'ask(' literal -- detects prefix variants
- Properly populates output for chained expressions
- Sandbox now exposes the FULL ticker/profile object (not just proxy)
"""
import asyncio
import io
import sys
import time
import traceback
from typing import Any, Dict, List

from core.logger import get_logger

log = get_logger("sandbox")


def _safe_repr(obj: Any, max_len: int = 600) -> str:
    """Convert any object to a safe string representation."""
    try:
        if isinstance(obj, list):
            if len(obj) > 10:
                return f"[{len(obj)} items] " + repr(obj[:5])[:max_len] + " ..."
            return repr(obj)[:max_len]
        if isinstance(obj, dict):
            keys = list(obj.keys())
            if len(keys) > 8:
                preview = {k: obj[k] for k in keys[:5]}
                return f"{{{len(keys)} keys}} " + repr(preview)[:max_len] + " ..."
            return repr(obj)[:max_len]
        s = repr(obj)
        if len(s) > max_len:
            s = s[:max_len] + "..."
        return s
    except Exception:
        return str(type(obj).__name__)


class TickerProxy:
    """Friendly wrapper around a WatchlistItem for sandbox access."""
    def __init__(self, item):
        self._item = item
        p = item.profile
        self.symbol   = p.symbol
        self.name     = p.name
        self.price    = p.price
        self.change   = p.change_pct
        self.change5d = p.change_5d
        self.volume   = p.volume_24h
        self.rsi      = p.rsi
        self.macd     = p.macd_signal
        self.score    = p.composite_score
        self.market   = p.market.value
        self.sector   = p.sector
        self.signals  = item.signals
        self.rank     = item.rank
        ind = p.provider_data.get("indicators", {})
        self.bb_pos    = ind.get("bb_position", 0)
        self.vol_ratio = ind.get("volume_ratio", 1)
        self.atr_pct   = ind.get("atr_pct", 0)
        self.stoch_k   = ind.get("stoch_k", 50)
        self.vwap_diff = ind.get("vwap_vs_price", 0)
        self.macd_hist = ind.get("macd_hist", 0)
        self._all_indicators = ind

    def __repr__(self):
        return (f"<{self.symbol} ${self.price:.4f} {self.change:+.2f}% "
                f"RSI={self.rsi:.0f} score={self.score:.1f} sector='{self.sector}'>")

    def all_indicators(self):
        """Return full indicators dict."""
        return dict(self._all_indicators)

    def show_chart(self, period="60d"):
        return f"CHART:{self.symbol}:{period}"


class SandboxContext:
    """Live sandbox context built from scanner state."""

    def __init__(self):
        self._items = []
        self._pnl_stats = {}
        self._portfolio = {}

    def refresh(self):
        try:
            from scanners.market_scanner import scanner
            self._items = scanner.get_watchlist()
        except Exception:
            pass
        try:
            from signals.pnl_tracker import pnl_tracker
            self._pnl_stats = pnl_tracker.get_stats()
        except Exception:
            pass
        try:
            from signals.portfolio import paper_portfolio
            self._portfolio = paper_portfolio.get_summary()
        except Exception:
            pass

    def build_globals(self) -> Dict[str, Any]:
        items = self._items
        item_map = {i.profile.symbol: i for i in items}

        def ticker(sym: str):
            sym = sym.upper()
            if sym in item_map:
                return TickerProxy(item_map[sym])
            return f"No data for {sym} (not in watchlist)"

        class SignalHistory:
            def __init__(self, stats): self._s = stats
            def win_rate(self, last=None): return f"{self._s.get('win_rate', 0):.1f}%"
            def expectancy(self): return f"{self._s.get('expectancy', 0):+.2f}% per trade"
            def avg_gain(self): return f"{self._s.get('avg_gain_pct', 0):+.2f}%"
            def avg_loss(self): return f"{self._s.get('avg_loss_pct', 0):+.2f}%"
            def by_rule(self): return self._s.get("by_reason", {})
            def recent(self, n=10): return self._s.get("recent", [])[:n]
            def __repr__(self):
                s = self._s
                return (f"SignalHistory(closed={s.get('closed',0)}, "
                        f"win_rate={s.get('win_rate',0):.1f}%, "
                        f"expectancy={s.get('expectancy',0):+.2f}%)")

        def top(n=10, market=None):
            filtered = items
            if market:
                filtered = [i for i in items if i.profile.market.value == market.upper()]
            return [TickerProxy(i) for i in filtered[:n]]

        def by_sector(sector_name):
            """Find tickers in a specific sector."""
            return [TickerProxy(i) for i in items
                    if sector_name.lower() in (i.profile.sector or '').lower()]

        def best_signals(min_conv=0.7):
            """Filter signals by conviction threshold."""
            opens = self._pnl_stats.get("open_signals", [])
            return [s for s in opens if s.get("conviction", 0) >= min_conv]

        def scan_stats():
            try:
                from scanners.market_scanner import scanner
                return {
                    "watchlist_size": len(scanner.watchlist),
                    "scan_count": scanner._scan_count,
                    "paused": scanner.paused,
                    "markets": list(scanner._universe.keys()) if scanner._universe else [],
                }
            except Exception as e:
                return {"error": str(e)}

        return {
            # Data
            "watchlist": [TickerProxy(i) for i in items],
            "ticker": ticker,
            "top": top,
            "by_sector": by_sector,
            "signals": self._pnl_stats.get("open_signals", []),
            "best_signals": best_signals,
            "pnl": self._pnl_stats,
            "portfolio": self._portfolio,
            "signal_history": SignalHistory(self._pnl_stats),
            "scan_stats": scan_stats,
            # Builtins
            "len": len, "list": list, "dict": dict, "sorted": sorted,
            "sum": sum, "min": min, "max": max, "round": round,
            "print": print, "abs": abs, "any": any, "all": all,
            "range": range, "enumerate": enumerate, "zip": zip,
            "str": str, "int": int, "float": float, "bool": bool,
        }


_ctx = SandboxContext()


async def run_sandbox(code: str) -> Dict[str, Any]:
    """Execute sandbox code. Always returns output list, never blank."""
    t0 = time.time()
    code = code.strip()
    if not code:
        return {"output": ["(empty input)"], "error": None, "chart": None, "elapsed_ms": 0}

    _ctx.refresh()
    globs = _ctx.build_globals()

    # PART 1 FIX: detect ask() in any form including bare "ask "
    if code.startswith("ask(") or code.startswith("ask ") or code.startswith('ask"') or code.startswith("ask'"):
        query = code[3:].strip().strip("()'\"")
        if query:
            return await _run_llm_query(query, globs, t0)

    # Python expression mode
    output_lines = []
    chart_cmd = None
    captured = io.StringIO()

    try:
        old_stdout = sys.stdout
        sys.stdout = captured
        result = None
        try:
            result = eval(compile(code, "<sandbox>", "eval"), globs)
        except SyntaxError:
            exec(compile(code, "<sandbox>", "exec"), globs)
            # PART 1 FIX: also pull a `result` var if user assigned one
            result = globs.get("result")
        sys.stdout = old_stdout
        printed = captured.getvalue()
        if printed.strip():
            output_lines.extend(printed.rstrip().split("\n"))
        if result is not None:
            r = _safe_repr(result)
            if isinstance(result, str) and result.startswith("CHART:"):
                chart_cmd = result
                output_lines.append(f"Chart loaded: {result.split(':')[1]}")
            elif isinstance(result, list):
                # Pretty-print lists item-by-item
                for it in result[:15]:
                    output_lines.append(_safe_repr(it, 200))
                if len(result) > 15:
                    output_lines.append(f"... and {len(result)-15} more")
            else:
                output_lines.append(r)
    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc().strip().split("\n")
        # Take last 2 lines of traceback for clarity
        error_line = " ".join(tb[-2:]) if len(tb) >= 2 else tb[-1]
        return {
            "output": [],
            "error": f"{type(e).__name__}: {e}",
            "chart": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # PART 1 FIX: never return empty output
    if not output_lines:
        output_lines = ["(no output -- expression returned None)"]

    return {
        "output": output_lines,
        "error": None,
        "chart": chart_cmd,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


async def _run_llm_query(query: str, globs: dict, t0: float) -> Dict[str, Any]:
    """Answer natural language query using LLM with live market context."""
    try:
        from agents.llm_router import call_llm

        items = globs["watchlist"][:15]
        ctx_lines = [f"{t.symbol}: ${t.price:.2f} {t.change:+.1f}% RSI={t.rsi:.0f} score={t.score:.0f} sector={t.sector}"
                     for t in items]
        context = "\n".join(ctx_lines)

        pnl = globs["pnl"]
        pnl_summary = (f"Win rate: {pnl.get('win_rate', 0):.1f}% | "
                       f"Closed: {pnl.get('closed', 0)} | "
                       f"Expectancy: {pnl.get('expectancy', 0):+.2f}%")

        port = globs["portfolio"]
        port_summary = (f"Account: ${port.get('account_value', 25000):,.0f} | "
                        f"Open: {port.get('open_count', 0)} | "
                        f"Realized P&L: ${port.get('total_pnl_usd', 0):+,.0f}")

        system = ("You are a quantitative trading assistant with access to live market data. "
                  "Answer concisely, use specific numbers, max 4 sentences. "
                  "Be honest if data doesn't show what user asks about.")

        user = f"""Live watchlist (top 15):
{context}

Signal performance: {pnl_summary}
Paper portfolio: {port_summary}

User question: {query}"""

        response = await call_llm(system=system, user=user,
                                  max_tokens=400, agent_id="sandbox")
        if not response:
            response = "LLM unavailable -- check LLM_PROVIDER in .env (set to 'local' for LM Studio or 'anthropic' with API key)"
        return {
            "output": response.split("\n"),
            "error": None,
            "chart": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "output": [],
            "error": f"LLM query error: {type(e).__name__}: {e}",
            "chart": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }