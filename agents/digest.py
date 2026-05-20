"""
agents/digest.py -- Pre-market daily digest generator

Checks every 60s. At 8:40-8:55 ET on weekdays, generates a 150-word
LLM brief covering yesterday's P&L, open positions, today's earnings,
and the current macro regime.

The digest is:
  - Stored at storage/digest.json
  - Broadcast via bus topic "digest.generated"
  - Readable at GET /api/digest/latest
  - Manually triggerable via POST /api/digest/generate
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Optional

from core.bus import emit
from core.logger import get_logger
from config.settings import SYS

log = get_logger("digest")
DIGEST_FILE = SYS.STORAGE_DIR / "digest.json"


def _et_hour_minute() -> tuple:
    """Return (hour, minute) in US/Eastern approximation."""
    try:
        import zoneinfo
        import datetime
        tz = zoneinfo.ZoneInfo("America/New_York")
        now = datetime.datetime.now(tz)
        return now.hour, now.minute
    except Exception:
        # Fallback: UTC-4 (EDT)
        import datetime
        utc = datetime.datetime.utcnow()
        et = utc - datetime.timedelta(hours=4)
        return et.hour, et.minute


class DigestAgent:
    def __init__(self):
        self._last_digest: Optional[Dict] = None
        self._last_date: str = ""
        self._load()

    def _load(self):
        try:
            if DIGEST_FILE.exists():
                data = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
                self._last_digest = data
                self._last_date = data.get("date", "")
                log.info(f"Digest loaded: {self._last_date}")
        except Exception:
            pass

    def _save(self, digest: Dict):
        try:
            DIGEST_FILE.write_text(json.dumps(digest, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"Digest save error: {e}")

    def get_latest(self) -> Optional[Dict]:
        return self._last_digest

    async def generate_digest(self) -> Dict:
        log.info("Generating pre-market digest...")
        ctx: Dict = {}

        # Yesterday's closed portfolio positions
        try:
            from signals.portfolio import paper_portfolio
            cutoff = time.time() - 86_400
            yesterday = [p for p in paper_portfolio.closed_positions if p.closed_ts > cutoff]
            ctx["yesterday_trades"] = [
                {"symbol": p.symbol, "signal_type": p.signal_type,
                 "pnl_pct": p.pnl_pct, "close_reason": p.close_reason}
                for p in yesterday[:10]
            ]
        except Exception:
            ctx["yesterday_trades"] = []

        # Current open positions with unrealized P&L
        try:
            from signals.portfolio import paper_portfolio
            from scanners.market_scanner import scanner
            positions = []
            for p in paper_portfolio.open_positions:
                item = scanner.watchlist.get(p.symbol)
                gap = 0.0
                if item and p.entry_price > 0:
                    gap = round((item.profile.price - p.entry_price) / p.entry_price * 100, 2)
                positions.append({
                    "symbol": p.symbol,
                    "unrealized_pct": p.unrealized_pct,
                    "overnight_gap_pct": gap,
                })
            ctx["open_positions"] = positions[:10]
        except Exception:
            ctx["open_positions"] = []

        # Today's earnings
        try:
            from signals.alerts import alerts_manager
            await alerts_manager.refresh_earnings()
            ctx["earnings_today"] = [
                {"symbol": e.get("symbol", ""), "date": e.get("date", "")}
                for e in alerts_manager.get_earnings(2)[:6]
            ]
        except Exception:
            ctx["earnings_today"] = []

        # Macro regime
        try:
            from signals.macro_monitor import macro_monitor
            ctx["macro"] = macro_monitor.get_status()
        except Exception:
            ctx["macro"] = {}

        # Portfolio summary
        try:
            from signals.portfolio import paper_portfolio
            s = paper_portfolio.get_summary()
            ctx["portfolio"] = {
                "account_value": s.get("account_value"),
                "total_pnl_pct": s.get("total_pnl_pct"),
                "open_count":    s.get("open_count"),
                "win_rate":      s.get("win_rate"),
                "cash":          s.get("cash"),
            }
        except Exception:
            ctx["portfolio"] = {}

        # Top signals last 24h
        try:
            from signals.pnl_tracker import pnl_tracker
            stats = pnl_tracker.get_stats()
            ctx["recent_signals"] = stats.get("recent", [])[:5]
        except Exception:
            ctx["recent_signals"] = []

        summary = await self._call_llm(self._build_prompt(ctx))

        digest = {
            "generated_at": time.time(),
            "date": time.strftime("%Y-%m-%d"),
            "time_et": time.strftime("%I:%M %p"),
            "summary": summary,
            "context": ctx,
        }
        self._last_digest = digest
        self._last_date = digest["date"]
        self._save(digest)
        await emit("digest.generated", {"summary": summary, "date": digest["date"],
                                        "generated_at": digest["generated_at"]}, "digest")
        log.info(f"Digest broadcast: {len(summary)} chars")
        return digest

    def _build_prompt(self, ctx: Dict) -> str:
        parts = [
            "You are a quantitative trading assistant. Write a concise 150-word pre-market brief "
            "for a systematic trader. Be specific, direct, and actionable. No bullet points — flowing prose.\n"
        ]

        p = ctx.get("portfolio", {})
        if p:
            parts.append(
                f"PORTFOLIO: ${p.get('account_value', 0):,.0f} account "
                f"({p.get('total_pnl_pct', 0):+.1f}% total P&L), "
                f"{p.get('open_count', 0)} open positions, "
                f"{p.get('win_rate', 0):.0f}% win rate."
            )

        yt = ctx.get("yesterday_trades", [])
        if yt:
            wins = [t for t in yt if t.get("pnl_pct", 0) > 0]
            losses = [t for t in yt if t.get("pnl_pct", 0) <= 0]
            parts.append(
                f"YESTERDAY: {len(wins)} wins, {len(losses)} losses. "
                + " ".join(f"{t['symbol']} {t['pnl_pct']:+.1f}%" for t in yt[:4]) + "."
            )

        ops = ctx.get("open_positions", [])
        if ops:
            parts.append(
                "OPEN: "
                + ", ".join(
                    f"{p['symbol']} ({p.get('unrealized_pct', 0):+.1f}% unrealized"
                    + (f", gap {p['overnight_gap_pct']:+.1f}%" if p.get("overnight_gap_pct") else "")
                    + ")"
                    for p in ops[:5]
                ) + "."
            )

        earnings = ctx.get("earnings_today", [])
        if earnings:
            syms = ", ".join(e["symbol"] for e in earnings)
            parts.append(f"EARNINGS TODAY: {syms} — elevated volatility expected.")

        macro = ctx.get("macro", {})
        if macro:
            parts.append(
                f"MACRO: {macro.get('regime', '?')} regime | "
                f"SPY {macro.get('spy_chg_pct', 0):+.1f}% | "
                f"VIX {macro.get('vix', 20):.1f}."
            )

        parts.append("\nWrite exactly 150 words. No headers. Plain prose. End with one actionable sentence.")
        return "\n".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        try:
            from agents.llm_router import call_llm
            result = await asyncio.wait_for(
                call_llm(
                    system="You are a concise, data-driven pre-market analyst.",
                    user=prompt,
                    max_tokens=300,
                    agent_id="digest",
                    temperature=0.4,
                ),
                timeout=60,
            )
            return (result or "").strip() or "Digest unavailable — LLM did not respond."
        except Exception as e:
            log.warning(f"Digest LLM error: {e}")
            return f"Pre-market digest unavailable at {time.strftime('%H:%M')} ET."

    async def start(self):
        """Fire at 8:40–8:55 ET every weekday, once per calendar day."""
        log.info("Digest agent started")
        while True:
            try:
                today = time.strftime("%Y-%m-%d")
                h, m = _et_hour_minute()
                if h == 8 and 40 <= m <= 55 and today != self._last_date:
                    await self.generate_digest()
            except Exception as e:
                log.error(f"Digest error: {e}")
            await asyncio.sleep(60)


digest_agent = DigestAgent()
