"""
agents/memory.py -- Persistent agent memory system

Agents learn from past analyses. This module:
  - Remembers which signals worked/failed per ticker
  - Tracks which analysis patterns correlate with outcomes
  - Feeds learned context into future analyses
  - Maintains per-ticker "conviction history" for trend detection
  - Stores LLM-generated insights as structured memory entries

Persists to storage/agent_memory.json
"""
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.bus import bus
from core.logger import get_logger
from config.settings import SYS

log = get_logger("memory")
MEMORY_FILE = SYS.STORAGE_DIR / "agent_memory.json"


@dataclass
class MemoryEntry:
    symbol: str
    entry_type: str      # "signal_outcome"|"pattern"|"insight"|"sector_note"
    content: str
    tags: List[str]
    confidence: float
    ts: float = field(default_factory=time.time)
    outcome: str = ""    # "correct"|"wrong"|"pending"
    related_signal: str = ""


class AgentMemory:
    """
    Structured long-term memory for agents.
    Enables learning from past performance.
    """

    def __init__(self):
        self._entries: List[MemoryEntry] = []
        self._ticker_stats: Dict[str, Dict] = defaultdict(lambda: {
            "analyses": 0, "correct_signals": 0, "wrong_signals": 0,
            "last_signal": "", "last_conviction": 0.0, "patterns": []
        })
        self._sector_insights: Dict[str, str] = {}
        self._load()

    def _load(self):
        try:
            if MEMORY_FILE.exists():
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                for e in data.get("entries", []):
                    self._entries.append(MemoryEntry(**e))
                self._ticker_stats.update(data.get("ticker_stats", {}))
                self._sector_insights.update(data.get("sector_insights", {}))
                self._entries = self._entries[-500:]  # cap at 500
                log.info(f"Agent memory loaded: {len(self._entries)} entries")
        except Exception as e:
            log.debug(f"Memory load: {e}")

    def _save(self):
        try:
            MEMORY_FILE.write_text(json.dumps({
                "entries": [asdict(e) for e in self._entries[-500:]],
                "ticker_stats": dict(self._ticker_stats),
                "sector_insights": self._sector_insights,
                "updated": time.time(),
            }, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug(f"Memory save: {e}")

    def remember(self, symbol: str, entry_type: str, content: str,
                 tags: List[str] = None, confidence: float = 0.5,
                 related_signal: str = "") -> MemoryEntry:
        entry = MemoryEntry(
            symbol=symbol, entry_type=entry_type, content=content,
            tags=tags or [], confidence=confidence,
            related_signal=related_signal,
        )
        self._entries.append(entry)
        self._ticker_stats[symbol]["analyses"] += 1
        if entry_type == "signal_outcome":
            self._ticker_stats[symbol]["last_signal"] = related_signal
        self._save()
        return entry

    def recall(self, symbol: str, entry_type: str = None,
               limit: int = 5) -> List[MemoryEntry]:
        """Retrieve memories for a symbol, most recent first."""
        entries = [e for e in reversed(self._entries)
                   if e.symbol == symbol
                   and (entry_type is None or e.entry_type == entry_type)]
        return entries[:limit]

    def recall_sector(self, sector: str) -> str:
        return self._sector_insights.get(sector, "")

    def record_outcome(self, symbol: str, signal_type: str,
                       outcome: str, pnl_pct: float = 0):
        """Record whether a signal was correct. outcome: correct | wrong | scratch"""
        stats = self._ticker_stats[symbol]
        if outcome == "correct":
            stats["correct_signals"] += 1
        elif outcome == "wrong":
            stats["wrong_signals"] += 1
        # scratch: no stat change — breakeven trades don't count against win rate

        # Update relevant memory entries
        for entry in reversed(self._entries):
            if entry.symbol == symbol and entry.outcome == "pending":
                entry.outcome = outcome
                break
        self._save()

    def update_sector_insight(self, sector: str, insight: str):
        self._sector_insights[sector] = insight
        self._save()

    def get_ticker_context(self, symbol: str) -> Dict:
        """Build context dict to inject into agent prompts."""
        stats = self._ticker_stats.get(symbol, {})
        recent = self.recall(symbol, limit=3)
        total = stats.get("correct_signals", 0) + stats.get("wrong_signals", 0)
        win_rate = (stats.get("correct_signals", 0) / total * 100) if total > 0 else None
        return {
            "past_analyses": stats.get("analyses", 0),
            "signal_win_rate": win_rate,
            "last_signal": stats.get("last_signal", ""),
            "recent_insights": [e.content[:100] for e in recent],
            "patterns": stats.get("patterns", []),
        }

    def get_stats(self) -> Dict:
        total_e = len(self._entries)
        correct = sum(1 for e in self._entries if e.outcome == "correct")
        wrong   = sum(1 for e in self._entries if e.outcome == "wrong")
        scratch = sum(1 for e in self._entries if e.outcome == "scratch")
        return {
            "total_entries": total_e,
            "correct_outcomes": correct,
            "wrong_outcomes": wrong,
            "scratch_outcomes": scratch,
            "tickers_tracked": len(self._ticker_stats),
            "sectors_tracked": len(self._sector_insights),
        }

    # ------------------------------------------------------------------
    # Pattern Memory (Feature B) — store indicator snapshots at signal time
    # ------------------------------------------------------------------

    def record_setup_pattern(self, symbol: str, indicators: Dict,
                              signal: str, conviction: float,
                              t0_score: int = 0, bias: str = "neutral"):
        """Store the indicator state when a signal was emitted."""
        snapshot = {
            "rsi":       round(indicators.get("rsi", 50), 1),
            "macd":      indicators.get("macd_cross", "neutral"),
            "bb":        round(indicators.get("bb_position", 0.5), 2),
            "vol_ratio": round(indicators.get("volume_ratio", 1.0), 1),
            "obv":       indicators.get("obv_trend", ""),
            "atr_pct":   round(indicators.get("atr_pct", 2.0), 1),
            "t0_score":  t0_score,
            "bias":      bias,
        }
        content = (
            f"Signal={signal} conv={conviction:.0%} | "
            f"RSI={snapshot['rsi']} MACD={snapshot['macd']} "
            f"BB={snapshot['bb']} Vol={snapshot['vol_ratio']}x "
            f"T0={t0_score}/3 bias={bias}"
        )
        stats = self._ticker_stats[symbol]
        patterns = stats.get("patterns", [])
        patterns.append({"ts": time.time(), "signal": signal,
                         "conviction": conviction, "snapshot": snapshot,
                         "outcome": "pending"})
        stats["patterns"] = patterns[-20:]  # keep last 20
        self.remember(symbol, "setup_pattern", content,
                      tags=[signal, bias, f"t0={t0_score}"],
                      confidence=conviction, related_signal=signal)

    def get_similar_setups(self, symbol: str, indicators: Dict,
                           t0_score: int = 0) -> str:
        """Return a human-readable summary of historically similar setups."""
        stats = self._ticker_stats.get(symbol, {})
        patterns = stats.get("patterns", [])
        if not patterns:
            return ""

        cur_rsi = indicators.get("rsi", 50)
        cur_macd = indicators.get("macd_cross", "")
        cur_t0 = t0_score

        matches = []
        for p in reversed(patterns):
            s = p.get("snapshot", {})
            rsi_diff = abs(s.get("rsi", 50) - cur_rsi)
            same_macd = s.get("macd", "") == cur_macd
            t0_close = abs(s.get("t0_score", 0) - cur_t0) <= 1
            if rsi_diff <= 10 and (same_macd or t0_close):
                matches.append(p)
            if len(matches) >= 5:
                break

        if not matches:
            return ""

        lines = []
        wins = sum(1 for m in matches if m.get("outcome") == "correct")
        total = len(matches)
        for m in matches[:5]:
            age_h = round((time.time() - m["ts"]) / 3600, 0)
            lines.append(
                f"  • {m['signal']} ({m['conviction']:.0%} conv) "
                f"{age_h:.0f}h ago → {m.get('outcome','pending')}"
            )
        summary = (
            f"Historical similar setups for {symbol} "
            f"({wins}/{total} won, RSI±10 + MACD/T0 match):\n"
            + "\n".join(lines)
        )
        return summary

    def record_pattern_outcome(self, symbol: str, outcome: str):
        """Mark the most recent pending pattern for a symbol as correct/wrong."""
        stats = self._ticker_stats.get(symbol, {})
        patterns = stats.get("patterns", [])
        for p in reversed(patterns):
            if p.get("outcome") == "pending":
                p["outcome"] = outcome
                break
        self._save()

    async def start(self):
        """Subscribe to portfolio outcomes to auto-learn."""
        q = await bus.subscribe("portfolio.closed")
        log.info("Agent memory started")
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=60)
                if event.topic == "portfolio.closed":
                    d = event.data
                    pnl_usd = d.get("pnl_usd", 0)
                    pnl_pct = d.get("pnl_pct", 0)
                    if pnl_usd > 0:
                        outcome = "correct"
                        verb = "won"
                    elif pnl_usd < -0.01:
                        outcome = "wrong"
                        verb = "lost"
                    else:
                        outcome = "scratch"
                        verb = "scratched"
                    self.record_outcome(
                        d.get("symbol", ""), d.get("signal_type", ""),
                        outcome, pnl_pct
                    )
                    insight = (f"Signal {d.get('signal_type','')} on {d.get('symbol','')} "
                               f"{verb} {pnl_pct:+.2f}% via {d.get('reason','')}")
                    self.remember(
                        d.get("symbol", ""), "signal_outcome", insight,
                        tags=[outcome, d.get("signal_type", "")],
                        confidence=abs(pnl_pct) / 10,
                        related_signal=d.get("signal_type", "")
                    )
                    log.info(f"Memory updated: {d.get('symbol','')} -> {outcome}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.debug(f"Memory loop: {e}")


agent_memory = AgentMemory()
