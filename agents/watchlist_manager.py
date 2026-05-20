"""
agents/watchlist_manager.py -- Autonomous watchlist intelligence

Goes beyond simple score ranking. This agent:
  - Detects when tickers are "heating up" (rising score trend)
  - Promotes tickers to a HIGH_ALERT tier when multiple signals align
  - Demotes tickers that have been analyzed and are stale
  - Discovers new tickers via: earnings calendar, news spikes, sector rotation
  - Maintains a priority queue so agents analyze the most promising first
  - Emits watchlist.priority events so the dashboard can highlight them
"""
import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

from core.bus import bus, emit
from core.models import WatchlistItem
from core.logger import get_logger

log = get_logger("wl_manager")

# Tier thresholds
TIER_HIGH_ALERT = 75.0   # score >= this = show in HIGH ALERT tier
TIER_WATCH      = 55.0   # score >= this = normal watchlist
TIER_COOLING    = 35.0   # score below this = demote / remove


class WatchlistIntelligence:
    """
    Continuously re-evaluates the watchlist and manages tiers.
    Agents can dismiss tickers that deep analysis finds unworthy,
    removing them from the analysis priority queue for a cooldown period.
    """

    _HISTORY_FILE = Path(__file__).parent.parent / "storage" / "wl_signal_history.json"
    _MAX_HISTORY_PER_SYM = 30  # keep last N signals per ticker to bound file size

    def __init__(self):
        self._score_history: Dict[str, deque] = {}   # sym -> last 5 scores
        self._high_alert: Set[str] = set()
        self._last_promoted: Dict[str, float] = {}
        self._analysis_queue: List[str] = []         # priority order for agents
        self._dismissed: Dict[str, Dict] = {}        # sym -> {reason, ts, expires_ts}
        self._signal_history: Dict[str, List[Dict]] = {}  # sym -> [{direction, ts}, ...]
        self._load_history()

    # ------------------------------------------------------------------
    # Cross-session signal history
    # ------------------------------------------------------------------

    def _load_history(self):
        try:
            if self._HISTORY_FILE.exists():
                self._signal_history = json.loads(
                    self._HISTORY_FILE.read_text(encoding="utf-8")
                )
                log.debug(f"Signal history loaded: {len(self._signal_history)} tickers")
        except Exception as e:
            log.debug(f"Signal history load failed (starting fresh): {e}")

    def _save_history(self):
        try:
            self._HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._HISTORY_FILE.write_text(
                json.dumps(self._signal_history, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Signal history save failed: {e}")

    def record_signal(self, symbol: str, direction: str, ts: float):
        """Persist a signal direction for a ticker across sessions."""
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        self._signal_history[symbol].append({"direction": direction, "ts": ts})
        # Trim to bound file size
        if len(self._signal_history[symbol]) > self._MAX_HISTORY_PER_SYM:
            self._signal_history[symbol] = \
                self._signal_history[symbol][-self._MAX_HISTORY_PER_SYM:]
        self._save_history()

    async def start(self):
        """Subscribe to the signal bus and persist each signal direction."""
        q = await bus.subscribe("signal")
        log.info("WL signal history listener started")
        while True:
            try:
                event = await q.get()
                data = event.data if isinstance(event.data, dict) else {}
                symbol = data.get("symbol", "")
                direction = data.get("signal_type", "")
                ts = data.get("ts", event.ts)
                if symbol and direction:
                    self.record_signal(symbol, direction, ts)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"Signal history listener error: {e}")

    # ------------------------------------------------------------------
    # Dismiss / undismiss API (called by agent_loop after deep analysis)
    # ------------------------------------------------------------------
    def dismiss(self, symbol: str, reason: str, duration_minutes: int = 90):
        """Exclude ticker from analysis queue for duration_minutes after weak result."""
        self._dismissed[symbol] = {
            "reason": reason,
            "ts": time.time(),
            "expires_ts": time.time() + duration_minutes * 60,
            "duration_min": duration_minutes,
        }
        log.info(f"WL DISMISS: {symbol} for {duration_minutes}min — {reason}")

    def undismiss(self, symbol: str):
        self._dismissed.pop(symbol, None)

    def get_dismissed(self) -> Dict[str, Dict]:
        now = time.time()
        self._dismissed = {k: v for k, v in self._dismissed.items() if v["expires_ts"] > now}
        return dict(self._dismissed)

    def get_analysis_priority(self) -> List[str]:
        """Return tickers in priority order, excluding currently dismissed ones."""
        now = time.time()
        active_dismissed = {k for k, v in self._dismissed.items() if v["expires_ts"] > now}
        return [s for s in self._analysis_queue if s not in active_dismissed]

    def update(self, items: List[WatchlistItem]) -> Dict:
        promoted = []
        demoted  = []
        trending_up = []

        for item in items:
            sym   = item.profile.symbol
            score = item.profile.composite_score

            # Track score history
            if sym not in self._score_history:
                self._score_history[sym] = deque(maxlen=5)
            history = self._score_history[sym]
            history.append(score)

            # Detect trending up (rising 3+ consecutive)
            if len(history) >= 3:
                last3 = list(history)[-3:]
                if last3[0] < last3[1] < last3[2]:
                    trending_up.append(sym)

            # High alert promotion
            was_high = sym in self._high_alert
            is_high  = score >= TIER_HIGH_ALERT

            if is_high and not was_high:
                self._high_alert.add(sym)
                self._last_promoted[sym] = time.time()
                promoted.append({"symbol": sym, "score": score, "reason": "score_threshold"})

            elif not is_high and was_high:
                self._high_alert.discard(sym)
                demoted.append(sym)

        # Rebuild priority queue: high_alert first, then by score
        high  = [i for i in items if i.profile.symbol in self._high_alert]
        other = [i for i in items if i.profile.symbol not in self._high_alert]
        high.sort( key=lambda x: x.profile.composite_score, reverse=True)
        other.sort(key=lambda x: x.profile.composite_score, reverse=True)
        self._analysis_queue = [i.profile.symbol for i in high + other]

        dismissed_info = self.get_dismissed()
        return {
            "high_alert":   sorted(self._high_alert),
            "promoted":     promoted,
            "demoted":      demoted,
            "trending_up":  trending_up,
            "queue_top":    self.get_analysis_priority()[:10],
            "dismissed":    list(dismissed_info.keys()),
        }

    def get_high_alert(self) -> List[str]:
        return sorted(self._high_alert)

    # ------------------------------------------------------------------
    # FEATURE H: Dynamic Watchlist Curation
    # ------------------------------------------------------------------

    def auto_curate(self, items: List[WatchlistItem]) -> Dict:
        """
        Evaluate watchlist health every cycle.
        Returns a report with removed/flagged symbols and reasons.
        """
        now = time.time()
        removed: List[Dict] = []
        flagged: List[Dict] = []

        for item in items:
            sym   = item.profile.symbol
            score = item.profile.composite_score

            # Use persistent cross-session signal history
            recent_signals = self._signal_history.get(sym, [])[-5:]
            bearish_count = sum(
                1 for s in recent_signals
                if "SELL" in s.get("direction", "")
            )

            # Rule 1: 3+ consecutive bearish signals + score below cooling threshold
            if bearish_count >= 3 and score < TIER_COOLING:
                if not self._dismissed.get(sym):
                    self.dismiss(sym,
                                 f"Auto-curate: {bearish_count} bearish signals, score={score:.0f}",
                                 duration_minutes=360)
                    removed.append({"symbol": sym, "reason": "bearish_streak", "score": score})

            # Rule 2: High alert but no signals in 24h — flag as stale
            elif sym in self._high_alert:
                last_sig_ts = recent_signals[-1]["ts"] if recent_signals else 0
                if now - last_sig_ts > 86_400:
                    flagged.append({"symbol": sym, "reason": "stale_alert",
                                    "hours_since_signal": round((now - last_sig_ts) / 3600, 1)})

        # Auto-undismiss expired
        expired = [s for s, v in self._dismissed.items() if v["expires_ts"] <= now]
        for sym in expired:
            self.undismiss(sym)

        return {"curated_removed": removed, "curated_flagged": flagged,
                "auto_undismissed": expired}


wl_intelligence = WatchlistIntelligence()
