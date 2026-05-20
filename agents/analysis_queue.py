"""
agents/analysis_queue.py -- Live analysis pipeline queue.

Tracks every ticker moving through Tier-1 / Tier-2 analysis so the
dashboard can display real-time queue state, elapsed times, and results.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.logger import get_logger

log = get_logger("analysis_queue")

_MAX_RECENT = 60   # completed entries kept in history


@dataclass
class QueueEntry:
    symbol: str
    tier: int                          # 1 or 2
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "queued"             # queued | analyzing | done | error | skipped
    signal: str = ""
    conviction: float = 0.0
    error: str = ""
    source: str = "auto"               # "auto" | "manual"

    def elapsed_s(self) -> int:
        end = self.completed_at or time.time()
        start = self.started_at or self.queued_at
        return max(0, int(end - start))

    def wait_s(self) -> int:
        end = self.started_at or time.time()
        return max(0, int(end - self.queued_at))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "tier": self.tier,
            "status": self.status,
            "signal": self.signal,
            "conviction": round(self.conviction, 3),
            "elapsed_s": self.elapsed_s(),
            "wait_s": self.wait_s(),
            "source": self.source,
            "error": self.error,
            "queued_at": self.queued_at,
        }


class AnalysisQueue:
    """Thread-safe (asyncio) queue with live status tracking.

    T1 and T2 active entries are tracked in separate dicts so a T2 enqueue
    never overwrites a running T1 entry (and vice-versa), eliminating the
    race condition where T1 completion would silently remove the T2 entry.
    """

    def __init__(self, max_size: int = 100):
        self._t1_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._t2_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._t1_active: Dict[str, QueueEntry] = {}   # symbol → T1 entry
        self._t2_active: Dict[str, QueueEntry] = {}   # symbol → T2 entry
        self._recent: List[QueueEntry] = []
        self._enqueued: Dict[str, float] = {}

    # ── Public API ──────────────────────────────────────────────────

    async def enqueue_tier1(self, symbol: str, source: str = "auto") -> bool:
        """Add ticker to Tier-1 queue. Returns False if already queued/active."""
        if symbol in self._t1_active or symbol in self._t2_active:
            return False
        entry = QueueEntry(symbol=symbol, tier=1, source=source)
        self._t1_active[symbol] = entry
        self._enqueued[symbol] = time.time()
        try:
            self._t1_queue.put_nowait(entry)
        except asyncio.QueueFull:
            del self._t1_active[symbol]
            return False
        return True

    async def enqueue_tier2(self, symbol: str, source: str = "auto") -> bool:
        """Add ticker to Tier-2 queue (deep dive).

        Allowed while T1 is still running for the same symbol (auto-promote
        pre-queues T2 before T1 finishes). Blocked only if T2 is already active.
        """
        if symbol in self._t2_active:
            return False
        entry = QueueEntry(symbol=symbol, tier=2, source=source)
        self._t2_active[symbol] = entry
        try:
            self._t2_queue.put_nowait(entry)
        except asyncio.QueueFull:
            del self._t2_active[symbol]
            return False
        return True

    async def get_tier1(self) -> QueueEntry:
        entry = await self._t1_queue.get()
        entry.status = "analyzing"
        entry.started_at = time.time()
        return entry

    async def get_tier2(self) -> QueueEntry:
        entry = await self._t2_queue.get()
        entry.status = "analyzing"
        entry.started_at = time.time()
        return entry

    def done_tier1(self, entry: QueueEntry):
        self._t1_queue.task_done()

    def done_tier2(self, entry: QueueEntry):
        self._t2_queue.task_done()

    def complete(self, entry: QueueEntry, signal: str = "", conviction: float = 0.0):
        entry.status = "done"
        entry.completed_at = time.time()
        entry.signal = signal
        entry.conviction = conviction
        _active = self._t1_active if entry.tier == 1 else self._t2_active
        _active.pop(entry.symbol, None)
        self._recent.insert(0, entry)
        self._recent = self._recent[:_MAX_RECENT]

    def fail(self, entry: QueueEntry, error: str):
        entry.status = "error"
        entry.completed_at = time.time()
        entry.error = error[:120]
        _active = self._t1_active if entry.tier == 1 else self._t2_active
        _active.pop(entry.symbol, None)
        self._recent.insert(0, entry)
        self._recent = self._recent[:_MAX_RECENT]

    def is_queued_or_active(self, symbol: str) -> bool:
        """True if symbol is in T1 or T2 active tracking (blocks feeder re-queue)."""
        return symbol in self._t1_active or symbol in self._t2_active

    def get_status(self) -> dict:
        all_active = list(self._t1_active.values()) + list(self._t2_active.values())
        return {
            "tier1_queue": self._t1_queue.qsize(),
            "tier2_queue": self._t2_queue.qsize(),
            "active": [e.to_dict() for e in all_active],
            "recent": [e.to_dict() for e in self._recent[:20]],
        }

    def get_stats(self) -> dict:
        done = [e for e in self._recent if e.status == "done"]
        errors = [e for e in self._recent if e.status == "error"]
        t1 = [e for e in done if e.tier == 1]
        t2 = [e for e in done if e.tier == 2]
        avg_t1 = int(sum(e.elapsed_s() for e in t1) / max(1, len(t1)))
        avg_t2 = int(sum(e.elapsed_s() for e in t2) / max(1, len(t2)))
        return {
            "total_done": len(done),
            "total_errors": len(errors),
            "tier1_done": len(t1),
            "tier2_done": len(t2),
            "avg_tier1_s": avg_t1,
            "avg_tier2_s": avg_t2,
        }


# Module-level singleton
analysis_queue = AnalysisQueue()
