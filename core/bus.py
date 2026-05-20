"""
core/bus.py -- Async in-process event bus (pub/sub)

All agents, scanners, and the dashboard communicate through this bus.
Topics:
  ticks            -> raw price updates
  scan.result      -> scanner output (new candidate tickers)
  watchlist.update -> current ranked watchlist
  signal           -> buy/sell signals with conviction
  agent.activity   -> agent thinking/tool-call log (for live viewer)
  news             -> raw news items
  error            -> system errors / warnings
  heartbeat        -> periodic system health
"""
import asyncio
import time
import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class Event:
    topic: str
    data: Any
    ts: float = field(default_factory=time.time)
    source: str = "system"

    def to_json(self) -> str:
        payload = {"topic": self.topic, "ts": self.ts, "source": self.source, "data": self.data}
        return json.dumps(payload, default=str)


class EventBus:
    """
    Async pub/sub bus. Subscribers receive events via asyncio.Queue.
    Multiple subscribers per topic are supported.
    """

    def __init__(self):
        self._subs: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._history: Dict[str, List[Event]] = defaultdict(list)
        self._history_limit = 200
        self._lock = asyncio.Lock()
        self._stats = defaultdict(int)

    async def subscribe(self, *topics: str, maxsize: int = 500) -> asyncio.Queue:
        """Subscribe to one or more topics, returns a Queue that receives Events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            for topic in topics:
                self._subs[topic].append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue, *topics: str):
        async with self._lock:
            for topic in topics:
                if q in self._subs[topic]:
                    self._subs[topic].remove(q)

    async def publish(self, topic: str, data: Any, source: str = "system") -> int:
        """Publish an event. Returns number of subscribers that received it."""
        event = Event(topic=topic, data=data, source=source, ts=time.time())
        # Save to history
        hist = self._history[topic]
        hist.append(event)
        if len(hist) > self._history_limit:
            hist.pop(0)
        self._stats[topic] += 1
        # Deliver to subscribers (wildcard match on prefix)
        count = 0
        async with self._lock:
            targets = list(self._subs.get(topic, []))
            # wildcard: subscribers on "agent.*" receive "agent.activity" etc.
            for sub_topic, qs in self._subs.items():
                if sub_topic.endswith("*") and topic.startswith(sub_topic[:-1]):
                    targets.extend(qs)
            targets = list(set(targets))
        for q in targets:
            try:
                q.put_nowait(event)
                count += 1
            except asyncio.QueueFull:
                # Drop oldest to keep live
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                    count += 1
                except Exception:
                    pass
        return count

    def get_history(self, topic: str, n: int = 50) -> List[Event]:
        return self._history.get(topic, [])[-n:]

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# Singleton bus instance
bus = EventBus()


async def emit(topic: str, data: Any, source: str = "system"):
    """Convenience shorthand for bus.publish."""
    await bus.publish(topic, data, source)
