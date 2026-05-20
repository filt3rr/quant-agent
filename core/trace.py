"""
core/trace.py -- Analysis trace system

Every agent layer emits structured trace steps that the dashboard
stores and replays. A trace captures: what the agent received,
what tools it called, what code it ran, what it returned, and
how long each step took.
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import deque

from core.bus import emit
from config.settings import SYS

TRACES_DIR = SYS.STORAGE_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store: symbol -> list of AnalysisTrace (last 20 per ticker)
_store: Dict[str, deque] = {}


@dataclass
class TraceStep:
    """One step inside an agent's analysis."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    layer: str = ""           # "price_structure" | "technicals" | "volume" | etc
    action: str = ""          # "tool_call" | "code_exec" | "llm_call" | "result"
    description: str = ""
    input_data: Any = None
    output_data: Any = None
    code: Optional[str] = None        # Python code written by agent
    code_output: Optional[str] = None # stdout/result of executed code
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    elapsed_ms: int = 0
    ts: float = field(default_factory=time.time)
    status: str = "ok"        # "ok" | "error" | "skip"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisTrace:
    """Complete analysis trace for one ticker, one run."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    started_ts: float = field(default_factory=time.time)
    finished_ts: float = 0.0
    steps: List[TraceStep] = field(default_factory=list)
    layers_run: List[str] = field(default_factory=list)
    final_signal: Optional[str] = None
    final_conviction: float = 0.0
    final_thesis: str = ""
    status: str = "running"   # "running" | "complete" | "error"

    def add_step(self, step: TraceStep):
        self.steps.append(step)

    def finish(self, signal: str, conviction: float, thesis: str):
        self.finished_ts = time.time()
        self.final_signal = signal
        self.final_conviction = conviction
        self.final_thesis = thesis
        self.status = "complete"

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "symbol": self.symbol,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "elapsed_ms": int((self.finished_ts - self.started_ts) * 1000) if self.finished_ts else 0,
            "layers_run": self.layers_run,
            "final_signal": self.final_signal,
            "final_conviction": self.final_conviction,
            "final_thesis": self.final_thesis,
            "status": self.status,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_summary(self) -> Dict:
        """Lightweight summary for watchlist view."""
        return {
            "trace_id": self.trace_id,
            "symbol": self.symbol,
            "started_ts": self.started_ts,
            "elapsed_ms": int((self.finished_ts - self.started_ts) * 1000) if self.finished_ts else 0,
            "layers_run": self.layers_run,
            "final_signal": self.final_signal,
            "final_conviction": self.final_conviction,
            "final_thesis": self.final_thesis[:120],
            "status": self.status,
            "step_count": len(self.steps),
        }


class TraceStore:
    def __init__(self):
        self._traces: Dict[str, deque] = {}  # symbol -> deque of AnalysisTrace
        self._active: Dict[str, AnalysisTrace] = {}  # trace_id -> active trace

    def start_trace(self, symbol: str) -> AnalysisTrace:
        trace = AnalysisTrace(symbol=symbol)
        self._active[trace.trace_id] = trace
        if symbol not in self._traces:
            self._traces[symbol] = deque(maxlen=10)
        return trace

    def get_active(self, trace_id: str) -> Optional[AnalysisTrace]:
        return self._active.get(trace_id)

    async def emit_step(self, trace: AnalysisTrace, step: TraceStep):
        """Add step to trace and broadcast to dashboard."""
        trace.add_step(step)
        await emit("trace.step", {
            "trace_id": trace.trace_id,
            "symbol": trace.symbol,
            "step": step.to_dict(),
        }, f"agent_{trace.symbol}")

    async def finish_trace(self, trace: AnalysisTrace,
                           signal: str, conviction: float, thesis: str):
        trace.finish(signal, conviction, thesis)
        self._traces[trace.symbol].appendleft(trace)
        self._active.pop(trace.trace_id, None)

        # Persist to disk
        try:
            path = TRACES_DIR / f"{trace.symbol}_{trace.trace_id}.json"
            path.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
        except Exception:
            pass

        await emit("trace.complete", trace.to_dict(), f"agent_{trace.symbol}")

    def get_traces(self, symbol: str) -> List[Dict]:
        return [t.to_summary() for t in self._traces.get(symbol, [])]

    def get_full_trace(self, trace_id: str) -> Optional[Dict]:
        for traces in self._traces.values():
            for t in traces:
                if t.trace_id == trace_id:
                    return t.to_dict()
        # Try loading from disk
        for f in TRACES_DIR.glob(f"*_{trace_id}.json"):
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        return None

    def get_all_recent(self, n: int = 30) -> List[Dict]:
        all_traces = []
        for traces in self._traces.values():
            all_traces.extend([t.to_summary() for t in traces])
        all_traces.sort(key=lambda x: x["started_ts"], reverse=True)
        return all_traces[:n]


# Singleton
trace_store = TraceStore()
