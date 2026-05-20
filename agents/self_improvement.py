"""
agents/self_improvement.py -- Self-Improvement Engine

Continuously analyses performance data and adjusts agent parameters:
  - Rule multipliers:  scale conviction per signal reason based on win rates
  - Layer weights:     reweight synthesis layers based on prediction accuracy
  - Conviction calibration: correct for systematic over/under-confidence
  - Market regime detection: adjust parameters for current market conditions
  - Learned context:   inject performance insights into LLM prompts

All parameters persist to storage/ so learning survives restarts.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger
from config.settings import SYS

log = get_logger("self_improvement")

STORAGE_PARAMS  = SYS.STORAGE_DIR / "learned_params.json"
STORAGE_LAYERS  = SYS.STORAGE_DIR / "layer_outcomes.json"

MIN_SAMPLES      = 3     # minimum closed trades to start any parameter adjustment
MIN_RULE_SAMPLES = 5     # minimum per-rule outcomes before adjusting that rule
_FULL_SAMPLES    = 10    # sample count where learning rate reaches full strength
LEARNING_RATE    = 0.30  # blend speed at full confidence (scaled down with fewer samples)
MULT_MIN         = 0.60  # floor for conviction rule multipliers
MULT_MAX         = 1.40  # ceiling for conviction rule multipliers
UPDATE_INTERVAL  = 1800  # seconds between full update cycles (30 min)
_PREDICTION_CAP  = 1000  # in-memory trace prediction window
_OUTCOME_WINDOW  = 259_200  # 72 hours — time to pair prediction to outcome

# Default synthesis layer weights (v9 pipeline: quick_screen → tech_deep → context → master_synthesis)
_DEFAULT_LAYER_WEIGHTS: Dict[str, float] = {
    "quick_screen":     0.40,
    "tech_deep":        0.35,
    "context":          0.20,
    "master_synthesis": 0.05,
}


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class LearnedParams:
    """All learnable parameters in one JSON-serialisable object."""

    # Per-rule conviction multipliers keyed by SignalReason.value.lower()
    rule_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "technical": 1.0, "fundamental": 1.0, "sentiment": 1.0,
        "news": 1.0, "momentum": 1.0, "volume_spike": 1.0,
        "breakout": 1.0, "multi_agent": 1.0,
    })

    # Synthesis layer weights (renormalised to sum=1)
    layer_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_LAYER_WEIGHTS)
    )

    # Single multiplier to correct systematic over/under-confidence
    conviction_calibration: float = 1.0

    # Per-sector minimum conviction gate  (sector_name -> threshold)
    sector_thresholds: Dict[str, float] = field(default_factory=dict)

    # Regime-specific conviction scale factors
    regime_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "trending_up": 1.05, "trending_down": 0.95,
        "sideways": 0.90, "volatile": 0.85,
    })

    # Per-regime, per-rule multipliers — learned independently per regime
    # Structure: {regime_name: {rule_name: multiplier}}
    regime_rule_multipliers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        r: {"technical": 1.0, "fundamental": 1.0, "sentiment": 1.0,
            "news": 1.0, "momentum": 1.0, "volume_spike": 1.0,
            "breakout": 1.0, "multi_agent": 1.0}
        for r in ("trending_up", "trending_down", "sideways", "volatile")
    })

    # Hot/cold sectors from last 30d signal performance
    hot_sectors:  List[str] = field(default_factory=list)
    cold_sectors: List[str] = field(default_factory=list)

    # Current detected regime (updated each cycle)
    current_regime: str = "unknown"

    # Text injected into master synthesis LLM prompt
    learned_context: str = ""

    # Metadata
    last_updated:            float = 0.0
    update_count:            int   = 0
    total_signals_analyzed:  int   = 0


@dataclass
class _LayerTracePrediction:
    """In-memory record of one completed deep-analysis trace."""
    symbol:       str
    ts:           float
    final_signal: str                  # BUY / SELL / HOLD / STRONG_BUY / …
    layer_biases: Dict[str, str]       # layer_name -> bias
    layer_confs:  Dict[str, float]     # layer_name -> confidence


# ======================================================================
# LAYER OUTCOME STORE
# ======================================================================

class LayerOutcomeStore:
    """
    Links deep-analysis layer predictions to eventual P&L outcomes.
    When a signal resolves as win/loss the store finds the matching trace
    and records whether each layer's bias was directionally correct.
    """

    def __init__(self):
        self._predictions: List[_LayerTracePrediction] = []
        self._outcomes: List[Dict] = []   # [{layer, correct, confidence, ts}]
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        try:
            if STORAGE_LAYERS.exists():
                data = json.loads(STORAGE_LAYERS.read_text())
                self._outcomes = data.get("outcomes", [])
                log.info(f"Layer store: loaded {len(self._outcomes)} outcomes")
        except Exception as e:
            log.debug(f"Layer store load: {e}")

    def _save(self):
        try:
            STORAGE_LAYERS.write_text(
                json.dumps({"outcomes": self._outcomes[-5000:]}, indent=2)
            )
        except Exception as e:
            log.debug(f"Layer store save: {e}")

    # ------------------------------------------------------------------ #
    def record_trace_prediction(self, symbol: str, final_signal: str,
                                layer_results: List[Dict]) -> None:
        """Called when deep analysis completes. Stores layer biases for later matching."""
        pred = _LayerTracePrediction(
            symbol=symbol, ts=time.time(), final_signal=final_signal,
            layer_biases={lr["layer"]: lr["bias"] for lr in layer_results},
            layer_confs={lr["layer"]: lr.get("confidence", 0.5) for lr in layer_results},
        )
        self._predictions.append(pred)
        self._predictions = self._predictions[-_PREDICTION_CAP:]

    def record_outcome(self, symbol: str, signal_type: str, won: bool) -> None:
        """
        Called when a P&L outcome is resolved for a symbol.
        Finds the most recent trace prediction and records per-layer correctness.
        """
        now = time.time()
        matching = [
            p for p in reversed(self._predictions)
            if p.symbol == symbol and (now - p.ts) < _OUTCOME_WINDOW  # within 72 h
        ]
        if not matching:
            return
        pred = matching[0]

        is_buy  = "BUY"  in signal_type
        is_sell = "SELL" in signal_type
        if not (is_buy or is_sell):
            return

        for layer, bias in pred.layer_biases.items():
            conf = pred.layer_confs.get(layer, 0.5)
            if is_buy:
                correct = (bias == "bullish") == won
            else:   # sell
                correct = (bias == "bearish") == won
            self._outcomes.append({
                "layer": layer, "correct": correct,
                "confidence": conf, "ts": now,
            })

        self._save()

    def get_layer_performance(self) -> Dict[str, Dict]:
        """Per-layer accuracy + avg confidence over the last 90 days."""
        cutoff = time.time() - 86_400 * 90
        perf: Dict[str, Dict] = {}
        for o in self._outcomes:
            if o.get("ts", 0) < cutoff:
                continue
            layer = o["layer"]
            if layer not in perf:
                perf[layer] = {"correct": 0, "total": 0, "conf_sum": 0.0}
            perf[layer]["total"] += 1
            if o["correct"]:
                perf[layer]["correct"] += 1
            perf[layer]["conf_sum"] += o.get("confidence", 0.5)

        result = {}
        for layer, d in perf.items():
            t = d["total"]
            result[layer] = {
                "correct": d["correct"],
                "total": t,
                "accuracy": round(d["correct"] / t, 3) if t > 0 else 0.5,
                "avg_confidence": round(d["conf_sum"] / t, 3) if t > 0 else 0.5,
            }
        return result


# ======================================================================
# SELF-IMPROVEMENT ENGINE
# ======================================================================

class SelfImprovementEngine:
    """
    Reads P&L performance data + layer outcomes every 30 minutes and
    adjusts LearnedParams that are consumed by the signal engine and
    deep analysis synthesis layer.
    """

    def __init__(self):
        self.params      = LearnedParams()
        self.layer_store = LayerOutcomeStore()
        self._last_report: Dict = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        try:
            if STORAGE_PARAMS.exists():
                data = json.loads(STORAGE_PARAMS.read_text())
                for k, v in data.items():
                    if hasattr(self.params, k):
                        setattr(self.params, k, v)
                log.info(
                    f"Self-improvement: loaded params (update #{self.params.update_count}, "
                    f"{self.params.total_signals_analyzed} signals analysed)"
                )
        except Exception as e:
            log.debug(f"Self-improvement load: {e}")

    def _save(self):
        try:
            STORAGE_PARAMS.write_text(json.dumps(asdict(self.params), indent=2))
        except Exception as e:
            log.warning(f"Self-improvement save: {e}")

    def get_params(self) -> LearnedParams:
        return self.params

    def _blend(self, current: float, target: float) -> float:
        """Full-strength EMA blend (used when samples >= _FULL_SAMPLES)."""
        return round(current * (1 - LEARNING_RATE) + target * LEARNING_RATE, 4)

    def _blend_graduated(self, current: float, target: float, samples: int) -> float:
        """Sample-weighted EMA blend.

        With few samples the learning rate is scaled down to avoid overfitting
        to noisy early data.  Reaches full LEARNING_RATE at _FULL_SAMPLES.

        samples=1  → lr ≈ 0.095  (31% of full rate)
        samples=3  → lr ≈ 0.164  (55% of full rate)  ← MIN_SAMPLES
        samples=5  → lr ≈ 0.212  (71% of full rate)  ← MIN_RULE_SAMPLES
        samples=10 → lr = 0.300  (100% of full rate) ← _FULL_SAMPLES
        """
        weight = min(1.0, (samples / _FULL_SAMPLES) ** 0.5)
        lr = LEARNING_RATE * weight
        return round(current * (1 - lr) + target * lr, 4)

    # ------------------------------------------------------------------ #
    # 1. RULE MULTIPLIERS
    # ------------------------------------------------------------------ #
    def _analyze_rule_performance(self, by_reason: Dict[str, Dict]) -> Dict:
        """
        Derive conviction multipliers from per-reason win/loss counts.
        Target formula: 1.0 at 50% win rate, 1.4 at 70%+, 0.6 at 30%-.
        """
        updates = {}
        for reason_raw, stats in by_reason.items():
            wins   = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            total  = wins + losses
            if total < MIN_RULE_SAMPLES:
                continue

            win_rate = wins / total
            target = max(MULT_MIN, min(MULT_MAX, win_rate * 2.0))

            key = reason_raw.lower().replace(" ", "_").replace("-", "_")
            current = self.params.rule_multipliers.get(key, 1.0)
            new_val = self._blend_graduated(current, target, total)
            self.params.rule_multipliers[key] = new_val
            updates[key] = {
                "old": round(current, 3), "new": round(new_val, 3),
                "win_rate": round(win_rate, 3), "samples": total,
            }
        return updates

    # ------------------------------------------------------------------ #
    # 2. CONVICTION CALIBRATION
    # ------------------------------------------------------------------ #
    def _analyze_conviction_calibration(self, records) -> float:
        """
        Compare average stated conviction to actual win rate.
        Returns a correction factor (multiply stated conviction by this).
        """
        closed = [r for r in records if r.outcome != "open" and r.conviction > 0]
        if len(closed) < MIN_SAMPLES:
            return self.params.conviction_calibration

        wins = sum(1 for r in closed if r.outcome == "win")
        actual_wr   = wins / len(closed)
        avg_conv    = sum(r.conviction for r in closed) / len(closed)

        if avg_conv > 0:
            cal = actual_wr / avg_conv
        else:
            cal = 1.0

        # Bound to prevent overcorrection
        cal = max(0.70, min(1.10, cal))
        new_val = self._blend_graduated(self.params.conviction_calibration, cal, len(closed))
        self.params.conviction_calibration = round(new_val, 4)
        return new_val

    # ------------------------------------------------------------------ #
    # 3. LAYER WEIGHTS
    # ------------------------------------------------------------------ #
    def _analyze_layer_weights(self) -> Dict:
        """
        Adjust synthesis layer weights based on prediction accuracy.
        More accurate layers get proportionally higher weight.
        All weights are renormalised to sum = 1.0.
        """
        layer_perf = self.layer_store.get_layer_performance()
        updates    = {}
        new_weights = dict(self.params.layer_weights)

        for layer, perf in layer_perf.items():
            if perf["total"] < MIN_SAMPLES:
                continue
            accuracy  = perf["accuracy"]
            # Target weight scales 0.02 … 0.35 proportionally to accuracy
            target_w  = max(0.02, min(0.35, accuracy * 0.30))
            current_w = new_weights.get(layer, 0.10)
            blended_w = self._blend_graduated(current_w, target_w, perf["total"])
            new_weights[layer] = blended_w
            updates[layer] = {
                "old": round(current_w, 4), "new": round(blended_w, 4),
                "accuracy": perf["accuracy"], "samples": perf["total"],
            }

        # Renormalise so weights always sum to 1
        total_w = sum(new_weights.values())
        if total_w > 0:
            new_weights = {k: round(v / total_w, 4) for k, v in new_weights.items()}

        self.params.layer_weights = new_weights
        return updates

    # ------------------------------------------------------------------ #
    # 4. SECTOR THRESHOLDS
    # ------------------------------------------------------------------ #
    def _analyze_sector_performance(self, records) -> Dict:
        """
        Compute per-sector win rates and set minimum conviction thresholds.
        Sectors with low win rates get higher conviction requirements.
        """
        sector_map: Dict[str, Dict] = {}
        for r in records:
            if r.outcome == "open":
                continue
            sector = getattr(r, "sector", "") or "unknown"
            if sector not in sector_map:
                sector_map[sector] = {"wins": 0, "total": 0}
            sector_map[sector]["total"] += 1
            if r.outcome == "win":
                sector_map[sector]["wins"] += 1

        updates = {}
        for sector, d in sector_map.items():
            if d["total"] < MIN_SAMPLES or not sector or sector == "unknown":
                continue
            win_rate = d["wins"] / d["total"]
            # Base threshold 0.70; low-win sectors need higher conviction bar
            # Formula: thresh = 0.70 + (0.55 - win_rate) * 0.40, clamped [0.55, 0.90]
            target = max(0.55, min(0.90, 0.70 + (0.55 - win_rate) * 0.40))
            current = self.params.sector_thresholds.get(sector, 0.70)
            new_val = self._blend_graduated(current, target, d["total"])
            self.params.sector_thresholds[sector] = round(new_val, 4)
            updates[sector] = {
                "old": round(current, 3), "new": round(new_val, 3),
                "win_rate": round(win_rate, 3), "samples": d["total"],
            }
        return updates

    # ------------------------------------------------------------------ #
    # 5. MARKET REGIME DETECTION
    # ------------------------------------------------------------------ #
    def _detect_market_regime(self, records) -> str:
        """
        Classify recent market conditions from signal outcomes.
        trending_up / trending_down / sideways / volatile / unknown
        """
        recent = [
            r for r in records
            if r.outcome != "open" and (time.time() - r.ts) < 86_400 * 7
        ]
        if len(recent) < 5:
            return self.params.current_regime or "unknown"

        buy_records  = [r for r in recent if "BUY"  in r.signal_type]
        sell_records = [r for r in recent if "SELL" in r.signal_type]

        def _wr(lst):
            if not lst: return 0.5
            return sum(1 for r in lst if r.outcome == "win") / len(lst)

        buy_wr  = _wr(buy_records)
        sell_wr = _wr(sell_records)

        if   buy_wr > 0.60 and sell_wr < 0.40:            regime = "trending_up"
        elif sell_wr > 0.60 and buy_wr < 0.40:            regime = "trending_down"
        elif buy_wr > 0.55 and sell_wr > 0.55:            regime = "volatile"
        else:                                               regime = "sideways"

        self.params.current_regime = regime
        return regime

    # ------------------------------------------------------------------ #
    # 5b. PER-REGIME RULE MULTIPLIERS
    # ------------------------------------------------------------------ #
    def _analyze_regime_rule_performance(self, records) -> Dict:
        """
        For each known regime, compute per-rule win rates from signals tagged
        with that regime and blend toward optimal multipliers.
        Only updates when MIN_RULE_SAMPLES outcomes exist per (regime, rule).
        """
        cutoff = time.time() - 86_400 * 30  # last 30 days

        # Build per-regime, per-rule outcome buckets
        buckets: Dict[str, Dict[str, Dict]] = {}
        for r in records:
            if r.outcome == "open" or (r.exit_ts or r.ts) < cutoff:
                continue
            regime = getattr(r, "regime_at_signal", None) or self.params.current_regime
            if not regime or regime == "unknown":
                continue
            rule = (r.reason or "technical").lower().replace("signalreason.", "")
            if regime not in buckets:
                buckets[regime] = {}
            if rule not in buckets[regime]:
                buckets[regime][rule] = {"wins": 0, "total": 0}
            buckets[regime][rule]["total"] += 1
            if r.outcome == "win":
                buckets[regime][rule]["wins"] += 1

        updates: Dict = {}
        for regime, rule_map in buckets.items():
            if regime not in self.params.regime_rule_multipliers:
                self.params.regime_rule_multipliers[regime] = {}
            for rule, d in rule_map.items():
                if d["total"] < MIN_RULE_SAMPLES:
                    continue
                win_rate = d["wins"] / d["total"]
                target = max(MULT_MIN, min(MULT_MAX, 0.5 + win_rate * 1.0))
                current = self.params.regime_rule_multipliers[regime].get(rule, 1.0)
                new_val = round(self._blend(current, target), 4)
                self.params.regime_rule_multipliers[regime][rule] = new_val
                updates.setdefault(regime, {})[rule] = {
                    "old": round(current, 3), "new": new_val,
                    "win_rate": round(win_rate, 3), "samples": d["total"],
                }
        return updates

    # ------------------------------------------------------------------ #
    # 5c. HOT / COLD SECTOR DETECTION
    # ------------------------------------------------------------------ #
    def _analyze_sector_hotness(self, records) -> Dict:
        """
        Classify sectors as hot (>60% WR, ≥10 samples) or cold (<40% WR, ≥10)
        based on last 30 days of closed signals.
        """
        cutoff = time.time() - 86_400 * 30
        sector_map: Dict[str, Dict] = {}
        for r in records:
            if r.outcome == "open" or (r.exit_ts or r.ts) < cutoff:
                continue
            sec = getattr(r, "sector", "") or "unknown"
            if sec == "unknown":
                continue
            if sec not in sector_map:
                sector_map[sec] = {"wins": 0, "total": 0}
            sector_map[sec]["total"] += 1
            if r.outcome == "win":
                sector_map[sec]["wins"] += 1

        hot, cold = [], []
        for sec, d in sector_map.items():
            if d["total"] < MIN_SAMPLES:
                continue
            wr = d["wins"] / d["total"]
            if wr >= 0.60:
                hot.append(sec)
            elif wr <= 0.40:
                cold.append(sec)

        self.params.hot_sectors  = sorted(hot)
        self.params.cold_sectors = sorted(cold)
        return {"hot": hot, "cold": cold}

    def get_hot_sectors(self) -> List[str]:
        return list(self.params.hot_sectors)

    def get_cold_sectors(self) -> List[str]:
        return list(self.params.cold_sectors)

    def get_regime_rule_multiplier(self, regime: str, rule: str) -> float:
        """Return the learned per-regime multiplier for a given rule (default 1.0)."""
        return self.params.regime_rule_multipliers.get(regime, {}).get(rule, 1.0)

    # ------------------------------------------------------------------ #
    # FEATURE F: Self-Narrated Learning Journal
    # ------------------------------------------------------------------ #
    def generate_journal_entry(self) -> str:
        """Build a concise natural-language summary of what the agent has learned."""
        p = self.params
        lines: List[str] = []

        lines.append(
            f"Learning update #{p.update_count} | "
            f"Regime: {p.current_regime} | "
            f"Signals analyzed: {p.total_signals_analyzed}"
        )

        # Best and worst rules
        mults = p.rule_multipliers
        if mults:
            best  = max(mults, key=mults.get)
            worst = min(mults, key=mults.get)
            lines.append(
                f"Most trusted rule: {best} (×{mults[best]:.2f}). "
                f"Least trusted: {worst} (×{mults[worst]:.2f})."
            )

        # Layer weights
        weights = p.layer_weights
        if weights:
            top_layer = max(weights, key=weights.get)
            lines.append(f"Highest-weighted analysis layer: {top_layer} ({weights[top_layer]:.2f}).")

        # Hot / cold sectors
        if p.hot_sectors:
            lines.append(f"Hot sectors: {', '.join(p.hot_sectors[:3])}.")
        if p.cold_sectors:
            lines.append(f"Cold sectors (suppressed): {', '.join(p.cold_sectors[:3])}.")

        # Regime adjustments
        adj = p.regime_adjustments
        if p.current_regime in adj:
            lines.append(
                f"Regime conviction scale: ×{adj[p.current_regime]:.2f} "
                f"({p.current_regime})."
            )

        # Calibration
        if abs(p.conviction_calibration - 1.0) > 0.05:
            direction = "overconfident" if p.conviction_calibration < 1.0 else "underconfident"
            lines.append(
                f"Conviction calibration: ×{p.conviction_calibration:.2f} "
                f"(agent has been {direction})."
            )

        return " ".join(lines)

    # ------------------------------------------------------------------ #
    # 6. LEARNED CONTEXT (text for LLM injection)
    # ------------------------------------------------------------------ #
    def _build_learned_context(self, rule_updates: Dict,
                                layer_updates: Dict, regime: str) -> str:
        parts: List[str] = []

        parts.append(
            f"[SELF-IMPROVEMENT v{self.params.update_count}] "
            f"Regime: {regime} | "
            f"{self.params.total_signals_analyzed} signals analysed"
        )

        mults = self.params.rule_multipliers
        if mults:
            best  = max(mults, key=lambda k: mults[k])
            worst = min(mults, key=lambda k: mults[k])
            if mults[best] > 1.05 or mults[worst] < 0.95:
                parts.append(
                    f"Best rule: {best} ({mults[best]:.2f}x) | "
                    f"Weakest: {worst} ({mults[worst]:.2f}x)"
                )

        cal = self.params.conviction_calibration
        if abs(cal - 1.0) > 0.03:
            direction = "over-confident" if cal < 1.0 else "under-confident"
            parts.append(f"Conviction calibration {cal:.2f} — signals have been {direction}")

        weights = self.params.layer_weights
        if weights:
            top    = max(weights, key=lambda k: weights[k])
            bottom = min(weights, key=lambda k: weights[k])
            parts.append(
                f"Highest-weight layer: {top} ({weights[top]:.2f}) | "
                f"Lowest: {bottom} ({weights[bottom]:.2f})"
            )

        regime_guidance = {
            "trending_up":
                "Market trending up: BUY signals have been more reliable. Lean bullish.",
            "trending_down":
                "Market trending down: SELL signals more reliable. Be cautious with BUYs.",
            "sideways":
                "Sideways market: require higher conviction before acting on signals.",
            "volatile":
                "Volatile market: both directions working. Tighten stops, size smaller.",
        }
        guidance = regime_guidance.get(regime, "")
        if guidance:
            parts.append(guidance)

        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    # MAIN UPDATE CYCLE
    # ------------------------------------------------------------------ #
    async def run_update(self) -> Dict:
        """Full self-improvement cycle. Returns a report dict."""
        report: Dict = {"ts": time.time(), "updates": {}}

        try:
            from signals.pnl_tracker import pnl_tracker
            stats   = pnl_tracker.get_stats()
            records = list(pnl_tracker._records.values())
        except Exception as e:
            log.warning(f"Self-improvement: cannot read pnl_tracker: {e}")
            return report

        total = stats.get("closed", 0)
        self.params.total_signals_analyzed = total
        log.info(f"Self-improvement running update #{self.params.update_count + 1} "
                 f"({total} closed signals)")

        lines: List[str] = []

        # 1. Rule multipliers
        rule_updates = self._analyze_rule_performance(stats.get("by_reason", {}))
        if rule_updates:
            report["updates"]["rule_multipliers"] = rule_updates
            lines.append(f"Rules adjusted: {list(rule_updates.keys())}")

        # 2. Conviction calibration
        cal = self._analyze_conviction_calibration(records)
        report["conviction_calibration"] = cal
        lines.append(f"Calibration: {cal:.3f}")

        # 3. Layer weights
        layer_updates = self._analyze_layer_weights()
        if layer_updates:
            report["updates"]["layer_weights"] = layer_updates
            lines.append(f"Layers adjusted: {list(layer_updates.keys())}")

        # 4. Sector thresholds
        sector_updates = self._analyze_sector_performance(records)
        if sector_updates:
            report["updates"]["sector_thresholds"] = sector_updates
            lines.append(f"Sectors adjusted: {list(sector_updates.keys())}")

        # 5. Market regime
        regime = self._detect_market_regime(records)
        report["regime"] = regime
        lines.append(f"Regime: {regime}")

        # 5b. Per-regime rule multipliers
        regime_rule_updates = self._analyze_regime_rule_performance(records)
        if regime_rule_updates:
            report["updates"]["regime_rule_multipliers"] = regime_rule_updates
            lines.append(f"Regime-rules adjusted: {list(regime_rule_updates.keys())}")

        # 5c. Hot / cold sectors
        sector_hotness = self._analyze_sector_hotness(records)
        report["hot_sectors"]  = sector_hotness.get("hot", [])
        report["cold_sectors"] = sector_hotness.get("cold", [])
        if sector_hotness["hot"] or sector_hotness["cold"]:
            lines.append(f"Hot sectors: {sector_hotness['hot']} | Cold: {sector_hotness['cold']}")

        # 6. Learned context for LLM
        context = self._build_learned_context(rule_updates, layer_updates, regime)
        self.params.learned_context = context
        report["learned_context"] = context

        # Persist
        self.params.last_updated = time.time()
        self.params.update_count += 1
        self._save()

        report["params"]  = asdict(self.params)
        report["summary"] = " | ".join(lines)
        self._last_report = report

        log.info(f"Self-improvement #{self.params.update_count} done: {report['summary']}")
        return report

    def get_report(self) -> Dict:
        """Return current params + layer performance as a dashboard-friendly dict."""
        return {
            "params":            asdict(self.params),
            "layer_performance": self.layer_store.get_layer_performance(),
            "summary":           self._last_report.get("summary", "No updates yet"),
            "last_updated":      self.params.last_updated,
        }

    async def start(self):
        """Background loop: run update every UPDATE_INTERVAL seconds."""
        log.info("Self-improvement engine starting...")
        await asyncio.sleep(120)   # let PnL tracker and signal engine warm up first
        while True:
            try:
                await self.run_update()
            except Exception as e:
                log.error(f"Self-improvement cycle error: {e}")
            await asyncio.sleep(UPDATE_INTERVAL)


# Singleton consumed by signal_engine and deep_analysis
self_improvement = SelfImprovementEngine()
