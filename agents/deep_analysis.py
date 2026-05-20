"""
agents/deep_analysis.py -- 2-Tier AI Analysis Pipeline (v9)

Tier 1 (Quick Screen):  1 LLM call  — all data in one prompt, ~60s target
Tier 2 (Deep Dive):     3 LLM calls — tech deep + context + synthesis, ~3min target

Replaces the original 8-layer × 1-LLM-call-per-layer design that was timing out
with slow local models.  Output format (LayerResult / synthesis dict) is unchanged
so downstream consumers (self_improvement, portfolio, memory) are unaffected.

Config: agents/analysis_config.py
Queue:  agents/analysis_queue.py
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from agents.llm_router import call_llm
from agents.code_executor import run_agent_code_with_retry
from core.trace import TraceStore, AnalysisTrace, TraceStep, trace_store
from core.bus import emit
from core.logger import get_logger
from core.models import WatchlistItem, TickerProfile, Market
from scanners.technicals import fetch_ohlcv_yf, compute_indicators, score_technicals
from providers.registry import registry
from config.settings import SYS

log = get_logger("deep_analysis")


# ── Shared data structures (backwards-compatible) ──────────────────────────

@dataclass
class LayerResult:
    layer: str
    signal_bias: str     # "bullish" | "bearish" | "neutral"
    confidence: float    # 0-1
    summary: str
    key_findings: List[str] = field(default_factory=list)
    code_used: bool = False
    data: Dict = field(default_factory=dict)


class BaseLayer:
    layer_name: str = "base"
    layer_num: int = 0

    def __init__(self, trace: AnalysisTrace):
        self.trace = trace

    async def _step(self, action: str, description: str, **kwargs) -> TraceStep:
        step = TraceStep(
            layer=self.layer_name,
            action=action,
            description=description,
            **{k: v for k, v in kwargs.items() if k in TraceStep.__dataclass_fields__}
        )
        await trace_store.emit_step(self.trace, step)
        return step

    async def _llm(self, system: str, user: str, description: str,
                   max_tokens: int = 400) -> Optional[str]:
        t0 = time.time()
        await self._step("llm_call", description, llm_prompt=user[:400])
        response = await call_llm(system=system, user=user, max_tokens=max_tokens,
                                  agent_id=f"{self.layer_name}_{self.trace.symbol}")
        elapsed = int((time.time() - t0) * 1000)
        if response:
            await trace_store.emit_step(self.trace, TraceStep(
                layer=self.layer_name, action="llm_response",
                description=f"LLM responded ({elapsed}ms)",
                llm_response=response, elapsed_ms=elapsed,
            ))
        return response

    async def _code(self, code: str, description: str,
                    df: Optional[pd.DataFrame], indicators: Dict,
                    profile: Dict) -> Dict:
        await trace_store.emit_step(self.trace, TraceStep(
            layer=self.layer_name, action="code_exec",
            description=description, code=code,
        ))
        result = await run_agent_code_with_retry(
            code, df, indicators, profile,
            self.trace.symbol, self.layer_name
        )
        await trace_store.emit_step(self.trace, TraceStep(
            layer=self.layer_name, action="code_result",
            description=f"Code executed in {result['elapsed_ms']}ms",
            code=code,
            code_output=result.get("output", ""),
            status="ok" if result["success"] else "error",
            elapsed_ms=result["elapsed_ms"],
        ))
        return result

    async def run(self, item: WatchlistItem,
                  df: Optional[pd.DataFrame], indicators: Dict) -> LayerResult:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# SHARED DATA COLLECTOR (pure Python, no LLM, ~2s)
# ══════════════════════════════════════════════════════════════════════════════

_COLLECT_CODE = """
import numpy as np
import json

close = df['close'].values if df is not None and len(df) > 0 else []
high  = df['high'].values  if df is not None and len(df) > 0 else []
low   = df['low'].values   if df is not None and len(df) > 0 else []
vol   = df['volume'].values if df is not None and len(df) > 0 else []

out = {}

if len(close) >= 20:
    # Price structure
    sma20 = float(np.mean(close[-20:]))
    sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else sma20
    current = float(close[-1])
    chg_1d = float((close[-1]-close[-2])/close[-2]*100) if len(close)>=2 else 0
    chg_5d = float((close[-1]-close[-6])/close[-6]*100) if len(close)>=6 else 0
    chg_20d= float((close[-1]-close[-21])/close[-21]*100) if len(close)>=21 else 0
    above20 = bool(current > sma20)
    above50 = bool(current > sma50)
    high20  = float(max(high[-21:-1])) if len(high)>=21 else current
    breakout = bool(current > high20)

    # OBV
    obv=[0]
    for i in range(1,len(close)):
        obv.append(obv[-1]+vol[i] if close[i]>close[i-1] else (obv[-1]-vol[i] if close[i]<close[i-1] else obv[-1]))
    obv_trend = 'rising' if obv[-1]>obv[-5] else 'falling'

    # Volume
    avg20v = float(np.mean(vol[-20:])) if len(vol)>=20 else 1
    vspike = float(vol[-1]/avg20v) if avg20v>0 else 1.0
    price_up5 = len(close)>=6 and close[-1]>close[-6]
    vol_up5   = len(vol)>=20 and float(np.mean(vol[-5:])) > float(np.mean(vol[-20:-5]))
    if price_up5 and not vol_up5:
        vol_div = 'distribution_warning'
    elif not price_up5 and vol_up5:
        vol_div = 'accumulation'
    else:
        vol_div = 'confirmed'

    # Risk
    atr_v = float(indicators.get('atr', current*0.02))
    atr_p = float(indicators.get('atr_pct', 2.0))
    stop  = round(current - atr_v*1.5, 4)
    tgt2r = round(current + atr_v*3.0, 4)
    mkt_cap = float(profile.get('market_cap', 0))
    red_flags = []
    if atr_p > 6: red_flags.append(f'High volatility ATR {atr_p:.1f}%')
    if vspike > 8: red_flags.append(f'Extreme volume {vspike:.0f}x')
    if 0 < mkt_cap < 50_000_000: red_flags.append('Micro-cap liquidity risk')
    if abs(chg_5d) > 50: red_flags.append(f'Extreme 5D move {chg_5d:+.0f}%')
    risk_level = 'EXTREME' if len(red_flags)>=3 else ('HIGH' if len(red_flags)>=2 else ('MEDIUM' if red_flags else 'LOW'))

    out = {
        'current': round(current, 4),
        'sma20': round(sma20, 4), 'sma50': round(sma50, 4),
        'above_sma20': above20, 'above_sma50': above50,
        'breakout_20d': breakout, 'high_20d': round(high20, 4),
        'chg_1d': round(chg_1d,2), 'chg_5d': round(chg_5d,2), 'chg_20d': round(chg_20d,2),
        'obv_trend': obv_trend, 'vol_spike': round(vspike,2),
        'vol_divergence': vol_div, 'avg_vol_20d': round(avg20v,0),
        'stop_loss': stop, 'target_2r': tgt2r,
        'red_flags': red_flags, 'risk_level': risk_level,
        'atr_pct': round(atr_p,2),
    }
else:
    out = {'error': 'insufficient_data'}
result = out
"""


async def _collect_data(item: WatchlistItem,
                        df: Optional[pd.DataFrame],
                        indicators: Dict,
                        trace: AnalysisTrace) -> Dict:
    """Run pure-Python data collection — no LLM needed."""
    import ast
    result = await run_agent_code_with_retry(
        _COLLECT_CODE, df, indicators, item.profile.to_dict(),
        item.profile.symbol, "data_collect"
    )
    if result["success"]:
        raw = result.get("result") or ""
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return parsed
                log.warning(f"[{item.profile.symbol}] _collect_data: unexpected type {type(parsed)}")
            except (ValueError, SyntaxError) as e:
                log.warning(f"[{item.profile.symbol}] _collect_data parse error: {e} — raw={raw[:80]}")
    elif result.get("error"):
        log.warning(f"[{item.profile.symbol}] _collect_data executor error: {result['error'][:120]}")
    return {}


def _extract_bias_confidence(text: str) -> tuple:
    """Parse BIAS/CONFIDENCE from LLM response."""
    bias = "neutral"
    confidence = 0.5
    if not text:
        return bias, confidence
    txt = text.lower()
    m = re.search(r"bias\s*[:\-]?\s*(bullish|bearish|neutral)", txt)
    if m:
        bias = m.group(1)
    elif txt.count("bullish") > txt.count("bearish"):
        bias = "bullish"
    elif txt.count("bearish") > txt.count("bullish"):
        bias = "bearish"
    m = re.search(r"confidence\s*[:\-]?\s*\[?\s*(0?\.\d+|1\.0|1)", txt)
    if m:
        try:
            confidence = max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            pass
    return bias, confidence


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — QUICK SCREEN  (1 LLM call, ~60s target)
# ══════════════════════════════════════════════════════════════════════════════

class QuickScreenLayer(BaseLayer):
    """Single LLM call combining all technical + volume + risk data."""
    layer_name = "quick_screen"
    layer_num = 1

    async def run(self, item: WatchlistItem, df, indicators) -> LayerResult:
        sym = item.profile.symbol
        p = item.profile
        await self._step("analyzing", f"Tier 1 quick screen: {sym}")

        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        # Pull computed data (fast, no LLM)
        data = await _collect_data(item, df, indicators, self.trace)

        rsi  = indicators.get("rsi", 50)
        macd = indicators.get("macd_cross", "neutral")
        bb   = indicators.get("bb_position", 0.5)
        vr   = indicators.get("volume_ratio", 1.0)
        roc5 = indicators.get("roc5", 0)

        # T0 multi-timeframe alignment summary
        t0_lines = []
        if p.h4_aligned is not None:
            t0_lines.append(f"4H-trend={'aligned' if p.h4_aligned else 'opposed'}")
        if p.h1_vwap_confirmed is not None:
            t0_lines.append(f"1H-VWAP={'confirmed' if p.h1_vwap_confirmed else 'rejected'}")
        if p.volume_expanding is not None:
            t0_lines.append(f"Vol-expansion={'yes' if p.volume_expanding else 'no'}")
        t0_str = " | ".join(t0_lines) if t0_lines else "pending"
        t0_score_str = f"{p.t0_score}/3" if t0_lines else "n/a"

        prompt = f"""Analyze {sym} ({p.market.value}) for a trade setup.

PRICE: ${p.price:.4f} | 1D: {data.get('chg_1d',0):+.2f}% | 5D: {data.get('chg_5d',0):+.2f}% | 20D: {data.get('chg_20d',0):+.2f}%
TREND: Above SMA20={data.get('above_sma20','?')} | Above SMA50={data.get('above_sma50','?')} | Breakout 20D={data.get('breakout_20d','?')}
TECHNICALS: RSI={rsi:.0f} | MACD={macd} | BB={bb:.2f} | ROC5={roc5:+.1f}%
VOLUME: {vr:.1f}x avg | Spike={data.get('vol_spike',1):.1f}x | Signal={data.get('vol_divergence','?')} | OBV={data.get('obv_trend','?')}
RISK: Level={data.get('risk_level','?')} | ATR%={data.get('atr_pct',2):.1f}% | Stop=${data.get('stop_loss',0):.4f} | Target2R=${data.get('target_2r',0):.4f}
MULTI-TIMEFRAME (T0 score {t0_score_str}): {t0_str}
FLAGS: {', '.join(data.get('red_flags',[])) or 'none'}
WATCHLIST SCORE: {p.composite_score:.0f}/100

In 2-3 sentences: What is the dominant signal? Is this setup actionable?
Note: A T0 score of 2-3 indicates strong intraday confirmation; 0-1 suggests the setup may be deteriorating on lower timeframes.
End with EXACTLY:
BIAS: <bullish|bearish|neutral>
CONFIDENCE: <0.0-1.0>"""

        response = await self._llm(
            "You are a concise quantitative trader. Analyze quickly and decisively.",
            prompt, f"Quick screen LLM for {sym}",
            max_tokens=cfg.tier1_max_tokens
        )

        bias, conf = _extract_bias_confidence(response)

        findings = [
            f"RSI {rsi:.0f} | MACD {macd} | Vol {vr:.1f}x",
            f"Trend: {'above' if data.get('above_sma20') else 'below'} SMA20/50",
            f"Risk: {data.get('risk_level','?')} | {data.get('vol_divergence','?')}",
        ]
        if t0_lines:
            findings.append(f"T0 ({t0_score_str}): {t0_str}")
        if data.get("red_flags"):
            findings.append(f"Flags: {'; '.join(data['red_flags'])}")

        return LayerResult(
            layer="quick_screen", signal_bias=bias, confidence=conf,
            summary=response or f"Quick screen: {bias}",
            key_findings=findings, code_used=True,
            data={**data, "rsi": rsi, "macd": macd, "vr": vr,
                  "llm_prompt": prompt, "llm_response": response or ""}
        )


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — DEEP DIVE  (3 LLM calls: tech + context + synthesis)
# ══════════════════════════════════════════════════════════════════════════════

class TechDeepLayer(BaseLayer):
    """Detailed technical analysis — 1 LLM call with richer prompt."""
    layer_name = "tech_deep"
    layer_num = 2

    async def run(self, item: WatchlistItem, df, indicators,
                  precomputed: Optional[Dict] = None) -> LayerResult:
        sym = item.profile.symbol
        p = item.profile
        await self._step("analyzing", f"Tier 2 tech deep dive: {sym}")

        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        data = precomputed or await _collect_data(item, df, indicators, self.trace)

        rsi  = indicators.get("rsi", 50)
        macd = indicators.get("macd_cross", "neutral")
        bb   = indicators.get("bb_position", 0.5)
        stoch= indicators.get("stoch_k", 50)
        vr   = indicators.get("volume_ratio", 1.0)
        atr  = indicators.get("atr_pct", 2.0)
        roc5 = indicators.get("roc5", 0)

        # Relative strength vs watchlist peers (no LLM, just code)
        rs_note = ""
        try:
            from scanners.market_scanner import scanner
            wl = scanner.get_watchlist()
            sector = p.sector or ""
            peers = [i for i in wl if i.profile.symbol != sym
                     and i.profile.market == p.market
                     and (not sector or i.profile.sector == sector)][:5]
            if peers:
                avg1d = sum(i.profile.change_pct for i in peers) / len(peers)
                avg5d = sum(i.profile.change_5d for i in peers) / len(peers)
                rs1d = p.change_pct - avg1d
                rs5d = p.change_5d - avg5d
                rs_note = f"RS vs {len(peers)} peers: 1D={rs1d:+.2f}% 5D={rs5d:+.2f}%"
        except Exception:
            pass

        prompt = f"""Deep technical analysis for {sym} ({p.market.value}).

PRICE ACTION:
  Current=${p.price:.4f} | 1D={data.get('chg_1d',0):+.2f}% | 5D={data.get('chg_5d',0):+.2f}% | 20D={data.get('chg_20d',0):+.2f}%
  Above SMA20={data.get('above_sma20','?')} | Above SMA50={data.get('above_sma50','?')}
  Breakout above 20D high={data.get('breakout_20d','?')} (20D high=${data.get('high_20d',0):.4f})

INDICATORS:
  RSI={rsi:.0f} | MACD={macd} | BB position={bb:.2f} | Stoch-K={stoch:.0f}
  ATR%={atr:.2f}% | ROC5={roc5:+.2f}% | Volume ratio={vr:.1f}x

VOLUME PROFILE:
  OBV={data.get('obv_trend','?')} | Spike={data.get('vol_spike',1):.1f}x | Signal={data.get('vol_divergence','?')}
  {rs_note}

RISK:
  Level={data.get('risk_level','?')} | Stop=${data.get('stop_loss',0):.4f} | Target 2:1=${data.get('target_2r',0):.4f}
  Flags: {', '.join(data.get('red_flags',[])) or 'none'}

Provide:
1. The dominant technical pattern or setup in 2 sentences
2. Key confirmation levels or conditions for entry
3. Main technical risk

End with EXACTLY:
BIAS: <bullish|bearish|neutral>
CONFIDENCE: <0.0-1.0>"""

        response = await self._llm(
            "You are a senior technical analyst. Be specific about levels, patterns, and signals.",
            prompt, f"Tech deep dive LLM for {sym}",
            max_tokens=cfg.tier2_max_tokens
        )

        bias, conf = _extract_bias_confidence(response)
        findings = [
            f"RSI {rsi:.0f} | MACD {macd} | Stoch {stoch:.0f}",
            f"BB pos: {bb:.2f} | ATR: {atr:.2f}% | OBV: {data.get('obv_trend','?')}",
            f"Volume: {vr:.1f}x | {data.get('vol_divergence','?')}",
            f"Risk: {data.get('risk_level','?')}",
        ]
        if rs_note:
            findings.append(rs_note)

        return LayerResult(
            layer="tech_deep", signal_bias=bias, confidence=conf,
            summary=response or "Technical deep dive",
            key_findings=findings, code_used=True,
            data={**data, "rs_note": rs_note,
                  "llm_prompt": prompt, "llm_response": response or ""}
        )


class ContextLayer(BaseLayer):
    """News + fundamentals context — 1 LLM call."""
    layer_name = "context"
    layer_num = 3

    async def run(self, item: WatchlistItem, df, indicators) -> LayerResult:
        sym = item.profile.symbol
        p = item.profile
        await self._step("analyzing", f"Tier 2 context layer: {sym}")

        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        # Collect news headlines
        headlines = []
        try:
            from providers.registry import registry
            fh = registry.get("finnhub")
            if fh:
                news = await asyncio.wait_for(fh.get_news(sym), timeout=8)
                if news:
                    headlines = [n.get("headline", "")[:80] for n in news[:5] if n.get("headline")]
        except Exception:
            pass

        # Fundamentals summary
        fund_lines = []
        if p.pe_ratio and p.pe_ratio > 0:
            fund_lines.append(f"P/E: {p.pe_ratio:.1f}")
        if p.market_cap and p.market_cap > 0:
            mc = p.market_cap
            mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
            fund_lines.append(f"Mkt Cap: {mc_str}")
        if p.sector:
            fund_lines.append(f"Sector: {p.sector}")
        if getattr(p, 'analyst_rating', None):
            fund_lines.append(f"Analyst: {p.analyst_rating}")

        # Memory context
        mem_note = ""
        try:
            from agents.memory import agent_memory
            ctx = agent_memory.get_ticker_context(sym)
            if ctx and ctx.get("past_analyses", 0) > 0:
                wr = ctx.get("signal_win_rate")
                mem_note = (f"Agent memory: {ctx['past_analyses']} prior analyses, "
                            f"win rate {wr:.0f}%" if wr else f"{ctx['past_analyses']} analyses")
        except Exception:
            pass

        news_block = "\n".join(f"  • {h}" for h in headlines) or "  (no recent news)"
        fund_block = " | ".join(fund_lines) or "No fundamental data"

        prompt = f"""Context analysis for {sym}:

FUNDAMENTALS: {fund_block}
RECENT NEWS:
{news_block}
{f"AGENT MEMORY: {mem_note}" if mem_note else ""}

Based on this context:
1. Is there a catalyst driving the current price action?
2. Does the fundamental/news picture support or contradict the technical setup?
3. What is the key risk from a context perspective?

End with EXACTLY:
BIAS: <bullish|bearish|neutral>
CONFIDENCE: <0.0-1.0>"""

        response = await self._llm(
            "You are a fundamental and news analyst. Connect macro context to price action.",
            prompt, f"Context LLM for {sym}",
            max_tokens=cfg.tier2_max_tokens
        )

        bias, conf = _extract_bias_confidence(response)

        findings = []
        if headlines:
            findings.append(f"News: {headlines[0][:60]}")
        findings += fund_lines[:3]
        if mem_note:
            findings.append(mem_note)

        return LayerResult(
            layer="context", signal_bias=bias, confidence=conf,
            summary=response or "Context analysis",
            key_findings=findings, code_used=False,
            data={"headlines": headlines, "fundamentals": fund_lines,
                  "llm_prompt": prompt, "llm_response": response or ""}
        )


# ══════════════════════════════════════════════════════════════════════════════
# MASTER SYNTHESIS (works for both Tier 1 and Tier 2)
# ══════════════════════════════════════════════════════════════════════════════

class MasterSynthesisLayer(BaseLayer):
    layer_name = "master_synthesis"
    layer_num = 9

    async def run_synthesis(self, item: WatchlistItem,
                            layer_results: List[LayerResult]) -> Dict:
        sym = item.profile.symbol
        p = item.profile
        await self._step("analyzing",
                         f"Synthesis: weighing {len(layer_results)} layers for {sym}")

        # Use learned layer weights when available
        try:
            from agents.self_improvement import self_improvement as _si
            weights = dict(_si.get_params().layer_weights)
        except Exception:
            weights = {
                "quick_screen": 0.40, "tech_deep": 0.35, "context": 0.25,
                # legacy names kept for backwards compat
                "price_structure": 0.15, "technicals": 0.20, "volume_profile": 0.15,
                "fundamentals": 0.15, "news_sentiment": 0.15, "sector_peers": 0.10,
                "risk": 0.05, "code_analysis": 0.05,
            }

        bias_scores = {"bullish": 1, "neutral": 0, "bearish": -1}
        weighted_sum = 0.0
        total_weight = 0.0
        votes = {"bullish": 0, "bearish": 0, "neutral": 0}
        layer_lines = []

        for lr in layer_results:
            w = weights.get(lr.layer, 0.10)
            bs = bias_scores.get(lr.signal_bias, 0)
            weighted_sum += bs * lr.confidence * w
            total_weight += w
            votes[lr.signal_bias] += 1
            layer_lines.append(
                f"[{lr.layer.upper()}] {lr.signal_bias.upper()} "
                f"({lr.confidence:.0%}): {lr.summary[:120]}"
            )

        net = weighted_sum / total_weight if total_weight > 0 else 0.0

        if net > 0.35:   signal = "STRONG_BUY"
        elif net > 0.10: signal = "BUY"
        elif net < -0.35:signal = "STRONG_SELL"
        elif net < -0.10:signal = "SELL"
        else:            signal = "HOLD"

        conviction = min(0.95, abs(net) + 0.30)
        vote_str = f"{votes['bullish']}B / {votes['neutral']}N / {votes['bearish']}Br"

        # ── Self-improvement context blocks ──────────────────────────
        learned_block = ""
        layer_perf_block = ""
        try:
            from agents.self_improvement import self_improvement as _si
            _params = _si.get_params()

            # General learned context (regime, rule performance, calibration)
            if _params.learned_context:
                learned_block = f"\n{_params.learned_context}\n"

            # Per-layer historical accuracy — tell LLM which sources to trust
            perf = _si.layer_store.get_layer_performance()
            perf_lines = []
            for layer_name, d in sorted(perf.items(),
                                        key=lambda x: x[1].get("accuracy", 0),
                                        reverse=True):
                if d.get("total", 0) >= 5:
                    acc = d["accuracy"]
                    n   = d["total"]
                    w   = _params.layer_weights.get(layer_name, 0.0)
                    perf_lines.append(
                        f"  {layer_name}: {acc:.0%} accurate over {n} outcomes "
                        f"(current weight {w:.2f})"
                    )
            if perf_lines:
                layer_perf_block = (
                    "\nLayer reliability (weight your interpretation accordingly):\n"
                    + "\n".join(perf_lines) + "\n"
                )
        except Exception:
            pass

        # ── Pattern Memory (Feature B) ────────────────────────────────
        pattern_block = ""
        try:
            from agents.memory import agent_memory as _am
            _ind = {}
            if layer_results:
                _d0 = layer_results[0].data
                _ind = {
                    "rsi":         _d0.get("rsi", 50),
                    "macd_cross":  _d0.get("macd", "neutral"),
                    "bb_position": _d0.get("bb", 0.5),
                    "volume_ratio":_d0.get("vr", 1.0),
                    "atr_pct":     _d0.get("atr_pct", 2.0),
                }
            t0 = p.t0_score or 0
            pattern_summary = _am.get_similar_setups(sym, _ind, t0_score=t0)
            if pattern_summary:
                pattern_block = f"\n{pattern_summary}\n"
            # Record this setup for future learning
            _am.record_setup_pattern(sym, _ind, signal, conviction, t0_score=t0,
                                     bias=votes.get("bullish", 0) > votes.get("bearish", 0)
                                          and "bullish" or "bearish")
        except Exception:
            pass

        summary_text = "\n".join(layer_lines)
        await self._step("reasoning",
                         f"Votes: {vote_str} | Net: {net:+.3f} | Signal: {signal}",
                         input_data={"votes": votes, "net_score": round(net, 3)})

        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        prompt = f"""Master analyst synthesis for {sym}:

{summary_text}
{layer_perf_block}{learned_block}{pattern_block}
Vote tally: {vote_str}
Weighted score: {net:+.3f}
Preliminary signal: {signal} ({conviction:.0%})

Price: ${p.price:.4f} | Market: {p.market.value}

Write a 2-3 sentence investment thesis:
1. Primary catalyst or pattern
2. Conflicting signals or key risk

Then end with EXACTLY:
FINAL_SIGNAL: <STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL>
FINAL_CONVICTION: <0.0-1.0>
ENTRY_CONTEXT: <one sentence>
RISK_NOTE: <one sentence>"""

        response = await self._llm(
            "You are the master trading agent. Be decisive and specific.",
            prompt, "Master synthesis LLM",
            max_tokens=cfg.tier2_max_tokens
        )

        final_signal = signal
        final_conviction = conviction
        thesis = entry_context = risk_note = ""

        if response:
            m = re.search(r"FINAL_SIGNAL\s*:\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)",
                          response, re.IGNORECASE)
            if m: final_signal = m.group(1).upper()
            m = re.search(r"FINAL_CONVICTION\s*:\s*(0?\.\d+|1\.0|1)", response)
            if m:
                try: final_conviction = max(0.0, min(1.0, float(m.group(1))))
                except Exception: pass
            m = re.search(r"ENTRY_CONTEXT\s*:\s*\[?(.*?)(?:\n|$)", response)
            if m: entry_context = m.group(1).strip("[] ")
            m = re.search(r"RISK_NOTE\s*:\s*\[?(.*?)(?:\n|$)", response)
            if m: risk_note = m.group(1).strip("[] ")
            thesis = " ".join(
                l.strip() for l in response.split("\n")
                if l.strip() and not any(x in l for x in ("FINAL_","ENTRY_","RISK_"))
            )[:400]

        if not thesis:
            thesis = (f"Composite signal {final_signal} at {final_conviction:.0%} "
                      f"conviction. {vote_str} vote split.")

        await self._step("signal_emitted",
                         f"FINAL: {final_signal} ({final_conviction:.0%}) — {thesis[:100]}",
                         output_data={"signal": final_signal, "conviction": final_conviction})

        return {
            "signal": final_signal,
            "conviction": final_conviction,
            "thesis": thesis,
            "entry_context": entry_context,
            "risk_note": risk_note,
            "net_score": round(net, 3),
            "votes": votes,
            "layer_results": [
                {
                    "layer": lr.layer, "bias": lr.signal_bias,
                    "confidence": lr.confidence, "summary": lr.summary,
                    "findings": lr.key_findings, "code_used": lr.code_used,
                    "data": {k: v for k, v in lr.data.items()
                             if k not in ("peers", "llm_prompt", "llm_response")},
                }
                for lr in layer_results
            ]
        }


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class DeepAnalysisOrchestrator:
    """Runs Tier-1 (quick) or Tier-2 (deep) analysis depending on config."""

    # Kept for legacy /api/settings/layers compatibility
    ENABLED_LAYERS = {
        "price_structure": True, "technicals": True, "volume_profile": True,
        "fundamentals": True, "news_sentiment": True, "sector_peers": True,
        "risk": True, "code_analysis": True,
    }

    async def analyze_tier1(self, item: WatchlistItem) -> Dict:
        """Fast screen: 1 LLM call. ~60s target."""
        sym = item.profile.symbol
        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        log.info(f"Tier-1 analysis: {sym}")
        trace = trace_store.start_trace(sym)
        trace.layers_run = ["quick_screen", "synthesis"]

        await emit("agent.activity", {
            "agent_id": f"t1_{sym}", "agent_type": "master",
            "symbol": sym, "action": "analyzing",
            "message": f"[T1] Quick screen starting for {sym}",
        }, "deep_analysis")

        try:
            df = await asyncio.wait_for(fetch_ohlcv_yf(sym), timeout=15)
        except Exception:
            df = None
        indicators = compute_indicators(df) if df is not None else {}

        layer_results: List[LayerResult] = []
        try:
            qs = QuickScreenLayer(trace)
            result = await asyncio.wait_for(
                qs.run(item, df, indicators),
                timeout=cfg.tier1_timeout
            )
            layer_results.append(result)

            await emit("agent.activity", {
                "agent_id": f"t1_{sym}", "agent_type": "quick",
                "symbol": sym, "action": "analyzing",
                "message": f"[T1] {sym}: {result.signal_bias} ({result.confidence:.0%}) — {result.summary[:100]}",
                "ts": time.time(),
                "layer_data": {
                    "bias": result.signal_bias,
                    "confidence": result.confidence,
                    "findings": result.key_findings[:4],
                    "llm_prompt": result.data.get("llm_prompt", "")[:400],
                    "llm_response": result.data.get("llm_response", "")[:400],
                },
            }, "deep_analysis")

        except asyncio.TimeoutError:
            log.warning(f"[T1] {sym} timed out after {cfg.tier1_timeout}s")
            await emit("agent.activity", {
                "agent_id": f"t1_{sym}", "agent_type": "quick",
                "symbol": sym, "action": "error",
                "message": f"[T1] {sym}: LLM timeout after {cfg.tier1_timeout}s — LM Studio may be busy, defaulting HOLD",
                "ts": time.time(),
            }, "deep_analysis")
            layer_results.append(LayerResult(
                "quick_screen", "neutral", 0.4,
                f"Tier-1 timeout after {cfg.tier1_timeout}s",
                key_findings=["Timeout — LLM may be slow"],
            ))

        except Exception as e:
            log.error(f"[T1] {sym} error: {e}")
            layer_results.append(LayerResult(
                "quick_screen", "neutral", 0.3,
                f"Error: {str(e)[:80]}",
            ))

        # Quick synthesis (math-only for Tier 1, skip master LLM call)
        if layer_results:
            lr = layer_results[0]
            net = {"bullish": 0.5, "neutral": 0.0, "bearish": -0.5}.get(lr.signal_bias, 0)
            conv = lr.confidence
            if net > 0.3 and conv > 0.5:   sig = "BUY"
            elif net > 0.15:                sig = "WATCH"
            elif net < -0.3 and conv > 0.5:sig = "SELL"
            elif net < -0.15:              sig = "WATCH"
            else:                           sig = "HOLD"
        else:
            sig, conv = "HOLD", 0.3

        synthesis = {
            "signal": sig, "conviction": round(conv, 3),
            "thesis": layer_results[0].summary if layer_results else "",
            "entry_context": "", "risk_note": "",
            "net_score": 0.0, "votes": {"bullish": 0, "neutral": 1, "bearish": 0},
            "layer_results": [
                {"layer": lr.layer, "bias": lr.signal_bias,
                 "confidence": lr.confidence, "summary": lr.summary,
                 "findings": lr.key_findings, "code_used": lr.code_used, "data": {}}
                for lr in layer_results
            ],
            "tier": 1,
        }

        trace_store.finish_trace(trace, synthesis["signal"], synthesis["conviction"],
                                 synthesis["thesis"])
        log.info(f"[T1] {sym} → {synthesis['signal']} ({synthesis['conviction']:.0%})")

        # Record for self-improvement
        try:
            from agents.self_improvement import self_improvement as _si
            _si.layer_store.record_trace_prediction(
                sym, synthesis["signal"], layer_results
            )
        except Exception as e:
            log.debug(f"SI trace record [{sym}]: {e}")

        await emit("agent.activity", {
            "agent_id": f"t1_{sym}", "agent_type": "master",
            "symbol": sym, "action": "signal",
            "message": (f"[T1] {sym}: {synthesis['signal']} "
                        f"({synthesis['conviction']:.0%}) — {synthesis['thesis'][:100]}"),
        }, "deep_analysis")

        return synthesis

    async def analyze_tier2(self, item: WatchlistItem) -> Dict:
        """Deep dive: 3 LLM calls. ~3min target."""
        sym = item.profile.symbol
        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        log.info(f"Tier-2 analysis: {sym}")
        trace = trace_store.start_trace(sym)
        trace.layers_run = ["tech_deep", "context", "master_synthesis"]

        # Build "why promoted to T2" context from existing signals / profile
        _t2_why = ""
        try:
            if item.latest_signal:
                _ls = item.latest_signal
                _t2_why = (f" T1={getattr(_ls.signal_type,'value',_ls.signal_type)}"
                           f"@{_ls.conviction:.0%}")
            if item.profile.composite_score:
                _t2_why += f" score={item.profile.composite_score:.0f}/100"
            if item.profile.t0_score:
                _t2_why += f" T0={item.profile.t0_score}/3"
        except Exception:
            pass

        await emit("agent.activity", {
            "agent_id": f"t2_{sym}", "agent_type": "master",
            "symbol": sym, "action": "analyzing",
            "message": (f"[T2] Deep dive starting for {sym}.{_t2_why} "
                        f"Running 3 layers: tech → context → synthesis."),
        }, "deep_analysis")

        try:
            df = await asyncio.wait_for(fetch_ohlcv_yf(sym), timeout=20)
        except Exception:
            df = None
        indicators = compute_indicators(df) if df is not None else {}
        precomputed = await _collect_data(item, df, indicators, trace)

        layer_results: List[LayerResult] = []
        _t2_start = time.time()   # track wall time so synthesis gets remaining budget

        # Layer A: Technical deep dive (1/3)
        await emit("agent.activity", {
            "agent_id": f"t2_{sym}", "agent_type": "technical",
            "symbol": sym, "action": "analyzing",
            "message": f"[T2-Tech] {sym}: running technical deep analysis (1/3)…",
        }, "deep_analysis")
        _tech_budget = max(60, cfg.tier2_timeout // 2)
        try:
            td = TechDeepLayer(trace)
            result = await asyncio.wait_for(
                td.run(item, df, indicators, precomputed=precomputed),
                timeout=_tech_budget
            )
            layer_results.append(result)
            await emit("agent.activity", {
                "agent_id": f"t2_{sym}", "agent_type": "technical",
                "symbol": sym, "action": "analyzing",
                "message": (f"[T2-Tech] {sym}: {result.signal_bias} ({result.confidence:.0%})"
                            f" — {result.summary[:100]}"),
                "layer_data": {
                    "bias": result.signal_bias, "confidence": result.confidence,
                    "findings": result.key_findings[:4],
                    "llm_prompt": result.data.get("llm_prompt", "")[:400],
                    "llm_response": result.data.get("llm_response", "")[:400],
                },
            }, "deep_analysis")
        except asyncio.TimeoutError:
            log.warning(f"[T2] {sym} tech layer timed out")
            layer_results.append(LayerResult("tech_deep", "neutral", 0.4,
                                             "Tech deep timeout", code_used=True))
        except Exception as e:
            log.error(f"[T2] {sym} tech error: {e}")
            layer_results.append(LayerResult("tech_deep", "neutral", 0.3,
                                             f"Tech error: {str(e)[:60]}"))

        # Layer B: Context (news + fundamentals) (2/3)
        await emit("agent.activity", {
            "agent_id": f"t2_{sym}", "agent_type": "context",
            "symbol": sym, "action": "analyzing",
            "message": f"[T2-Ctx] {sym}: fetching news & fundamentals (2/3)…",
        }, "deep_analysis")
        _ctx_budget = max(45, min(90, cfg.tier2_timeout - int(time.time() - _t2_start) - 60))
        try:
            cl = ContextLayer(trace)
            result = await asyncio.wait_for(
                cl.run(item, df, indicators),
                timeout=_ctx_budget
            )
            layer_results.append(result)
            await emit("agent.activity", {
                "agent_id": f"t2_{sym}", "agent_type": "context",
                "symbol": sym, "action": "analyzing",
                "message": (f"[T2-Ctx] {sym}: {result.signal_bias} ({result.confidence:.0%})"
                            f" — {result.summary[:100]}"),
                "layer_data": {
                    "bias": result.signal_bias, "confidence": result.confidence,
                    "findings": result.key_findings[:4],
                    "llm_prompt": result.data.get("llm_prompt", "")[:400],
                    "llm_response": result.data.get("llm_response", "")[:400],
                },
            }, "deep_analysis")
        except asyncio.TimeoutError:
            log.warning(f"[T2] {sym} context layer timed out")
            layer_results.append(LayerResult("context", "neutral", 0.3,
                                             "Context timeout"))
        except Exception as e:
            log.debug(f"[T2] {sym} context error: {e}")
            layer_results.append(LayerResult("context", "neutral", 0.3,
                                             f"Context error: {str(e)[:60]}"))

        # Master synthesis (LLM call 3/3) — uses remaining budget, guaranteed at least 45s
        _elapsed = int(time.time() - _t2_start)
        _synth_budget = max(45, cfg.tier2_timeout - _elapsed - 10)
        _tech_bias = layer_results[0].signal_bias if layer_results else "?"
        _ctx_bias  = layer_results[1].signal_bias if len(layer_results) > 1 else "?"
        await emit("agent.activity", {
            "agent_id": f"t2_{sym}", "agent_type": "master",
            "symbol": sym, "action": "reasoning",
            "message": (f"[T2-Synth] {sym}: synthesizing (tech={_tech_bias}, "
                        f"ctx={_ctx_bias}) → final signal (3/3)… budget={_synth_budget}s"),
        }, "deep_analysis")
        synthesis_layer = MasterSynthesisLayer(trace)
        try:
            synthesis = await asyncio.wait_for(
                synthesis_layer.run_synthesis(item, layer_results),
                timeout=_synth_budget
            )
        except Exception as e:
            log.error(f"[T2] {sym} synthesis error: {e}")
            lr = layer_results[0] if layer_results else None
            synthesis = {
                "signal": "HOLD", "conviction": 0.3,
                "thesis": f"Synthesis error: {str(e)[:80]}",
                "entry_context": "", "risk_note": "",
                "net_score": 0.0, "votes": {"bullish": 0, "neutral": 1, "bearish": 0},
                "layer_results": [],
            }

        synthesis["tier"] = 2
        trace_store.finish_trace(trace, synthesis["signal"], synthesis["conviction"],
                                 synthesis["thesis"])
        log.info(f"[T2] {sym} → {synthesis['signal']} ({synthesis['conviction']:.0%})")

        try:
            from agents.self_improvement import self_improvement as _si
            _si.layer_store.record_trace_prediction(
                sym, synthesis["signal"], layer_results
            )
        except Exception as e:
            log.debug(f"SI trace record [{sym}]: {e}")

        await emit("agent.activity", {
            "agent_id": f"t2_{sym}", "agent_type": "master",
            "symbol": sym, "action": "signal",
            "message": (f"[T2] {sym}: {synthesis['signal']} "
                        f"({synthesis['conviction']:.0%}) — {synthesis['thesis'][:120]}"),
        }, "deep_analysis")

        return synthesis

    async def analyze(self, item: WatchlistItem) -> Dict:
        """Entry point: routes to Tier 1, Tier 2, or legacy based on config."""
        from agents.analysis_config import analysis_config
        cfg = analysis_config.get()

        if cfg.legacy_mode:
            # Fall through to Tier 2 as a reasonable default
            return await self.analyze_tier2(item)

        if cfg.tier1_enabled:
            result = await self.analyze_tier1(item)
            conv = result.get("conviction", 0)
            if cfg.tier2_enabled and conv >= cfg.tier2_threshold:
                log.info(f"[AUTO] {item.profile.symbol} Tier-1 conv {conv:.0%} "
                         f">= {cfg.tier2_threshold:.0%} → promoting to Tier-2")
                result = await self.analyze_tier2(item)
            return result

        if cfg.tier2_enabled:
            return await self.analyze_tier2(item)

        # Both tiers disabled — return stub
        return {
            "signal": "HOLD", "conviction": 0.0,
            "thesis": "Analysis disabled", "tier": 0,
        }


orchestrator = DeepAnalysisOrchestrator()
