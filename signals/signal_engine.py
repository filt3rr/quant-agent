"""
signals/signal_engine.py -- Rule-based signal generation

Runs on watchlist items and produces buy/sell signals based on
technical indicators, volume patterns, and cross-indicator confluence.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.models import (
    Signal, SignalType, SignalReason, TickerProfile, WatchlistItem, Market
)
from core.bus import emit
from core.logger import get_logger
from core.staleness_guard import staleness_guard

log = get_logger("signal_engine")

# ── Signal arbitration ─────────────────────────────────────────────────────
# Rule-engine and AI signals use separate dedup namespaces so they don't
# compete. AI signals (higher priority) can always emit, and they also gate
# the rule engine from re-emitting the opposing direction.
#
# Priority: ai_t2 > ai_t1 > rule_engine
# Behaviour:
#   AI emits BUY  → blocks rule from emitting SELL for the dedup window
#   AI emits SELL → blocks rule from emitting BUY  for the dedup window
#   Rule emits BUY  → does NOT block AI from emitting SELL (AI overrides rule)
#   Same direction, AI recently fired → rule emission suppressed (5-min window)
_rule_dedup: Dict[str, float] = {}   # symbol_DIRECTION → ts (rule engine)
_ai_dedup:   Dict[str, float] = {}   # symbol_DIRECTION → ts (AI, any tier)
_SIGNAL_DEDUP_SECS = 1800            # 30 minutes
_AI_COVER_SECS     = 300             # 5-min window: rule suppressed if AI just covered same direction

_DEDUP_FILE      = Path(__file__).parent.parent / "storage" / "signal_dedup.json"
_last_dedup_prune: float = 0.0


def _load_dedup():
    """Restore dedup state from prior session, ignoring entries older than the window."""
    try:
        if _DEDUP_FILE.exists():
            data = json.loads(_DEDUP_FILE.read_text(encoding="utf-8"))
            cutoff = time.time() - _SIGNAL_DEDUP_SECS
            for k, ts in data.get("rule", {}).items():
                if ts > cutoff:
                    _rule_dedup[k] = ts
            for k, ts in data.get("ai", {}).items():
                if ts > cutoff:
                    _ai_dedup[k] = ts
    except Exception as e:
        log.debug(f"Dedup load failed (starting fresh): {e}")


def _save_dedup():
    try:
        _DEDUP_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DEDUP_FILE.write_text(
            json.dumps({"rule": dict(_rule_dedup), "ai": dict(_ai_dedup)}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.debug(f"Dedup save failed: {e}")


def _prune_dedup():
    """Remove stale dedup entries to prevent memory growth over long sessions."""
    global _last_dedup_prune
    now = time.time()
    if now - _last_dedup_prune < 300:
        return
    _last_dedup_prune = now
    cutoff = now - _SIGNAL_DEDUP_SECS
    for d in (_rule_dedup, _ai_dedup):
        stale = [k for k, ts in list(d.items()) if ts < cutoff]
        for k in stale:
            del d[k]


_load_dedup()


def _calc_target_stop(price: float, signal_type: SignalType, atr: float) -> Tuple[float, float]:
    """Calculate target price and stop loss using ATR."""
    if atr <= 0:
        atr = price * 0.02  # fallback 2%
    if signal_type in (SignalType.BUY, SignalType.STRONG_BUY):
        target = price + (atr * 3)
        stop   = price - (atr * 1.5)
    else:
        target = price - (atr * 3)
        stop   = price + (atr * 1.5)
    return round(target, 4), round(stop, 4)


class SignalEngine:
    """
    Multi-rule signal detection system.
    Each rule returns a (signal_type, conviction, reason, summary) tuple or None.
    """

    def __init__(self):
        self.rules = [
            self._rule_volume_breakout,
            self._rule_rsi_reversal,
            self._rule_macd_cross,
            self._rule_bb_squeeze_breakout,
            self._rule_momentum_surge,
            self._rule_multi_indicator_confluence,
            self._rule_vwap_reclaim,
            self._rule_gap_up,
            self._rule_accumulation_pattern,
            self._rule_earnings_momentum,
            self._rule_mean_reversion,
        ]

    def _get_indicators(self, profile: TickerProfile) -> Dict:
        return profile.provider_data.get("indicators", {})

    def _rule_volume_breakout(self, profile: TickerProfile) -> Optional[Tuple]:
        """Volume spike + positive price action = breakout signal."""
        vr = profile.volume_ratio
        chg = profile.change_pct
        if vr >= 3.0 and chg > 2.0:
            conviction = min(0.9, 0.5 + (vr - 3) * 0.1 + chg * 0.02)
            return (
                SignalType.STRONG_BUY if chg > 5 else SignalType.BUY,
                conviction,
                SignalReason.VOLUME_SPIKE,
                f"Volume {vr:.1f}x avg with +{chg:.1f}% move -- classic breakout pattern"
            )
        if vr >= 2.0 and chg < -3.0:
            return (
                SignalType.SELL,
                0.6,
                SignalReason.VOLUME_SPIKE,
                f"Volume {vr:.1f}x avg with {chg:.1f}% dump -- distribution signal"
            )
        return None

    def _rule_rsi_reversal(self, profile: TickerProfile) -> Optional[Tuple]:
        """RSI oversold/overbought reversal."""
        rsi = profile.rsi
        if 0 < rsi < 25 and profile.change_pct > -1:  # Oversold + stabilizing (rsi>0 = valid data)
            return (
                SignalType.BUY,
                0.70,
                SignalReason.TECHNICAL,
                f"RSI deeply oversold at {rsi:.0f} with price stabilization -- mean reversion setup"
            )
        if 0 < rsi < 30:
            return (
                SignalType.WATCH,
                0.55,
                SignalReason.TECHNICAL,
                f"RSI oversold at {rsi:.0f} -- monitoring for reversal confirmation"
            )
        if rsi > 75 and profile.change_pct < 0.5 and rsi <= 100:
            return (
                SignalType.SELL,
                0.65,
                SignalReason.TECHNICAL,
                f"RSI overbought at {rsi:.0f} with momentum stalling -- potential reversal"
            )
        return None

    def _rule_macd_cross(self, profile: TickerProfile) -> Optional[Tuple]:
        """MACD crossover signals."""
        macd_cross = profile.macd_signal
        ind = self._get_indicators(profile)
        hist = ind.get("macd_hist", 0)

        if macd_cross == "bullish" and abs(hist) > 0:
            rsi_boost = 0.1 if profile.rsi < 55 else 0
            return (
                SignalType.BUY,
                0.65 + rsi_boost,
                SignalReason.TECHNICAL,
                f"MACD bullish crossover (hist: {hist:.4f}) -- trend shift signal"
            )
        if macd_cross == "bearish" and abs(hist) > 0:
            return (
                SignalType.SELL,
                0.62,
                SignalReason.TECHNICAL,
                f"MACD bearish crossover -- momentum turning negative"
            )
        return None

    def _rule_bb_squeeze_breakout(self, profile: TickerProfile) -> Optional[Tuple]:
        """Bollinger Band squeeze followed by breakout."""
        ind = self._get_indicators(profile)
        bb_pos = profile.bb_position
        squeeze = ind.get("bb_squeeze", False)
        chg = profile.change_pct

        if squeeze and chg > 1.5:
            return (
                SignalType.BUY,
                0.72,
                SignalReason.BREAKOUT,
                f"BB squeeze breakout (+{chg:.1f}%) -- volatility expansion after compression"
            )
        if bb_pos < 0.05 and chg > 0:
            return (
                SignalType.BUY,
                0.60,
                SignalReason.TECHNICAL,
                f"Price bouncing off lower Bollinger Band (pos: {bb_pos:.2f})"
            )
        if bb_pos > 0.95 and chg < 0:
            return (
                SignalType.SELL,
                0.58,
                SignalReason.TECHNICAL,
                f"Price rejected at upper Bollinger Band (pos: {bb_pos:.2f})"
            )
        return None

    def _rule_momentum_surge(self, profile: TickerProfile) -> Optional[Tuple]:
        """5-day momentum surge."""
        chg5 = profile.change_5d
        ind  = self._get_indicators(profile)
        roc5 = ind.get("roc5", chg5)

        if roc5 > 10 and profile.volume_ratio > 1.5:
            return (
                SignalType.BUY,
                0.68,
                SignalReason.MOMENTUM,
                f"Strong 5-day momentum (+{roc5:.1f}%) with volume confirmation ({profile.volume_ratio:.1f}x)"
            )
        if roc5 < -10 and profile.volume_ratio > 1.5:
            return (
                SignalType.SELL,
                0.65,
                SignalReason.MOMENTUM,
                f"Strong downside momentum ({roc5:.1f}%) with volume -- avoid or short"
            )
        return None

    def _rule_multi_indicator_confluence(self, profile: TickerProfile) -> Optional[Tuple]:
        """Multiple bullish/bearish signals in alignment = high conviction."""
        bullish_count = 0
        bearish_count = 0

        if profile.rsi < 40: bullish_count += 1
        if profile.rsi > 60: bearish_count += 1
        if profile.macd_signal == "bullish": bullish_count += 1
        if profile.macd_signal == "bearish": bearish_count += 1
        if profile.bb_position < 0.35: bullish_count += 1
        if profile.bb_position > 0.75: bearish_count += 1
        if profile.change_pct > 1: bullish_count += 1
        if profile.change_pct < -1: bearish_count += 1
        if profile.volume_ratio > 1.5: bullish_count += 1
        if profile.sentiment_score > 0.3: bullish_count += 1
        if profile.sentiment_score < -0.3: bearish_count += 1

        if bullish_count >= 4:
            return (
                SignalType.STRONG_BUY if bullish_count >= 5 else SignalType.BUY,
                min(0.92, 0.55 + bullish_count * 0.07),
                SignalReason.MULTI_AGENT,
                f"Multi-indicator confluence: {bullish_count}/6 bullish signals aligned"
            )
        if bearish_count >= 4:
            return (
                SignalType.STRONG_SELL if bearish_count >= 5 else SignalType.SELL,
                min(0.90, 0.55 + bearish_count * 0.07),
                SignalReason.MULTI_AGENT,
                f"Multi-indicator confluence: {bearish_count}/6 bearish signals aligned"
            )
        return None

    def _rule_vwap_reclaim(self, profile: TickerProfile) -> Optional[Tuple]:
        """Price reclaiming VWAP is a bullish intraday signal."""
        vwap_diff = profile.vwap_vs_price
        if 0.5 < vwap_diff < 5.0 and profile.change_pct > 0.5:
            return (
                SignalType.BUY,
                0.60,
                SignalReason.TECHNICAL,
                f"Price {vwap_diff:.1f}% above VWAP with positive momentum -- VWAP reclaim"
            )
        if -10.0 < vwap_diff < -3.0 and profile.change_pct < -1:
            return (
                SignalType.SELL,
                0.58,
                SignalReason.TECHNICAL,
                f"Price {abs(vwap_diff):.1f}% below VWAP with selling pressure"
            )
        return None

    async def evaluate(self, item: WatchlistItem) -> List[Signal]:
        """
        Multi-validation signal evaluation.
        BUY/SELL signals require 2+ independent rules to agree on direction.
        STRONG_BUY/SELL requires 3+ rules + conviction threshold.
        Single-rule results are downgraded to WATCH (non-actionable).
        Stale data (provider refresh too old) is skipped entirely.
        """
        profile = item.profile
        if staleness_guard.is_stale(profile.symbol):
            log.debug(
                f"[{profile.symbol}] skipped — stale data "
                f"({staleness_guard.age_seconds(profile.symbol):.0f}s old)"
            )
            return []
        ind = self._get_indicators(profile)
        atr = ind.get("atr", profile.price * 0.02)

        # Collect all raw rule hits
        raw: List[Tuple] = []
        for rule in self.rules:
            try:
                result = rule(profile)
                if result:
                    raw.append(result)
            except Exception as e:
                log.debug(f"Rule error [{profile.symbol}]: {e}")

        # ---- SELF-IMPROVEMENT: apply learned rule multipliers + regime factor ----
        try:
            from agents.self_improvement import self_improvement as _si
            _p = _si.get_params()
            _regime = _p.current_regime
            _regime_adj = _p.regime_adjustments.get(_regime, 1.0)
            adjusted_raw = []
            for _t, _c, _r, _s in raw:
                _rule_key = _r.value.lower()
                # Base rule multiplier (global, across all regimes)
                _base_mult = _p.rule_multipliers.get(_rule_key, 1.0)
                # Per-regime rule multiplier (layered on top of base)
                _regime_rule_mult = _si.get_regime_rule_multiplier(_regime, _rule_key)
                _new_c = min(0.95, _c * _base_mult * _regime_rule_mult * _regime_adj)
                adjusted_raw.append((_t, _new_c, _r, _s))
            raw = adjusted_raw
        except Exception:
            pass   # self-improvement is optional — never break signal generation

        # Bucket by direction
        buy_hits  = [(t, c, r, s) for t, c, r, s in raw if t in (SignalType.BUY, SignalType.STRONG_BUY)]
        sell_hits = [(t, c, r, s) for t, c, r, s in raw if t in (SignalType.SELL, SignalType.STRONG_SELL)]
        watch_hits = [(t, c, r, s) for t, c, r, s in raw if t == SignalType.WATCH]

        confirmed: List[Signal] = []
        now = time.time()

        # ---- CONFIRMED BUY: 2+ rules must agree --------------------------------
        if len(buy_hits) >= 2:
            avg_conv = sum(c for _, c, _, _ in buy_hits) / len(buy_hits)
            # +4% per extra rule beyond 2 (max +12%)
            agreement_bonus = min(0.12, (len(buy_hits) - 2) * 0.04)
            # Cross-validate: penalise if contradicted by indicators
            rsi_penalty  = 0.08 if profile.rsi > 78 else 0          # overbought
            vol_penalty  = 0.05 if profile.volume_ratio < 0.7 else 0 # no volume support
            final_conv = min(0.95, avg_conv + agreement_bonus - rsi_penalty - vol_penalty)

            # STRONG_BUY only when 3+ rules + high conviction
            sig_type = (SignalType.STRONG_BUY
                        if len(buy_hits) >= 3 and final_conv >= 0.78
                        else SignalType.BUY)

            reasons  = list(dict.fromkeys(r for _, _, r, _ in buy_hits))  # preserve order, dedupe
            summaries = [s for _, _, _, s in sorted(buy_hits, key=lambda x: -x[1])]
            combined  = f"[{len(buy_hits)}/11 rules agree] " + " · ".join(summaries[:2])

            target, stop = _calc_target_stop(profile.price, sig_type, atr)
            rr = abs(target - profile.price) / abs(profile.price - stop) if abs(profile.price - stop) > 0 else 0

            confirmed.append(Signal(
                symbol=profile.symbol, signal_type=sig_type,
                conviction=final_conv, price_at_signal=profile.price,
                target_price=target, stop_loss=stop,
                risk_reward=round(rr, 2),
                reason=SignalReason.MULTI_AGENT if len(buy_hits) >= 3 else reasons[0],
                summary=combined, agent="signal_engine",
                market=profile.market, ts=now, expires_ts=now + 3600,
            ))

        # ---- CONFIRMED SELL: 2+ rules must agree --------------------------------
        if len(sell_hits) >= 2:
            avg_conv = sum(c for _, c, _, _ in sell_hits) / len(sell_hits)
            agreement_bonus = min(0.12, (len(sell_hits) - 2) * 0.04)
            rsi_penalty = 0.08 if profile.rsi < 22 else 0  # already deeply oversold
            final_conv = min(0.95, avg_conv + agreement_bonus - rsi_penalty)

            sig_type = (SignalType.STRONG_SELL
                        if len(sell_hits) >= 3 and final_conv >= 0.75
                        else SignalType.SELL)

            reasons  = list(dict.fromkeys(r for _, _, r, _ in sell_hits))
            summaries = [s for _, _, _, s in sorted(sell_hits, key=lambda x: -x[1])]
            combined  = f"[{len(sell_hits)}/11 rules agree] " + " · ".join(summaries[:2])

            target, stop = _calc_target_stop(profile.price, sig_type, atr)
            rr = abs(target - profile.price) / abs(profile.price - stop) if abs(profile.price - stop) > 0 else 0

            confirmed.append(Signal(
                symbol=profile.symbol, signal_type=sig_type,
                conviction=final_conv, price_at_signal=profile.price,
                target_price=None, stop_loss=None,
                risk_reward=round(rr, 2),
                reason=SignalReason.MULTI_AGENT if len(sell_hits) >= 3 else reasons[0],
                summary=combined, agent="signal_engine",
                market=profile.market, ts=now, expires_ts=now + 3600,
            ))

        # ---- WATCH: pass through best single-rule hit when nothing confirmed ----
        if watch_hits and not confirmed:
            t, c, r, s = max(watch_hits, key=lambda x: x[1])
            target, stop = _calc_target_stop(profile.price, t, atr)
            confirmed.append(Signal(
                symbol=profile.symbol, signal_type=t,
                conviction=c, price_at_signal=profile.price,
                target_price=target, stop_loss=stop,
                risk_reward=0.0, reason=r, summary=s,
                agent="signal_engine", market=profile.market,
                ts=now, expires_ts=now + 3600,
            ))

        # ---- SELF-IMPROVEMENT: conviction calibration + sector threshold gate ----
        try:
            from agents.self_improvement import self_improvement as _si2
            _p2 = _si2.get_params()
            _cal = _p2.conviction_calibration
            _sec_thresh = _p2.sector_thresholds.get(profile.sector or "", 0.0)
            _filtered = []
            for _sig in confirmed:
                # Apply calibration multiplier
                _sig.conviction = max(0.0, min(0.95, round(_sig.conviction * _cal, 3)))
                # Sector gate: drop actionable signals below sector-specific threshold
                if _sec_thresh > 0 and _sig.conviction < _sec_thresh and \
                   _sig.signal_type not in (SignalType.WATCH,):
                    log.debug(
                        f"Sector gate [{profile.symbol}/{profile.sector}]: "
                        f"conviction {_sig.conviction:.2f} < threshold {_sec_thresh:.2f} — skipped"
                    )
                    continue
                _filtered.append(_sig)
            confirmed = _filtered
        except Exception:
            pass

        # ---- MTF CONFLICT: reduce conviction when 4h trend opposes daily signal ----
        try:
            mtf = profile.provider_data.get("indicators", {}).get("mtf_trend", "neutral")
            if mtf != "neutral" and confirmed:
                adjusted = []
                for sig in confirmed:
                    is_buy  = sig.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)
                    is_sell = sig.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)
                    if (is_buy and mtf == "down") or (is_sell and mtf == "up"):
                        old_c = sig.conviction
                        sig.conviction = round(sig.conviction * 0.60, 3)
                        sig.summary = f"[MTF_CONFLICT 4h={mtf}] " + sig.summary
                        log.info(
                            f"MTF conflict [{profile.symbol}]: {sig.signal_type.value} "
                            f"but 4h={mtf} → conviction {old_c:.0%} → {sig.conviction:.0%}"
                        )
                        if sig.signal_type == SignalType.STRONG_BUY:
                            sig.signal_type = SignalType.BUY
                        elif sig.signal_type == SignalType.STRONG_SELL:
                            sig.signal_type = SignalType.SELL
                    adjusted.append(sig)
                confirmed = adjusted
        except Exception:
            pass

        # ---- DEEP ANALYSIS OVERLAY: cross-reference with recent AI traces ----
        try:
            from core.trace import trace_store
            traces = trace_store.get_traces(profile.symbol)
            if traces and confirmed:
                latest = traces[0]
                age = now - latest.get("finished_ts", 0)
                if age <= 900:  # only use traces < 15 minutes old
                    deep_signal = latest.get("final_signal", "HOLD")
                    deep_conv   = latest.get("final_conviction", 0.5)
                    surviving = []
                    for sig in confirmed:
                        is_buy  = sig.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)
                        is_sell = sig.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)
                        deep_buy  = "BUY"  in deep_signal
                        deep_sell = "SELL" in deep_signal
                        if (is_buy and deep_sell) or (is_sell and deep_buy):
                            # Hard gate: high-conviction opposing AI overrides rule entirely
                            if deep_conv >= 0.78:
                                log.info(
                                    f"Deep-rule VETO [{profile.symbol}]: "
                                    f"rules={sig.signal_type.value} deep={deep_signal}({deep_conv:.0%}) "
                                    f"— signal dropped"
                                )
                                continue
                            if deep_conv > 0.55:
                                penalty = min(0.30, deep_conv * 0.35)
                                sig.conviction = max(0.25, round(sig.conviction - penalty, 3))
                                log.info(
                                    f"Deep-rule conflict [{profile.symbol}]: "
                                    f"rules={sig.signal_type.value} deep={deep_signal}({deep_conv:.0%})"
                                    f" -> conviction={sig.conviction:.0%}"
                                )
                        elif (is_buy and deep_buy) or (is_sell and deep_sell):
                            if deep_conv >= 0.60:
                                boost = min(0.15, (deep_conv - 0.60) * 0.30)
                                sig.conviction = min(0.95, round(sig.conviction + boost, 3))
                                log.info(
                                    f"Deep-rule agree [{profile.symbol}]: "
                                    f"both {sig.signal_type.value}({deep_conv:.0%})"
                                    f" -> conviction={sig.conviction:.0%}"
                                )
                        # Downgrade STRONG if conviction dropped below threshold
                        if sig.signal_type == SignalType.STRONG_BUY and sig.conviction < 0.78:
                            sig.signal_type = SignalType.BUY
                        elif sig.signal_type == SignalType.STRONG_SELL and sig.conviction < 0.75:
                            sig.signal_type = SignalType.SELL
                        surviving.append(sig)
                    confirmed = surviving
        except Exception as e:
            log.debug(f"Deep overlay [{profile.symbol}]: {e}")

        confirmed.sort(key=lambda s: s.conviction, reverse=True)

        _prune_dedup()
        now_ts = time.time()
        to_emit = []
        for sig in confirmed[:2]:
            direction = "BUY" if sig.signal_type in (SignalType.BUY, SignalType.STRONG_BUY) else "SELL"
            dedup_key = f"{sig.symbol}_{direction}"

            # Block if rule already emitted same direction recently
            if now_ts - _rule_dedup.get(dedup_key, 0) < _SIGNAL_DEDUP_SECS:
                log.debug(f"Rule dedup [{sig.symbol}] {direction} — already signaled recently")
                continue

            # Suppress if AI recently covered the same direction (AI is authoritative)
            if now_ts - _ai_dedup.get(dedup_key, 0) < _AI_COVER_SECS:
                log.debug(f"Rule suppressed [{sig.symbol}] {direction} — AI covered same direction <{_AI_COVER_SECS}s ago")
                continue

            _rule_dedup[dedup_key] = now_ts
            to_emit.append(sig)
        if to_emit:
            _save_dedup()

        for sig in to_emit:
            await emit("signal", sig.to_dict(), "signal_engine")
            log.info(
                f" SIGNAL [{sig.symbol}] {sig.signal_type.value} "
                f"conviction={sig.conviction:.0%} | {sig.summary[:80]}"
            )

        return confirmed

    async def emit_ai_signal(
        self,
        item: WatchlistItem,
        ai_signal: str,
        ai_conviction: float,
        tier: int = 1,
    ):
        """Emit a Signal directly from AI analysis when conviction is high enough."""
        if ai_conviction < 0.65 or ai_signal.upper() in ("HOLD", "WATCH", ""):
            return
        profile = item.profile
        now = time.time()
        ind = self._get_indicators(profile)
        atr = ind.get("atr", profile.price * 0.02)
        sig_map = {
            "STRONG_BUY":  SignalType.STRONG_BUY,
            "BUY":         SignalType.BUY,
            "SELL":        SignalType.SELL,
            "STRONG_SELL": SignalType.STRONG_SELL,
        }
        sig_type = sig_map.get(ai_signal.upper())
        if not sig_type:
            return
        target, stop = _calc_target_stop(profile.price, sig_type, atr)
        rr = abs(target - profile.price) / max(abs(profile.price - stop), 0.0001)
        sig = Signal(
            symbol=profile.symbol,
            signal_type=sig_type,
            conviction=round(ai_conviction, 3),
            price_at_signal=profile.price,
            target_price=target,
            stop_loss=stop,
            risk_reward=round(rr, 2),
            reason=SignalReason.MULTI_AGENT,
            summary=f"[Tier-{tier} AI] {ai_signal} at {ai_conviction:.0%} conviction",
            agent="deep_analysis",
            market=profile.market,
            ts=now,
            expires_ts=now + 3600,
        )
        try:
            from agents.self_improvement import self_improvement as _si
            cal = _si.get_params().conviction_calibration
            sig.conviction = max(0.0, min(0.95, round(sig.conviction * cal, 3)))
        except Exception:
            pass

        # AI arbitration: AI has priority over rule engine.
        # - AI uses its own dedup namespace (_ai_dedup), not blocked by rule signals.
        # - On emit, AI blocks the rule from emitting the opposing direction
        #   (prevents rule from contradicting a fresh AI signal for the full window).
        # - AI also marks its own direction in _rule_dedup so rule doesn't re-emit
        #   the same direction redundantly.
        now_ts = time.time()
        direction    = "BUY" if sig_type in (SignalType.BUY, SignalType.STRONG_BUY) else "SELL"
        opposing_dir = "SELL" if direction == "BUY" else "BUY"
        dedup_key    = f"{profile.symbol}_{direction}"
        opposing_key = f"{profile.symbol}_{opposing_dir}"

        if now_ts - _ai_dedup.get(dedup_key, 0) < _SIGNAL_DEDUP_SECS:
            log.debug(
                f"AI signal dedup [{profile.symbol}] {direction} tier={tier} "
                f"— already signaled {(now_ts - _ai_dedup.get(dedup_key,0))/60:.0f}min ago"
            )
            return

        _ai_dedup[dedup_key]      = now_ts   # AI dedup for this direction
        _rule_dedup[dedup_key]    = now_ts   # Block rule from duplicate same-direction emit
        _rule_dedup[opposing_key] = now_ts   # Block rule from contradicting AI for full window
        _save_dedup()

        await emit("signal", sig.to_dict(), "deep_analysis")
        log.info(
            f" AI SIGNAL [{sig.symbol}] {sig.signal_type.value} "
            f"conviction={sig.conviction:.0%} tier={tier}"
        )

    async def run_on_watchlist(self, watchlist: List[WatchlistItem]):
        """Run signal evaluation on the entire watchlist."""
        log.info(f"Running signal engine on {len(watchlist)} tickers")
        sem = asyncio.Semaphore(10)

        async def _eval(item):
            async with sem:
                signals = await self.evaluate(item)
                if signals:
                    item.signals.extend(signals)
                    item.signals = item.signals[-20:]  # keep last 20
                    item.latest_signal = signals[0]

        await asyncio.gather(*[_eval(item) for item in watchlist], return_exceptions=True)
        log.info("Signal engine pass complete")




    # -- NEW RULES (v8) ---------------------------------------------

    def _rule_gap_up(self, profile: TickerProfile) -> Optional[Tuple]:
        """True overnight gap: today's open vs previous close (not intraday move)."""
        ind = self._get_indicators(profile)
        gap = ind.get("gap_pct", 0)   # populated by get_technicals() using open vs prev close
        vr  = profile.volume_ratio
        if gap == 0:
            return None  # gap_pct not available — skip rather than use intraday proxy
        if gap > 3.0 and vr > 1.5:
            return (
                SignalType.BUY,
                min(0.82, 0.55 + gap * 0.015 + vr * 0.02),
                SignalReason.BREAKOUT,
                f"Gap-up {gap:+.1f}% at open (prev close vs today open) on {vr:.1f}x volume"
            )
        if gap < -3.0 and vr > 1.5:
            return (
                SignalType.SELL,
                min(0.80, 0.55 + abs(gap) * 0.012),
                SignalReason.MOMENTUM,
                f"Gap-down {gap:+.1f}% at open on {vr:.1f}x volume -- avoid, gap fill risk"
            )
        return None

    def _rule_accumulation_pattern(self, profile: TickerProfile) -> Optional[Tuple]:
        """Quiet accumulation: price rising in neutral RSI zone with non-extreme volume.

        Replaces the insider rule which relied on provider_data fields that are never populated.
        Uses only indicators computed from OHLCV data (always available).
        """
        ind     = self._get_indicators(profile)
        rsi     = profile.rsi
        roc5    = ind.get("roc5", profile.change_5d)
        vr      = profile.volume_ratio
        stoch_k = ind.get("stoch_k", 50)

        # Accumulation: positive 5D trend, RSI in neutral zone (not overbought/oversold),
        # stoch not extreme, moderate volume (excludes blow-off tops and distribution)
        if (45 < rsi < 63 and roc5 > 4.0 and
                0.6 < vr < 2.5 and 25 < stoch_k < 72 and 0 < rsi):
            conviction = min(0.72, 0.50 + roc5 * 0.012 + (rsi - 45) * 0.004)
            return (
                SignalType.BUY,
                conviction,
                SignalReason.TECHNICAL,
                f"Accumulation pattern: +{roc5:.1f}% 5D, RSI {rsi:.0f} neutral, "
                f"stoch {stoch_k:.0f}, vol {vr:.1f}x -- steady building"
            )
        return None

    def _rule_earnings_momentum(self, profile: TickerProfile) -> Optional[Tuple]:
        """Post-earnings momentum -- extreme moves with volume after earnings."""
        chg = profile.change_pct
        chg5 = profile.change_5d
        vr   = profile.volume_ratio
        news = profile.news_count_24h
        # Earnings beats typically: big 1D move + elevated volume + news activity
        if chg > 8 and vr > 2 and news >= 3:
            return (
                SignalType.BUY,
                min(0.85, 0.6 + chg * 0.008),
                SignalReason.NEWS,
                f"Earnings-driven surge {chg:+.1f}% on {vr:.1f}x vol + {news} news items -- beat signal"
            )
        if chg < -8 and vr > 2 and news >= 3:
            return (
                SignalType.SELL,
                min(0.82, 0.6 + abs(chg) * 0.007),
                SignalReason.NEWS,
                f"Earnings miss {chg:+.1f}% on {vr:.1f}x vol -- avoid, downside risk"
            )
        return None

    def _rule_mean_reversion(self, profile: TickerProfile) -> Optional[Tuple]:
        """Extreme oversold with stabilization -- mean reversion play."""
        ind  = self._get_indicators(profile)
        rsi  = profile.rsi
        chg5 = profile.change_5d
        chg1 = profile.change_pct
        bb   = profile.bb_position
        # Extreme oversold: RSI < 20, BB at bottom, 5D down hard but today flat/up
        if (0 < rsi < 22 and bb < 0.1 and chg5 < -15 and chg1 > -1):
            return (
                SignalType.BUY,
                0.72,
                SignalReason.TECHNICAL,
                f"Extreme mean-reversion setup: RSI {rsi:.0f}, BB {bb:.2f}, "
                f"5D {chg5:+.1f}% but stabilizing today -- bounce candidate"
            )
        return None


signal_engine = SignalEngine()
