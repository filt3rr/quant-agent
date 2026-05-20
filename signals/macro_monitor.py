"""
signals/macro_monitor.py -- Macro regime overlay

Fetches SPY, QQQ, and VIX daily data every 15 minutes to directly update
self_improvement's current_regime without waiting for trade outcomes.

Regime rules:
  volatile:      VIX > 30
  trending_down: SPY 1D < -1.5% OR SPY 5D < -5%
  trending_up:   SPY 1D > +1.5% AND QQQ 1D > +1.0%
  sideways:      VIX < 15 AND |SPY 5D| < 2%
  (unknown kept when signal is ambiguous — preserves existing outcome-based regime)
"""
import asyncio
import time
from typing import Dict

from core.logger import get_logger

log = get_logger("macro_monitor")

REFRESH_INTERVAL = 900  # 15 minutes


class MacroMonitor:
    def __init__(self):
        self._last_regime: str = "unknown"
        self._last_ts: float = 0
        self._spy_chg: float = 0.0
        self._spy_5d: float = 0.0
        self._qqq_chg: float = 0.0
        self._vix: float = 20.0

    async def _fetch_macro_data(self) -> Dict:
        try:
            from scanners.technicals import fetch_ohlcv_yf
            spy_df, qqq_df, vix_df = await asyncio.gather(
                fetch_ohlcv_yf("SPY", period="10d"),
                fetch_ohlcv_yf("QQQ", period="10d"),
                fetch_ohlcv_yf("^VIX", period="10d"),
                return_exceptions=True,
            )
            result: Dict = {}
            if not isinstance(spy_df, Exception) and spy_df is not None and len(spy_df) >= 2:
                c0, c1 = float(spy_df["close"].iloc[-1]), float(spy_df["close"].iloc[-2])
                c5 = float(spy_df["close"].iloc[max(0, len(spy_df) - 6)])
                result["spy_chg"]  = (c0 - c1) / c1 * 100 if c1 else 0.0
                result["spy_5d"]   = (c0 - c5) / c5 * 100 if c5 else 0.0
            if not isinstance(qqq_df, Exception) and qqq_df is not None and len(qqq_df) >= 2:
                c0, c1 = float(qqq_df["close"].iloc[-1]), float(qqq_df["close"].iloc[-2])
                result["qqq_chg"] = (c0 - c1) / c1 * 100 if c1 else 0.0
            if not isinstance(vix_df, Exception) and vix_df is not None and len(vix_df) >= 1:
                result["vix"] = float(vix_df["close"].iloc[-1])
            return result
        except Exception as e:
            log.error(f"Macro data fetch error: {e}")
            return {}

    def _classify_regime(self, data: Dict) -> str:
        spy_chg = data.get("spy_chg", 0.0)
        spy_5d  = data.get("spy_5d",  0.0)
        qqq_chg = data.get("qqq_chg", 0.0)
        vix     = data.get("vix",    20.0)

        if vix > 30:
            return "volatile"
        if spy_chg < -1.5 or spy_5d < -5.0:
            return "trending_down"
        if spy_chg > 1.5 and qqq_chg > 1.0:
            return "trending_up"
        if vix < 15 and abs(spy_5d) < 2.0:
            return "sideways"
        return "unknown"

    async def update(self) -> str:
        data = await self._fetch_macro_data()
        if not data:
            return self._last_regime

        self._spy_chg = round(data.get("spy_chg", 0.0), 2)
        self._spy_5d  = round(data.get("spy_5d",  0.0), 2)
        self._qqq_chg = round(data.get("qqq_chg", 0.0), 2)
        self._vix     = round(data.get("vix", 20.0),    2)

        regime = self._classify_regime(data)
        if regime != "unknown":
            self._last_regime = regime
            try:
                from agents.self_improvement import self_improvement
                old = self_improvement.params.current_regime
                if old != regime:
                    self_improvement.params.current_regime = regime
                    self_improvement._save()
                    log.info(
                        f"MACRO REGIME SHIFT: {old} → {regime} | "
                        f"SPY {self._spy_chg:+.1f}% (5D {self._spy_5d:+.1f}%) "
                        f"QQQ {self._qqq_chg:+.1f}% VIX {self._vix:.1f}"
                    )
                else:
                    log.debug(
                        f"Macro: regime={regime} SPY {self._spy_chg:+.1f}% "
                        f"QQQ {self._qqq_chg:+.1f}% VIX {self._vix:.1f}"
                    )
            except Exception as e:
                log.warning(f"Regime update failed: {e}")

        self._last_ts = time.time()
        return regime

    def get_status(self) -> Dict:
        return {
            "regime":       self._last_regime,
            "spy_chg_pct":  self._spy_chg,
            "spy_5d_pct":   self._spy_5d,
            "qqq_chg_pct":  self._qqq_chg,
            "vix":          self._vix,
            "last_update":  self._last_ts,
        }

    async def start(self):
        log.info("Macro monitor started")
        while True:
            try:
                await self.update()
            except Exception as e:
                log.error(f"Macro monitor error: {e}")
            await asyncio.sleep(REFRESH_INTERVAL)


macro_monitor = MacroMonitor()
