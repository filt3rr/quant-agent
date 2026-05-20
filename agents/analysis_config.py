"""
agents/analysis_config.py -- Runtime-adjustable analysis pipeline configuration.

All settings are persisted to storage/analysis_config.json and can be
updated live from the dashboard without restarting.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from core.logger import get_logger

log = get_logger("analysis_config")

_STORAGE = Path(__file__).parent.parent / "storage" / "analysis_config.json"


@dataclass
class AnalysisConfig:
    # ── Tier 1: Fast AI Screen (1 LLM call per ticker) ───────────────
    tier1_enabled: bool = True
    tier1_workers: int = 3          # parallel Tier-1 workers
    tier1_max_tokens: int = 300     # LLM response budget
    tier1_timeout: int = 120        # seconds per ticker

    # ── Tier 2: Deep Dive (2-3 LLM calls per ticker) ─────────────────
    tier2_enabled: bool = True
    tier2_workers: int = 2           # parallel Tier-2 workers (2 is safe — calls don't overlap)
    tier2_threshold: float = 0.65   # min Tier-1 conviction to qualify
    tier2_max_tokens: int = 250     # per LLM call in Tier 2
    tier2_timeout: int = 360        # seconds per ticker

    # ── General pipeline ──────────────────────────────────────────────
    analysis_cooldown: int = 600    # 10 min between re-analyses of same ticker
    tickers_per_cycle: int = 5      # how many tickers to pull per agent cycle
    min_composite_score: float = 40.0  # min watchlist score to enter queue

    # ── Legacy deep analysis (used when both tiers disabled) ──────────
    legacy_mode: bool = False       # fall back to 8-layer analysis


class AnalysisConfigManager:
    def __init__(self):
        self._config = AnalysisConfig()
        self._load()

    def get(self) -> AnalysisConfig:
        return self._config

    def update(self, **kwargs) -> dict:
        """Update one or more config fields. Unknown keys are silently ignored."""
        changed = {}
        for k, v in kwargs.items():
            if not hasattr(self._config, k):
                continue
            current = getattr(self._config, k)
            try:
                cast = type(current)
                # bool needs special handling since bool("false") == True
                if cast is bool:
                    if isinstance(v, str):
                        v = v.lower() in ("1", "true", "yes")
                    else:
                        v = bool(v)
                else:
                    v = cast(v)
                setattr(self._config, k, v)
                changed[k] = v
            except (TypeError, ValueError) as e:
                log.warning(f"Config update skipped {k}={v}: {e}")
        if changed:
            self._save()
        return changed

    def to_dict(self) -> dict:
        return asdict(self._config)

    def _load(self):
        try:
            if _STORAGE.exists():
                data = json.loads(_STORAGE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(self._config, k):
                        try:
                            current = getattr(self._config, k)
                            cast = type(current)
                            if cast is bool and isinstance(v, str):
                                v = v.lower() in ("1", "true", "yes")
                            setattr(self._config, k, cast(v))
                        except Exception:
                            pass
                log.debug("Analysis config loaded from disk")
        except Exception as e:
            log.debug(f"Analysis config load failed (using defaults): {e}")

    def _save(self):
        try:
            _STORAGE.parent.mkdir(parents=True, exist_ok=True)
            _STORAGE.write_text(
                json.dumps(asdict(self._config), indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Analysis config save failed: {e}")


analysis_config = AnalysisConfigManager()
