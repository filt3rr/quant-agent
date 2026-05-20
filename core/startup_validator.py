"""
core/startup_validator.py — Pre-flight validation

Runs before any trading logic starts. Checks that the environment is
correctly configured and prints a formatted report. Returns False and
sets _failed=True when a CRITICAL check fails (system should not start).

Critical (abort):
  • Storage directory writable
  • SQLite database accessible (can create / open)
  • At least one LLM provider configured

Warning (degraded):
  • Polygon API key missing  → limited market data
  • Finnhub API key missing  → no fundamentals / sentiment
  • Tavily API key missing   → LLM research degraded
  • Alpaca API key missing   → live/paper brokerage unavailable

Usage:
    from core.startup_validator import startup_validator
    ok = await startup_validator.validate(mock_mode=False)
    if not ok:
        sys.exit(1)
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

from core.logger import get_logger
from config.settings import KEYS, SYS, LLM

log = get_logger("startup")

_PASS  = "PASS"
_WARN  = "WARN"
_FAIL  = "FAIL"
_INFO  = "INFO"

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"

_COLORS = {_PASS: _GREEN, _WARN: _YELLOW, _FAIL: _RED, _INFO: _CYAN}


def _fmt(level: str, label: str, detail: str = "") -> str:
    color = _COLORS.get(level, "")
    badge = f"{color}{level:4s}{_RESET}"
    tail  = f"  {detail}" if detail else ""
    return f"  {badge}  {label:<42}{tail}"


class StartupValidator:
    def __init__(self):
        self._results: List[Tuple[str, str, str]] = []
        self.passed = False

    def _record(self, level: str, label: str, detail: str = ""):
        self._results.append((level, label, detail))

    # ── Individual checks ─────────────────────────────────────────────────

    def _check_storage(self) -> bool:
        path = SYS.STORAGE_DIR
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            self._record(_PASS, "Storage directory writable", str(path))
            return True
        except Exception as e:
            self._record(_FAIL, "Storage directory NOT writable", str(e))
            return False

    def _check_log_dir(self) -> bool:
        path = SYS.LOG_DIR
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._record(_PASS, "Log directory writable", str(path))
            return True
        except Exception as e:
            self._record(_WARN, "Log directory issue", str(e))
            return True  # not critical

    def _check_db(self) -> bool:
        try:
            import sqlite3
            from core.db import DB_PATH
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("SELECT 1")
            conn.close()
            self._record(_PASS, "SQLite database accessible", DB_PATH.name)
            return True
        except Exception as e:
            self._record(_FAIL, "SQLite database NOT accessible", str(e))
            return False

    async def _check_llm(self, mock_mode: bool) -> bool:
        if mock_mode:
            self._record(_INFO, "LLM check skipped", "MOCK_MODE")
            return True

        if LLM.PROVIDER == "anthropic":
            if KEYS.ANTHROPIC:
                self._record(_PASS, "LLM provider configured", "anthropic (Claude)")
                return True
            else:
                self._record(_FAIL, "ANTHROPIC_API_KEY missing",
                             "set LLM_PROVIDER=local or add key to .env")
                return False

        # Local LM Studio
        if LLM.PROVIDER == "local":
            try:
                import aiohttp
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=4)
                ) as sess:
                    async with sess.get(
                        f"{LLM.LOCAL_URL.rstrip('/')}/models"
                    ) as resp:
                        if resp.status == 200:
                            self._record(_PASS, "LLM provider configured",
                                         f"local ({LLM.LOCAL_URL})")
                            return True
                        self._record(_WARN, "LM Studio unreachable",
                                     f"{LLM.LOCAL_URL} → HTTP {resp.status} — agents degraded")
                        return True  # warn, not fatal — data collection still works
            except Exception as e:
                self._record(_WARN, "LM Studio unreachable",
                             f"{str(e)[:60]} — start LM Studio to enable agents")
                return True  # warn, not fatal

        self._record(_WARN, f"Unknown LLM provider '{LLM.PROVIDER}'",
                     "expected 'anthropic' or 'local'")
        return True  # unknown provider is a warn, not a hard abort

    def _check_api_key(self, key_name: str, key_val, label: str, detail: str = ""):
        if key_val:
            self._record(_PASS, f"{label} configured")
        else:
            self._record(_WARN, f"{label} missing", detail or f"{key_name} not set in .env")

    def _check_python_version(self) -> bool:
        vi = sys.version_info
        if vi >= (3, 9):
            self._record(_PASS, "Python version", f"{vi.major}.{vi.minor}.{vi.micro}")
            return True
        self._record(_FAIL, "Python 3.9+ required",
                     f"found {vi.major}.{vi.minor}.{vi.micro}")
        return False

    def _check_required_packages(self) -> bool:
        missing = []
        for pkg in ("fastapi", "uvicorn", "aiohttp", "yfinance"):
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        if missing:
            self._record(_FAIL, "Required packages missing",
                         "pip install " + " ".join(missing))
            return False
        self._record(_PASS, "Core packages installed")
        return True

    # ── Main entry point ──────────────────────────────────────────────────

    async def validate(self, mock_mode: bool = False) -> bool:
        self._results.clear()
        width = 62

        print()
        print("=" * width)
        print("  QUANT AGENT — Pre-flight Validation")
        print("=" * width)

        critical_ok = True
        critical_ok &= self._check_python_version()
        critical_ok &= self._check_required_packages()
        critical_ok &= self._check_storage()
        self._check_log_dir()
        critical_ok &= self._check_db()
        critical_ok &= await self._check_llm(mock_mode)

        # Non-critical API keys
        self._check_api_key("POLYGON_API_KEY",  KEYS.POLYGON,
                             "Polygon (market data)", "limited to yfinance fallback")
        self._check_api_key("FINNHUB_API_KEY",  KEYS.FINNHUB,
                             "Finnhub (fundamentals)", "sentiment/insiders unavailable")
        self._check_api_key("TAVILY_API_KEY",   KEYS.TAVILY,
                             "Tavily (research)", "LLM research degraded")
        self._check_api_key("ALPACA_API_KEY",   KEYS.ALPACA_KEY,
                             "Alpaca (broker)", "live execution unavailable")

        # Print results
        for level, label, detail in self._results:
            print(_fmt(level, label, detail))

        warnings = sum(1 for r in self._results if r[0] == _WARN)
        failures = sum(1 for r in self._results if r[0] == _FAIL)

        print()
        if failures:
            print(f"  {_RED}[X] {failures} critical failure(s). System cannot start.{_RESET}")
        elif warnings:
            print(f"  {_YELLOW}[!] {warnings} warning(s). Starting in degraded mode.{_RESET}")
        else:
            print(f"  {_GREEN}[OK] All checks passed.{_RESET}")
        print("=" * width)
        print()

        self.passed = critical_ok
        return critical_ok


startup_validator = StartupValidator()
