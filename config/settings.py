"""
config/settings.py -- Centralized configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

def _get(key: str, default=None, required=False):
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val

class APIKeys:
    POLYGON    = _get("POLYGON_API_KEY")
    ALPACA_KEY = _get("ALPACA_API_KEY")
    ALPACA_SEC = _get("ALPACA_SECRET_KEY")
    ALPACA_URL = _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    ALPACA_DATA= _get("ALPACA_DATA_URL", "https://data.alpaca.markets")
    COINGECKO  = _get("COINGECKO_API_KEY")
    FINNHUB    = _get("FINNHUB_API_KEY")
    TAVILY     = _get("TAVILY_API_KEY")
    ANTHROPIC  = _get("ANTHROPIC_API_KEY")

class LLMConfig:
    """LLM routing: 'local' uses LM Studio, 'anthropic' uses Claude API."""
    PROVIDER      = _get("LLM_PROVIDER", "anthropic").lower()   # "local" | "anthropic"
    LOCAL_URL     = _get("LLM_LOCAL_URL", "http://localhost:1234/v1")
    LOCAL_MODEL   = _get("LLM_LOCAL_MODEL", "qwen2.5-14b-instruct")
    LOCAL_TIMEOUT = int(_get("LLM_LOCAL_TIMEOUT", 120))
    MAX_TOKENS    = int(_get("LLM_MAX_TOKENS", 800))

    @classmethod
    def is_local(cls) -> bool:
        return cls.PROVIDER == "local"

    @classmethod
    def display_name(cls) -> str:
        if cls.is_local():
            return cls.LOCAL_MODEL.split("/")[-1][:20]
        return "claude-sonnet"

class SystemConfig:
    DASHBOARD_PORT   = int(_get("DASHBOARD_PORT", 8765))
    SCAN_INTERVAL    = int(_get("SCAN_INTERVAL_SECONDS", 60))
    WATCHLIST_SIZE   = int(_get("WATCHLIST_SIZE", 150))
    MAX_WORKERS      = int(_get("MAX_AGENT_WORKERS", 6))
    LOG_LEVEL        = _get("LOG_LEVEL", "INFO")
    ROOT_DIR         = _ROOT
    LOG_DIR          = _ROOT / "logs"
    STORAGE_DIR      = _ROOT / "storage"
    # Alert email delivery (optional — leave blank to disable)
    ALERT_EMAIL_TO   = _get("ALERT_EMAIL_TO", "")
    SMTP_HOST        = _get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT        = int(_get("SMTP_PORT", 587))
    SMTP_USER        = _get("SMTP_USER", "")
    SMTP_PASS        = _get("SMTP_PASS", "")
    # Dashboard API key (optional — leave blank to disable auth)
    DASHBOARD_API_KEY = _get("DASHBOARD_API_KEY", "")

class MarketConfig:
    ENABLE_US_STOCKS    = _get("ENABLE_US_STOCKS", "true").lower() == "true"
    ENABLE_PENNY_STOCKS = _get("ENABLE_PENNY_STOCKS", "true").lower() == "true"
    ENABLE_CRYPTO       = _get("ENABLE_CRYPTO", "true").lower() == "true"
    ENABLE_INTERNATIONAL= _get("ENABLE_INTERNATIONAL", "true").lower() == "true"
    ENABLE_ETF          = _get("ENABLE_ETF", "true").lower() == "true"
    PENNY_MAX_PRICE     = 5.0
    MIN_ADV_US          = 50_000
    MIN_ADV_PENNY       = 10_000
    CRYPTO_TOP_N        = 100
    US_TICKER_CAP       = 800
    PENNY_TICKER_CAP    = 400
    INTL_TICKER_CAP     = 200
    CRYPTO_CAP          = 100
    ETF_CAP             = 100

KEYS   = APIKeys()
LLM    = LLMConfig()
SYS    = SystemConfig()
MARKET = MarketConfig()

SYS.LOG_DIR.mkdir(parents=True, exist_ok=True)
SYS.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
