"""
core/market_hours.py -- Market hours awareness

US stock market: Mon-Fri 09:30-16:00 ET
Extended hours: 04:00-20:00 ET (pre/post market)
Crypto: 24/7

Timezone: uses stdlib zoneinfo (Python 3.9+) with pytz fallback.
"""
import datetime
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        import pytz
        _ET = pytz.timezone("America/New_York")
    except ImportError:
        _ET = None  # fallback: assume market is open

_MARKET_OPEN  = datetime.time(9, 30)
_MARKET_CLOSE = datetime.time(16, 0)
_EXTENDED_OPEN  = datetime.time(4, 0)
_EXTENDED_CLOSE = datetime.time(20, 0)


def _now_et() -> Optional[datetime.datetime]:
    if _ET is None:
        return None
    return datetime.datetime.now(tz=_ET)


def is_market_open(extended: bool = False) -> bool:
    """Return True if US equities market is currently open."""
    now = _now_et()
    if now is None:
        return True  # no timezone support — assume open
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now.time()
    if extended:
        return _EXTENDED_OPEN <= t < _EXTENDED_CLOSE
    return _MARKET_OPEN <= t < _MARKET_CLOSE


def is_crypto_trading() -> bool:
    """Crypto trades 24/7."""
    return True


def market_status() -> dict:
    """Return a status dict for the dashboard."""
    now = _now_et()
    if now is None:
        return {"us_stocks": True, "crypto": True, "timezone": "unknown", "time_et": "unknown"}
    t = now.time()
    is_weekday = now.weekday() < 5
    regular = is_weekday and _MARKET_OPEN <= t < _MARKET_CLOSE
    extended = is_weekday and _EXTENDED_OPEN <= t < _EXTENDED_CLOSE
    return {
        "us_stocks": regular,
        "extended_hours": extended,
        "crypto": True,
        "timezone": "America/New_York",
        "time_et": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }
