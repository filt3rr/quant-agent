"""
providers/base.py -- Abstract base class for all market data providers
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from core.models import Tick, TickerProfile, Market


class BaseProvider(ABC):
    name: str = "base"
    markets: List[Market] = []

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Tick]:
        """Fetch a single quote for symbol."""
        ...

    @abstractmethod
    async def get_batch_quotes(self, symbols: List[str]) -> List[Tick]:
        """Fetch quotes for a list of symbols efficiently."""
        ...

    @abstractmethod
    async def get_universe(self, market: Market) -> List[str]:
        """Return list of ticker symbols for a given market segment."""
        ...

    @abstractmethod
    async def get_profile(self, symbol: str) -> Optional[TickerProfile]:
        """Return enriched profile with fundamentals, sector, etc."""
        ...

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Return recent news items. Optional -- return [] if unsupported."""
        return []

    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        return True
