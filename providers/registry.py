"""
providers/registry.py -- Provider orchestration layer

Routes requests to the best available provider per market.
Handles fallback chains and health monitoring.
"""
import asyncio
from typing import Dict, List, Optional

from providers.base import BaseProvider
from providers.polygon_provider import PolygonProvider
from providers.alpaca_provider import AlpacaProvider
from providers.finnhub_provider import FinnhubProvider
from providers.coingecko_provider import CoinGeckoProvider
from providers.tavily_provider import TavilyProvider
from core.models import Tick, TickerProfile, Market
from core.logger import get_logger

log = get_logger("registry")


class ProviderRegistry:
    """
    Manages all providers and routes requests intelligently.
    Primary -> fallback chain per market.
    """

    def __init__(self):
        # Initialize all providers
        self.polygon   = PolygonProvider()
        self.alpaca    = AlpacaProvider()
        self.finnhub   = FinnhubProvider()
        self.coingecko = CoinGeckoProvider()
        self.tavily    = TavilyProvider()

        # Primary provider per market
        self._market_primary: Dict[Market, BaseProvider] = {
            Market.US_STOCK: self.polygon,
            Market.PENNY:    self.polygon,
            Market.NASDAQ:   self.polygon,
            Market.ETF:      self.polygon,
            Market.CRYPTO:   self.coingecko,
            Market.INTL:     self.finnhub,
        }

        # Fallback chains: if primary fails, try these in order
        self._fallbacks: Dict[Market, List[BaseProvider]] = {
            Market.US_STOCK: [self.alpaca, self.finnhub],
            Market.PENNY:    [self.alpaca, self.finnhub],
            Market.NASDAQ:   [self.alpaca, self.finnhub],
            Market.ETF:      [self.alpaca],
            Market.CRYPTO:   [],
            Market.INTL:     [],
        }

        self._health: Dict[str, bool] = {}

    async def startup_check(self):
        """Check all providers on startup."""
        providers = [self.polygon, self.alpaca, self.finnhub, self.coingecko]
        results = await asyncio.gather(
            *[p.health_check() for p in providers],
            return_exceptions=True
        )
        for provider, result in zip(providers, results):
            ok = result is True
            self._health[provider.name] = ok
            status = "OK OK" if ok else "FAIL FAIL"
            log.info(f"  {provider.name:<12} {status}")

    def get_provider(self, market: Market) -> BaseProvider:
        return self._market_primary.get(market, self.polygon)

    async def get_universe(self, market: Market) -> List[str]:
        provider = self.get_provider(market)
        try:
            return await provider.get_universe(market)
        except Exception as e:
            log.error(f"Universe fetch failed for {market}: {e}")
            for fb in self._fallbacks.get(market, []):
                try:
                    return await fb.get_universe(market)
                except:
                    continue
        return []

    async def get_batch_quotes(self, symbols: List[str], market: Market) -> List[Tick]:
        provider = self.get_provider(market)
        try:
            return await provider.get_batch_quotes(symbols)
        except Exception as e:
            log.error(f"Batch quotes failed [{market}]: {e}")
            for fb in self._fallbacks.get(market, []):
                try:
                    return await fb.get_batch_quotes(symbols)
                except:
                    continue
        return []

    async def get_quote(self, symbol: str, market: Market = Market.US_STOCK) -> Optional[Tick]:
        provider = self.get_provider(market)
        try:
            return await provider.get_quote(symbol)
        except Exception as e:
            log.error(f"Quote failed {symbol}: {e}")
        return None

    async def get_profile(self, symbol: str, market: Market = Market.US_STOCK) -> Optional[TickerProfile]:
        """Enriched profile: Polygon base + Finnhub fundamentals merged."""
        tasks = []
        if market in (Market.US_STOCK, Market.PENNY, Market.NASDAQ, Market.ETF):
            tasks = [
                self.polygon.get_profile(symbol),
                self.finnhub.get_profile(symbol),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            poly_p = results[0] if isinstance(results[0], TickerProfile) else None
            finn_p = results[1] if isinstance(results[1], TickerProfile) else None
            # Merge: Polygon is primary, Finnhub fills gaps
            if poly_p and finn_p:
                if not poly_p.sector and finn_p.sector:
                    poly_p.sector = finn_p.sector
                if not poly_p.name and finn_p.name:
                    poly_p.name = finn_p.name
                poly_p.sentiment_score = finn_p.sentiment_score
                poly_p.news_count_24h = finn_p.news_count_24h
                poly_p.pe_ratio = finn_p.pe_ratio
            return poly_p or finn_p
        elif market == Market.CRYPTO:
            return await self.coingecko.get_profile(symbol)
        elif market == Market.INTL:
            return await self.finnhub.get_profile(symbol)
        return None

    async def get_news(self, symbol: str) -> List[Dict]:
        """Get news from multiple sources merged."""
        tasks = [
            self.polygon.get_news(symbol, limit=5),
            self.finnhub.get_news(symbol, limit=5),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged = []
        for r in results:
            if isinstance(r, list):
                merged.extend(r)
        return merged[:10]

    async def close_all(self):
        for p in [self.polygon, self.alpaca, self.finnhub, self.coingecko, self.tavily]:
            try:
                await p.close()
            except:
                pass
        log.info("All providers closed")


# Module-level singleton
registry = ProviderRegistry()
