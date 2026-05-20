"""
providers/polygon_provider.py -- Polygon.io adapter
Handles: US equities, penny stocks, NASDAQ, ETFs
Real-time quotes, snapshots, news, financial details
"""
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional

from providers.base import BaseProvider
from core.models import Tick, TickerProfile, Market
from core.logger import get_logger
from core.rate_limiter import polygon_limiter
from config.settings import KEYS, MARKET

log = get_logger("polygon")

BASE = "https://api.polygon.io"


class PolygonProvider(BaseProvider):
    name = "polygon"
    markets = [Market.US_STOCK, Market.PENNY, Market.NASDAQ, Market.ETF]

    def __init__(self):
        self._key = KEYS.POLYGON
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit = polygon_limiter

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Authorization": f"Bearer {self._key}"}
            )
        return self._session

    async def _get(self, path: str, params: Dict = None) -> Optional[Dict]:
        async with polygon_limiter:
            try:
                s = await self._session_get()
                async with s.get(f"{BASE}{path}", params=params or {}) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        log.warning("Polygon rate limit hit -- backing off")
                        await asyncio.sleep(2)
                    else:
                        log.debug(f"Polygon {path} -> {resp.status}")
            except Exception as e:
                log.error(f"Polygon request error: {e}")
        return None

    async def get_quote(self, symbol: str) -> Optional[Tick]:
        data = await self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        if not data or "ticker" not in data:
            return None
        t = data["ticker"]
        day = t.get("day", {})
        prev = t.get("prevDay", {})
        price = day.get("c") or t.get("lastTrade", {}).get("p", 0)
        prev_close = prev.get("c", price)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        return Tick(
            symbol=symbol,
            price=price,
            volume=day.get("v", 0),
            bid=t.get("lastQuote", {}).get("P", 0),
            ask=t.get("lastQuote", {}).get("p", 0),
            change_pct=change_pct,
            market=Market.PENNY if price < MARKET.PENNY_MAX_PRICE else Market.US_STOCK,
            provider="polygon",
        )

    async def get_batch_quotes(self, symbols: List[str]) -> List[Tick]:
        """Use snapshot endpoint for batches."""
        tickers_str = ",".join(symbols[:100])
        data = await self._get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"tickers": tickers_str}
        )
        results = []
        if not data or "tickers" not in data:
            return results
        for t in data["tickers"]:
            sym = t.get("ticker", "")
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            price = day.get("c") or t.get("lastTrade", {}).get("p", 0)
            prev_close = prev.get("c", price)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            results.append(Tick(
                symbol=sym,
                price=price,
                volume=day.get("v", 0),
                change_pct=change_pct,
                market=Market.PENNY if price < MARKET.PENNY_MAX_PRICE else Market.US_STOCK,
                provider="polygon",
            ))
        return results

    async def get_universe(self, market: Market) -> List[str]:
        """Fetch full ticker list from Polygon reference endpoint."""
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1000,
        }
        if market == Market.ETF:
            params["type"] = "ETF"
        elif market == Market.PENNY:
            # Will filter by price post-fetch
            params["market"] = "stocks"

        all_tickers = []
        cursor = None
        pages = 0
        max_pages = 3  # Cap to avoid long startup

        while pages < max_pages:
            if cursor:
                params["cursor"] = cursor
            data = await self._get("/v3/reference/tickers", params)
            if not data:
                break
            results = data.get("results", [])
            for r in results:
                all_tickers.append(r["ticker"])
            cursor = data.get("next_url", "")
            if not cursor:
                break
            # Extract cursor param from next_url
            if "cursor=" in cursor:
                cursor = cursor.split("cursor=")[1].split("&")[0]
            pages += 1

        log.info(f"Polygon universe [{market.value}]: {len(all_tickers)} tickers")
        return all_tickers

    async def get_profile(self, symbol: str) -> Optional[TickerProfile]:
        # Fetch details and snapshot in parallel
        details_task = self._get(f"/v3/reference/tickers/{symbol}")
        snap_task = self._get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        )
        details, snap = await asyncio.gather(details_task, snap_task)

        profile = TickerProfile(symbol=symbol, market=Market.US_STOCK)

        if details and "results" in details:
            r = details["results"]
            profile.name = r.get("name", "")
            profile.sector = r.get("sic_description", "")
            profile.market_cap = r.get("market_cap", 0) or 0

        if snap and "ticker" in snap:
            t = snap["ticker"]
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            price = day.get("c") or t.get("lastTrade", {}).get("p", 0)
            prev_close = prev.get("c", price) or price
            profile.price = price
            profile.volume_24h = day.get("v", 0)
            profile.change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
            if price < MARKET.PENNY_MAX_PRICE:
                profile.market = Market.PENNY

        return profile

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        data = await self._get(f"/v2/reference/news", {"ticker": symbol, "limit": limit})
        if not data:
            return []
        return [
            {
                "title": n.get("title", ""),
                "publisher": n.get("publisher", {}).get("name", ""),
                "url": n.get("article_url", ""),
                "published": n.get("published_utc", ""),
                "sentiment": n.get("insights", [{}])[0].get("sentiment", "neutral")
                              if n.get("insights") else "neutral",
            }
            for n in data.get("results", [])
        ]

    async def health_check(self) -> bool:
        data = await self._get("/v1/marketstatus/now")
        return data is not None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
