"""
providers/coingecko_provider.py -- CoinGecko adapter
Handles: Crypto top-N, detailed coin data, market trends
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional

from providers.base import BaseProvider
from core.models import Tick, TickerProfile, Market
from core.logger import get_logger
from config.settings import KEYS, MARKET

log = get_logger("coingecko")

BASE = "https://api.coingecko.com/api/v3"


class CoinGeckoProvider(BaseProvider):
    name = "coingecko"
    markets = [Market.CRYPTO]

    def __init__(self):
        self._key = KEYS.COINGECKO
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate = asyncio.Semaphore(4)
        self._coin_list: Optional[List[Dict]] = None
        self._id_map: Dict[str, str] = {}  # symbol -> coin_id

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"x-cg-demo-api-key": self._key}
            )
        return self._session

    async def _get(self, path: str, params: Dict = None) -> Optional[any]:
        async with self._rate:
            try:
                # aiohttp serializes bool as 'True'/'False'; CoinGecko wants 'true'/'false'
                safe_params = {}
                for k, v in (params or {}).items():
                    if isinstance(v, bool):
                        safe_params[k] = "true" if v else "false"
                    else:
                        safe_params[k] = v

                s = await self._sess()
                async with s.get(f"{BASE}{path}", params=safe_params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        log.warning("CoinGecko rate limit -- sleeping 30s")
                        await asyncio.sleep(30)
                    else:
                        log.debug(f"CoinGecko {path} -> {resp.status}")
            except Exception as e:
                log.error(f"CoinGecko error: {e}")
        return None

    async def _ensure_id_map(self):
        if self._id_map:
            return
        data = await self._get("/coins/list")
        if data:
            self._coin_list = data
            for coin in data:
                sym = coin["symbol"].upper()
                if sym not in self._id_map:
                    self._id_map[sym] = coin["id"]
        log.info(f"CoinGecko ID map built: {len(self._id_map)} coins")

    async def get_universe(self, market: Market) -> List[str]:
        data = await self._get("/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": MARKET.CRYPTO_TOP_N,
            "page": 1,
            "sparkline": False,
        })
        if not data:
            return []
        symbols = [c["symbol"].upper() for c in data]
        for c in data:
            self._id_map[c["symbol"].upper()] = c["id"]
        log.info(f"CoinGecko universe: {len(symbols)} coins")
        return symbols

    async def get_batch_quotes(self, symbols: List[str]) -> List[Tick]:
        data = await self._get("/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(max(len(symbols), 50), 250),
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h",
        })
        if not data:
            return []
        sym_set = {s.upper() for s in symbols}
        ticks = []
        for c in data:
            sym = c["symbol"].upper()
            if not sym_set or sym in sym_set:
                self._id_map[sym] = c["id"]
                ticks.append(Tick(
                    symbol=sym,
                    price=c.get("current_price", 0) or 0,
                    volume=c.get("total_volume", 0) or 0,
                    change_pct=c.get("price_change_percentage_24h", 0) or 0,
                    market=Market.CRYPTO,
                    provider="coingecko",
                ))
        return ticks

    async def get_quote(self, symbol: str) -> Optional[Tick]:
        ticks = await self.get_batch_quotes([symbol])
        return ticks[0] if ticks else None

    async def get_profile(self, symbol: str) -> Optional[TickerProfile]:
        await self._ensure_id_map()
        coin_id = self._id_map.get(symbol.upper())
        if not coin_id:
            return None
        data = await self._get(f"/coins/{coin_id}", {
            "localization": False,
            "tickers": False,
            "market_data": True,
            "community_data": True,
            "developer_data": False,
        })
        if not data:
            return None
        md = data.get("market_data", {})
        price = md.get("current_price", {}).get("usd", 0) or 0
        return TickerProfile(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            market=Market.CRYPTO,
            price=price,
            volume_24h=md.get("total_volume", {}).get("usd", 0) or 0,
            market_cap=md.get("market_cap", {}).get("usd", 0) or 0,
            change_pct=md.get("price_change_percentage_24h", 0) or 0,
            change_5d=md.get("price_change_percentage_7d", 0) or 0,
            sentiment_score=self._sentiment_from_votes(data),
            provider_data={
                "coingecko_rank": data.get("market_cap_rank", 0),
                "description": data.get("description", {}).get("en", "")[:300],
            }
        )

    def _sentiment_from_votes(self, data: Dict) -> float:
        up = data.get("sentiment_votes_up_percentage", 50) or 50
        return (up / 100 - 0.5) * 2

    async def get_trending(self) -> List[Dict]:
        data = await self._get("/search/trending")
        if not data:
            return []
        return [
            {
                "symbol": c["item"]["symbol"].upper(),
                "name": c["item"]["name"],
                "rank": c["item"].get("market_cap_rank", 999),
            }
            for c in data.get("coins", [])
        ]

    async def health_check(self) -> bool:
        data = await self._get("/ping")
        return data is not None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
