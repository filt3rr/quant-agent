"""
providers/tavily_provider.py -- Tavily search adapter
Powers AI agent research: news, earnings context, macro events
"""
import asyncio
import aiohttp
import datetime
from typing import Dict, List, Optional

from core.logger import get_logger
from core.rate_limiter import tavily_limiter
from config.settings import KEYS

log = get_logger("tavily")

BASE = "https://api.tavily.com"


class TavilyProvider:
    name = "tavily"

    def __init__(self):
        self._key = KEYS.TAVILY
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate = tavily_limiter

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)
            )
        return self._session

    async def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> List[Dict]:
        """
        Returns list of: {title, url, content, score, published_date}
        search_depth: "basic" (fast) | "advanced" (more thorough)
        """
        async with tavily_limiter:
            try:
                s = await self._sess()
                payload = {
                    "api_key": self._key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "include_answer": True,
                    "include_images": False,
                }
                async with s.post(f"{BASE}/search", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for r in data.get("results", []):
                            results.append({
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "content": r.get("content", "")[:500],
                                "score": r.get("score", 0),
                                "published": r.get("published_date", ""),
                            })
                        return results
                    log.debug(f"Tavily search -> {resp.status}")
            except Exception as e:
                log.error(f"Tavily error: {e}")
        return []

    async def search_news(self, symbol: str, company_name: str = "") -> List[Dict]:
        """Targeted news search for a ticker."""
        year = datetime.date.today().year
        q = f"{symbol} {company_name} stock news analysis {year}".strip()
        return await self.search(q, max_results=5)

    async def search_macro(self, topic: str) -> List[Dict]:
        """Search macro/market events."""
        year = datetime.date.today().year
        return await self.search(f"{topic} market impact {year}", max_results=5, search_depth="advanced")

    async def search_earnings(self, symbol: str) -> List[Dict]:
        """Search for earnings-related information."""
        return await self.search(f"{symbol} earnings results guidance analyst", max_results=5)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
