"""
providers/finnhub_provider.py -- Finnhub adapter
Handles: Fundamentals, earnings calendar, insider transactions,
         sentiment scores, analyst recommendations, international stocks
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional

from providers.base import BaseProvider
from core.models import Tick, TickerProfile, Market
from core.logger import get_logger
from core.rate_limiter import finnhub_limiter
from config.settings import KEYS

log = get_logger("finnhub")

BASE = "https://finnhub.io/api/v1"

# Major international ETFs / indices for international coverage
INTL_SYMBOLS = [
    "BABA", "TSM", "ASML", "SAP", "NVO", "TCEHY", "AZN",  # ADRs
    "EEM", "VEA", "IEFA", "EFA", "VWO",                    # Intl ETFs
    "SONY", "TM", "HMC", "SNE", "NTDOY",                   # Japan
    "BTI", "RIO", "BP", "SHEL", "GSK",                     # UK/Europe
]


class FinnhubProvider(BaseProvider):
    name = "finnhub"
    markets = [Market.US_STOCK, Market.INTL, Market.NASDAQ]

    def __init__(self):
        self._key = KEYS.FINNHUB
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate = finnhub_limiter

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def _get(self, path: str, params: Dict = None) -> Optional[Dict]:
        async with finnhub_limiter:
            try:
                p = params or {}
                p["token"] = self._key
                s = await self._sess()
                async with s.get(f"{BASE}{path}", params=p) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        log.warning("Finnhub rate limit -- sleeping")
                        await asyncio.sleep(5)
                    log.debug(f"Finnhub {path} -> {resp.status}")
            except Exception as e:
                log.error(f"Finnhub error: {e}")
        return None

    async def get_quote(self, symbol: str) -> Optional[Tick]:
        data = await self._get("/quote", {"symbol": symbol})
        if not data or not data.get("c"):
            return None
        price = data["c"]
        prev = data.get("pc", price)
        return Tick(
            symbol=symbol,
            price=price,
            volume=data.get("v", 0),
            change_pct=data.get("dp", 0),
            provider="finnhub",
        )

    async def get_batch_quotes(self, symbols: List[str]) -> List[Tick]:
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Tick)]

    async def get_universe(self, market: Market) -> List[str]:
        if market == Market.INTL:
            return INTL_SYMBOLS
        # For US, return a curated high-quality list
        data = await self._get("/stock/symbol", {"exchange": "US"})
        if not data:
            return []
        symbols = [d["symbol"] for d in data if d.get("type") in ("Common Stock", "EQS")]
        log.info(f"Finnhub universe [US]: {len(symbols)} symbols")
        return symbols[:500]

    async def get_profile(self, symbol: str) -> Optional[TickerProfile]:
        # Fetch profile2, quote, and recency-weighted sentiment in parallel
        profile_task = self._get("/stock/profile2", {"symbol": symbol})
        quote_task   = self._get("/quote", {"symbol": symbol})
        sent_task    = self._get("/news-sentiment", {"symbol": symbol})
        news_task    = self._get("/company-news", {
            "symbol": symbol, "from": self._days_ago(3), "to": self._today()
        })

        profile_data, quote_data, sent_data, news_data = await asyncio.gather(
            profile_task, quote_task, sent_task, news_task
        )

        tp = TickerProfile(symbol=symbol)
        if profile_data:
            tp.name = profile_data.get("name", "")
            tp.sector = profile_data.get("finnhubIndustry", "")
            tp.market_cap = (profile_data.get("marketCapitalization", 0) or 0) * 1e6
        if quote_data and quote_data.get("c"):
            tp.price = quote_data["c"]
            tp.volume_24h = quote_data.get("v", 0)
            tp.change_pct = quote_data.get("dp", 0)

        # Recency-weighted sentiment (prefer individual articles over aggregate score)
        import math, datetime as _dt
        now_ts = _dt.datetime.utcnow().timestamp()
        if news_data and isinstance(news_data, list) and len(news_data) > 0:
            total_w = total_s = 0.0
            for art in news_data[:20]:
                pub_ts = art.get("datetime", 0)
                if not pub_ts:
                    continue
                hours_old = max(0, (now_ts - pub_ts) / 3600)
                weight = math.exp(-hours_old * math.log(2) / 12)
                raw = art.get("sentiment", 0.0)
                if isinstance(raw, (int, float)):
                    total_w += weight
                    total_s += weight * raw
            if total_w > 0.01:
                tp.sentiment_score = round(total_s / total_w, 4)
            elif sent_data:
                score = sent_data.get("companyNewsScore", 0.5)
                tp.sentiment_score = round((score - 0.5) * 2, 4)
            tp.news_count_24h = len(news_data)
        elif sent_data:
            score = sent_data.get("companyNewsScore", 0.5)
            tp.sentiment_score = round((score - 0.5) * 2, 4)
            tp.news_count_24h = len(sent_data.get("buzz", {}).get("articlesInLastWeek", []))

        return tp

    async def get_recency_weighted_sentiment(self, symbol: str) -> float:
        """
        Compute recency-weighted news sentiment using exponential decay (half-life = 12h).
        Returns score in [-1, +1]. Falls back to companyNewsScore if no articles found.
        """
        import datetime, math
        data = await self._get("/company-news", {
            "symbol": symbol,
            "from": self._days_ago(7),
            "to": self._today(),
        })
        if not data:
            sent = await self._get("/news-sentiment", {"symbol": symbol})
            if sent:
                raw = sent.get("companyNewsScore", 0.5)
                return round((raw - 0.5) * 2, 4)
            return 0.0

        now_ts = datetime.datetime.utcnow().timestamp()
        half_life_s = 12 * 3600  # 12-hour half-life
        total_w = 0.0
        total_s = 0.0
        for article in data[:30]:
            pub_ts = article.get("datetime", 0)
            if not pub_ts:
                continue
            hours_old = max(0, (now_ts - pub_ts) / 3600)
            weight = math.exp(-hours_old * math.log(2) / 12)  # exp decay
            raw_sentiment = article.get("sentiment", 0.0)
            if isinstance(raw_sentiment, (int, float)):
                total_w += weight
                total_s += weight * raw_sentiment

        if total_w < 0.01:
            sent = await self._get("/news-sentiment", {"symbol": symbol})
            if sent:
                raw = sent.get("companyNewsScore", 0.5)
                return round((raw - 0.5) * 2, 4)
            return 0.0

        return round(total_s / total_w, 4)

    async def get_sentiment(self, symbol: str) -> Dict:
        data = await self._get("/news-sentiment", {"symbol": symbol})
        return data or {}

    async def get_insider_transactions(self, symbol: str) -> List[Dict]:
        data = await self._get("/stock/insider-transactions", {"symbol": symbol})
        if not data or "data" not in data:
            return []
        return data["data"][:10]

    async def get_earnings_calendar(self) -> List[Dict]:
        """Upcoming earnings -- high volatility plays."""
        import datetime
        today = datetime.date.today().isoformat()
        next_wk = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
        data = await self._get("/calendar/earnings", {"from": today, "to": next_wk})
        return data.get("earningsCalendar", []) if data else []

    async def get_analyst_recommendation(self, symbol: str) -> Dict:
        data = await self._get("/stock/recommendation", {"symbol": symbol})
        if not data or not isinstance(data, list):
            return {}
        return data[0] if data else {}

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        data = await self._get("/company-news", {
            "symbol": symbol,
            "from": self._days_ago(7),
            "to": self._today()
        })
        if not data:
            return []
        return [
            {
                "title": n.get("headline", ""),
                "publisher": n.get("source", ""),
                "url": n.get("url", ""),
                "published": str(n.get("datetime", "")),
                "summary": n.get("summary", "")[:200],
                "sentiment": "positive" if n.get("sentiment", 0) > 0 else
                             "negative" if n.get("sentiment", 0) < 0 else "neutral",
            }
            for n in data[:limit]
        ]

    def _today(self):
        import datetime
        return datetime.date.today().isoformat()

    def _days_ago(self, n: int):
        import datetime
        return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()

    async def health_check(self) -> bool:
        data = await self._get("/quote", {"symbol": "AAPL"})
        return data is not None and "c" in data

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
