"""
providers/mock_provider.py -- Realistic mock data for testing

Used when live APIs are unavailable (network restrictions, testing).
Generates realistic price data with proper market dynamics.
"""
import asyncio
import random
import math
import time
from typing import Dict, List, Optional

from core.models import Tick, TickerProfile, Market

# Realistic seed data
MOCK_STOCKS = [
    ("AAPL",  "Apple Inc",              Market.US_STOCK, 193.42, "Technology",    "Consumer Electronics", 3_000_000_000_000),
    ("NVDA",  "NVIDIA Corp",            Market.US_STOCK, 875.20, "Technology",    "Semiconductors",       2_100_000_000_000),
    ("MSFT",  "Microsoft Corp",         Market.US_STOCK, 415.80, "Technology",    "Software",             3_100_000_000_000),
    ("TSLA",  "Tesla Inc",              Market.US_STOCK, 243.60, "Consumer Disc", "Auto Manufacturers",     775_000_000_000),
    ("META",  "Meta Platforms",         Market.US_STOCK, 492.10, "Technology",    "Internet Content",     1_250_000_000_000),
    ("AMZN",  "Amazon.com Inc",         Market.US_STOCK, 180.90, "Consumer Disc", "Internet Retail",      1_880_000_000_000),
    ("GOOGL", "Alphabet Inc",           Market.US_STOCK, 163.40, "Technology",    "Internet Services",    2_020_000_000_000),
    ("JPM",   "JPMorgan Chase",         Market.US_STOCK, 207.80, "Financial",     "Banks",                  590_000_000_000),
    ("BAC",   "Bank of America",        Market.US_STOCK,  39.20, "Financial",     "Banks",                  298_000_000_000),
    ("XOM",   "Exxon Mobil",            Market.US_STOCK, 117.30, "Energy",        "Oil & Gas",              470_000_000_000),
    ("SPY",   "SPDR S&P 500 ETF",       Market.ETF,      520.80, "ETF",           "Large Blend",                        0),
    ("QQQ",   "Invesco QQQ Trust",      Market.ETF,      442.30, "ETF",           "Large Growth",                       0),
    ("SOXL",  "Direxion Semi Bull 3x",  Market.ETF,       32.60, "ETF",           "Leveraged",                          0),
    # Penny stocks
    ("MULN",  "Mullen Automotive",      Market.PENNY,     0.042, "Consumer Disc", "Auto",                    15_000_000),
    ("BBIG",  "Vinco Ventures",         Market.PENNY,     0.089, "Consumer Disc", "Entertainment",           22_000_000),
    ("PROG",  "Progenity Inc",          Market.PENNY,     1.240, "Healthcare",    "Diagnostics",             85_000_000),
    ("CENN",  "Cenntro Electric",       Market.PENNY,     0.310, "Industrials",   "Vehicles",                45_000_000),
    ("EVGO",  "EVgo Inc",               Market.US_STOCK,  3.820, "Utilities",     "EV Charging",            730_000_000),
    # Crypto
    ("BTC",   "Bitcoin",                Market.CRYPTO, 63_420.0, "Crypto",        "Store of Value",   1_240_000_000_000),
    ("ETH",   "Ethereum",               Market.CRYPTO,  3_124.0, "Crypto",        "Smart Contracts",    375_000_000_000),
    ("SOL",   "Solana",                 Market.CRYPTO,    142.3, "Crypto",        "Layer 1",             62_000_000_000),
    ("DOGE",  "Dogecoin",               Market.CRYPTO,    0.162, "Crypto",        "Meme",                22_000_000_000),
    ("ADA",   "Cardano",                Market.CRYPTO,    0.449, "Crypto",        "Layer 1",             15_800_000_000),
    ("AVAX",  "Avalanche",              Market.CRYPTO,     35.8, "Crypto",        "Layer 1",             14_600_000_000),
    ("LINK",  "Chainlink",              Market.CRYPTO,     14.2, "Crypto",        "Oracle",               8_300_000_000),
    # International ADRs
    ("BABA",  "Alibaba Group",          Market.INTL,      73.40, "Technology",    "Internet Retail",     188_000_000_000),
    ("TSM",   "Taiwan Semiconductor",   Market.INTL,     143.20, "Technology",    "Semiconductors",      742_000_000_000),
    ("ASML",  "ASML Holding",           Market.INTL,     818.60, "Technology",    "Semiconductor Equip", 321_000_000_000),
    ("SONY",  "Sony Group Corp",        Market.INTL,      80.10, "Technology",    "Consumer Electronics", 97_000_000_000),
    ("NVO",   "Novo Nordisk",           Market.INTL,      81.20, "Healthcare",    "Drug Manufacturers",  366_000_000_000),
]


class MockProvider:
    """Generates realistic mock market data with live-like volatility."""

    _prices: Dict[str, float] = {}
    _start_ts: float = time.time()

    def __init__(self):
        # Initialize prices from seed
        for sym, name, mkt, price, sector, industry, mcap in MOCK_STOCKS:
            self._prices[sym] = price

    def _live_price(self, sym: str, base_price: float) -> float:
        """Simulate realistic price movement using sine + random walk."""
        if sym not in self._prices:
            self._prices[sym] = base_price
        t = time.time() - self._start_ts
        # Intraday sine wave (mimics market open/close dynamics)
        trend = math.sin(t / 3600) * 0.005
        # Random walk step
        step = random.gauss(0, 0.002)
        # Occasional spike
        if random.random() < 0.02:
            step += random.gauss(0, 0.015)
        self._prices[sym] *= (1 + trend + step)
        return round(self._prices[sym], 6)

    def _volume(self, base_price: float, mkt: Market) -> float:
        multiplier = {
            Market.US_STOCK: 5_000_000,
            Market.PENNY:    2_000_000,
            Market.ETF:      10_000_000,
            Market.CRYPTO:   500_000_000,
            Market.INTL:     2_000_000,
        }.get(mkt, 1_000_000)
        return random.uniform(0.3, 3.5) * multiplier

    def get_all_ticks(self) -> List[Tick]:
        ticks = []
        for sym, name, mkt, base, sector, industry, mcap in MOCK_STOCKS:
            price = self._live_price(sym, base)
            base_price_ref = list([s for s in MOCK_STOCKS if s[0] == sym])[0][3]
            change_pct = (price - base_price_ref) / base_price_ref * 100
            ticks.append(Tick(
                symbol=sym, price=price, volume=self._volume(price, mkt),
                change_pct=change_pct, market=mkt, provider="mock",
            ))
        return ticks

    def get_profiles(self) -> List[TickerProfile]:
        profiles = []
        ticks_map = {t.symbol: t for t in self.get_all_ticks()}

        for sym, name, mkt, base, sector, industry, mcap in MOCK_STOCKS:
            tick = ticks_map.get(sym)
            if not tick:
                continue
            rsi = random.uniform(25, 75)
            vol_ratio = random.uniform(0.5, 4.5)
            change_5d = random.uniform(-12, 12)
            bb_pos = random.uniform(0, 1)

            profile = TickerProfile(
                symbol=sym,
                name=name,
                market=mkt,
                price=tick.price,
                volume_24h=tick.volume,
                avg_volume=tick.volume / max(0.5, vol_ratio),
                volume_ratio=vol_ratio,
                change_pct=tick.change_pct,
                change_5d=change_5d,
                market_cap=mcap,
                sector=sector,
                industry=industry,
                rsi=rsi,
                macd_signal="bullish" if rsi < 50 else "bearish",
                bb_position=bb_pos,
                vwap_vs_price=random.uniform(-3, 3),
                sentiment_score=random.uniform(-0.5, 0.8),
                news_count_24h=random.randint(0, 12),
                provider_data={
                    "indicators": {
                        "rsi": rsi,
                        "macd_hist": random.uniform(-0.5, 0.5),
                        "macd_cross": "bullish" if rsi < 50 else "bearish",
                        "bb_position": bb_pos,
                        "bb_squeeze": random.random() < 0.15,
                        "volume_ratio": vol_ratio,
                        "avg_volume": tick.volume / max(0.5, vol_ratio),
                        "vwap_vs_price": random.uniform(-3, 3),
                        "atr_pct": random.uniform(1, 5),
                        "stoch_k": random.uniform(10, 90),
                        "stoch_d": random.uniform(10, 90),
                        "roc5": change_5d,
                        "roc10": random.uniform(-15, 15),
                        "ema9": tick.price * random.uniform(0.97, 1.03),
                        "ema21": tick.price * random.uniform(0.95, 1.05),
                        "ema50": tick.price * random.uniform(0.92, 1.08),
                        "atr": tick.price * 0.025,
                    }
                }
            )
            profiles.append(profile)
        return profiles


mock_provider = MockProvider()
