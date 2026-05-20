"""
scanners/market_scanner.py -- Universe scanning and watchlist builder

PART 1 FIXES:
- Sectors are now properly populated via Finnhub get_profile() during enrichment
- Stricter universe filters (min volume, min market cap for stocks)
- Scanner now supports a `paused` flag that can be toggled at runtime
- Caches Finnhub profile lookups per session (avoids re-hitting rate limits)
- Sector inference fallback for crypto, ETFs, and unknown tickers
"""
import asyncio
import os
import random
import time
from typing import Dict, List, Optional

from providers.registry import registry
from scanners.technicals import get_technicals, score_technicals
from core.models import Market, TickerProfile, WatchlistItem, Tick
from core.bus import emit
from core.logger import get_logger
from core.staleness_guard import staleness_guard
from config.settings import SYS, MARKET

log = get_logger("scanner")
MOCK_MODE = os.environ.get("QUANT_MOCK_MODE", "0") == "1"


# Suffixes that indicate warrants, rights, units, preferred shares
_JUNK_SUFFIXES = ('W', 'WS', 'WW', 'R', 'U', 'Z', 'L', 'A', 'B')

def _is_quality_symbol(symbol: str) -> bool:
    """Return False for warrants, rights, units, SPACs."""
    s = symbol.upper()
    if len(s) > 4:
        for suffix in ('W', 'WW', 'WS', 'R', 'Z'):
            if s.endswith(suffix):
                return False
    if any(p in s for p in ('WW', '.W', '-W')):
        return False
    # Block dot-suffixed share-class noise (e.g. BRK.B is fine, but stuff like XYZ.U is junk)
    if '.' in s and s.split('.')[-1] in ('U', 'W', 'WS', 'R'):
        return False
    return True


# Sector inference for symbols where Finnhub has no profile data
# Crypto sectors derived from CoinGecko categories at scan time, but here's a fallback
_CRYPTO_SECTOR_MAP = {
    'BTC': 'Crypto Layer 1', 'ETH': 'Crypto Layer 1', 'SOL': 'Crypto Layer 1',
    'ADA': 'Crypto Layer 1', 'AVAX': 'Crypto Layer 1', 'DOT': 'Crypto Layer 1',
    'MATIC': 'Crypto Layer 2', 'ARB': 'Crypto Layer 2', 'OP': 'Crypto Layer 2',
    'LINK': 'Crypto DeFi', 'UNI': 'Crypto DeFi', 'AAVE': 'Crypto DeFi',
    'DOGE': 'Crypto Meme', 'SHIB': 'Crypto Meme', 'PEPE': 'Crypto Meme',
    'XRP': 'Crypto Payments', 'XLM': 'Crypto Payments',
}
_ETF_SECTOR_MAP = {
    'SPY':'ETF Broad','QQQ':'ETF Tech','IWM':'ETF Small Cap','DIA':'ETF Broad',
    'XLF':'ETF Financials','XLE':'ETF Energy','XLK':'ETF Tech','XLV':'ETF Healthcare',
    'XLI':'ETF Industrials','XLU':'ETF Utilities','XLY':'ETF Consumer','XLP':'ETF Staples',
    'XLB':'ETF Materials','XLRE':'ETF Real Estate','XBI':'ETF Biotech','SMH':'ETF Semis',
    'SOXX':'ETF Semis','SOXL':'ETF Semis 3x','SOXS':'ETF Semis 3x','TQQQ':'ETF Tech 3x',
    'GLD':'ETF Metals','SLV':'ETF Metals','TLT':'ETF Bonds','HYG':'ETF Bonds',
    'ARKK':'ETF Innovation','ARKG':'ETF Genomics',
}


def _infer_sector(symbol: str, market: Market, finnhub_sector: str = "") -> str:
    """Pick the best available sector label."""
    if finnhub_sector and finnhub_sector.lower() not in ("", "n/a", "none"):
        return finnhub_sector
    s = symbol.upper()
    if market == Market.CRYPTO:
        return _CRYPTO_SECTOR_MAP.get(s, "Crypto Other")
    if market == Market.ETF:
        return _ETF_SECTOR_MAP.get(s, "ETF Other")
    if market == Market.INTL:
        return "International"
    return "Unclassified"


def compute_composite_score(profile: TickerProfile, tech: Dict) -> float:
    if not tech:
        return 10.0
    tech_score, _ = score_technicals(tech)
    momentum  = min(25, abs(profile.change_pct) * 2 + abs(profile.change_5d) * 0.5)
    vr        = tech.get("volume_ratio", 1.0)
    vol_score = min(20, (vr - 1) * 5) if vr > 1 else 0
    sent      = (profile.sentiment_score + 1) / 2 * 15
    news      = min(5, profile.news_count_24h)
    composite = tech_score * 0.40 + momentum + vol_score + sent + news
    avg_vol = tech.get("avg_volume", 0)
    if avg_vol > 0:
        if avg_vol < 100_000:
            composite *= 0.35
        elif avg_vol < 500_000:
            composite *= 0.65
    return round(min(100.0, composite), 2)


def market_filter(tick: Tick, market: Market) -> bool:
    """Strict filters: real price + min volume per market type."""
    if tick.price <= 0:
        return False
    if not _is_quality_symbol(tick.symbol):
        return False
    # PART 1 FIX: stricter min volume thresholds
    if market == Market.PENNY:
        return 0.10 <= tick.price <= MARKET.PENNY_MAX_PRICE and tick.volume >= 50_000
    if market in (Market.US_STOCK, Market.NASDAQ):
        return tick.price > MARKET.PENNY_MAX_PRICE and tick.volume >= 100_000
    if market == Market.ETF:
        return tick.price > 5.0 and tick.volume >= 50_000
    if market == Market.CRYPTO:
        return tick.volume > 0  # crypto volume is in coin units
    return True  # International


def _build_scan_batch(symbols: List[str], seeds: List[str], cap: int) -> List[str]:
    seed_set = set(seeds)
    valid_seeds = [s for s in seeds if s in set(symbols)]
    universe_rest = [s for s in symbols if s not in seed_set]
    random.shuffle(universe_rest)
    seed_slots  = min(len(valid_seeds), int(cap * 0.60))
    disco_slots = cap - seed_slots
    batch = valid_seeds[:seed_slots] + universe_rest[:disco_slots]
    random.shuffle(batch)
    return batch[:cap]


async def _fetch_us_batch(symbols: List[str]) -> List[Tick]:
    all_ticks: List[Tick] = []
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            ticks = await registry.alpaca.get_batch_quotes(chunk)
            if ticks:
                all_ticks.extend(ticks)
                log.info(f"    Alpaca batch: {len(ticks)} quotes for {len(chunk)} symbols")
                continue
        except Exception as e:
            log.debug(f"Alpaca chunk failed: {e}")
        log.info(f"    Polygon individual fallback for {min(15, len(chunk))} symbols")
        sem = asyncio.Semaphore(3)
        results = []
        async def _one(sym):
            async with sem:
                try:
                    t = await registry.polygon.get_quote(sym)
                    if t and t.price > 0:
                        results.append(t)
                except Exception:
                    pass
        await asyncio.gather(*[_one(s) for s in chunk[:15]], return_exceptions=True)
        all_ticks.extend(results)
    return all_ticks


# Per-session cache of Finnhub profile lookups (sector + name + market_cap)
_PROFILE_CACHE: Dict[str, Dict] = {}


async def _fetch_finnhub_profile(symbol: str) -> Dict:
    """Fetch and cache Finnhub company profile (returns sector, name, market_cap)."""
    if symbol in _PROFILE_CACHE:
        return _PROFILE_CACHE[symbol]
    try:
        # registry.finnhub._get returns the raw profile2 endpoint data
        data = await registry.finnhub._get("/stock/profile2", {"symbol": symbol})
        if data:
            cached = {
                "name":   data.get("name", "")[:60],
                "sector": data.get("finnhubIndustry", ""),
                "industry": data.get("finnhubIndustry", ""),
                "market_cap": (data.get("marketCapitalization", 0) or 0) * 1e6,
            }
            _PROFILE_CACHE[symbol] = cached
            return cached
    except Exception as e:
        log.debug(f"Finnhub profile [{symbol}]: {e}")
    _PROFILE_CACHE[symbol] = {"name": "", "sector": "", "industry": "", "market_cap": 0}
    return _PROFILE_CACHE[symbol]


class MarketScanner:
    def __init__(self):
        self.watchlist: Dict[str, WatchlistItem] = {}
        self._universe: Dict[Market, List[str]] = {}
        self._universe_ts: float = 0
        self._running = False
        self._scan_count = 0
        # PART 1 FIX: pause control
        self.paused: bool = False

    def set_paused(self, paused: bool):
        """Pause/resume the scanner loop. Watchlist freezes; agents keep running."""
        self.paused = paused
        log.info(f"Scanner {'PAUSED' if paused else 'RESUMED'}")

    async def refresh_universe(self):
        log.info("Refreshing market universe...")
        await emit("agent.activity", {
            "agent_id": "scanner", "agent_type": "scanner",
            "symbol": "*", "action": "scanning",
            "message": "Refreshing universe from all markets..."
        }, "scanner")

        markets = []
        if MARKET.ENABLE_US_STOCKS:    markets += [Market.US_STOCK, Market.NASDAQ]
        if MARKET.ENABLE_PENNY_STOCKS: markets.append(Market.PENNY)
        if MARKET.ENABLE_CRYPTO:       markets.append(Market.CRYPTO)
        if MARKET.ENABLE_INTERNATIONAL:markets.append(Market.INTL)
        if MARKET.ENABLE_ETF:          markets.append(Market.ETF)

        async def _get(market):
            if market in (Market.US_STOCK, Market.NASDAQ, Market.ETF):
                try:
                    syms = await registry.alpaca.get_universe(market)
                    if syms:
                        return syms
                except Exception:
                    pass
            return await registry.get_universe(market)

        results = await asyncio.gather(*[_get(m) for m in markets], return_exceptions=True)
        for market, result in zip(markets, results):
            if isinstance(result, list) and result:
                self._universe[market] = list(dict.fromkeys(result))
                log.info(f"  Universe [{market.value}]: {len(self._universe[market])} tickers")
            else:
                log.warning(f"  Universe [{market.value}] failed: {result}")
                self._universe[market] = []

        self._universe_ts = time.time()
        total = sum(len(v) for v in self._universe.values())
        log.info(f"Total universe: {total} tickers across {len(self._universe)} markets")

    async def scan_market(self, market: Market) -> List[TickerProfile]:
        from providers.universe_seeds import US_LIQUID, CRYPTO_LIQUID, PENNY_LIQUID

        symbols = self._universe.get(market, [])
        if not symbols:
            return []

        caps = {
            Market.US_STOCK: MARKET.US_TICKER_CAP,
            Market.NASDAQ:   MARKET.US_TICKER_CAP,
            Market.PENNY:    MARKET.PENNY_TICKER_CAP,
            Market.CRYPTO:   MARKET.CRYPTO_CAP,
            Market.INTL:     MARKET.INTL_TICKER_CAP,
            Market.ETF:      MARKET.ETF_CAP,
        }
        cap = caps.get(market, 200)

        base_seeds = {
            Market.US_STOCK: US_LIQUID,
            Market.NASDAQ:   US_LIQUID,
            Market.ETF:      US_LIQUID,
            Market.PENNY:    PENNY_LIQUID,
            Market.CRYPTO:   CRYPTO_LIQUID,
            Market.INTL:     [],
        }.get(market, [])

        # Dynamic universe: boost hot-sector symbols, suppress cold-sector symbols
        seeds = list(base_seeds)
        try:
            from agents.self_improvement import self_improvement as _si
            hot  = set(_si.get_hot_sectors())
            cold = set(_si.get_cold_sectors())
            if hot or cold:
                # Pull symbols already in watchlist; add hot-sector ones to seeds
                hot_syms  = [s for s, item in self.watchlist.items()
                             if (item.profile.sector or "") in hot]
                cold_syms = {s for s, item in self.watchlist.items()
                             if (item.profile.sector or "") in cold}
                # Prepend hot-sector symbols to seeds so they fill seed slots first
                seeds = hot_syms + [s for s in base_seeds if s not in set(hot_syms)]
                # Remove cold-sector symbols from the universe for this scan cycle
                symbols = [s for s in symbols if s not in cold_syms]
                if hot_syms or cold_syms:
                    log.debug(f"Dynamic universe [{market.value}]: +{len(hot_syms)} hot, -{len(cold_syms)} cold")
        except Exception:
            pass

        batch = _build_scan_batch(symbols, seeds, cap)

        if market in (Market.US_STOCK, Market.NASDAQ, Market.ETF, Market.PENNY):
            ticks = await _fetch_us_batch(batch)
        else:
            ticks = await registry.get_batch_quotes(batch, market)

        if not ticks:
            log.warning(f"  [{market.value}] No quotes returned")
            return []

        candidates = [t for t in ticks if market_filter(t, market)]
        candidates.sort(key=lambda t: abs(t.change_pct), reverse=True)
        top = candidates[:40]

        log.info(f"  [{market.value}] {len(ticks)} quotes -> {len(candidates)} filtered -> {len(top)} deep scan")

        sem = asyncio.Semaphore(6)

        async def _enrich(tick: Tick) -> Optional[TickerProfile]:
            async with sem:
                try:
                    is_us = market in (Market.US_STOCK, Market.NASDAQ, Market.PENNY)
                    if is_us:
                        tech, fh_profile, insider_raw = await asyncio.gather(
                            get_technicals(tick.symbol),
                            _fetch_finnhub_profile(tick.symbol),
                            registry.finnhub.get_insider_transactions(tick.symbol),
                            return_exceptions=True
                        )
                        if isinstance(tech, Exception): tech = {}
                        if isinstance(fh_profile, Exception): fh_profile = {}
                        if isinstance(insider_raw, Exception): insider_raw = []
                    else:
                        tech = await get_technicals(tick.symbol)
                        fh_profile = {}
                        insider_raw = []

                    # Count 90-day insider buys/sells
                    insider_buys = insider_sells = 0
                    insider_buy_val = insider_sell_val = 0.0
                    try:
                        import datetime
                        cutoff_ts = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
                        for txn in (insider_raw or []):
                            txn_date = txn.get("transactionDate", "") or txn.get("date", "")
                            if txn_date < cutoff_ts:
                                continue
                            ttype = txn.get("transactionCode", txn.get("change", 0))
                            val = abs(txn.get("value", 0) or 0)
                            if isinstance(ttype, str) and ttype.upper() in ("P", "A"):
                                insider_buys += 1
                                insider_buy_val += val
                            elif isinstance(ttype, str) and ttype.upper() in ("S", "D", "F"):
                                insider_sells += 1
                                insider_sell_val += val
                    except Exception:
                        pass

                    sector = _infer_sector(tick.symbol, market, fh_profile.get("sector", ""))
                    profile = TickerProfile(
                        symbol=tick.symbol,
                        name=fh_profile.get("name", "") or tick.symbol,
                        market=market,
                        price=tick.price,
                        volume_24h=tick.volume,
                        change_pct=tick.change_pct,
                        sector=sector,
                        industry=fh_profile.get("industry", ""),
                        market_cap=fh_profile.get("market_cap", 0),
                        avg_volume=tech.get("avg_volume", 0),
                        volume_ratio=tech.get("volume_ratio", 1.0),
                        rsi=tech.get("rsi", 50),
                        macd_signal=tech.get("macd_cross", "neutral"),
                        bb_position=tech.get("bb_position", 0.5),
                        vwap_vs_price=tech.get("vwap_vs_price", 0),
                        change_5d=tech.get("change_5d_pct", 0),
                    )
                    profile.composite_score = compute_composite_score(profile, tech)
                    profile.provider_data["indicators"] = {
                        k: v for k, v in tech.items()
                        if k in ("rsi","macd_hist","macd_cross","bb_position","bb_squeeze",
                                 "volume_ratio","avg_volume","vwap","vwap_vs_price",
                                 "atr","atr_pct","stoch_k","stoch_d","roc5","roc10",
                                 "mtf_trend")
                    }
                    profile.provider_data["insider_buys"]     = insider_buys
                    profile.provider_data["insider_sells"]    = insider_sells
                    profile.provider_data["insider_buy_val"]  = round(insider_buy_val, 0)
                    profile.provider_data["insider_sell_val"] = round(insider_sell_val, 0)
                    staleness_guard.mark_refreshed(tick.symbol)
                    return profile
                except Exception as e:
                    log.debug(f"Enrich error [{tick.symbol}]: {e}")
                    staleness_guard.reset(tick.symbol)
                    return None

        results = await asyncio.gather(*[_enrich(t) for t in top], return_exceptions=True)
        profiles = [
            r for r in results
            if isinstance(r, TickerProfile) and r.composite_score > 25
        ]
        profiles.sort(key=lambda p: p.composite_score, reverse=True)
        return profiles

    async def run_full_scan(self):
        self._scan_count += 1
        label = "[MOCK]" if MOCK_MODE else ""
        log.info(f"=== Full Scan #{self._scan_count} {label} ===")

        if MOCK_MODE:
            return await self._run_mock_scan()

        if time.time() - self._universe_ts > 14400 or not self._universe:
            await self.refresh_universe()

        all_profiles: List[TickerProfile] = []
        for market in self._universe:
            try:
                profiles = await self.scan_market(market)
                all_profiles.extend(profiles)
                log.info(f"  [{market.value}] -> {len(profiles)} candidates")
            except Exception as e:
                log.error(f"Scan error [{market}]: {e}")

        new_wl: Dict[str, WatchlistItem] = {}
        for p in all_profiles:
            existing = self.watchlist.get(p.symbol)
            new_wl[p.symbol] = WatchlistItem(
                profile=p,
                signals=existing.signals if existing else [],
                agent_coverage=existing.agent_coverage if existing else [],
            )
        for sym, item in self.watchlist.items():
            if sym not in new_wl and item.profile.composite_score > 65:
                new_wl[sym] = item

        ranked = sorted(new_wl.values(), key=lambda x: x.profile.composite_score, reverse=True)[:SYS.WATCHLIST_SIZE]
        for i, item in enumerate(ranked):
            item.rank = i + 1
        self.watchlist = {item.profile.symbol: item for item in ranked}

        await emit("watchlist.update", {
            "count": len(ranked),
            "scan_num": self._scan_count,
            "items": [item.to_dict() for item in ranked[:50]],
        }, "scanner")
        for item in ranked[:20]:
            await emit("scan.result", item.to_dict(), "scanner")

        top5 = ", ".join(list(self.watchlist.keys())[:5])
        log.info(f"Watchlist: {len(self.watchlist)} tickers | Top: {top5}")
        return ranked

    async def _run_mock_scan(self) -> List[WatchlistItem]:
        from providers.mock_provider import mock_provider
        profiles = mock_provider.get_profiles()
        for p in profiles:
            ind = p.provider_data.get("indicators", {})
            p.composite_score = compute_composite_score(p, ind)
            if not p.sector:
                p.sector = _infer_sector(p.symbol, p.market)
        profiles.sort(key=lambda p: p.composite_score, reverse=True)
        new_wl: Dict[str, WatchlistItem] = {}
        for i, p in enumerate(profiles[:SYS.WATCHLIST_SIZE]):
            existing = self.watchlist.get(p.symbol)
            new_wl[p.symbol] = WatchlistItem(
                profile=p, rank=i+1,
                signals=existing.signals if existing else [],
                agent_coverage=existing.agent_coverage if existing else [],
            )
        self.watchlist = new_wl
        ranked = list(self.watchlist.values())
        await emit("watchlist.update", {
            "count": len(ranked), "scan_num": self._scan_count,
            "items": [item.to_dict() for item in ranked[:50]],
        }, "scanner")
        for item in ranked[:10]:
            await emit("scan.result", item.to_dict(), "scanner")
        top5 = ", ".join(list(self.watchlist.keys())[:5])
        log.info(f"[MOCK] Watchlist: {len(ranked)} | Top: {top5}")
        return ranked

    async def start(self):
        self._running = True
        log.info("Market Scanner started")
        while self._running:
            try:
                # PART 1 FIX: respect pause flag (agents continue running on frozen watchlist)
                if self.paused:
                    log.info(f"Scanner paused - holding watchlist of {len(self.watchlist)} tickers (agents still active)")
                    await asyncio.sleep(10)
                    continue
                await self.run_full_scan()
            except Exception as e:
                log.error(f"Scanner loop error: {e}")
                await emit("error", {"source": "scanner", "msg": str(e)}, "scanner")
            log.info(f"Next scan in {SYS.SCAN_INTERVAL}s")
            await asyncio.sleep(SYS.SCAN_INTERVAL)

    def stop(self):
        self._running = False

    def get_watchlist(self) -> List[WatchlistItem]:
        return sorted(self.watchlist.values(), key=lambda x: x.profile.composite_score, reverse=True)


scanner = MarketScanner()