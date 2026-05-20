"""
providers/alpaca_provider.py -- Alpaca Markets adapter
Handles: Real-time US stock quotes, bars, snapshots
Paper trading ready for signal execution simulation
"""
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional

from providers.base import BaseProvider
from core.models import Tick, TickerProfile, Market
from core.logger import get_logger
from config.settings import KEYS, MARKET

log = get_logger("alpaca")

DATA_BASE  = "https://data.alpaca.markets"
PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE  = "https://api.alpaca.markets"


class AlpacaProvider(BaseProvider):
    name = "alpaca"
    markets = [Market.US_STOCK, Market.PENNY, Market.NASDAQ, Market.ETF]

    def __init__(self):
        self._headers = {
            "APCA-API-KEY-ID": KEYS.ALPACA_KEY,
            "APCA-API-SECRET-KEY": KEYS.ALPACA_SEC,
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate = asyncio.Semaphore(8)

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=self._headers
            )
        return self._session

    def _trade_base(self, paper: bool) -> str:
        return PAPER_BASE if paper else LIVE_BASE

    async def _get(self, base: str, path: str, params: Dict = None) -> Optional[any]:
        async with self._rate:
            try:
                s = await self._sess()
                async with s.get(f"{base}{path}", params=params or {}) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        log.warning("Alpaca rate limit -- backing off 2s")
                        await asyncio.sleep(2)
                    else:
                        text = await resp.text()
                        log.debug(f"Alpaca {path} -> {resp.status}: {text[:100]}")
            except Exception as e:
                log.error(f"Alpaca error: {e}")
        return None

    async def get_quote(self, symbol: str) -> Optional[Tick]:
        # Use snapshot for single symbol -- includes trade + bar + quote
        data = await self._get(DATA_BASE, "/v2/stocks/snapshots",
                               {"symbols": symbol, "feed": "iex"})
        if not data or symbol not in data:
            return None
        snap = data[symbol]
        trade = snap.get("latestTrade", {})
        bar   = snap.get("dailyBar", {})
        prev  = snap.get("prevDailyBar", {})
        price = trade.get("p", 0) or bar.get("c", 0)
        prev_close = prev.get("c", price) or price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        return Tick(
            symbol=symbol,
            price=price,
            volume=bar.get("v", 0),
            change_pct=change_pct,
            market=Market.PENNY if price < MARKET.PENNY_MAX_PRICE else Market.US_STOCK,
            provider="alpaca",
        )

    async def get_batch_quotes(self, symbols: List[str], market=None) -> List[Tick]:
        """
        Alpaca v2 multi-stock snapshots.
        Max ~1000 symbols per call; split into chunks of 100.
        Uses IEX feed (free, real-time, slightly less coverage than SIP).
        """
        all_ticks: List[Tick] = []
        chunk_size = 100

        for i in range(0, min(len(symbols), 500), chunk_size):
            chunk = symbols[i:i + chunk_size]
            syms_str = ",".join(chunk)
            data = await self._get(DATA_BASE, "/v2/stocks/snapshots", {
                "symbols": syms_str,
                "feed": "iex",   # free real-time feed
            })
            if not data:
                continue

            for sym, snap in data.items():
                trade = snap.get("latestTrade", {})
                bar   = snap.get("dailyBar", {})
                prev  = snap.get("prevDailyBar", {})
                price = trade.get("p", 0) or bar.get("c", 0)
                if not price:
                    continue
                prev_close = prev.get("c", price) or price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                all_ticks.append(Tick(
                    symbol=sym,
                    price=price,
                    volume=bar.get("v", 0),
                    change_pct=change_pct,
                    market=Market.PENNY if price < MARKET.PENNY_MAX_PRICE else Market.US_STOCK,
                    provider="alpaca",
                ))

        return all_ticks

    async def get_universe(self, market: Market) -> List[str]:
        """Get tradable US equity assets from Alpaca."""
        params = {
            "status": "active",
            "tradable": "true",
            "asset_class": "us_equity",
        }
        data = await self._get(PAPER_BASE, "/v2/assets", params)
        if not data:
            return []
        symbols = [
            a["symbol"] for a in data
            if a.get("tradable") and a.get("status") == "active"
            and "." not in a["symbol"]  # skip share classes like BRK.A
        ]
        log.info(f"Alpaca universe: {len(symbols)} assets")
        return symbols[:1000]

    async def get_profile(self, symbol: str) -> Optional[TickerProfile]:
        snap_task  = self._get(DATA_BASE, "/v2/stocks/snapshots",
                               {"symbols": symbol, "feed": "iex"})
        asset_task = self._get(PAPER_BASE, f"/v2/assets/{symbol}")
        snaps, asset = await asyncio.gather(snap_task, asset_task, return_exceptions=True)

        profile = TickerProfile(symbol=symbol)
        if isinstance(asset, dict):
            profile.name = asset.get("name", "")
        if isinstance(snaps, dict) and symbol in snaps:
            snap = snaps[symbol]
            trade = snap.get("latestTrade", {})
            bar   = snap.get("dailyBar", {})
            prev  = snap.get("prevDailyBar", {})
            price = trade.get("p", 0) or bar.get("c", 0)
            prev_c = prev.get("c", price) or price
            profile.price = price
            profile.volume_24h = bar.get("v", 0)
            profile.change_pct = ((price - prev_c) / prev_c * 100) if prev_c else 0
            if price < MARKET.PENNY_MAX_PRICE:
                profile.market = Market.PENNY
        return profile

    async def _post(self, base: str, path: str, body: Dict = None) -> Optional[Dict]:
        async with self._rate:
            try:
                s = await self._sess()
                async with s.post(f"{base}{path}", json=body or {}) as resp:
                    data = await resp.json()
                    if resp.status in (200, 201):
                        return data
                    log.warning(f"Alpaca POST {path} → {resp.status}: {str(data)[:120]}")
                    return {"_error": str(data), "_status": resp.status}
            except Exception as e:
                log.error(f"Alpaca POST error: {e}")
        return None

    async def _delete_json(self, base: str, path: str) -> Optional[any]:
        """DELETE that returns a JSON response body."""
        async with self._rate:
            try:
                s = await self._sess()
                async with s.delete(f"{base}{path}") as resp:
                    if resp.status in (200, 201, 204, 207):
                        try:
                            return await resp.json()
                        except Exception:
                            return {"status": "ok"}
                    text = await resp.text()
                    log.debug(f"Alpaca DELETE {path} → {resp.status}: {text[:100]}")
                    return {"_error": text[:100], "_status": resp.status}
            except Exception as e:
                log.error(f"Alpaca DELETE error: {e}")
        return None

    # ── Account & Positions ────────────────────────────────────────────────────

    async def get_account(self, paper: bool = True) -> Dict:
        """Fetch account equity, cash, buying power, and status."""
        data = await self._get(self._trade_base(paper), "/v2/account")
        if not isinstance(data, dict):
            return {}
        return {
            "equity":           float(data.get("equity", 0)),
            "cash":             float(data.get("cash", 0)),
            "buying_power":     float(data.get("buying_power", 0)),
            "portfolio_value":  float(data.get("portfolio_value", 0)),
            "daytrade_count":   int(data.get("daytrade_count", 0)),
            "pdt":              bool(data.get("pattern_day_trader", False)),
            "trading_blocked":  bool(data.get("trading_blocked", False)),
            "account_blocked":  bool(data.get("account_blocked", False)),
            "currency":         data.get("currency", "USD"),
            "status":           data.get("status", ""),
        }

    async def get_positions(self, paper: bool = True) -> List[Dict]:
        """Return all open broker positions."""
        data = await self._get(self._trade_base(paper), "/v2/positions")
        if not isinstance(data, list):
            return []
        return [
            {
                "symbol":          p.get("symbol", ""),
                "qty":             float(p.get("qty", 0)),
                "side":            p.get("side", "long"),
                "avg_entry":       float(p.get("avg_entry_price", 0)),
                "current_price":   float(p.get("current_price", 0)),
                "market_value":    float(p.get("market_value", 0)),
                "unrealized_pl":   float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
            }
            for p in data
        ]

    async def get_orders(self, status: str = "open", limit: int = 50,
                         paper: bool = True) -> List[Dict]:
        """Fetch orders filtered by status: open | closed | all."""
        data = await self._get(self._trade_base(paper), "/v2/orders",
                               {"status": status, "limit": limit})
        if not isinstance(data, list):
            return []
        return [
            {
                "order_id":        o.get("id", ""),
                "symbol":          o.get("symbol", ""),
                "qty":             float(o.get("qty", 0)),
                "side":            o.get("side", ""),
                "type":            o.get("type", ""),
                "status":          o.get("status", ""),
                "submitted_at":    o.get("submitted_at", ""),
                "filled_at":       o.get("filled_at"),
                "filled_avg_price":float(o.get("filled_avg_price") or 0),
                "limit_price":     float(o.get("limit_price") or 0),
            }
            for o in data
        ]

    # ── Order Execution ────────────────────────────────────────────────────────

    async def place_order(self, symbol: str, qty: float, side: str,
                          order_type: str = "market", limit_price: float = None,
                          time_in_force: str = "day", paper: bool = True) -> Dict:
        """
        Place a buy or sell order.
        Returns order dict with order_id on success, or {'_error': ...} on failure.
        """
        body: Dict = {
            "symbol":        symbol.upper(),
            "qty":           str(round(abs(qty), 4)),
            "side":          side.lower(),
            "type":          order_type.lower(),
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price:
            body["limit_price"] = str(round(limit_price, 2))

        data = await self._post(self._trade_base(paper), "/v2/orders", body)
        if not data:
            return {"_error": "No response from Alpaca", "_status": 0}
        return {
            "order_id":         data.get("id", ""),
            "client_order_id":  data.get("client_order_id", ""),
            "symbol":           data.get("symbol", symbol),
            "qty":              float(data.get("qty", qty)),
            "side":             data.get("side", side),
            "type":             data.get("type", order_type),
            "status":           data.get("status", ""),
            "submitted_at":     data.get("submitted_at", ""),
            "filled_at":        data.get("filled_at"),
            "filled_avg_price": float(data.get("filled_avg_price") or 0),
            "limit_price":      float(data.get("limit_price") or limit_price or 0),
            "time_in_force":    data.get("time_in_force", time_in_force),
            "_error":           data.get("_error", ""),
        }

    async def cancel_order(self, order_id: str, paper: bool = True) -> bool:
        """Cancel a pending order by Alpaca order ID."""
        result = await self._delete_json(self._trade_base(paper), f"/v2/orders/{order_id}")
        return isinstance(result, dict) and "_error" not in result

    async def cancel_all_orders(self, paper: bool = True) -> bool:
        """Cancel all open orders."""
        result = await self._delete_json(self._trade_base(paper), "/v2/orders")
        return result is not None

    async def close_position(self, symbol: str, paper: bool = True) -> Dict:
        """Close an entire position by symbol (market order)."""
        result = await self._delete_json(
            self._trade_base(paper), f"/v2/positions/{symbol.upper()}"
        )
        return result or {"_error": f"Failed to close {symbol}"}

    async def close_all_positions(self, paper: bool = True) -> List[Dict]:
        """Close ALL open positions at market price."""
        result = await self._delete_json(self._trade_base(paper), "/v2/positions")
        if isinstance(result, list):
            return result
        return []

    # ── Health ─────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        data = await self._get(PAPER_BASE, "/v2/clock")
        return isinstance(data, dict) and "is_open" in data

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
