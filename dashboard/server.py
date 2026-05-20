"""
dashboard/server.py -- FastAPI + WebSocket dashboard server (v8.1)

PART 2 FIXES:
- Removes duplicate route registrations from prior versions
- Adds /api/scanner/pause endpoint to toggle scanner pause
- Adds /api/scanner/status for paused state
- Better error responses on /api/analyze/{symbol} and /api/trace/{trace_id}
- Trace endpoints now always return valid JSON (never 404 on empty)
- /api/chart/{symbol} validates symbol exists in watchlist OR allows direct lookup
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Security, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from core.bus import bus, Event
from core.logger import get_logger
from config.settings import LLM, SYS

log = get_logger("dashboard")

app = FastAPI(title="QuantAgent", version="8.1.0")

# ------------------------------------------------------------------
# API KEY AUTH (optional — skip check when DASHBOARD_API_KEY not set)
# ------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request, api_key: str = Security(_api_key_header)):
    """Dependency: validates X-API-Key header when DASHBOARD_API_KEY is configured."""
    required = SYS.DASHBOARD_API_KEY
    if not required:
        return  # auth disabled — dev mode
    # Allow dashboard HTML and WebSocket through without a key
    path = request.url.path
    if path in ("/", "/ws") or path.startswith("/static"):
        return
    if api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app.dependency_overrides  # no-op reference to ensure FastAPI picks up the dep

# Register auth as a global dependency on all /api/* routes via middleware
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/"):
        required = SYS.DASHBOARD_API_KEY
        if required:
            key = request.headers.get("X-API-Key", "")
            if key != required:
                return JSONResponse(
                    {"error": "Unauthorized — set X-API-Key header"},
                    status_code=401
                )
    return await call_next(request)

_state = {
    "watchlist": [],
    "signals": [],
    "agent_activity": [],
    "stats": {},
    "heartbeat": 0,
}


class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info(f"Client connected ({len(self.active)} total)")

    async def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data, default=str)
        dead = set()
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()


# ------------------------------------------------------------------
# CORE
# ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    p = Path(__file__).parent / "templates" / "dashboard.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html missing</h1>")


@app.get("/api/watchlist")
async def get_watchlist():
    return JSONResponse(_state["watchlist"])


@app.get("/api/signals")
async def get_signals():
    return JSONResponse(_state["signals"][-200:])


@app.get("/api/activity")
async def get_activity():
    return JSONResponse(_state["agent_activity"][-500:])


@app.get("/api/stats")
async def get_stats():
    from agents.llm_router import get_llm_stats
    llm = get_llm_stats()
    try:
        from scanners.market_scanner import scanner
        paused = scanner.paused
    except Exception:
        paused = False
    return JSONResponse({
        **_state["stats"],
        "clients": len(manager.active),
        "uptime_ts": _state["heartbeat"],
        "scanner_paused": paused,
        "llm_provider": LLM.PROVIDER,
        "llm_model": LLM.display_name(),
        "llm_calls": llm.get("calls", 0),
        "llm_tok_per_sec": llm.get("tok_per_sec", 0),
        "llm_errors": llm.get("errors", 0),
    })


# ------------------------------------------------------------------
# CHART
# ------------------------------------------------------------------
@app.get("/api/chart/{symbol}")
async def get_chart(symbol: str, period: str = "3mo"):
    try:
        # Map frontend period strings to yfinance-valid values + pick best interval
        _period_map = {
            "5d": "5d", "60d": "3mo", "90d": "3mo", "180d": "6mo",
            "2mo": "3mo", "4mo": "6mo", "9mo": "1y", "2y": "2y",
        }
        _interval_map = {
            "5d": "1h",   # intraday resolution for 5-day view
            "3mo": "1d",
            "6mo": "1d",
            "1y":  "1d",
            "2y":  "1wk",
        }
        period = _period_map.get(period, period)
        interval = _interval_map.get(period, "1d")

        from scanners.technicals import fetch_ohlcv_yf
        df = await fetch_ohlcv_yf(symbol.upper(), period=period, interval=interval)
        if df is None or df.empty:
            return JSONResponse({"error": f"No price data for {symbol}", "bars": []})
        bars = []
        intraday = interval in ("1h", "30m", "15m", "5m", "1m")
        for ts, row in df.iterrows():
            ts_str = str(ts)
            # Keep full YYYY-MM-DD (10) or YYYY-MM-DD HH:MM (16) so the
            # JS slice(5) produces "MM-DD" or "MM-DD HH:MM" respectively.
            label = ts_str[:16] if intraday else ts_str[:10]
            bars.append({
                "t": label,
                "o": round(float(row.get("open", row["close"])), 4),
                "h": round(float(row.get("high", row["close"])), 4),
                "l": round(float(row.get("low",  row["close"])), 4),
                "c": round(float(row["close"]), 4),
                "v": int(row.get("volume", 0)),
            })
        return JSONResponse({
            "symbol": symbol.upper(), "period": period,
            "interval": interval, "bars": bars,
        })
    except Exception as e:
        log.error(f"chart error {symbol}: {e}")
        return JSONResponse({"error": str(e), "bars": []})


# ------------------------------------------------------------------
# SCANNER PAUSE CONTROL  (NEW)
# ------------------------------------------------------------------
class PauseRequest(BaseModel):
    paused: bool


@app.post("/api/scanner/pause")
async def set_scanner_pause(req: PauseRequest):
    """Pause/resume the watchlist scanner. Agents continue running."""
    try:
        from scanners.market_scanner import scanner
        scanner.set_paused(req.paused)
        return JSONResponse({"status": "ok", "paused": scanner.paused})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/scanner/status")
async def scanner_status():
    try:
        from scanners.market_scanner import scanner
        return JSONResponse({
            "paused": scanner.paused,
            "scan_count": scanner._scan_count,
            "watchlist_size": len(scanner.watchlist),
            "universe_age_s": int(time.time() - scanner._universe_ts) if scanner._universe_ts else 0,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# SANDBOX
# ------------------------------------------------------------------
class SandboxRequest(BaseModel):
    code: str


@app.post("/api/sandbox")
async def run_sandbox_endpoint(req: SandboxRequest):
    try:
        from dashboard.sandbox import run_sandbox
        result = await run_sandbox(req.code.strip())
        if result.get("chart"):
            sym = result["chart"].split(":")[1]
            await manager.broadcast({"type": "chart_request", "symbol": sym, "data": result})
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"output": [], "error": str(e), "chart": None, "elapsed_ms": 0})


# ------------------------------------------------------------------
# DEEP ANALYSIS / TRACES
# ------------------------------------------------------------------
@app.get("/api/traces")
async def get_traces():
    try:
        from core.trace import trace_store
        return JSONResponse(trace_store.get_all_recent(50))
    except Exception as e:
        return JSONResponse([], status_code=200)


@app.get("/api/traces/{symbol}")
async def get_symbol_traces(symbol: str):
    try:
        from core.trace import trace_store
        traces = trace_store.get_traces(symbol.upper())
        return JSONResponse(traces or [])
    except Exception as e:
        log.debug(f"traces/{symbol} error: {e}")
        return JSONResponse([])


@app.get("/api/trace/{trace_id}")
async def get_full_trace(trace_id: str):
    try:
        from core.trace import trace_store
        t = trace_store.get_full_trace(trace_id)
        if not t:
            return JSONResponse({"error": "Trace not found", "trace_id": trace_id}, status_code=404)
        return JSONResponse(t)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze/{symbol}")
async def trigger_deep_analysis(symbol: str):
    """Manually trigger deep analysis on a ticker."""
    try:
        from scanners.market_scanner import scanner
        from agents.deep_analysis import orchestrator
        sym = symbol.upper()
        item = scanner.watchlist.get(sym)
        if not item:
            return JSONResponse(
                {"error": f"{sym} not in current watchlist (try a top-50 ticker)"},
                status_code=404
            )
        # Fire-and-forget, run in background
        asyncio.create_task(orchestrator.analyze(item))
        return JSONResponse({"status": "started", "symbol": sym,
                             "estimated_seconds": 90})
    except Exception as e:
        log.error(f"analyze {symbol} failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
class LayerConfig(BaseModel):
    layers: dict


@app.post("/api/settings/layers")
async def update_layer_config(cfg: LayerConfig):
    try:
        from agents.deep_analysis import orchestrator
        for k, v in cfg.layers.items():
            if k in orchestrator.ENABLED_LAYERS:
                orchestrator.ENABLED_LAYERS[k] = bool(v)
        return JSONResponse({"status": "ok", "layers": orchestrator.ENABLED_LAYERS})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/settings")
async def get_settings():
    from agents.deep_analysis import orchestrator
    from agents.llm_router import get_llm_stats
    return JSONResponse({
        "layers": orchestrator.ENABLED_LAYERS,
        "llm": {
            "provider": LLM.PROVIDER,
            "model": LLM.display_name(),
            "local_url": LLM.LOCAL_URL if LLM.is_local() else None,
            **get_llm_stats(),
        },
        "scan_interval": SYS.SCAN_INTERVAL,
        "watchlist_size": SYS.WATCHLIST_SIZE,
        "max_workers": SYS.MAX_WORKERS,
    })


# ------------------------------------------------------------------
# PNL
# ------------------------------------------------------------------
@app.get("/api/pnl")
async def get_pnl():
    try:
        from signals.pnl_tracker import pnl_tracker
        return JSONResponse(pnl_tracker.get_stats())
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/pnl/advanced")
async def get_pnl_advanced():
    try:
        from signals.pnl_tracker import pnl_tracker
        return JSONResponse(pnl_tracker.get_stats())
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ------------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------------
@app.get("/api/portfolio")
async def get_portfolio():
    try:
        from signals.portfolio import paper_portfolio
        return JSONResponse(paper_portfolio.get_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/api/portfolio/close/{pos_id}")
async def close_position(pos_id: str):
    try:
        from signals.portfolio import paper_portfolio
        from scanners.market_scanner import scanner
        pos = paper_portfolio._positions.get(pos_id)
        if not pos or pos.status != "open":
            return JSONResponse({"error": "Position not found"}, status_code=404)
        price = pos.entry_price
        item = scanner.watchlist.get(pos.symbol)
        if item:
            price = item.profile.price
        await paper_portfolio._close_position(pos, price, "manual")
        return JSONResponse({"status": "closed", "pnl": pos.pnl_usd})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/portfolio/settings")
async def get_portfolio_settings():
    try:
        from signals.portfolio import paper_portfolio
        return JSONResponse(paper_portfolio.get_settings())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class PortfolioSettingsRequest(BaseModel):
    enabled: bool = None
    min_conviction: float = None
    max_positions: int = None


@app.post("/api/portfolio/settings")
async def update_portfolio_settings(req: PortfolioSettingsRequest):
    try:
        from signals.portfolio import paper_portfolio
        paper_portfolio.update_settings(
            enabled=req.enabled,
            min_conviction=req.min_conviction,
            max_positions=req.max_positions,
        )
        return JSONResponse(paper_portfolio.get_settings())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/portfolio/reset")
async def reset_portfolio():
    try:
        from signals.portfolio import paper_portfolio, ACCOUNT_SIZE
        paper_portfolio._positions.clear()
        paper_portfolio._cash = ACCOUNT_SIZE
        paper_portfolio._save()
        return JSONResponse({"status": "reset", "cash": ACCOUNT_SIZE})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# ALERTS
# ------------------------------------------------------------------
class AlertRequest(BaseModel):
    symbol: str
    alert_type: str
    threshold: float = 0.0
    signal_type: str = ""
    note: str = ""
    notify_once: bool = True


@app.get("/api/alerts")
async def get_alerts():
    try:
        from signals.alerts import alerts_manager
        return JSONResponse({
            "alerts":   alerts_manager.get_alerts(),
            "earnings": alerts_manager.get_earnings(7),
        })
    except Exception as e:
        return JSONResponse({"alerts": [], "earnings": []})


@app.post("/api/alerts")
async def add_alert(req: AlertRequest):
    try:
        from signals.alerts import alerts_manager
        alert = alerts_manager.add_alert(
            req.symbol, req.alert_type, req.threshold,
            req.signal_type, req.note, req.notify_once
        )
        return JSONResponse({"status": "ok", "alert_id": alert.alert_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    try:
        from signals.alerts import alerts_manager
        ok = alerts_manager.remove_alert(alert_id)
        return JSONResponse({"status": "ok" if ok else "not_found"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/earnings")
async def get_earnings():
    try:
        from signals.alerts import alerts_manager
        await alerts_manager.refresh_earnings()
        return JSONResponse(alerts_manager.get_earnings(14))
    except Exception as e:
        return JSONResponse([])


# ------------------------------------------------------------------
# BACKTESTING
# ------------------------------------------------------------------
@app.get("/api/backtest/{symbol}")
async def backtest(symbol: str, period: str = "90d", conviction_threshold: float = 0.60):
    """
    Simulate signal engine rules on historical OHLCV data.
    Returns per-bar signals, equity curve, and summary stats.
    """
    try:
        import asyncio
        from scanners.technicals import fetch_ohlcv_yf, compute_indicators
        from signals.signal_engine import SignalEngine
        from core.models import WatchlistItem, TickerProfile, Market

        sym = symbol.upper()
        df = await asyncio.wait_for(fetch_ohlcv_yf(sym, period=period), timeout=30)
        if df is None or df.empty:
            return JSONResponse({"error": f"No data for {sym}"}, status_code=404)

        engine = SignalEngine()
        trades = []
        equity = 1.0
        equity_curve = []
        open_trade = None

        bars = df.reset_index()
        for i in range(20, len(bars)):
            bar_df = df.iloc[:i+1]
            ind    = compute_indicators(bar_df)
            row    = bars.iloc[i]
            price  = float(row["close"])
            date   = str(row.get("Date", row.get("Datetime", row.name)))[:10]

            # Build a minimal TickerProfile from bar data
            profile = TickerProfile(
                symbol=sym,
                name=sym,
                price=price,
                change_pct=float((row["close"] - bars.iloc[i-1]["close"]) / bars.iloc[i-1]["close"] * 100),
                change_5d=float((row["close"] - bars.iloc[max(0,i-5)]["close"]) / bars.iloc[max(0,i-5)]["close"] * 100),
                volume=int(row.get("volume", 0)),
                volume_ratio=ind.get("volume_ratio", 1.0),
                market_cap=0,
                rsi=ind.get("rsi", 50.0),
                macd_signal=ind.get("macd_cross", "neutral"),
                bb_position=ind.get("bb_position", 0.5),
                vwap_vs_price=0.0,
                sentiment_score=0.0,
                market=Market.US_STOCK,
                provider_data={"indicators": ind},
            )
            item = WatchlistItem(profile=profile)

            # Check if open trade should close
            if open_trade:
                is_long = "BUY" in open_trade["signal_type"]
                pnl = (price - open_trade["entry"]) / open_trade["entry"] * (1 if is_long else -1)
                if (open_trade.get("stop") and is_long and price <= open_trade["stop"]) or \
                   (open_trade.get("target") and is_long and price >= open_trade["target"]) or \
                   (i - open_trade["bar_idx"] >= 5):
                    outcome = "win" if pnl > 0 else "loss"
                    equity *= (1 + pnl)
                    trades.append({**open_trade, "exit": price, "exit_date": date,
                                   "pnl_pct": round(pnl * 100, 2), "outcome": outcome})
                    open_trade = None

            # Evaluate signals (synchronous-ish via run)
            if not open_trade:
                sigs = await engine.evaluate(item)
                for sig in sigs:
                    if sig.conviction >= conviction_threshold and \
                       sig.signal_type.value in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
                        open_trade = {
                            "symbol": sym, "date": date, "entry": price,
                            "signal_type": sig.signal_type.value,
                            "conviction": round(sig.conviction, 3),
                            "stop": sig.stop_loss, "target": sig.target_price,
                            "bar_idx": i,
                        }
                        break

            equity_curve.append({"date": date, "equity": round((equity - 1) * 100, 2)})

        wins   = [t for t in trades if t["outcome"] == "win"]
        losses = [t for t in trades if t["outcome"] == "loss"]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        return JSONResponse({
            "symbol": sym,
            "period": period,
            "bars_analyzed": len(bars) - 20,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_return_pct": round((equity - 1) * 100, 2),
            "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0,
            "equity_curve": equity_curve[-120:],
            "trades": trades[-50:],
        })
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Data fetch timed out"}, status_code=504)
    except Exception as e:
        log.error(f"Backtest error [{symbol}]: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# SELF-IMPROVEMENT / LEARNING
# ------------------------------------------------------------------
@app.get("/api/learning")
async def get_learning():
    """Return the current learned parameters + per-layer performance report."""
    try:
        from agents.self_improvement import self_improvement
        return JSONResponse(self_improvement.get_report())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/learning/reset")
async def reset_learning():
    """Reset all learned parameters back to defaults and clear layer outcomes."""
    try:
        from agents.self_improvement import self_improvement, LearnedParams, STORAGE_PARAMS, STORAGE_LAYERS
        self_improvement.params = LearnedParams()
        self_improvement._save()
        if STORAGE_LAYERS.exists():
            STORAGE_LAYERS.write_text('{"outcomes":[]}')
        self_improvement.layer_store._outcomes = []
        return JSONResponse({"status": "reset", "message": "Learned params cleared to defaults"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/learning/run")
async def trigger_learning_update():
    """Manually trigger a self-improvement update cycle."""
    try:
        from agents.self_improvement import self_improvement
        report = await self_improvement.run_update()
        return JSONResponse({"status": "ok", "summary": report.get("summary", "")})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# MARKET STATUS
# ------------------------------------------------------------------
@app.get("/api/market/status")
async def get_market_status():
    try:
        from core.market_hours import market_status
        return JSONResponse(market_status())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------------
@app.get("/api/compare")
async def compare_tickers(symbols: str = ""):
    try:
        from scanners.market_scanner import scanner
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:6]
        result = []
        for sym in syms:
            item = scanner.watchlist.get(sym)
            if item:
                result.append(item.to_dict())
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse([])


# ------------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------------
@app.get("/api/export/signals")
async def export_signals():
    from signals.pnl_tracker import pnl_tracker
    import io, csv
    stats = pnl_tracker.get_stats()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["symbol", "type", "outcome", "entry", "exit", "pnl", "agent", "ts"])
    writer.writeheader()
    for r in stats.get("recent", []):
        writer.writerow({
            "symbol": r.get("symbol", ""), "type": r.get("type", ""),
            "outcome": r.get("outcome", ""), "entry": r.get("entry", 0),
            "exit": r.get("exit", 0), "pnl": r.get("pnl", 0),
            "agent": r.get("agent", ""), "ts": r.get("ts", 0),
        })
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        output.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=signals.csv"}
    )


# ------------------------------------------------------------------
# MEMORY / NOTES
# ------------------------------------------------------------------
@app.get("/api/memory")
async def get_memory():
    try:
        from agents.memory import agent_memory
        stats = agent_memory.get_stats()
        recent = [
            {"symbol": e.symbol, "type": e.entry_type,
             "content": e.content[:120], "confidence": e.confidence,
             "outcome": e.outcome, "ts": e.ts}
            for e in list(reversed(agent_memory._entries))[:50]
        ]
        return JSONResponse({"stats": stats, "recent": recent,
                             "sectors": agent_memory._sector_insights})
    except Exception as e:
        return JSONResponse({"stats": {}, "recent": [], "sectors": {}})


@app.get("/api/memory/{symbol}")
async def get_symbol_memory(symbol: str):
    try:
        from agents.memory import agent_memory
        entries = agent_memory.recall(symbol.upper(), limit=10)
        ctx = agent_memory.get_ticker_context(symbol.upper())
        return JSONResponse({
            "context": ctx,
            "entries": [{"content": e.content, "type": e.entry_type,
                         "outcome": e.outcome, "ts": e.ts} for e in entries]
        })
    except Exception as e:
        return JSONResponse({"context": {}, "entries": []})


@app.get("/api/notes/{symbol}")
async def get_notes(symbol: str):
    try:
        from agents.memory import agent_memory
        entries = agent_memory.recall(symbol.upper(), limit=20)
        return JSONResponse([
            {"content": e.content, "type": e.entry_type,
             "ts": e.ts, "outcome": e.outcome}
            for e in entries
        ])
    except Exception as e:
        return JSONResponse([])


class NoteRequest(BaseModel):
    content: str
    entry_type: str = "insight"


@app.post("/api/notes/{symbol}")
async def add_note(symbol: str, req: NoteRequest):
    try:
        from agents.memory import agent_memory
        agent_memory.remember(symbol.upper(), req.entry_type, req.content,
                              tags=["user_note"], confidence=1.0)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/watchlist/intelligence")
async def get_wl_intelligence():
    try:
        from agents.watchlist_manager import wl_intelligence
        return JSONResponse({
            "high_alert":     wl_intelligence.get_high_alert(),
            "priority_queue": wl_intelligence.get_analysis_priority()[:20],
            "dismissed":      wl_intelligence.get_dismissed(),
        })
    except Exception as e:
        return JSONResponse({"high_alert": [], "priority_queue": [], "dismissed": {}})


@app.get("/api/llm/stats")
async def llm_stats():
    from agents.llm_router import get_llm_stats
    return JSONResponse(get_llm_stats())


# ------------------------------------------------------------------
# ONBOARDING
# ------------------------------------------------------------------

_ONBOARDING_FILE = SYS.STORAGE_DIR / "onboarding.json"


@app.get("/api/onboarding/status")
async def onboarding_status():
    from config.settings import KEYS, LLM
    completed = False
    try:
        if _ONBOARDING_FILE.exists():
            data = json.loads(_ONBOARDING_FILE.read_text(encoding="utf-8"))
            completed = data.get("completed", False)
    except Exception:
        pass

    from signals.portfolio import paper_portfolio
    return JSONResponse({
        "completed": completed,
        "paper_trading_enabled": paper_portfolio.get_settings().get("enabled", False),
        "checks": {
            "alpaca":       bool(KEYS.ALPACA_KEY and KEYS.ALPACA_SEC),
            "polygon":      bool(KEYS.POLYGON),
            "finnhub":      bool(KEYS.FINNHUB),
            "anthropic":    bool(KEYS.ANTHROPIC) and not LLM.is_local(),
            "lm_studio":    LLM.is_local(),
            "llm_provider": LLM.PROVIDER,
            "llm_model":    LLM.display_name(),
            "email_alerts": bool(SYS.SMTP_USER and SYS.ALERT_EMAIL_TO),
        },
    })


class OnboardingCompleteReq(BaseModel):
    enable_paper_trading: bool = False


@app.post("/api/onboarding/complete")
async def onboarding_complete(req: OnboardingCompleteReq):
    try:
        if req.enable_paper_trading:
            from signals.portfolio import paper_portfolio
            paper_portfolio.update_settings(enabled=True)
        _ONBOARDING_FILE.write_text(json.dumps({
            "completed": True,
            "completed_at": time.time(),
            "paper_trading_enabled": req.enable_paper_trading,
        }, indent=2), encoding="utf-8")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/onboarding/reset")
async def onboarding_reset():
    """Re-show the onboarding wizard (dev/debug use)."""
    try:
        if _ONBOARDING_FILE.exists():
            _ONBOARDING_FILE.unlink()
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol:   str
    days:     int   = 90
    risk_pct: float = 0.01


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    sym  = req.symbol.upper().strip()
    days = max(30, min(365, req.days))
    try:
        from agents.backtest import run_backtest as _bt
        result = await _bt(sym, days=days, risk_pct=req.risk_pct)
        from dataclasses import asdict
        d = asdict(result)
        # Convert enum-based signal strings (already str in BtTrade)
        return JSONResponse(d)
    except Exception as e:
        log.error(f"Backtest error [{sym}]: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# ANALYSIS PIPELINE CONFIG & QUEUE  (v9)
# ------------------------------------------------------------------

class AnalysisCfgRequest(BaseModel):
    tier1_enabled: bool = None
    tier1_workers: int = None
    tier1_max_tokens: int = None
    tier1_timeout: int = None
    tier2_enabled: bool = None
    tier2_threshold: float = None
    tier2_max_tokens: int = None
    tier2_timeout: int = None
    analysis_cooldown: int = None
    tickers_per_cycle: int = None
    min_composite_score: float = None
    legacy_mode: bool = None


@app.get("/api/pipeline/config")
async def get_pipeline_config():
    try:
        from agents.analysis_config import analysis_config
        return JSONResponse(analysis_config.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/pipeline/config")
async def update_pipeline_config(req: AnalysisCfgRequest):
    try:
        from agents.analysis_config import analysis_config
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        changed = analysis_config.update(**updates)
        return JSONResponse({"status": "ok", "changed": changed,
                             "config": analysis_config.to_dict()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pipeline/queue")
async def get_pipeline_queue():
    try:
        from agents.analysis_queue import analysis_queue
        return JSONResponse(analysis_queue.get_status())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pipeline/stats")
async def get_pipeline_stats():
    try:
        from agents.analysis_queue import analysis_queue
        from agents.analysis_config import analysis_config
        return JSONResponse({
            "queue": analysis_queue.get_status(),
            "stats": analysis_queue.get_stats(),
            "config": analysis_config.to_dict(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class ManualAnalyzeRequest(BaseModel):
    symbol: str
    tier: int = 1  # 1 or 2


@app.post("/api/pipeline/analyze")
async def manual_analyze(req: ManualAnalyzeRequest):
    """Manually queue any ticker for Tier-1 or Tier-2 analysis."""
    try:
        from agents.analysis_queue import analysis_queue
        from scanners.market_scanner import scanner
        from agents.deep_analysis import orchestrator

        sym = req.symbol.strip().upper()
        if not sym:
            return JSONResponse({"error": "Symbol required"}, status_code=400)

        # For manual analysis of tickers not in watchlist, run directly
        item = scanner.watchlist.get(sym)
        if not item:
            # Try to create a minimal item from live data
            from core.models import WatchlistItem, TickerProfile, Market
            try:
                from scanners.technicals import fetch_ohlcv_yf, compute_indicators, score_technicals
                df = await asyncio.wait_for(fetch_ohlcv_yf(sym), timeout=10)
                if df is not None and len(df) > 0:
                    ind = compute_indicators(df)
                    price = float(df["close"].iloc[-1])
                    chg1d = float((df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100) if len(df) > 1 else 0
                    chg5d = float((df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100) if len(df) > 5 else 0
                    profile = TickerProfile(
                        symbol=sym, name=sym, price=price,
                        change_pct=chg1d, change_5d=chg5d,
                        volume=int(df["volume"].iloc[-1]),
                        volume_ratio=ind.get("volume_ratio", 1.0),
                        market_cap=0, rsi=ind.get("rsi", 50.0),
                        macd_signal=ind.get("macd_cross", "neutral"),
                        bb_position=ind.get("bb_position", 0.5),
                        vwap_vs_price=0.0, sentiment_score=0.0,
                        market=Market.US_STOCK,
                        provider_data={"indicators": ind},
                    )
                    item = WatchlistItem(profile=profile)
                else:
                    return JSONResponse({"error": f"No data found for {sym}"}, status_code=404)
            except Exception as e:
                return JSONResponse({"error": f"Could not load data for {sym}: {e}"}, status_code=404)

        # Run analysis in background
        if req.tier == 2:
            asyncio.create_task(orchestrator.analyze_tier2(item))
            msg = f"Tier-2 deep dive started for {sym}"
        else:
            asyncio.create_task(orchestrator.analyze_tier1(item))
            msg = f"Tier-1 quick screen started for {sym}"

        return JSONResponse({"status": "started", "symbol": sym, "tier": req.tier, "message": msg})

    except Exception as e:
        log.error(f"Manual analyze {req.symbol}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/pipeline/queue/clear")
async def clear_pipeline_queue():
    """Clear the pending queue (does not cancel in-progress analyses)."""
    try:
        from agents.analysis_queue import analysis_queue, AnalysisQueue
        # Replace the queues to clear them
        analysis_queue._t1_queue = asyncio.Queue(maxsize=100)
        analysis_queue._t2_queue = asyncio.Queue(maxsize=30)
        return JSONResponse({"status": "cleared"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# ANALYTICS (Pillar C)
# ------------------------------------------------------------------
@app.get("/api/analytics")
async def get_analytics():
    """Full analytics payload: equity curve, attribution, daily P&L, rolling stats."""
    try:
        from signals.pnl_tracker import pnl_tracker
        return JSONResponse(pnl_tracker.get_analytics())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Real-time risk: sector concentration, VaR, beta, largest position, daily P&L at risk."""
    try:
        from signals.portfolio import paper_portfolio, ACCOUNT_SIZE
        from scanners.market_scanner import scanner
        from scanners.technicals import fetch_ohlcv_yf

        open_pos = paper_portfolio.open_positions
        account  = paper_portfolio.account_value

        # Sector concentration
        sector_map: dict = {}
        for p in open_pos:
            sec = p.sector or "Unclassified"
            sector_map[sec] = sector_map.get(sec, 0.0) + p.notional
        sector_pct = {
            s: round(v / account * 100, 1) for s, v in sector_map.items()
        } if account > 0 else {}

        # Largest single position exposure
        max_exposure = max((p.notional / account * 100 for p in open_pos), default=0.0)

        # VaR estimate (95% 1-day, parametric) — assume 2% daily vol per position
        SECTOR_BETA = {
            "Technology": 1.25, "Financials": 1.10, "Healthcare": 0.80,
            "Energy": 1.15, "Consumer Cyclical": 1.10, "Industrials": 1.00,
            "ETF Tech": 1.20, "ETF Broad": 1.00, "Crypto Layer 1": 1.80,
            "Crypto DeFi": 2.00, "Crypto Meme": 2.50,
        }
        daily_vol_usd = 0.0
        port_beta = 0.0
        for p in open_pos:
            sec = p.sector or ""
            beta = SECTOR_BETA.get(sec, 1.0)
            vol_pct = 0.02  # 2% daily vol baseline
            daily_vol_usd += (p.notional * vol_pct) ** 2
            port_beta += beta * (p.notional / account) if account > 0 else 0

        var_95 = round(1.645 * (daily_vol_usd ** 0.5), 2)
        var_95_pct = round(var_95 / account * 100, 2) if account > 0 else 0

        # Unrealized P&L at risk (open positions)
        daily_pnl_at_risk = round(sum(
            p.notional * 0.02 for p in open_pos
        ), 2)

        # SPY benchmark equity (fetch last 90d)
        spy_return = None
        try:
            df = await asyncio.wait_for(fetch_ohlcv_yf("SPY", period="90d"), timeout=8)
            if df is not None and len(df) >= 2:
                spy_return = round(
                    (float(df["close"].iloc[-1]) - float(df["close"].iloc[0]))
                    / float(df["close"].iloc[0]) * 100, 2
                )
        except Exception:
            pass

        return JSONResponse({
            "open_positions":    len(open_pos),
            "account_value":     account,
            "sector_pct":        sector_pct,
            "max_exposure_pct":  round(max_exposure, 1),
            "portfolio_beta":    round(port_beta, 2),
            "var_95_usd":        var_95,
            "var_95_pct":        var_95_pct,
            "daily_pnl_at_risk": daily_pnl_at_risk,
            "spy_90d_return_pct": spy_return,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# DIGEST (Pillar C)
# ------------------------------------------------------------------
@app.get("/api/digest/latest")
async def get_digest_latest():
    try:
        from agents.digest import digest_agent
        d = digest_agent.get_latest()
        if not d:
            return JSONResponse({"summary": None, "date": None})
        return JSONResponse(d)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/digest/generate")
async def trigger_digest():
    """Manually generate a fresh pre-market digest."""
    try:
        from agents.digest import digest_agent
        asyncio.create_task(digest_agent.generate_digest())
        return JSONResponse({"status": "generating"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# WEBHOOKS (Pillar C)
# ------------------------------------------------------------------
class WebhookCreateRequest(BaseModel):
    url: str
    name: str = ""
    events: list = None


@app.get("/api/webhooks")
async def list_webhooks():
    try:
        from signals.webhooks import webhook_manager
        return JSONResponse(webhook_manager.get_all())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/webhooks")
async def create_webhook(req: WebhookCreateRequest):
    try:
        from signals.webhooks import webhook_manager
        if not req.url or not req.url.startswith(("http://", "https://")):
            return JSONResponse({"error": "url must start with http:// or https://"}, status_code=400)
        wh = webhook_manager.add(req.url, req.name, req.events or ["all"])
        from dataclasses import asdict
        return JSONResponse(asdict(wh), status_code=201)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    try:
        from signals.webhooks import webhook_manager
        ok = webhook_manager.remove(webhook_id)
        if not ok:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    try:
        from signals.webhooks import webhook_manager
        result = await webhook_manager.test_webhook(webhook_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# NEWS FEED
# ------------------------------------------------------------------
@app.get("/api/news/{symbol}")
async def get_symbol_news(symbol: str, limit: int = 15):
    """Combine Finnhub + Polygon news, deduplicate, sort by recency."""
    sym = symbol.upper().strip()
    articles = []
    seen_titles: set = set()

    # Detect if this is a crypto symbol — skip stock-only news providers for crypto
    is_crypto = False
    try:
        from scanners.market_scanner import scanner
        from core.models import Market
        item = scanner.watchlist.get(sym)
        if item and item.profile.market == Market.CRYPTO:
            is_crypto = True
    except Exception:
        pass

    if not is_crypto:
        # Finnhub company news (stocks only)
        try:
            from providers.registry import registry
            fh_news = await registry.finnhub.get_news(sym, limit=limit)
            for art in fh_news:
                title = art.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    articles.append({**art, "source_provider": "finnhub"})
        except Exception as e:
            log.debug(f"Finnhub news [{sym}]: {e}")

        # Polygon news (stocks only)
        try:
            from providers.registry import registry
            poly_news = await registry.polygon.get_news(sym, limit=limit)
            for art in poly_news:
                title = art.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    articles.append({**art, "source_provider": "polygon"})
        except Exception:
            pass

    # Sort by published descending (string ISO or unix ts both sort correctly)
    articles.sort(key=lambda a: str(a.get("published", "")), reverse=True)
    return JSONResponse({"symbol": sym, "articles": articles[:limit], "is_crypto": is_crypto})


@app.get("/api/macro/status")
async def get_macro_status():
    """Return current macro regime overlay data."""
    try:
        from signals.macro_monitor import macro_monitor
        return JSONResponse(macro_monitor.get_status())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# EXECUTION ENGINE
# ------------------------------------------------------------------

@app.get("/api/execution/status")
async def get_execution_status():
    """Full execution engine state: mode, circuit breakers, pending slots, orders."""
    try:
        from signals.execution_engine import execution_engine
        return JSONResponse(execution_engine.get_status())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class ExecModeRequest(BaseModel):
    mode: str          # "paper" | "live"
    coordination: bool = None


@app.post("/api/execution/mode")
async def set_execution_mode(req: ExecModeRequest):
    """Switch between paper (JSON sim) and live (Alpaca) modes."""
    try:
        from signals.execution_engine import execution_engine
        if req.mode not in ("paper", "live"):
            return JSONResponse({"error": "mode must be 'paper' or 'live'"}, status_code=400)
        execution_engine.set_mode(req.mode)
        if req.coordination is not None:
            execution_engine.set_coordination(req.coordination)
        return JSONResponse({"status": "ok", "mode": execution_engine._mode,
                             "coordination": execution_engine._coordination})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class CBConfigRequest(BaseModel):
    daily_loss_pct:  float = None
    max_drawdown_pct: float = None
    max_positions:   int   = None


@app.post("/api/execution/circuit-breaker/config")
async def update_circuit_breaker_config(req: CBConfigRequest):
    """Update circuit breaker thresholds."""
    try:
        from signals.execution_engine import execution_engine
        execution_engine.update_circuit_breaker(
            daily_loss_pct   = req.daily_loss_pct,
            max_drawdown_pct = req.max_drawdown_pct,
            max_positions    = req.max_positions,
        )
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/execution/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Reset a tripped circuit breaker and resume trading."""
    try:
        from signals.execution_engine import execution_engine
        execution_engine.reset_circuit_breaker()
        return JSONResponse({"status": "ok", "paused": False})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/execution/kill")
async def kill_switch():
    """Emergency: cancel all orders and close all positions immediately."""
    try:
        from signals.execution_engine import execution_engine
        result = await execution_engine.emergency_close_all()
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        log.error(f"Kill switch error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/execution/reconcile")
async def trigger_reconcile():
    """Manually trigger broker reconciliation."""
    try:
        from signals.execution_engine import execution_engine
        result = await execution_engine.reconcile_with_broker()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/execution/broker/account")
async def get_broker_account():
    """Fetch live Alpaca account state (equity, buying power, etc.)."""
    try:
        from providers.alpaca_provider import AlpacaProvider
        alpaca = AlpacaProvider()
        account = await alpaca.get_account(paper=True)
        positions = await alpaca.get_positions(paper=True)
        orders    = await alpaca.get_orders(status="open", paper=True)
        await alpaca.close()
        return JSONResponse({
            "account":   account,
            "positions": positions,
            "open_orders": orders,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
# WEBSOCKET
# ------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "init",
            "data": {
                "watchlist": _state["watchlist"][:80],
                "signals":   _state["signals"][-100:],
                "activity":  _state["agent_activity"][-200:],
                "stats":     _state["stats"],
            },
        }, default=str))

        q = await bus.subscribe("watchlist.update", "signal", "agent.activity",
                                "scan.result", "heartbeat", "error",
                                "trace.step", "trace.complete",
                                "alert.triggered", "portfolio.opened", "portfolio.closed",
                                "watchlist.intelligence", "digest.generated")
        try:
            while True:
                try:
                    event: Event = await asyncio.wait_for(q.get(), timeout=25)
                    await ws.send_text(json.dumps({
                        "type": event.topic, "data": event.data, "ts": event.ts,
                    }, default=str))
                except asyncio.TimeoutError:
                    await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
        finally:
            await bus.unsubscribe(q, "watchlist.update", "signal", "agent.activity",
                                  "scan.result", "heartbeat", "error",
                                  "trace.step", "trace.complete",
                                  "alert.triggered", "portfolio.opened", "portfolio.closed",
                                  "watchlist.intelligence", "digest.generated")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f"WS error: {e}")
    finally:
        await manager.disconnect(ws)


# ------------------------------------------------------------------
# STATE UPDATER (background task started by main.py)
# ------------------------------------------------------------------
async def state_updater():
    q = await bus.subscribe("watchlist.update", "signal", "agent.activity", "heartbeat")
    while True:
        try:
            event: Event = await asyncio.wait_for(q.get(), timeout=5)
            if event.topic == "watchlist.update":
                _state["watchlist"] = event.data.get("items", [])
                _state["stats"]["scan_count"] = event.data.get("scan_num", 0)
                _state["stats"]["watchlist_count"] = event.data.get("count", 0)
            elif event.topic == "signal":
                _state["signals"].append(event.data)
                _state["signals"] = _state["signals"][-1000:]
            elif event.topic == "agent.activity":
                entry = event.data if isinstance(event.data, dict) else {}
                if "ts" not in entry:
                    entry = {**entry, "ts": event.ts}
                _state["agent_activity"].append(entry)
                _state["agent_activity"] = _state["agent_activity"][-2000:]
            elif event.topic == "heartbeat":
                _state["heartbeat"] = time.time()
                if isinstance(event.data, dict):
                    _state["stats"].update(event.data)
        except asyncio.TimeoutError:
            _state["heartbeat"] = time.time()
        except Exception as e:
            log.debug(f"State updater: {e}")