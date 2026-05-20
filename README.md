# QuantAgent

> Autonomous multi-market trading intelligence powered by a two-tier AI analysis pipeline.

[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Claude%20%7C%20LM%20Studio-purple)](https://anthropic.com)

QuantAgent scans thousands of tickers across stocks, crypto, ETFs, and international markets, scores them with a composite algorithm, and routes candidates through a two-tier AI pipeline that produces buy/sell signals with conviction scores, reasoning, and risk parameters — all visible in a real-time dashboard.

---

## How It Works

### Scanning
Every 60 seconds, QuantAgent fetches live price, volume, and technical data across your configured markets. Each ticker is scored 0–100 using a weighted composite (technicals 40%, momentum 20%, volume 20%, sentiment 15%, news 5%) and ranked into tiers: **HIGH_ALERT**, **WATCH**, or **COOLING**.

### Two-Tier AI Pipeline

**Tier 1 — Quick Screen**
High-scoring candidates are queued for a single LLM call (~60s). The agent receives current price, indicators, and the composite score, then returns a signal (BUY / SELL / HOLD / WATCH) and a conviction score 0–1. Any ticker with conviction ≥ 0.65 is automatically promoted to Tier 2.

**Tier 2 — Deep Dive**
Three sequential LLM calls build a layered analysis:
1. **Technical Deep** — code execution on OHLCV data, price structure, volume divergence, OBV trend
2. **Context Layer** — macro regime, peer comparison, earnings calendar, sector rotation
3. **Master Synthesis** — weighs all layers using learned weights, generates a thesis, sets target/stop via ATR

### Signal Arbitration
Rule-based signals (11 rules: volume breakout, RSI reversal, MACD cross, BB squeeze, momentum surge, etc.) run in parallel with the AI pipeline. AI signals take priority. A 30-minute deduplication window prevents repeat signals in the same direction. All signals are stored to SQLite with outcome tracking at 1h, 4h, and 1D intervals.

### Self-Learning
The agent continuously updates its own parameters based on signal outcomes:
- **Conviction multipliers** per signal rule (scales down if systematically overconfident)
- **Layer weights** (rebalanced based on which layers' calls were directionally correct)
- **Regime adjustments** (more aggressive in uptrends, tighter in volatile regimes)
- **Sector thresholds** (lower conviction gate for hot sectors, higher for cold)

---

## Features

- **Multi-market scanning** — US stocks, penny stocks, crypto (top 100), ETFs, international ADRs
- **Two-tier LLM pipeline** — fast screen + deep dive with code execution
- **11 rule-based signal generators** — volume, RSI, MACD, Bollinger, momentum, VWAP, gap, OBV, earnings, mean reversion, confluence
- **Macro regime detection** — SPY/QQQ/VIX-based (volatile / trending_up / trending_down / sideways)
- **Persistent agent memory** — signal outcomes, patterns, sector insights (500 entries, cross-session)
- **Self-improvement loop** — learned parameters updated every 30 minutes from closed trade outcomes
- **Paper trading simulation** — full position lifecycle with circuit breakers and P&L tracking
- **Real-time dashboard** — FastAPI + WebSocket, 5 views, live queue status, trace replay
- **Autonomous commentary** — plain-English market summary every 5 minutes
- **Email + webhook alerts** — price thresholds, signal fires, volume spikes
- **Docker-ready** — multi-stage build with optional LM Studio sidecar

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│              DASHBOARD  (FastAPI + WebSocket)          │
│  Watchlist | Signals | Agent Activity | Traces | Stats │
└─────────────────────────┬──────────────────────────────┘
                          │ WebSocket (push every 100ms)
┌─────────────────────────▼──────────────────────────────┐
│                   ASYNC EVENT BUS                      │
│  ticks · scan.result · signal · agent.activity · trace │
└────┬──────────┬──────────┬──────────┬──────────────────┘
     │          │          │          │
┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────┐
│SCANNER │ │SIGNAL  │ │TIER-1  │ │TIER-2        │
│LOOP    │ │ENGINE  │ │WORKERS │ │WORKERS       │
│        │ │(Rules) │ │        │ │              │
│Polygon │ │11 rules│ │1 LLM   │ │3 LLM calls   │
│Alpaca  │ │→Signal │ │→Signal │ │Tech+Context  │
│Finnhub │ │+target │ │+conv.  │ │+Synthesis    │
│CoinGec.│ │+stop   │ │        │ │→final signal │
└────────┘ └────────┘ └────────┘ └──────┬───────┘
                                         │
              ┌─────────────┬────────────┘
              │             │
        ┌─────▼──┐   ┌──────▼───────┐
        │PAPER   │   │SELF-         │
        │TRADING │   │IMPROVEMENT   │
        │P&L     │   │MEMORY        │
        │Circuit │   │MACRO MONITOR │
        │Breakers│   │LEARNING LOOP │
        └────────┘   └──────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.13+
- API keys: [Polygon.io](https://polygon.io), [Alpaca](https://alpaca.markets), [Finnhub](https://finnhub.io), [CoinGecko](https://coingecko.com/en/api)
- LLM: [LM Studio](https://lmstudio.ai) (local, free) or an [Anthropic](https://console.anthropic.com) API key

### 1. Clone and install

```bash
git clone https://github.com/filt3rr/quant-agent.git
cd quant-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API keys. At minimum you need:
- `POLYGON_API_KEY` (stock data)
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` (quotes + paper trading)
- `FINNHUB_API_KEY` (company fundamentals)
- `COINGECKO_API_KEY` (crypto data)
- An LLM backend (see below)

### 3. Choose an LLM backend

**Option A — Local (free, no usage costs)**

1. Download [LM Studio](https://lmstudio.ai) and load `Qwen2.5-Coder-7B-Instruct`
2. Start the LM Studio server on port 1234
3. In `.env` set `LLM_PROVIDER=local`

**Option B — Anthropic Claude**

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. In `.env` set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=sk-ant-...`

### 4. Run

```bash
python main.py
```

Dashboard available at **http://localhost:8765**

---

## Docker

```bash
cd docker
docker-compose up --build
```

The compose file includes an optional LM Studio sidecar. Set `LLM_PROVIDER=local` and `LLM_LOCAL_URL=http://lmstudio:1234/v1` in your `.env`.

---

## Dashboard

Five real-time views updated via WebSocket:

| View | Description |
|------|-------------|
| **Watchlist** | All tracked tickers ranked by composite score with tier badges |
| **Signals Feed** | Live signal stream — symbol, direction, conviction, rule, timestamp |
| **Agent Activity** | Per-worker status: queued, active, completed, avg time |
| **Traces** | Step-by-step replay of every Tier-1 and Tier-2 analysis |
| **Stats** | Heartbeat data — LLM stats, macro regime, pipeline health, P&L |

The scanner and analysis pipeline can be paused and reconfigured from the dashboard without restarting.

---

## Signal Rules

The rule-based engine runs 11 generators in parallel on every watchlist tick:

| Rule | Trigger | Default Conviction |
|------|---------|-------------------|
| Volume Breakout | Vol ratio ≥ 3.0 + price ↑2% | 0.90 |
| RSI Reversal | RSI < 25 (oversold) or > 75 (overbought) | 0.70 |
| MACD Cross | Bullish or bearish MACD crossover | 0.65 |
| BB Squeeze | Squeeze + breakout above/below band | 0.75 |
| Momentum Surge | 1D ≥ 5% and 5D ≥ 10% | 0.80 |
| Multi-Indicator Confluence | RSI + MACD + VWAP all aligned | 0.80 |
| VWAP Reclaim | Close above VWAP + expanding volume | 0.65 |
| Gap Up | Open > 3% above prev close + vol spike | 0.70 |
| Accumulation | OBV rising + price consolidating | 0.60 |
| Earnings Momentum | BUY signal within 3 days of earnings | +0.15 boost |
| Mean Reversion | 5D down > 7% + RSI recovering above 40 | 0.65 |

All conviction scores are continuously adjusted by the self-improvement system based on per-rule win rates.

---

## Configuration Reference

All pipeline settings can be changed at runtime via the dashboard (`/api/config`) without restarting.

### Analysis Pipeline

| Setting | Default | Description |
|---------|---------|-------------|
| `tier1_enabled` | `true` | Enable Tier-1 quick screen |
| `tier1_workers` | `3` | Parallel Tier-1 workers |
| `tier1_timeout` | `60` | LLM timeout per ticker (seconds) |
| `tier2_enabled` | `true` | Enable Tier-2 deep dive |
| `tier2_workers` | `2` | Parallel Tier-2 workers |
| `tier2_threshold` | `0.65` | Min Tier-1 conviction to qualify for Tier-2 |
| `analysis_cooldown` | `600` | Min seconds before re-analyzing same ticker |
| `min_composite_score` | `55` | Watchlist entry threshold (0–100) |

### Execution & Risk

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `paper` | `paper` or `live` (Alpaca) |
| `daily_loss_limit_pct` | `0.02` | Pause at 2% daily loss |
| `max_drawdown_pct` | `0.05` | Pause at 5% peak drawdown |
| `max_open_positions` | `10` | Refuse new entries at limit |
| `min_conviction` | `0.70` | Min conviction to auto-trade |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13, asyncio |
| API server | FastAPI + Uvicorn + WebSocket |
| Database | SQLite (WAL mode) |
| Data analysis | Pandas, NumPy, pandas-ta |
| Market data | Polygon.io, Alpaca, Finnhub, CoinGecko, yfinance |
| LLM (cloud) | Anthropic Claude (claude-sonnet-4) |
| LLM (local) | LM Studio — Qwen2.5-Coder-7B-Instruct |
| News search | Tavily (optional) |
| Deployment | Docker + Docker Compose |

---

## Project Structure

```
quant-agent/
├── main.py                     # Orchestrator — 25+ async tasks
├── requirements.txt
├── run.sh
│
├── config/
│   └── settings.py             # Env var loading, all config dataclasses
│
├── core/
│   ├── models.py               # Tick, Signal, WatchlistItem, AgentActivity
│   ├── bus.py                  # Async in-memory pub/sub event bus
│   ├── db.py                   # SQLite persistence (signals, positions)
│   ├── rate_limiter.py         # Per-provider API rate limiting
│   ├── staleness_guard.py      # Price feed freshness monitoring
│   └── startup_validator.py    # Pre-flight API connectivity checks
│
├── providers/                  # Pluggable market data adapters
│   ├── polygon_provider.py
│   ├── alpaca_provider.py
│   ├── finnhub_provider.py
│   ├── coingecko_provider.py
│   └── registry.py             # Provider orchestration + fallback chains
│
├── scanners/
│   ├── market_scanner.py       # Universe scanning, composite scoring
│   ├── technicals.py           # RSI, MACD, Bollinger, VWAP, ATR, OBV
│   └── t0_filter.py            # Multi-timeframe pre-filter (H4/H1/volume)
│
├── agents/
│   ├── deep_analysis.py        # Two-tier LLM pipeline
│   ├── llm_router.py           # Unified routing (local or Anthropic)
│   ├── code_executor.py        # Sandboxed Python execution
│   ├── memory.py               # Persistent agent memory
│   ├── self_improvement.py     # Parameter learning and regime adaptation
│   └── watchlist_manager.py    # Intelligent watchlist curation
│
├── signals/
│   ├── signal_engine.py        # 11 rule-based signal generators
│   ├── execution_engine.py     # Order coordination, circuit breakers
│   ├── portfolio.py            # Paper trading simulation
│   ├── pnl_tracker.py          # Signal outcome tracking, win rates
│   └── macro_monitor.py        # SPY/QQQ/VIX regime detection
│
├── dashboard/
│   ├── server.py               # FastAPI + WebSocket server
│   └── templates/dashboard.html
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── tests/
    ├── test_core.py
    ├── test_db.py
    ├── test_production.py
    ├── test_rate_limiter.py
    └── test_staleness.py
```

---

## Security

- API keys live in `.env` only — gitignored, never committed
- Agent-generated code runs in a sandboxed namespace (no file I/O, no imports)
- Optional dashboard auth via `DASHBOARD_API_KEY` environment variable
- LLM circuit breaker auto-pauses analysis after 3 failures in 2 minutes
- Paper trading uses Alpaca's paper environment — no real orders

---

## Tests

```bash
pytest tests/
```

Covers the event bus, database layer, rate limiter, staleness guard, and end-to-end pipeline.

---

## Disclaimer

QuantAgent is a research and educational tool. It is not financial advice. Paper trading mode is the default — enabling live trading is your own decision and responsibility. Past signal performance does not guarantee future results.

---

## License

MIT
