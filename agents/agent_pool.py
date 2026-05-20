"""
agents/agent_pool.py -- Multi-agent AI analysis system

Specialized agents that run Claude AI on watchlist items:
  - TechnicalAgent:    deep technical analysis + chart pattern recognition
  - FundamentalAgent:  earnings, PE, sector analysis
  - SentimentAgent:    news sentiment, social, analyst ratings
  - NewsAgent:         Tavily search for breaking developments
  - RiskAgent:         position sizing, risk/reward, red flags
  - MasterAgent:       synthesizes all agents into final conviction signal

Each agent emits agent.activity events so the dashboard can show
live thinking and tool calls.
"""
import asyncio
import json
import time
import httpx
from typing import Any, Dict, List, Optional

from core.models import (
    WatchlistItem, Signal, SignalType, SignalReason, AgentActivity, Market
)
from core.bus import emit
from core.logger import get_logger
from providers.registry import registry
from providers.tavily_provider import TavilyProvider
from config.settings import KEYS, SYS

log = get_logger("agents")


async def call_claude(system: str, messages: List[Dict],
                       max_tokens: int = 800, agent_id: str = "agent") -> Optional[str]:
    from agents.llm_router import call_llm
    user = messages[-1]["content"] if messages else ""
    return await call_llm(system=system, user=user, max_tokens=max_tokens, agent_id=agent_id)


class BaseAgent:
    agent_type: str = "base"
    tavily: TavilyProvider = TavilyProvider()

    async def _emit_activity(self, agent_id: str, symbol: str, action: str,
                              message: str, tool_name: str = None,
                              tool_input: Dict = None, tool_output: str = None):
        activity = AgentActivity(
            agent_id=agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            action=action,
            message=message,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
        )
        await emit("agent.activity", activity.to_dict(), agent_id)

    async def analyze(self, item: WatchlistItem) -> Optional[Dict]:
        raise NotImplementedError


class TechnicalAgent(BaseAgent):
    agent_type = "technical"

    async def analyze(self, item: WatchlistItem) -> Optional[Dict]:
        sym     = item.profile.symbol
        profile = item.profile
        ind     = profile.provider_data.get("indicators", {})
        agent_id = f"technical_{sym}"

        await self._emit_activity(agent_id, sym, "analyzing",
            f"Starting deep technical analysis for {sym}")

        # Gather data
        await self._emit_activity(agent_id, sym, "tool_call",
            f"Fetching extended technical indicators",
            tool_name="get_technicals",
            tool_input={"symbol": sym, "period": "90d"})

        from scanners.technicals import get_technicals, score_technicals
        tech = await get_technicals(sym)
        tech_score, tech_summary = score_technicals(tech)

        await self._emit_activity(agent_id, sym, "tool_call",
            f"Technical score computed: {tech_score:.0f}/100",
            tool_name="score_technicals",
            tool_output=f"Score: {tech_score:.0f} | {tech_summary[:100]}")

        # Claude analysis
        prompt = f"""Analyze the following technical data for {sym}:

Price: ${profile.price:.4f}
Change 1D: {profile.change_pct:+.2f}%
Change 5D: {profile.change_5d:+.2f}%
Volume Ratio: {profile.volume_ratio:.2f}x avg

Indicators:
- RSI(14): {tech.get('rsi', 'N/A'):.1f}
- MACD Cross: {tech.get('macd_cross', 'N/A')}
- MACD Hist: {tech.get('macd_hist', 0):.6f}
- BB Position: {tech.get('bb_position', 0.5):.2f} (0=lower, 1=upper band)
- BB Squeeze: {tech.get('bb_squeeze', False)}
- VWAP vs Price: {tech.get('vwap_vs_price', 0):+.2f}%
- ATR%: {tech.get('atr_pct', 0):.2f}%
- Stoch K/D: {tech.get('stoch_k', 50):.0f}/{tech.get('stoch_d', 50):.0f}
- ROC 5D: {tech.get('roc5', 0):+.2f}%
- ROC 10D: {tech.get('roc10', 0):+.2f}%
- EMA9: {tech.get('ema9', 0):.4f}
- EMA21: {tech.get('ema21', 0):.4f}
- EMA50: {tech.get('ema50', 0):.4f}
- Volume Ratio: {tech.get('volume_ratio', 1):.2f}x

Technical Score: {tech_score:.0f}/100

In 3-4 sentences: What is the technical picture telling us? Is this bullish, bearish, or neutral?
What is the highest-probability near-term move? State a specific bias (BUY/SELL/HOLD) with brief justification.
End with: SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]"""

        await self._emit_activity(agent_id, sym, "reasoning",
            f"Consulting Claude AI for pattern recognition...")

        response = await call_claude(
            system="""You are a professional technical analyst with 20 years of experience.
You analyze charts and indicators with precision. Be concise, direct, and data-driven.
Always end with: SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]""",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            agent_id=agent_id
        )

        signal = "HOLD"
        conviction = 0.5
        if response:
            if "SIGNAL:" in response:
                parts = response.split("SIGNAL:")[-1].strip().split()
                signal = parts[0].strip() if parts else "HOLD"
            if "CONVICTION:" in response:
                try:
                    conviction = float(response.split("CONVICTION:")[-1].strip().split()[0])
                except:
                    conviction = 0.5

            await self._emit_activity(agent_id, sym, "signal_emitted",
                f"Technical verdict: {signal} ({conviction:.0%})\n{response[:200]}",
                tool_output=response)

        return {
            "agent": "technical",
            "symbol": sym,
            "signal": signal,
            "conviction": conviction,
            "tech_score": tech_score,
            "summary": tech_summary,
            "ai_analysis": response or "No AI response",
            "indicators": tech,
        }


class FundamentalAgent(BaseAgent):
    agent_type = "fundamental"

    async def analyze(self, item: WatchlistItem) -> Optional[Dict]:
        sym = item.profile.symbol
        profile = item.profile
        agent_id = f"fundamental_{sym}"

        if profile.market == Market.CRYPTO:
            return await self._crypto_fundamental(item, agent_id)

        await self._emit_activity(agent_id, sym, "analyzing",
            f"Fetching fundamental data for {sym}")

        # Fetch fundamentals from Finnhub
        await self._emit_activity(agent_id, sym, "tool_call",
            "Fetching analyst recommendations",
            tool_name="finnhub.get_analyst_recommendation",
            tool_input={"symbol": sym})

        analyst = await registry.finnhub.get_analyst_recommendation(sym)
        insider = await registry.finnhub.get_insider_transactions(sym)

        await self._emit_activity(agent_id, sym, "tool_call",
            f"Got analyst data: {analyst}",
            tool_name="finnhub.get_insider_transactions",
            tool_output=f"Insider transactions: {len(insider)} records")

        # Format insider summary
        insider_summary = "No recent insider activity"
        if insider:
            buys  = sum(1 for t in insider if t.get("transactionType") in ("P", "Buy"))
            sells = sum(1 for t in insider if t.get("transactionType") in ("S", "Sell"))
            insider_summary = f"Recent: {buys} insider buys, {sells} insider sells"

        await self._emit_activity(agent_id, sym, "reasoning",
            f"Analyzing fundamentals with AI...")

        prompt = f"""Fundamental analysis for {sym}:

Company: {profile.name}
Sector: {profile.sector}
Market Cap: ${profile.market_cap/1e9:.2f}B
PE Ratio: {profile.pe_ratio or 'N/A'}
Price: ${profile.price:.4f}

Analyst Recommendation (most recent): {analyst}
Insider Activity: {insider_summary}

In 2-3 sentences: What does the fundamental picture look like?
Is valuation attractive? Any fundamental red flags or catalysts?
SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]"""

        response = await call_claude(
            system="""You are a fundamental equity analyst. Be direct and concise.
Focus on valuation, sector dynamics, and insider signals.
Always end with: SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]""",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            agent_id=agent_id
        )

        signal = "HOLD"
        conviction = 0.5
        if response:
            if "SIGNAL:" in response:
                parts = response.split("SIGNAL:")[-1].strip().split()
                signal = parts[0].strip() if parts else "HOLD"
            if "CONVICTION:" in response:
                try:
                    conviction = float(response.split("CONVICTION:")[-1].strip().split()[0])
                except:
                    pass
            await self._emit_activity(agent_id, sym, "signal_emitted",
                f"Fundamental verdict: {signal} ({conviction:.0%})")

        return {
            "agent": "fundamental",
            "symbol": sym,
            "signal": signal,
            "conviction": conviction,
            "analyst": analyst,
            "insider_summary": insider_summary,
            "ai_analysis": response or "No response",
        }

    async def _crypto_fundamental(self, item: WatchlistItem, agent_id: str) -> Dict:
        sym = item.profile.symbol
        profile = item.profile
        await self._emit_activity(agent_id, sym, "analyzing",
            f"Crypto on-chain fundamental analysis for {sym}")
        data = await registry.coingecko.get_profile(sym)
        prompt = f"""Crypto fundamental analysis for {sym}:

Market Cap: ${(data.market_cap if data else profile.market_cap)/1e9:.3f}B
24h Volume: ${profile.volume_24h/1e6:.1f}M
Community Sentiment: {profile.sentiment_score:+.2f} (-1 bearish to +1 bullish)
Market Cap Rank: {data.provider_data.get('coingecko_rank', 'N/A') if data else 'N/A'}

In 2 sentences: Is this crypto fundamentally attractive at current levels?
SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]"""

        response = await call_claude(
            system="You are a crypto analyst. Be concise. Always end with SIGNAL and CONVICTION.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            agent_id=agent_id
        )
        signal = "HOLD"
        conviction = 0.5
        if response:
            if "SIGNAL:" in response:
                parts = response.split("SIGNAL:")[-1].strip().split()
                signal = parts[0].strip() if parts else "HOLD"
            if "CONVICTION:" in response:
                try:
                    conviction = float(response.split("CONVICTION:")[-1].strip().split()[0])
                except:
                    pass
        return {"agent": "fundamental", "symbol": sym, "signal": signal,
                "conviction": conviction, "ai_analysis": response or ""}


class NewsAgent(BaseAgent):
    agent_type = "news"
    tavily = TavilyProvider()

    async def analyze(self, item: WatchlistItem) -> Optional[Dict]:
        sym = item.profile.symbol
        profile = item.profile
        agent_id = f"news_{sym}"

        await self._emit_activity(agent_id, sym, "analyzing",
            f"Searching for latest news and developments on {sym}")

        # Search with Tavily
        await self._emit_activity(agent_id, sym, "tool_call",
            f"Running Tavily search for {sym} news",
            tool_name="tavily.search_news",
            tool_input={"symbol": sym, "company": profile.name})

        news_results = await self.tavily.search_news(sym, profile.name)

        # Also get polygon/finnhub news
        api_news = await registry.get_news(sym)

        all_headlines = []
        for n in news_results[:4]:
            all_headlines.append(f"• [{n.get('published','')[:10]}] {n.get('title','')}")
        for n in api_news[:3]:
            all_headlines.append(f"• [{n.get('published','')[:10]}] {n.get('title','')}")

        news_text = "\n".join(all_headlines) if all_headlines else "No recent news found"

        await self._emit_activity(agent_id, sym, "tool_call",
            f"Found {len(all_headlines)} news items",
            tool_name="aggregate_news",
            tool_output=news_text[:300])

        if not all_headlines:
            return {"agent": "news", "symbol": sym, "signal": "HOLD",
                    "conviction": 0.3, "headlines": [], "ai_analysis": "No news"}

        await self._emit_activity(agent_id, sym, "reasoning",
            "Analyzing news sentiment and market impact...")

        prompt = f"""News analysis for {sym} ({profile.name}):

Recent Headlines:
{news_text}

Current Price: ${profile.price:.4f} ({profile.change_pct:+.1f}% today)

In 2-3 sentences: What is the news sentiment? Any material catalysts or risks?
Is the news bullish, bearish, or neutral for the stock near-term?
SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]"""

        response = await call_claude(
            system="""You are a news and sentiment analyst. Parse headlines for market-moving events.
Separate signal from noise. Always end with: SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]""",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            agent_id=agent_id
        )

        signal = "HOLD"
        conviction = 0.4
        if response:
            if "SIGNAL:" in response:
                parts = response.split("SIGNAL:")[-1].strip().split()
                signal = parts[0].strip() if parts else "HOLD"
            if "CONVICTION:" in response:
                try:
                    conviction = float(response.split("CONVICTION:")[-1].strip().split()[0])
                except:
                    pass
            await self._emit_activity(agent_id, sym, "signal_emitted",
                f"News verdict: {signal} ({conviction:.0%})\n{(response or '')[:150]}")

        return {
            "agent": "news",
            "symbol": sym,
            "signal": signal,
            "conviction": conviction,
            "headlines": all_headlines,
            "ai_analysis": response or "",
        }


class RiskAgent(BaseAgent):
    agent_type = "risk"

    async def analyze(self, item: WatchlistItem) -> Optional[Dict]:
        sym = item.profile.symbol
        profile = item.profile
        agent_id = f"risk_{sym}"
        ind = profile.provider_data.get("indicators", {})

        await self._emit_activity(agent_id, sym, "analyzing",
            f"Running risk assessment for {sym}")

        atr_pct = ind.get("atr_pct", 2.0)
        price = profile.price

        # Position size suggestion (1% account risk, $10k account)
        account_size = 10000
        risk_per_trade = account_size * 0.01  # 1%
        stop_distance = price * (atr_pct / 100) * 1.5
        position_size = risk_per_trade / stop_distance if stop_distance > 0 else 0

        await self._emit_activity(agent_id, sym, "tool_call",
            f"Calculating position size and risk metrics",
            tool_name="risk_calculator",
            tool_input={"price": price, "atr_pct": atr_pct, "account": account_size},
            tool_output=f"Suggested shares: {position_size:.0f} | Stop: ${stop_distance:.4f}/share")

        # Red flags
        red_flags = []
        if profile.market == Market.PENNY:
            red_flags.append("Penny stock -- elevated manipulation/volatility risk")
        if atr_pct > 5:
            red_flags.append(f"High volatility: ATR {atr_pct:.1f}% -- wide stops required")
        if profile.volume_ratio > 5:
            red_flags.append(f"Extreme volume spike ({profile.volume_ratio:.0f}x) -- could be pump")
        if profile.market_cap < 50e6 and profile.market_cap > 0:
            red_flags.append(f"Micro-cap (${profile.market_cap/1e6:.0f}M) -- low liquidity risk")

        prompt = f"""Risk assessment for {sym}:

Price: ${price:.4f}
Volatility (ATR%): {atr_pct:.2f}%
Volume Ratio: {profile.volume_ratio:.1f}x
Market: {profile.market.value}
Market Cap: ${profile.market_cap/1e6:.0f}M

Red Flags Detected: {', '.join(red_flags) if red_flags else 'None'}

Suggested position size ($10k account, 1% risk): {position_size:.0f} shares

In 2 sentences: What are the key risk factors? Is the risk/reward favorable?
Rate overall risk: LOW / MEDIUM / HIGH / EXTREME
SIGNAL: [BUY/SELL/HOLD] CONVICTION: [0.0-1.0]"""

        await self._emit_activity(agent_id, sym, "reasoning",
            "Evaluating risk/reward and position sizing...")

        response = await call_claude(
            system="""You are a risk manager. Focus on capital preservation and position sizing.
Be direct about dangers. Always end with risk rating, SIGNAL, and CONVICTION.""",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            agent_id=agent_id
        )

        signal = "HOLD"
        conviction = 0.4
        risk_level = "MEDIUM"
        if response:
            for level in ("EXTREME", "HIGH", "MEDIUM", "LOW"):
                if level in response.upper():
                    risk_level = level
                    break
            if "SIGNAL:" in response:
                parts = response.split("SIGNAL:")[-1].strip().split()
                signal = parts[0].strip() if parts else "HOLD"
            if "CONVICTION:" in response:
                try:
                    conviction = float(response.split("CONVICTION:")[-1].strip().split()[0])
                except:
                    pass
            await self._emit_activity(agent_id, sym, "signal_emitted",
                f"Risk level: {risk_level} | {signal} ({conviction:.0%})")

        return {
            "agent": "risk",
            "symbol": sym,
            "signal": signal,
            "conviction": conviction,
            "risk_level": risk_level,
            "red_flags": red_flags,
            "position_size": position_size,
            "ai_analysis": response or "",
        }


class MasterAgent(BaseAgent):
    """Synthesizes all agent outputs into final actionable signal."""
    agent_type = "master"

    async def synthesize(
        self,
        item: WatchlistItem,
        agent_results: List[Dict]
    ) -> Optional[Signal]:
        sym = item.profile.symbol
        profile = item.profile
        agent_id = f"master_{sym}"

        await self._emit_activity(agent_id, sym, "analyzing",
            f"Synthesizing {len(agent_results)} agent reports for {sym}")

        # Tally votes
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0, "STRONG_BUY": 0, "STRONG_SELL": 0}
        total_conviction = 0
        summaries = []

        for r in agent_results:
            sig = r.get("signal", "HOLD").upper().replace(" ", "_")
            if "STRONG" not in sig:
                conv = r.get("conviction", 0.5)
            else:
                conv = r.get("conviction", 0.7)
            votes[sig if sig in votes else "HOLD"] += 1
            total_conviction += conv
            agent_name = r.get("agent", "unknown")
            analysis = r.get("ai_analysis", "")[:150]
            summaries.append(f"[{agent_name.upper()}] {sig} ({conv:.0%}): {analysis}")

        avg_conviction = total_conviction / len(agent_results) if agent_results else 0.5
        vote_summary = " | ".join(f"{k}:{v}" for k, v in votes.items() if v > 0)

        await self._emit_activity(agent_id, sym, "reasoning",
            f"Agent vote tally: {vote_summary} | Avg conviction: {avg_conviction:.0%}")

        prompt = f"""You are the Master Trading Agent. Synthesize these specialist reports for {sym}:

Price: ${profile.price:.4f} | Market: {profile.market.value}
Score: {profile.composite_score:.0f}/100

Agent Reports:
{chr(10).join(summaries)}

Vote Tally: {vote_summary}
Average Conviction: {avg_conviction:.0%}

Based on ALL evidence, what is your FINAL verdict?
Provide:
1. Overall signal (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL)
2. Final conviction (0.0-1.0)
3. 2-sentence thesis (why this is actionable or not)
4. Entry strategy (specific price levels if BUY)

Format:
FINAL_SIGNAL: [signal]
FINAL_CONVICTION: [0.0-1.0]
THESIS: [2 sentences]
ENTRY: [price context]"""

        response = await call_claude(
            system="""You are the Master Trading Agent -- the final decision maker.
You synthesize multiple specialists' views into one clear actionable verdict.
Be decisive. The market rewards conviction backed by data.""",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            agent_id=agent_id
        )

        # Parse master response
        final_signal = SignalType.HOLD
        final_conviction = avg_conviction
        thesis = ""
        entry = ""

        if response:
            lines = response.split("\n")
            for line in lines:
                if "FINAL_SIGNAL:" in line:
                    sig_str = line.split("FINAL_SIGNAL:")[-1].strip()
                    sig_map = {
                        "STRONG_BUY": SignalType.STRONG_BUY,
                        "BUY": SignalType.BUY,
                        "HOLD": SignalType.HOLD,
                        "SELL": SignalType.SELL,
                        "STRONG_SELL": SignalType.STRONG_SELL,
                    }
                    final_signal = sig_map.get(sig_str, SignalType.HOLD)
                elif "FINAL_CONVICTION:" in line:
                    try:
                        final_conviction = float(line.split("FINAL_CONVICTION:")[-1].strip())
                    except:
                        pass
                elif "THESIS:" in line:
                    thesis = line.split("THESIS:")[-1].strip()
                elif "ENTRY:" in line:
                    entry = line.split("ENTRY:")[-1].strip()

            await self._emit_activity(agent_id, sym, "signal_emitted",
                f"MASTER VERDICT: {final_signal.value} ({final_conviction:.0%})\n{thesis}",
                tool_output=response)

        # Build final signal
        from signals.signal_engine import _calc_target_stop
        ind = profile.provider_data.get("indicators", {})
        atr = ind.get("atr", profile.price * 0.02)
        target, stop = _calc_target_stop(profile.price, final_signal, atr)
        rr = abs(target - profile.price) / abs(profile.price - stop) if abs(profile.price - stop) > 0 else 0

        signal = Signal(
            symbol=sym,
            signal_type=final_signal,
            conviction=final_conviction,
            price_at_signal=profile.price,
            target_price=target if final_signal in (SignalType.BUY, SignalType.STRONG_BUY) else None,
            stop_loss=stop if final_signal in (SignalType.BUY, SignalType.STRONG_BUY) else None,
            risk_reward=round(rr, 2),
            reason=SignalReason.MULTI_AGENT,
            summary=thesis or f"Multi-agent consensus: {final_signal.value}",
            details=entry,
            tags=["multi_agent", "ai_synthesized"],
            agent="master_agent",
            market=profile.market,
        )

        await emit("signal", signal.to_dict(), "master_agent")
        log.info(
            f" MASTER [{sym}] -> {final_signal.value} "
            f"conviction={final_conviction:.0%} | {thesis[:80]}"
        )
        return signal


class AgentPool:
    """Manages concurrent execution of all agents on watchlist items."""

    def __init__(self):
        self.technical   = TechnicalAgent()
        self.fundamental = FundamentalAgent()
        self.news        = NewsAgent()
        self.risk        = RiskAgent()
        self.master      = MasterAgent()
        self._sem = asyncio.Semaphore(SYS.MAX_WORKERS)
        self._active: Dict[str, bool] = {}

    async def analyze_ticker(self, item: WatchlistItem):
        """Run full agent pipeline on one ticker."""
        sym = item.profile.symbol
        if self._active.get(sym):
            return  # Already being analyzed

        self._active[sym] = True
        async with self._sem:
            try:
                log.info(f" Agent pipeline starting: {sym}")
                # Run specialist agents in parallel
                results = await asyncio.gather(
                    self.technical.analyze(item),
                    self.fundamental.analyze(item),
                    self.news.analyze(item),
                    self.risk.analyze(item),
                    return_exceptions=True
                )
                valid_results = [r for r in results if isinstance(r, dict)]

                if valid_results:
                    # Master synthesis
                    final_signal = await self.master.synthesize(item, valid_results)
                    if final_signal:
                        item.signals.append(final_signal)
                        item.signals = item.signals[-20:]
                        item.latest_signal = final_signal
                        item.agent_coverage = [r["agent"] for r in valid_results] + ["master"]

                log.info(f"OK Agent pipeline complete: {sym}")
            except Exception as e:
                log.error(f"Agent pipeline error [{sym}]: {e}")
            finally:
                self._active[sym] = False

    async def run_on_watchlist(self, watchlist: List[WatchlistItem], max_tickers: int = 10):
        """Analyze top N watchlist items concurrently."""
        targets = [
            item for item in watchlist[:max_tickers]
            if not self._active.get(item.profile.symbol)
        ]
        log.info(f"Starting agent analysis on {len(targets)} tickers")
        await asyncio.gather(
            *[self.analyze_ticker(item) for item in targets],
            return_exceptions=True
        )


# Singleton
agent_pool = AgentPool()
