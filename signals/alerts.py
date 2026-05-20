"""
signals/alerts.py -- Price alerts and earnings calendar watcher

Alert types:
  - Price threshold (above/below)
  - % move alert (e.g. "notify if AAPL moves >3% in a day")
  - Signal alert (notify when a specific ticker gets a BUY/SELL)
  - Volume spike alert
  - Earnings upcoming (pre-event warning)

Alerts persist to storage/alerts.json
Earnings calendar fetched from Finnhub every 6 hours
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.bus import bus, emit
from core.logger import get_logger
from config.settings import SYS

import smtplib
import email.mime.text
import email.mime.multipart

log = get_logger("alerts")
ALERTS_FILE   = SYS.STORAGE_DIR / "alerts.json"
EARNINGS_FILE = SYS.STORAGE_DIR / "earnings.json"


@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    alert_type: str = "price_above"  # price_above|price_below|pct_move|signal|volume_spike
    threshold: float = 0.0
    signal_type: str = ""  # for alert_type=signal
    active: bool = True
    triggered: bool = False
    triggered_at: float = 0.0
    triggered_value: float = 0.0
    note: str = ""
    created_at: float = field(default_factory=time.time)
    notify_once: bool = True


class AlertsManager:
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._earnings: List[Dict] = []
        self._earnings_ts: float = 0
        self._email_enabled = bool(
            SYS.ALERT_EMAIL_TO and SYS.SMTP_USER and SYS.SMTP_PASS
        )
        self._load()

    async def _send_email(self, alert_data: dict, value: float):
        """Send email notification in a thread pool to avoid blocking."""
        if not self._email_enabled:
            return
        try:
            subject = (
                f"QuantAgent Alert: {alert_data.get('symbol','?')} "
                f"{alert_data.get('alert_type','?')} @ {value:.4f}"
            )
            body_lines = [
                f"Symbol:     {alert_data.get('symbol','')}",
                f"Type:       {alert_data.get('alert_type','')}",
                f"Value:      {value:.4f}",
                f"Threshold:  {alert_data.get('threshold','')}",
                f"Note:       {alert_data.get('note','')}",
                f"Signal:     {alert_data.get('signal_type','')}",
                f"Conviction: {alert_data.get('conviction','')}",
                f"Summary:    {alert_data.get('summary','')[:200]}",
                "",
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            ]
            body = "\n".join(body_lines)

            msg = email.mime.multipart.MIMEMultipart()
            msg["From"]    = SYS.SMTP_USER
            msg["To"]      = SYS.ALERT_EMAIL_TO
            msg["Subject"] = subject
            msg.attach(email.mime.text.MIMEText(body, "plain"))

            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_sync, msg)
        except Exception as e:
            log.warning(f"Email send failed: {e}")

    def _send_sync(self, msg):
        with smtplib.SMTP(SYS.SMTP_HOST, SYS.SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(SYS.SMTP_USER, SYS.SMTP_PASS)
            s.send_message(msg)

    def _load(self):
        try:
            if ALERTS_FILE.exists():
                data = json.loads(ALERTS_FILE.read_text(encoding='utf-8'))
                for k, v in data.items():
                    self._alerts[k] = Alert(**v)
                log.info(f"Loaded {len(self._alerts)} alerts")
        except Exception as e:
            log.warning(f"Alerts load error: {e}")

    def _save(self):
        try:
            ALERTS_FILE.write_text(
                json.dumps({k: asdict(v) for k, v in self._alerts.items()}, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            log.warning(f"Alerts save error: {e}")

    def add_alert(self, symbol: str, alert_type: str, threshold: float = 0,
                  signal_type: str = "", note: str = "", notify_once: bool = True) -> Alert:
        alert = Alert(
            symbol=symbol.upper(), alert_type=alert_type,
            threshold=threshold, signal_type=signal_type,
            note=note, notify_once=notify_once,
        )
        self._alerts[alert.alert_id] = alert
        self._save()
        log.info(f"Alert added: {symbol} {alert_type} @ {threshold}")
        return alert

    def remove_alert(self, alert_id: str) -> bool:
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            self._save()
            return True
        return False

    def get_alerts(self, active_only: bool = False) -> List[Dict]:
        alerts = list(self._alerts.values())
        if active_only:
            alerts = [a for a in alerts if a.active and not a.triggered]
        return [asdict(a) for a in sorted(alerts, key=lambda x: x.created_at, reverse=True)]

    async def check_price_alerts(self, prices: Dict[str, float], changes: Dict[str, float],
                                  volume_ratios: Optional[Dict[str, float]] = None):
        """Check all active alerts against current prices."""
        vol = volume_ratios or {}
        triggered = []
        for alert_id, alert in list(self._alerts.items()):
            if not alert.active:
                continue
            if alert.triggered and alert.notify_once:
                continue

            sym = alert.symbol
            price = prices.get(sym)
            if price is None:
                continue

            fired = False
            value = price

            if alert.alert_type == "price_above" and price >= alert.threshold:
                fired = True
            elif alert.alert_type == "price_below" and price <= alert.threshold:
                fired = True
            elif alert.alert_type == "pct_move":
                chg = abs(changes.get(sym, 0))
                if chg >= alert.threshold:
                    fired = True
                    value = chg
            elif alert.alert_type == "volume_spike":
                ratio = vol.get(sym, 1.0)
                threshold = alert.threshold if alert.threshold > 1.0 else 2.0
                if ratio >= threshold:
                    fired = True
                    value = ratio

            if fired:
                alert.triggered = True
                alert.triggered_at = time.time()
                alert.triggered_value = value
                self._save()
                triggered.append(alert)

                payload = {
                    "alert_id": alert_id,
                    "symbol": sym,
                    "alert_type": alert.alert_type,
                    "threshold": alert.threshold,
                    "current_value": value,
                    "note": alert.note,
                    "ts": time.time(),
                }
                await emit("alert.triggered", payload, "alerts")
                await self._send_email(payload, value)
                log.info(f"ALERT TRIGGERED: {sym} {alert.alert_type} @ {value:.4f} (threshold={alert.threshold})")

        return triggered

    async def check_signal_alerts(self, signal: Dict):
        """Trigger alerts watching for specific signal types on specific tickers."""
        sym = signal.get("symbol", "")
        sig_type = signal.get("signal_type", "")

        for alert in self._alerts.values():
            if not alert.active: continue
            if alert.triggered and alert.notify_once: continue
            if alert.alert_type != "signal": continue
            if alert.symbol != sym: continue
            if alert.signal_type and alert.signal_type not in sig_type: continue

            alert.triggered = True
            alert.triggered_at = time.time()
            alert.triggered_value = signal.get("conviction", 0)
            self._save()

            payload = {
                "alert_id": alert.alert_id,
                "symbol": sym,
                "alert_type": "signal",
                "signal_type": sig_type,
                "conviction": signal.get("conviction", 0),
                "summary": signal.get("summary", ""),
                "note": alert.note,
                "ts": time.time(),
            }
            await emit("alert.triggered", payload, "alerts")
            await self._send_email(payload, signal.get("conviction", 0))

    async def refresh_earnings(self):
        """Fetch upcoming earnings from Finnhub."""
        if time.time() - self._earnings_ts < 21600:  # 6h cache
            return
        try:
            from providers.registry import registry
            earnings = await registry.finnhub.get_earnings_calendar()
            self._earnings = earnings[:50]
            self._earnings_ts = time.time()
            # Save
            EARNINGS_FILE.write_text(
                json.dumps({"updated": self._earnings_ts, "earnings": self._earnings}, indent=2),
                encoding='utf-8'
            )
            log.info(f"Earnings calendar: {len(self._earnings)} upcoming events")
        except Exception as e:
            log.warning(f"Earnings fetch error: {e}")
            # Try loading cached
            try:
                if EARNINGS_FILE.exists():
                    data = json.loads(EARNINGS_FILE.read_text(encoding='utf-8'))
                    self._earnings = data.get("earnings", [])
            except Exception:
                pass

    def get_earnings(self, days_ahead: int = 7) -> List[Dict]:
        """Return earnings events within N days."""
        import datetime
        cutoff = (datetime.date.today() + datetime.timedelta(days=days_ahead)).isoformat()
        today  = datetime.date.today().isoformat()
        return [
            e for e in self._earnings
            if e.get("date", "") >= today and e.get("date", "") <= cutoff
        ]

    async def check_earnings_alerts(self, watchlist_symbols: set):
        """Fire alerts for tickers reporting earnings today or tomorrow."""
        import datetime
        today    = datetime.date.today().isoformat()
        tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

        for event in self._earnings:
            sym   = event.get("symbol", "").upper()
            edate = event.get("date", "")
            if edate not in (today, tomorrow) or sym not in watchlist_symbols:
                continue

            # Fire user-created earnings alerts for this symbol
            for alert in self._alerts.values():
                if (alert.alert_type == "earnings" and alert.symbol == sym
                        and alert.active and not (alert.triggered and alert.notify_once)):
                    alert.triggered = True
                    alert.triggered_at = time.time()
                    alert.triggered_value = 0
                    self._save()
                    payload = {
                        "alert_id": alert.alert_id,
                        "symbol": sym,
                        "alert_type": "earnings",
                        "note": f"Earnings report: {edate}",
                        "ts": time.time(),
                    }
                    await emit("alert.triggered", payload, "alerts")
                    await self._send_email(payload, 0)
                    log.info(f"EARNINGS ALERT: {sym} reports {edate}")

            # Auto-broadcast for any watchlist ticker with earnings today/tomorrow
            await emit("alert.triggered", {
                "symbol": sym,
                "alert_type": "earnings_auto",
                "note": f"Earnings {'today' if edate == today else 'tomorrow'} — {edate}",
                "ts": time.time(),
            }, "alerts")

    async def start(self):
        """Subscribe to signals and prices, check alerts continuously."""
        q = await bus.subscribe("signal", "watchlist.update")
        log.info("Alerts manager started")
        last_earnings_check = 0
        last_earnings_alert = 0

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)

                if event.topic == "signal":
                    await self.check_signal_alerts(event.data)

                elif event.topic == "watchlist.update":
                    items = event.data.get("items", [])
                    prices        = {i["profile"]["symbol"]: i["profile"]["price"] for i in items}
                    changes       = {i["profile"]["symbol"]: i["profile"]["change_pct"] for i in items}
                    volume_ratios = {i["profile"]["symbol"]: i["profile"].get("volume_ratio", 1.0) for i in items}
                    await self.check_price_alerts(prices, changes, volume_ratios)

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.debug(f"Alerts loop error: {e}")

            now = time.time()
            # Refresh earnings calendar every 6h
            if now - last_earnings_check > 3600:
                await self.refresh_earnings()
                last_earnings_check = now

            # Check earnings date alerts every hour
            if now - last_earnings_alert > 3600 and self._earnings:
                try:
                    from scanners.market_scanner import scanner
                    syms = set(scanner.watchlist.keys())
                    if syms:
                        await self.check_earnings_alerts(syms)
                except Exception as e:
                    log.debug(f"Earnings alert check: {e}")
                last_earnings_alert = now


alerts_manager = AlertsManager()
