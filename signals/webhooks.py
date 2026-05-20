"""
signals/webhooks.py -- Outbound webhook dispatcher

Registers webhook URLs and POSTs a JSON payload for every:
  signal, portfolio.opened, portfolio.closed, alert.triggered, digest.generated

Webhook URLs are persisted to storage/webhooks.json.
Register via POST /api/webhooks. Plug the URL into Slack, Discord, n8n, Zapier.
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import aiohttp

from core.bus import bus
from core.logger import get_logger
from config.settings import SYS

log = get_logger("webhooks")
WEBHOOKS_FILE = SYS.STORAGE_DIR / "webhooks.json"

SUPPORTED_EVENTS = frozenset({
    "signal", "portfolio.opened", "portfolio.closed",
    "alert.triggered", "digest.generated", "all",
})


@dataclass
class Webhook:
    webhook_id: str
    url: str
    name: str
    events: List[str]
    created_at: float
    active: bool = True
    last_fired: float = 0.0
    fire_count: int = 0
    error_count: int = 0
    last_error: str = ""
    last_status: int = 0


class WebhookManager:
    def __init__(self):
        self._hooks: Dict[str, Webhook] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if WEBHOOKS_FILE.exists():
                data = json.loads(WEBHOOKS_FILE.read_text(encoding="utf-8"))
                for wh in data.get("webhooks", []):
                    w = Webhook(**wh)
                    self._hooks[w.webhook_id] = w
                log.info(f"Webhooks loaded: {len(self._hooks)}")
        except Exception as e:
            log.warning(f"Webhook load error: {e}")

    def _save(self):
        try:
            WEBHOOKS_FILE.write_text(
                json.dumps({"webhooks": [asdict(w) for w in self._hooks.values()]}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning(f"Webhook save error: {e}")

    # ── Public API ───────────────────────────────────────────────────────────

    def add(self, url: str, name: str = "", events: List[str] = None) -> Webhook:
        events = events or ["all"]
        valid  = [e for e in events if e in SUPPORTED_EVENTS]
        if not valid:
            valid = ["all"]
        wh = Webhook(
            webhook_id=str(uuid.uuid4())[:12],
            url=url.strip(),
            name=(name or url[:50]).strip(),
            events=valid,
            created_at=time.time(),
        )
        self._hooks[wh.webhook_id] = wh
        self._save()
        log.info(f"Webhook registered [{wh.webhook_id}]: {url}")
        return wh

    def remove(self, webhook_id: str) -> bool:
        if webhook_id in self._hooks:
            del self._hooks[webhook_id]
            self._save()
            log.info(f"Webhook removed: {webhook_id}")
            return True
        return False

    def toggle(self, webhook_id: str, active: bool) -> bool:
        wh = self._hooks.get(webhook_id)
        if wh:
            wh.active = active
            self._save()
            return True
        return False

    def get_all(self) -> List[Dict]:
        return [asdict(w) for w in self._hooks.values()]

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    def _matches(self, wh: Webhook, event_type: str) -> bool:
        if not wh.active:
            return False
        return "all" in wh.events or event_type in wh.events

    async def _fire(self, wh: Webhook, event_type: str, payload: Dict):
        try:
            sess = await self._get_session()
            body = {
                "event":      event_type,
                "ts":         time.time(),
                "webhook_id": wh.webhook_id,
                "data":       payload,
            }
            async with sess.post(
                wh.url, json=body,
                headers={"Content-Type": "application/json", "X-QuantAgent-Event": event_type},
            ) as resp:
                wh.last_fired  = time.time()
                wh.fire_count += 1
                wh.last_status = resp.status
                if resp.status >= 400:
                    wh.error_count += 1
                    wh.last_error = f"HTTP {resp.status}"
                    log.warning(f"Webhook [{wh.webhook_id}] {event_type} → HTTP {resp.status}")
                else:
                    log.debug(f"Webhook [{wh.webhook_id}] {event_type} → OK")
        except Exception as e:
            wh.error_count += 1
            wh.last_error = str(e)[:120]
            log.debug(f"Webhook [{wh.webhook_id}] error: {e}")
        finally:
            self._save()

    async def dispatch(self, event_type: str, payload: Dict):
        targets = [w for w in self._hooks.values() if self._matches(w, event_type)]
        if not targets:
            return
        await asyncio.gather(
            *[self._fire(w, event_type, payload) for w in targets],
            return_exceptions=True,
        )

    async def test_webhook(self, webhook_id: str) -> Dict:
        """Fire a test payload to a single webhook and return result."""
        wh = self._hooks.get(webhook_id)
        if not wh:
            return {"error": "webhook not found"}
        await self._fire(wh, "test", {"message": "QuantAgent webhook test", "ts": time.time()})
        return {"status": wh.last_status, "error": wh.last_error}

    # ── Bus listener ─────────────────────────────────────────────────────────

    async def start(self):
        q = await bus.subscribe(
            "signal", "portfolio.opened", "portfolio.closed",
            "alert.triggered", "digest.generated",
        )
        log.info("Webhook dispatcher started")
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30)
                if self._hooks:
                    await self.dispatch(event.topic, event.data)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.debug(f"Webhook dispatch error: {e}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


webhook_manager = WebhookManager()
