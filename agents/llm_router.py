"""
agents/llm_router.py -- Unified LLM router: LM Studio (local) or Anthropic

Features:
  - Automatic retry with backoff on failure
  - Token counting and budget tracking
  - Streaming support (yields chunks)
  - Provider health monitoring
  - Per-call timing stats
"""
import asyncio
import time
from typing import AsyncGenerator, Optional
import httpx

from config.settings import KEYS, LLM
from core.logger import get_logger

log = get_logger("llm_router")

_stats = {
    "calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0,
    "total_ms": 0, "provider": LLM.PROVIDER, "model": LLM.display_name(),
    "tok_per_sec": 0.0, "retries": 0, "last_error": "",
}
_warned_no_key = False
_MAX_RETRIES = 2
_RETRY_DELAY = 2.0


class _CircuitBreaker:
    FAIL_THRESHOLD = 3
    RESET_AFTER_S  = 120

    def __init__(self):
        self._fails = 0
        self._opened: Optional[float] = None

    def success(self):
        self._fails = 0
        self._opened = None

    def failure(self):
        self._fails += 1
        if self._fails >= self.FAIL_THRESHOLD and self._opened is None:
            self._opened = time.time()
            log.warning(
                f"LLM circuit breaker OPEN after {self.FAIL_THRESHOLD} failures "
                f"— pausing {self.RESET_AFTER_S}s"
            )
            self._emit_event(
                "error",
                f"[LLM Circuit Breaker] OPEN — {self.FAIL_THRESHOLD} consecutive LLM "
                f"failures detected. Analysis paused for {self.RESET_AFTER_S}s.",
            )

    def is_open(self) -> bool:
        if self._opened is None:
            return False
        if (time.time() - self._opened) >= self.RESET_AFTER_S:
            trip_s = int(time.time() - self._opened)
            log.info("LLM circuit breaker CLOSED — resuming")
            self._opened = None
            self._fails = 0
            self._emit_event(
                "planning",
                f"[LLM Circuit Breaker] CLOSED — resuming after {trip_s}s pause.",
            )
            return False
        return True

    def secs_until_reset(self) -> int:
        if not self._opened:
            return 0
        return max(0, int(self.RESET_AFTER_S - (time.time() - self._opened)))

    @staticmethod
    def _emit_event(action: str, message: str):
        try:
            from core.bus import emit as _bus_emit
            asyncio.get_event_loop().create_task(
                _bus_emit("agent.activity", {
                    "agent_id":   "llm_circuit",
                    "agent_type": "narrator",
                    "symbol":     "—",
                    "action":     action,
                    "message":    message,
                    "ts":         time.time(),
                }, "llm_router")
            )
        except Exception:
            pass


_circuit = _CircuitBreaker()

# KV cache pressure tracking: warn when many unique system prompts are in use.
# LM Studio assigns a KV cache slot per unique prompt prefix; too many unique
# system prompts exhaust VRAM before inference begins.
# Fix: keep system= a static role template; put ticker data in the user= message.
_unique_sys_hashes: set = set()
_MAX_UNIQUE_SYS    = 6
_kv_warned         = False


def get_llm_stats() -> dict:
    s = dict(_stats)
    s["circuit_open"] = _circuit.is_open()
    s["circuit_reset_in"] = _circuit.secs_until_reset()
    return s


async def call_llm(
    system: str,
    user: str,
    max_tokens: int = None,
    agent_id: str = "agent",
    temperature: float = 0.3,
    retries: int = _MAX_RETRIES,
) -> Optional[str]:
    """Route to configured LLM. Retries on transient failures."""
    mt = max_tokens or LLM.MAX_TOKENS
    last_err = None

    for attempt in range(retries + 1):
        if attempt > 0:
            _stats["retries"] += 1
            await asyncio.sleep(_RETRY_DELAY * attempt)
            log.debug(f"LLM retry {attempt}/{retries} [{agent_id}]")

        try:
            if LLM.is_local():
                result = await _call_local(system, user, mt, agent_id, temperature)
            else:
                result = await _call_anthropic(system, user, mt, agent_id)

            if result is not None:
                return result

        except Exception as e:
            last_err = str(e)
            log.debug(f"LLM attempt {attempt} failed [{agent_id}]: {e}")

    if last_err:
        _stats["last_error"] = last_err
    return None


async def stream_llm(
    system: str,
    user: str,
    max_tokens: int = None,
    agent_id: str = "agent",
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    """Stream LLM response token by token. Falls back to single call if unsupported."""
    mt = max_tokens or LLM.MAX_TOKENS

    if LLM.is_local():
        async for chunk in _stream_local(system, user, mt, agent_id, temperature):
            yield chunk
    else:
        # Anthropic streaming
        async for chunk in _stream_anthropic(system, user, mt, agent_id):
            yield chunk


async def _call_local(system: str, user: str, max_tokens: int,
                       agent_id: str, temperature: float) -> Optional[str]:
    """Use streaming internally so slow local models don't hit the read timeout.
    With stream=True, httpx read_timeout applies per-chunk not total response,
    so a model generating at 0.2 tok/s won't disconnect after 120s."""
    global _kv_warned
    if _circuit.is_open():
        log.debug(f"LLM circuit open — skipping [{agent_id}] ({_circuit.secs_until_reset()}s remaining)")
        return None

    # Track unique system prompt hashes to detect KV cache pressure.
    import hashlib
    sys_hash = hashlib.md5(system[:300].encode("utf-8", errors="ignore")).hexdigest()[:8]
    _unique_sys_hashes.add(sys_hash)
    if len(_unique_sys_hashes) > _MAX_UNIQUE_SYS and not _kv_warned:
        _kv_warned = True
        log.warning(
            f"KV cache pressure: {len(_unique_sys_hashes)} unique system prompts detected. "
            "Each occupies ~75MB VRAM in LM Studio. "
            "Keep system= a static template — put ticker data in user= messages."
        )

    url = f"{LLM.LOCAL_URL}/chat/completions"
    payload = {
        "model": LLM.LOCAL_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    t0 = time.time()
    full_text = ""
    tokens_out = 0
    try:
        # connect=10s, read=per-chunk timeout (keeps connection alive for slow models)
        timeout = httpx.Timeout(connect=10.0, read=float(LLM.LOCAL_TIMEOUT),
                                write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code == 503:
                    log.warning("LM Studio not ready -- is the model loaded?")
                    _stats["errors"] += 1
                    return None
                if resp.status_code == 404:
                    log.warning(f"LM Studio model not found: {LLM.LOCAL_MODEL}")
                    _stats["errors"] += 1
                    return None
                if resp.status_code != 200:
                    log.debug(f"LM Studio {resp.status_code} [{agent_id}]")
                    _stats["errors"] += 1
                    return None
                import json as _json
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_text += delta
                                tokens_out += 1
                        except Exception:
                            pass
        elapsed = int((time.time() - t0) * 1000)
        if full_text:
            _update_stats(elapsed, 0, tokens_out)
            _circuit.success()
            return full_text
        log.debug(f"LM Studio returned empty response [{agent_id}]")
        return None
    except httpx.ConnectError:
        log.warning(
            f"LM Studio not reachable at {LLM.LOCAL_URL} -- "
            "open LM Studio -> Local Server -> Start Server"
        )
        _stats["errors"] += 1
        _circuit.failure()
    except httpx.TimeoutException:
        log.warning(f"LM Studio chunk timeout [{agent_id}] — model may have stalled")
        _stats["errors"] += 1
        _circuit.failure()
        if full_text:
            return full_text  # return whatever we got before the stall
    except Exception as e:
        log.debug(f"LM Studio error [{agent_id}]: {type(e).__name__}: {e}")
        _stats["errors"] += 1
        _circuit.failure()
    return None


async def _stream_local(system: str, user: str, max_tokens: int,
                         agent_id: str, temperature: float) -> AsyncGenerator[str, None]:
    url = f"{LLM.LOCAL_URL}/chat/completions"
    payload = {
        "model": LLM.LOCAL_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=LLM.LOCAL_TIMEOUT) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            pass
    except Exception as e:
        log.debug(f"Stream error [{agent_id}]: {e}")


async def _call_anthropic(system: str, user: str, max_tokens: int,
                           agent_id: str) -> Optional[str]:
    global _warned_no_key
    if _circuit.is_open():
        log.debug(f"LLM circuit open — skipping [{agent_id}]")
        return None
    if not KEYS.ANTHROPIC:
        if not _warned_no_key:
            log.warning(
                "ANTHROPIC_API_KEY not set. Set LLM_PROVIDER=local for LM Studio "
                "or add ANTHROPIC_API_KEY to .env"
            )
            _warned_no_key = True
        return None

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": KEYS.ANTHROPIC,
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload
            )
            elapsed = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"]
                usage = data.get("usage", {})
                _update_stats(elapsed, usage.get("input_tokens", 0),
                              usage.get("output_tokens", len(text.split())))
                _circuit.success()
                return text
            elif resp.status_code == 401:
                if not _warned_no_key:
                    log.warning("Anthropic 401 -- check ANTHROPIC_API_KEY")
                    _warned_no_key = True
            elif resp.status_code == 429:
                log.warning("Anthropic rate limit hit")
                _stats["errors"] += 1
                raise Exception("rate_limit")  # trigger retry
            else:
                log.debug(f"Anthropic {resp.status_code} [{agent_id}]")
            _stats["errors"] += 1
            _circuit.failure()
    except Exception as e:
        if "rate_limit" in str(e):
            raise
        log.debug(f"Anthropic error [{agent_id}]: {type(e).__name__}")
        _stats["errors"] += 1
        _circuit.failure()
    return None


async def _stream_anthropic(system: str, user: str, max_tokens: int,
                              agent_id: str) -> AsyncGenerator[str, None]:
    if not KEYS.ANTHROPIC:
        return
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": KEYS.ANTHROPIC,
        "anthropic-beta": "messages-2023-06-01",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system,
        "stream": True,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            import json
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {}).get("text", "")
                                if delta:
                                    yield delta
                        except Exception:
                            pass
    except Exception as e:
        log.debug(f"Anthropic stream error [{agent_id}]: {e}")


def _update_stats(elapsed_ms: int, tokens_in: int, tokens_out: int):
    _stats["calls"] += 1
    _stats["tokens_in"] += tokens_in
    _stats["tokens_out"] += tokens_out
    _stats["total_ms"] += elapsed_ms
    if elapsed_ms > 0 and tokens_out > 0:
        _stats["tok_per_sec"] = round(tokens_out / (elapsed_ms / 1000), 1)


def count_tokens(text: str) -> int:
    """Rough token count (4 chars ~= 1 token)."""
    return max(1, len(text) // 4)


def format_stats() -> str:
    s = _stats
    return (
        f"{s['provider']} | {s['model']} | "
        f"calls={s['calls']} | "
        f"tokens={s['tokens_in']}in/{s['tokens_out']}out | "
        f"{s['tok_per_sec']} tok/s | "
        f"errors={s['errors']} | retries={s['retries']}"
    )
