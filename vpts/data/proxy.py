"""Rotating proxy pool — keep high-volume free-feed pulls from getting IP-blocked.

Free market-data endpoints (Yahoo's v8 chart JSON, yfinance) rate-limit and then
**block by source IP**: scan a few hundred names and the 429s start. Routing requests
through a pool of proxies — rotating per request and backing off any IP that gets
throttled — spreads the load so a full audit completes without a block.

What this is (and isn't)
------------------------
A proxy changes the *source IP*; it does **not** execute JavaScript. It therefore helps
against **IP rate-limiting** (Yahoo/yfinance) but does **not** defeat a JavaScript
proof-of-work wall such as Stooq's live CSV endpoint (and datacenter IPs tend to get
*challenged more*, not less). Scope here is the rate-limited endpoints only.

Credentials never live in the repo
-----------------------------------
Proxies are loaded at runtime from, in order: ``$VPTS_PROXIES`` (inline list) →
``$VPTS_PROXY_FILE`` (a path) → a git-ignored ``proxies.txt`` at the repo root. Each
line is the Webshare export format ``host:port:user:pass`` (full ``http://user:pass@host:port``
URLs are also accepted). With nothing configured, :meth:`from_env` returns ``None`` and
callers fetch directly — fully backward compatible.

Anti-block hygiene
------------------
* **Rotation** — round-robin across all proxies, so load is spread evenly.
* **Cooldown** — a proxy that returns 403/407/429 or a transport error is parked in
  **exponential-backoff cooldown** (30s → 60s → … capped) and skipped until it recovers.
* **Jitter** — a small randomized sleep between requests avoids bursty patterns.
* **User-Agent rotation** — each request carries a realistic, rotating UA.

The network transport is **injectable** (``transport=`` / ``opener_factory=``), so the
rotation/cooldown logic is unit-tested with no network — mirroring the ``http_get`` seam
in :class:`~vpts.data.sources.polygon_source.PolygonSource`.
"""
from __future__ import annotations

import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

#: A small pool of realistic desktop User-Agents, rotated per request.
_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
)

#: HTTP statuses that mean "this IP is being throttled/blocked" → cool the proxy down.
_BLOCK_STATUSES: frozenset[int] = frozenset({403, 407, 429, 503})


class AllProxiesCoolingError(RuntimeError):
    """Raised when every proxy is in cooldown (all recently blocked/errored)."""


class DataFetchProxyError(RuntimeError):
    """Raised when a request fails across all attempted proxies."""


@dataclass
class _Proxy:
    """One proxy endpoint plus its live health state."""

    url: str                              # http://user:pass@host:port
    label: str                            # host:port (safe to log — no credentials)
    fails: int = 0                        # consecutive failures (drives backoff)
    cooldown_until: float = 0.0           # monotonic time before which to skip it

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until


def _parse_spec(spec: str) -> Optional[_Proxy]:
    """Parse one proxy spec (``host:port:user:pass`` or a full URL) → :class:`_Proxy`."""
    spec = spec.strip()
    if not spec or spec.startswith("#"):
        return None
    if "://" in spec:                                   # already a URL
        rest = spec.split("://", 1)[1]
        hostport = rest.split("@")[-1]
        return _Proxy(url=spec, label=hostport)
    parts = spec.split(":")
    if len(parts) == 4:                                 # Webshare: host:port:user:pass
        host, port, user, pwd = parts
        return _Proxy(url=f"http://{user}:{pwd}@{host}:{port}", label=f"{host}:{port}")
    if len(parts) == 2:                                 # host:port (no auth)
        host, port = parts
        return _Proxy(url=f"http://{host}:{port}", label=f"{host}:{port}")
    raise ValueError(
        f"Unrecognised proxy spec {spec!r}; expected 'host:port:user:pass', "
        "'host:port', or a full 'http://user:pass@host:port' URL."
    )


def _split_specs(blob: str) -> list[str]:
    """Split an inline blob on newlines/commas/whitespace into individual specs."""
    return [s for s in blob.replace(",", "\n").split("\n") if s.strip()]


@dataclass
class ProxyPool:
    """A rotating pool of proxies with per-proxy cooldown and request hygiene.

    Parameters
    ----------
    specs:
        Iterable of proxy specs (``host:port:user:pass`` or URLs).
    shuffle:
        Shuffle the order on construction (default True) so concurrent runs don't all
        start on the same IP.
    jitter:
        ``(min, max)`` seconds of randomized sleep before each request (default
        ``(0.3, 1.2)``). Set ``(0, 0)`` for deterministic tests.
    max_tries:
        Max proxies to try for a single request before giving up (default 4).
    cooldown_base / cooldown_cap:
        Exponential-backoff cooldown after a block: ``base * 2**(fails-1)``, capped.
    transport:
        Injectable ``(url, proxy_url, headers, timeout) -> bytes`` for testing. Defaults
        to a stdlib ``urllib`` opener built per proxy.
    rng:
        Optional ``random.Random`` for deterministic jitter/UA/shuffle in tests.
    """

    specs: Sequence[str]
    shuffle: bool = True
    jitter: tuple[float, float] = (0.3, 1.2)
    max_tries: int = 4
    cooldown_base: float = 30.0
    cooldown_cap: float = 600.0
    transport: Optional[Callable[[str, str, dict, float], bytes]] = None
    rng: random.Random = field(default_factory=random.Random)

    _proxies: list[_Proxy] = field(init=False, default_factory=list)
    _idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for spec in self.specs:
            p = _parse_spec(spec)
            if p is not None and p.url not in seen:
                seen.add(p.url)
                self._proxies.append(p)
        if not self._proxies:
            raise ValueError("ProxyPool needs at least one valid proxy spec.")
        if self.shuffle:
            self.rng.shuffle(self._proxies)
        self._fetch = self.transport or _urllib_transport

    # ------------------------------------------------------------------ #
    # Construction from the environment (the normal entry point)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **kwargs) -> Optional["ProxyPool"]:
        """Build from ``$VPTS_PROXIES`` → ``$VPTS_PROXY_FILE`` → ``proxies.txt``.

        Returns ``None`` when nothing is configured, so callers can simply do
        ``pool = ProxyPool.from_env()`` and fetch directly when it is ``None``.
        """
        inline = os.environ.get("VPTS_PROXIES")
        if inline and inline.strip():
            return cls(_split_specs(inline), **kwargs)
        path = os.environ.get("VPTS_PROXY_FILE") or "proxies.txt"
        p = Path(path)
        if p.exists():
            return cls.from_file(p, **kwargs)
        return None

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "ProxyPool":
        """Build from a file of one proxy spec per line (``#`` comments allowed)."""
        text = Path(path).read_text("utf-8")
        return cls(_split_specs(text), **kwargs)

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self._proxies)

    @property
    def labels(self) -> list[str]:
        """Credential-free ``host:port`` labels (safe to log)."""
        return [p.label for p in self._proxies]

    def _next_available(self, now: float) -> Optional[_Proxy]:
        """Round-robin to the next proxy whose cooldown has elapsed."""
        n = len(self._proxies)
        for _ in range(n):
            p = self._proxies[self._idx % n]
            self._idx += 1
            if p.available(now):
                return p
        return None

    def _cool(self, p: _Proxy) -> None:
        p.fails += 1
        delay = min(self.cooldown_base * (2 ** (p.fails - 1)), self.cooldown_cap)
        p.cooldown_until = time.monotonic() + delay

    def _ok(self, p: _Proxy) -> None:
        p.fails = 0
        p.cooldown_until = 0.0

    def _ua(self) -> str:
        return _USER_AGENTS[self.rng.randrange(len(_USER_AGENTS))]

    def _sleep_jitter(self) -> None:
        lo, hi = self.jitter
        if hi > 0:
            time.sleep(self.rng.uniform(lo, hi))

    # ------------------------------------------------------------------ #
    def urlopen(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> bytes:
        """GET *url* through a healthy proxy, rotating + cooling on blocks.

        Tries up to :attr:`max_tries` distinct healthy proxies; a 403/407/429/503 or a
        transport error parks the offending proxy in exponential-backoff cooldown and
        moves on. Raises :class:`AllProxiesCoolingError` if none are available, or the
        last error after exhausting :attr:`max_tries`.
        """
        last_err: Optional[Exception] = None
        for _ in range(max(1, self.max_tries)):
            now = time.monotonic()
            p = self._next_available(now)
            if p is None:
                raise AllProxiesCoolingError(
                    f"all {self.size} proxies are cooling down (recent blocks/errors)."
                )
            req_headers = {"User-Agent": self._ua(), **(headers or {})}
            self._sleep_jitter()
            try:
                body = self._fetch(url, p.url, req_headers, timeout)
            except urllib.error.HTTPError as exc:        # got a response with a status
                last_err = exc
                if exc.code in _BLOCK_STATUSES:
                    self._cool(p)
                    continue
                raise                                    # 404 etc. — a real answer, not a block
            except Exception as exc:                     # noqa: BLE001 — transport/timeout
                last_err = exc
                self._cool(p)
                continue
            self._ok(p)
            return body
        raise DataFetchProxyError(
            f"request failed after trying {self.max_tries} prox(ies): {last_err}"
        ) from last_err

    # ------------------------------------------------------------------ #
    # requests.Session for clients that take one (e.g. yfinance via session=)
    # ------------------------------------------------------------------ #
    def requests_session(self):
        """Return a ``requests.Session`` pinned to one healthy proxy (rotating UA).

        For clients like yfinance that accept ``session=``. Use :meth:`rotate_session`
        to repin to a different proxy after a block. Requires ``requests`` (already a
        transitive dependency via yfinance).
        """
        import requests  # local import: optional for the urllib path

        session = requests.Session()
        return self.rotate_session(session)

    def rotate_session(self, session):
        """Point an existing ``requests.Session`` at the next healthy proxy."""
        p = self._next_available(time.monotonic())
        if p is None:
            raise AllProxiesCoolingError(
                f"all {self.size} proxies are cooling down (recent blocks/errors)."
            )
        session.proxies.update({"http": p.url, "https": p.url})
        session.headers.update({"User-Agent": self._ua()})
        session._vpts_proxy = p  # type: ignore[attr-defined]  # so cool_session() can park it
        return session

    def cool_session(self, session) -> None:
        """Park the session's current proxy in cooldown (e.g. after a failed fetch).

        Lets clients that hold a long-lived proxied ``Session`` (yfinance) apply the
        same exponential-backoff-and-skip that :meth:`urlopen` applies automatically,
        so a rate-limited proxy isn't immediately rotated back into use.
        """
        p = getattr(session, "_vpts_proxy", None)
        if p is not None:
            self._cool(p)


def _urllib_transport(url: str, proxy_url: str, headers: dict, timeout: float) -> bytes:
    """Default transport: a stdlib opener bound to one proxy. (Network — not unit-tested.)"""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    )
    req = urllib.request.Request(url, headers=headers)
    with opener.open(req, timeout=timeout) as resp:  # pragma: no cover - needs network
        return resp.read()
