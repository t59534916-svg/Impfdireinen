"""Tests for the rotating ProxyPool — offline via an injected transport.

The load-bearing checks: it parses Webshare/URL specs, rotates round-robin, and — the
point of the whole thing — when a proxy gets blocked (403/429) it **cools that proxy
down, rotates to a healthy one, and the request still succeeds**. Plus the Yahoo v8
loader routes through the pool. All with zero network.

    python tests/test_proxy.py
    pytest tests/test_proxy.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import pandas as pd  # noqa: E402

from vpts.data import (  # noqa: E402
    AllProxiesCoolingError,
    DataFetchProxyError,
    ProxyPool,
)
from vpts.data.proxy import _parse_spec  # noqa: E402

_SPECS = [
    "166.0.40.167:7175:user:pass",
    "82.23.88.111:7867:user:pass",
    "45.56.183.188:8510:user:pass",
]


def _pool(transport, **kw):
    """A deterministic pool: fixed order, no jitter, fixed RNG."""
    import random
    return ProxyPool(_SPECS, shuffle=False, jitter=(0, 0), transport=transport,
                     rng=random.Random(0), **kw)


# --------------------------------------------------------------------------- #
def test_parse_webshare_and_url_specs() -> None:
    p = _parse_spec("166.0.40.167:7175:adnaiwqf:secret")
    assert p.url == "http://adnaiwqf:secret@166.0.40.167:7175"
    assert p.label == "166.0.40.167:7175"          # label carries no credentials
    p2 = _parse_spec("http://u:pw@1.2.3.4:9000")
    assert p2.url == "http://u:pw@1.2.3.4:9000" and p2.label == "1.2.3.4:9000"
    p3 = _parse_spec("1.2.3.4:8080")               # no-auth host:port
    assert p3.url == "http://1.2.3.4:8080"
    assert _parse_spec("   ") is None and _parse_spec("# comment") is None


def test_labels_never_leak_credentials() -> None:
    pool = _pool(lambda *a: b"{}")
    joined = " ".join(pool.labels)
    assert "user" not in joined and "pass" not in joined
    assert pool.size == 3


def test_round_robin_rotation() -> None:
    seen = []

    def transport(url, proxy_url, headers, timeout):
        seen.append(proxy_url)
        return b"{}"

    pool = _pool(transport)
    for _ in range(4):
        pool.urlopen("https://example.com")
    # Three distinct proxies, cycling in order, wrapping on the 4th call.
    assert len(set(seen)) == 3
    assert seen[0] != seen[1] != seen[2] and seen[3] == seen[0]


def test_block_cools_proxy_and_rotates_to_healthy() -> None:
    blocked_url = _parse_spec(_SPECS[0]).url
    calls = []

    def transport(url, proxy_url, headers, timeout):
        calls.append(proxy_url)
        if proxy_url == blocked_url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return b'{"ok": true}'

    pool = _pool(transport)
    body = pool.urlopen("https://example.com")        # first proxy 429s → rotate
    assert body == b'{"ok": true}'
    assert calls[0] == blocked_url and calls[1] != blocked_url

    # The blocked proxy is now cooling: a fresh request skips it entirely.
    calls.clear()
    pool.urlopen("https://example.com")
    assert blocked_url not in calls


def test_user_agent_is_set_and_rotates() -> None:
    uas = []

    def transport(url, proxy_url, headers, timeout):
        uas.append(headers.get("User-Agent"))
        return b"{}"

    pool = _pool(transport)
    for _ in range(6):
        pool.urlopen("https://example.com")
    assert all(uas) and len(set(uas)) > 1              # a UA is always present, and it varies


def test_non_block_http_error_propagates() -> None:
    def transport(url, proxy_url, headers, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    pool = _pool(transport)
    try:
        pool.urlopen("https://example.com")            # 404 is a real answer, not a block
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:  # pragma: no cover
        raise AssertionError("404 should propagate, not be retried as a block")


def test_all_cooling_raises() -> None:
    def transport(url, proxy_url, headers, timeout):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    pool = _pool(transport, max_tries=3)
    try:
        pool.urlopen("https://example.com")            # every proxy 403s
    except (AllProxiesCoolingError, DataFetchProxyError):
        pass
    else:  # pragma: no cover
        raise AssertionError("expected an error once all proxies are cooling")


def test_from_env_inline_and_none(monkeypatch=None) -> None:
    import os
    saved = {k: os.environ.pop(k, None) for k in ("VPTS_PROXIES", "VPTS_PROXY_FILE")}
    try:
        # Nothing configured + no proxies.txt in CWD → None (callers fetch direct).
        cwd = os.getcwd()
        os.chdir(Path(__file__).resolve().parent)      # a dir with no proxies.txt
        try:
            assert ProxyPool.from_env() is None
        finally:
            os.chdir(cwd)
        # Inline list wins.
        os.environ["VPTS_PROXIES"] = "1.2.3.4:80:u:p, 5.6.7.8:81:u:p"
        pool = ProxyPool.from_env(jitter=(0, 0))
        assert pool is not None and pool.size == 2
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_from_file(tmp_path) -> None:
    f = tmp_path / "proxies.txt"
    f.write_text("# my proxies\n166.0.40.167:7175:u:p\n\nhttp://a:b@9.9.9.9:8000\n")
    pool = ProxyPool.from_file(f, jitter=(0, 0))
    assert pool.size == 2 and "166.0.40.167:7175" in pool.labels


def test_rejects_empty_pool() -> None:
    try:
        ProxyPool(["# only a comment", "  "])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a pool with no valid proxies")


# --- Yahoo v8 loader routes through the pool ------------------------------- #
def _v8_json(n=300) -> bytes:
    idx = pd.date_range("2007-01-03", periods=n, freq="B")
    ts = [int(t.timestamp()) for t in idx]
    o = [10.0 + i * 0.01 for i in range(n)]
    doc = {"chart": {"result": [{
        "timestamp": ts,
        "indicators": {
            "quote": [{"open": o, "high": [x + 0.5 for x in o], "low": [x - 0.5 for x in o],
                       "close": o, "volume": [1000 + i for i in range(n)]}],
            "adjclose": [{"adjclose": o}],
        },
    }]}}
    return json.dumps(doc).encode()


def test_yahoo_v8_loader_uses_pool_and_rotates_on_block() -> None:
    from real_delisted_audit import yahoo_v8_loader

    blocked_url = _parse_spec(_SPECS[0]).url
    used = []

    def transport(url, proxy_url, headers, timeout):
        used.append(proxy_url)
        if proxy_url == blocked_url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return _v8_json()

    pool = _pool(transport)
    fetch = yahoo_v8_loader("2007-01-01", "2008-01-01", pool=pool)
    df = fetch("LEH")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) > 100
    assert used[0] == blocked_url and used[1] != blocked_url   # rotated past the block


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import tempfile
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} ProxyPool tests …\n")
    for t in tests:
        try:
            if "tmp_path" in t.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
