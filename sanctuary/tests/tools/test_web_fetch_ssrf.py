"""Regression guards: web_fetch cannot be used for SSRF.

web_fetch used to fetch any entity-supplied URL with follow_redirects=True and
no host/scheme filtering -- reachable targets included http://127.0.0.1, the
cloud-metadata endpoint 169.254.169.254, and internal LAN services. These pin
the guard down: only http/https, never internal addresses, and (when fetching
directly) hostnames are resolved and their IPs checked.

The redirect-hop validation is structural (web_fetch validates `current` before
every hop); these test the validator that gates each hop.

Authored by Fable 5 (adversarial seat), 2026-07-02.
"""
from __future__ import annotations

import httpx
import pytest

from sanctuary.tools import builtin
from sanctuary.tools.builtin import _ip_is_blocked, _validate_fetch_url, _web_fetch


# ---------------------------------------------------------------------------
# Scheme allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "gopher://example.com",
    "data:text/plain,hi",
    "//example.com/no-scheme",
])
def test_non_http_schemes_blocked(url):
    reason = _validate_fetch_url(url, resolve=True)
    assert reason is not None
    assert "blocked" in reason


# ---------------------------------------------------------------------------
# Literal internal IPs blocked (always, even via proxy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private
    "192.168.1.1",      # private
    "172.16.0.1",       # private
    "169.254.169.254",  # link-local -- cloud metadata
    "0.0.0.0",          # unspecified
    "[::1]",            # ipv6 loopback
])
def test_literal_internal_ips_blocked(host):
    for resolve in (True, False):  # blocked even when a proxy defers DNS
        reason = _validate_fetch_url(f"http://{host}/", resolve=resolve)
        assert reason is not None, (host, resolve)
        assert "blocked" in reason


def test_public_literal_ip_allowed():
    assert _validate_fetch_url("http://8.8.8.8/", resolve=True) is None
    assert _validate_fetch_url("https://1.1.1.1/", resolve=False) is None


# ---------------------------------------------------------------------------
# Hostname resolution (localhost resolves locally -> loopback -> blocked)
# ---------------------------------------------------------------------------


def test_localhost_name_resolves_and_is_blocked():
    reason = _validate_fetch_url("http://localhost/admin", resolve=True)
    assert reason is not None
    assert "blocked" in reason


def test_name_resolution_skipped_under_proxy():
    # With a proxy (resolve=False) a name is NOT resolved here (the proxy is
    # the egress boundary) -- so a name that only the proxy can reach is not
    # pre-rejected. Literal internal IPs are still blocked (tested above).
    assert _validate_fetch_url("http://some-internal-name/", resolve=False) is None


# ---------------------------------------------------------------------------
# _ip_is_blocked unit
# ---------------------------------------------------------------------------


def test_ip_is_blocked_classification():
    for blocked in ("127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.169.254",
                    "::1", "fc00::1", "224.0.0.1", "0.0.0.0"):
        assert _ip_is_blocked(blocked), blocked
    for allowed in ("8.8.8.8", "1.1.1.1", "93.184.216.34"):
        assert not _ip_is_blocked(allowed), allowed
    # A non-IP string is not an IP -> not "blocked" by this helper (the caller
    # handles names via resolution).
    assert _ip_is_blocked("example.com") is False


# ---------------------------------------------------------------------------
# Alternate IPv4 encodings (octal / integer / short form) -> loopback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["0177.0.0.1", "2130706433", "127.1"])
def test_alternate_ipv4_encodings_blocked(host):
    # These bypass ipaddress.ip_address but inet_aton canonicalizes them to
    # 127.0.0.1. Blocked even under a proxy (resolve=False) with no DNS.
    reason = _validate_fetch_url(f"http://{host}/", resolve=False)
    assert reason is not None
    assert "blocked" in reason


def test_genuinely_public_octal_allowed():
    # 010.0.0.1 is octal 8.0.0.1 -- a real public address, must NOT be blocked.
    assert _validate_fetch_url("http://010.0.0.1/", resolve=False) is None


# ---------------------------------------------------------------------------
# Integration: the actual _web_fetch redirect loop (offline via MockTransport)
# ---------------------------------------------------------------------------


def _inject_client(monkeypatch, handler):
    def factory(proxy):
        return httpx.AsyncClient(
            follow_redirects=False, transport=httpx.MockTransport(handler)
        )
    monkeypatch.setattr(builtin, "_make_web_client", factory)


@pytest.mark.asyncio
async def test_web_fetch_denies_redirect_to_internal(monkeypatch):
    reached = {"internal": False}

    def handler(request):
        host = request.url.host
        if host == "8.8.8.8":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        if host == "127.0.0.1":
            reached["internal"] = True  # must never happen
            return httpx.Response(200, text="SECRET")
        return httpx.Response(200, text="?")

    _inject_client(monkeypatch, handler)
    r = await _web_fetch({"url": "http://8.8.8.8/"})
    assert r.success is False
    assert "blocked" in r.error
    assert reached["internal"] is False  # the internal hop was never fetched


@pytest.mark.asyncio
async def test_web_fetch_follows_safe_redirect(monkeypatch):
    def handler(request):
        if request.url.host == "8.8.8.8":
            return httpx.Response(301, headers={"location": "http://1.1.1.1/final"})
        return httpx.Response(200, text="FINAL CONTENT")

    _inject_client(monkeypatch, handler)
    r = await _web_fetch({"url": "http://8.8.8.8/"})
    assert r.success is True, r.error
    assert r.output["status"] == 200
    assert "FINAL CONTENT" in r.output["content"]
