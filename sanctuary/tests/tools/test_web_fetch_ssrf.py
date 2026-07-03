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

import pytest

from sanctuary.tools.builtin import _ip_is_blocked, _validate_fetch_url


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
