"""Validation for caller-supplied URLs the live browser is asked to load.

The one caller today is the login bootstrap (``POST /api/login``), whose URL
is operator-supplied but travels the same bearer-authenticated surface as
everything else — so the check is defense-in-depth, not an auth boundary.
The bar it enforces is the one every other navigation target already meets
(``plan_model`` requires http(s)) plus the network fact a scheme check
cannot see: this pod sits on a flat cluster network a hop away from the
browser's own control plane, the cluster DNS and any other workload, so a
target that resolves into private, loopback, link-local or otherwise
non-global address space is refused before Chrome ever sees it.

Resolution is deliberate and fail-closed: a name that does not resolve is
refused, because an unresolvable name is exactly what a typo and a
rebinding attack have in common. An operator who mistyped retries; the
browser never gets halfway into somewhere it should not go.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = ("http", "https")


class UrlRejected(ValueError):
    """The URL is not a navigable public http(s) target."""


def _resolve(host: str) -> list[str]:
    """Resolve ``host`` to addresses. Split out so tests can stub DNS."""
    return [info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)]


def _addresses_allowed(addresses: list[str]) -> str | None:
    """A rejection reason for the first non-global address, or None."""
    for raw in addresses:
        ip = ipaddress.ip_address(raw)
        # `is_global` is the umbrella (private, loopback, link-local — the
        # metadata address 169.254.169.254 is link-local — reserved and
        # shared space such as 100.64.0.0/10 are all non-global); the
        # explicit flags keep the intent readable and guard against a
        # Python version classifying any of them as global.
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return f"resolves to a non-public address ({ip})"
    return None


def validate_public_http_url(url: str) -> str:
    """Return ``url`` if the live browser may be navigated to it.

    Raises :class:`UrlRejected` otherwise. Blocking (DNS) — call it from a
    thread in async contexts.
    """
    candidate = (url or "").strip()
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.hostname:
        raise UrlRejected(f"not an http(s) URL with a host: {candidate!r}")
    host = parts.hostname
    try:
        ipaddress.ip_address(host)
        addresses = [host]  # a literal IP needs no lookup
    except ValueError:
        try:
            addresses = _resolve(host)
        except OSError as exc:
            raise UrlRejected(f"host does not resolve: {host} ({exc})") from exc
    if not addresses:
        raise UrlRejected(f"host does not resolve: {host}")
    reason = _addresses_allowed(addresses)
    if reason:
        raise UrlRejected(f"{host} {reason}")
    return candidate
