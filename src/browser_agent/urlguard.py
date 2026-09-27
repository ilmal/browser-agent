"""Validation for caller-supplied URLs the live browser is asked to load.

Two callers, one bar. The login bootstrap (``POST /api/login``) has always
held an operator-supplied URL to the full check below. Every *other*
navigation target — the freeform runner's start page (``tasks._start_url_for``,
which reads it out of the payload, the previous attempt or the prose) and a
plan's ``navigate`` steps — reached Chrome unfiltered until 2026-09-27, so a
task naming ``http://169.254.169.254/…`` (the metadata service) or a
cluster/loopback address was queued and the browser went there, and
``agent.task`` then read the page back into the thread. The proxy is no
backstop: its ``bypass`` list exempts loopback by design.

The bar a target meets is the scheme http(s) — ``plan_model`` already requires
it — plus the network fact a scheme check cannot see: this pod sits on a flat
cluster network a hop away from the browser's own control plane, the cluster
DNS and any other workload, so a target that resolves into private, loopback,
link-local or otherwise non-global address space is refused.

:func:`validate_public_http_url` is the complete check and resolves DNS, so it
is blocking and fail-closed (an unresolvable name is refused — what a typo and
a rebinding attack have in common). :func:`public_url_reason` is its
non-blocking half: it catches a literal private/loopback IP and every
non-http(s) scheme, but not a *name* that resolves into private space, because
it cannot resolve without blocking the caller. The navigation paths that run
in async context use the full check in a thread; the ones that cannot (a sync
start-URL helper, a pure plan validator) use the structural half, which still
refuses the literal-IP metadata and loopback targets and every ``file:``/
``data:`` scheme.
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


def _address_not_allowed(address: str) -> str | None:
    """A rejection reason if this single address is not global, else None."""
    ip = ipaddress.ip_address(address)
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


def _addresses_allowed(addresses: list[str]) -> str | None:
    """A rejection reason for the first non-global address, or None."""
    for raw in addresses:
        reason = _address_not_allowed(raw)
        if reason:
            return reason
    return None


def public_url_reason(url: str) -> str | None:
    """Why ``url`` may not be navigated to, or None — without touching DNS.

    The non-blocking half of :func:`validate_public_http_url`, for callers that
    must stay synchronous. It refuses a non-http(s) scheme, a URL with no host,
    and a *literal* non-global address (the metadata service by IP, a loopback
    literal, ``[::1]``); a hostname that would resolve into private space is
    not caught here and is left to the full check where the caller can afford
    one.
    """
    candidate = (url or "").strip()
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.hostname:
        return f"not an http(s) URL with a host: {candidate!r}"
    try:
        literal = ipaddress.ip_address(parts.hostname)
    except ValueError:
        return None
    return _address_not_allowed(str(literal))


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
