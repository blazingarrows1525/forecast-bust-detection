"""Fetch a public web page without letting a URL reach anything else.

The pages this module opens come from a search result, which means from the
internet, which means from anyone. Every URL is checked at every hop:

* scheme http or https only; no credentials in the URL; ports 80 and 443 only
* the hostname is resolved HERE, every address it resolves to must be public
  (no loopback, private, link-local -- which covers 169.254.169.254, the cloud
  metadata endpoint -- carrier-grade NAT, multicast, reserved or unspecified),
  and the connection is then made to that checked address. Connecting by name
  after checking by address would let a hostile DNS server answer twice
  ("DNS rebinding"); pinning the address closes that.
* at most three redirects, each one re-validated from the top
* connect/read timeouts, a byte cap enforced while reading, and a content-type
  allowlist (HTML and plain text); anything else is refused, not parsed

Nothing here follows instructions found in a page. That is the extractor's
and the guardrails' business; this module only decides whether bytes may be
read at all.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import urllib.parse
from dataclasses import dataclass

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (80, 443)
ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
MAX_REDIRECTS = 3
USER_AGENT = "ForecastBustIntelligence-ResearchAssistant/1.0 (+read-only; respects robots meta)"

# Hostnames that are never fetched whatever they resolve to.
_DENY_HOSTS = {"localhost", "metadata.google.internal", "metadata", "instance-data"}


class UnsafeURL(ValueError):
    """The URL may not be fetched. The message says why, safely."""


@dataclass
class Fetched:
    url: str              # final URL after redirects
    status: int
    content_type: str
    text: str
    truncated: bool


def canonical(url: str) -> str:
    """A stable form for de-duplication: lower-case host, no fragment, no
    default port, no tracking parameters. Not used for fetching."""
    p = urllib.parse.urlsplit(url.strip())
    host = (p.hostname or "").lower().rstrip(".")
    port = f":{p.port}" if p.port and p.port not in (80, 443) else ""
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith(("utm_", "fbclid", "gclid", "mc_"))]
    path = p.path or "/"
    return urllib.parse.urlunsplit((p.scheme.lower(), host + port, path, urllib.parse.urlencode(q), ""))


def _public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
                or addr.is_reserved or addr.is_unspecified
                or (isinstance(addr, ipaddress.IPv4Address) and addr in ipaddress.ip_network("100.64.0.0/10")))


def validate(url: str, resolver=socket.getaddrinfo) -> tuple[urllib.parse.SplitResult, str]:
    """Return (parts, a checked public IP to connect to), or raise UnsafeURL."""
    if not isinstance(url, str) or len(url) > 2048:
        raise UnsafeURL("not a URL of acceptable length")
    p = urllib.parse.urlsplit(url.strip())
    if p.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURL(f"scheme {p.scheme or 'none'!r} is not allowed")
    if p.username or p.password:
        raise UnsafeURL("credentials in a URL are not allowed")
    host = (p.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURL("no host")
    try:
        port = p.port or (443 if p.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeURL("malformed port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed")
    if host in _DENY_HOSTS or host.endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("internal hostname")
    # A literal IP is checked as-is; a name is resolved and EVERY answer checked.
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
        addrs = [str(literal)]
    except ValueError:
        try:
            infos = resolver(host, port, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError) as exc:
            raise UnsafeURL("host does not resolve") from exc
        addrs = sorted({info[4][0] for info in infos})
    if not addrs:
        raise UnsafeURL("host does not resolve")
    bad = [a for a in addrs if not _public(a)]
    if bad:
        raise UnsafeURL("host resolves to a non-public address")
    return p, addrs[0]


class _PinnedHTTPS(http.client.HTTPSConnection):
    """HTTPS to a checked IP, with SNI and certificate checks on the hostname."""

    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


def fetch(url: str, *, max_bytes: int = 1_500_000, timeout: float = 8.0,
          resolver=socket.getaddrinfo) -> Fetched:
    """GET a public page under every rule above. Raises UnsafeURL or OSError."""
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        p, ip = validate(current, resolver)
        host = p.hostname
        port = p.port or (443 if p.scheme.lower() == "https" else 80)
        conn_cls = _PinnedHTTPS if p.scheme.lower() == "https" else _PinnedHTTP
        conn = conn_cls(host, ip, port, timeout)
        path = urllib.parse.urlunsplit(("", "", p.path or "/", p.query, ""))
        try:
            conn.request("GET", path, headers={"Host": host, "User-Agent": USER_AGENT,
                                               "Accept": "text/html,text/plain;q=0.8",
                                               "Accept-Encoding": "identity"})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location")
                if not loc:
                    raise UnsafeURL("redirect without a location")
                current = urllib.parse.urljoin(current, loc)
                continue
            ctype = (resp.getheader("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_TYPES:
                raise UnsafeURL(f"content type {ctype or 'unknown'!r} is not fetched")
            length = resp.getheader("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes * 4:
                raise UnsafeURL("page is far larger than the reading budget")
            body = resp.read(max_bytes + 1)
            truncated = len(body) > max_bytes
            charset = "utf-8"
            for part in (resp.getheader("Content-Type") or "").split(";")[1:]:
                if part.strip().lower().startswith("charset="):
                    charset = part.split("=", 1)[1].strip() or "utf-8"
            try:
                text = body[:max_bytes].decode(charset, "replace")
            except LookupError:
                text = body[:max_bytes].decode("utf-8", "replace")
            return Fetched(url=current, status=resp.status, content_type=ctype, text=text,
                           truncated=truncated)
        finally:
            conn.close()
    raise UnsafeURL("too many redirects")
