import asyncio
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from urllib.parse import urlparse


_DNS_TIMEOUT_SECONDS = 5.0

# Numeric spellings that are not accepted by ipaddress, such as 0177.0.0.1,
# 127.1, 2130706433, and 0x7f000001.
_DOTTED_NUMERIC = re.compile(
    r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){1,3}$",
    re.IGNORECASE,
)
_INTEGER_OR_HEX = re.compile(r"^(?:0x[0-9a-f]+|\d+)$", re.IGNORECASE)
_SIXTO4_NETWORK = ipaddress.IPv6Network("2002::/16")

_AddressInfo = Tuple


@dataclass(frozen=True)
class PinnedConnection:
    """A connection target whose IP was already checked."""

    ip: str
    host_header: str
    sni_hostname: Optional[str]


def validate_public_url(url: str) -> bool:
    try:
        resolve_pinned_connection(url)
    except ValueError:
        return False
    return True


def resolve_pinned_connection(url: str) -> PinnedConnection:
    """Resolve a URL and return the IP the HTTP client must connect to."""
    parsed, port, literal_ip = _prepare_target(url)

    if literal_ip is None:
        try:
            addresses = socket.getaddrinfo(
                _dns_hostname(parsed.hostname),
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror:
            raise ValueError("Invalid or unsafe URL.")
        ip = _first_safe_ip(addresses)
    else:
        ip = str(literal_ip)

    return _pinned_connection(parsed.hostname, parsed.scheme, port, ip)


async def resolve_pinned_connection_async(url: str) -> PinnedConnection:
    """Async version of resolve_pinned_connection for the downloader."""
    parsed, port, literal_ip = _prepare_target(url)

    if literal_ip is None:
        try:
            addresses = await _async_getaddrinfo(
                _dns_hostname(parsed.hostname),
                port,
            )
        except (socket.gaierror, asyncio.TimeoutError, TimeoutError):
            raise ValueError("Invalid or unsafe URL.")
        ip = _first_safe_ip(addresses)
    else:
        ip = str(literal_ip)

    return _pinned_connection(parsed.hostname, parsed.scheme, port, ip)


async def _async_getaddrinfo(hostname: str, port: int):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        ),
        timeout=_DNS_TIMEOUT_SECONDS,
    )


def _prepare_target(url: str):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid or unsafe URL.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid or unsafe URL.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    normalized = hostname[:-1] if hostname.endswith(".") else hostname
    literal_ip = _literal_ip(normalized)

    if literal_ip is not None:
        if _is_unsafe_ip(literal_ip):
            raise ValueError("Invalid or unsafe URL.")
        return parsed, port, literal_ip

    if _is_disguised_ip(normalized):
        raise ValueError("Invalid or unsafe URL.")

    dns_name = _dns_hostname(hostname)
    if dns_name == "localhost" or dns_name.endswith(".local"):
        raise ValueError("Invalid or unsafe URL.")

    return parsed, port, None


def _literal_ip(hostname: str) -> Optional[ipaddress._BaseAddress]:
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return None

    if hostname.lower() != str(ip).lower():
        raise ValueError("Invalid or unsafe URL.")

    return ip


def _is_disguised_ip(hostname: str) -> bool:
    if ":" in hostname:
        return True
    if _DOTTED_NUMERIC.match(hostname) or _INTEGER_OR_HEX.match(hostname):
        return True
    return False


def _dns_hostname(hostname: str) -> str:
    if hostname.endswith("."):
        hostname = hostname[:-1]

    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("Invalid or unsafe URL.")


def _first_safe_ip(addresses: Sequence[_AddressInfo]) -> str:
    chosen = None

    for address in addresses:
        try:
            ip_text = address[4][0].split("%", 1)[0]
            ip = ipaddress.ip_address(ip_text)
        except (IndexError, ValueError):
            raise ValueError("Invalid or unsafe URL.")

        if _is_unsafe_ip(ip):
            raise ValueError("Invalid or unsafe URL.")

        if chosen is None:
            chosen = ip

    if chosen is None:
        raise ValueError("Invalid or unsafe URL.")

    return str(chosen)


def _sixto4_ipv4(
    ip: ipaddress.IPv6Address,
) -> Optional[ipaddress.IPv4Address]:
    if ip not in _SIXTO4_NETWORK:
        return None
    return ipaddress.IPv4Address(ip.packed[2:6])


def _is_unsafe_ip(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _sixto4_ipv4(ip)
        if embedded is not None and _is_unsafe_ip(embedded):
            return True
        if ip.ipv4_mapped is not None:
            return _is_unsafe_ip(ip.ipv4_mapped)

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return True

    return False


def _pinned_connection(
    original_host: str,
    scheme: str,
    port: int,
    ip: str,
) -> PinnedConnection:
    try:
        literal = ipaddress.ip_address(original_host)
    except ValueError:
        host = _dns_hostname(original_host)
        sni_hostname = host if scheme == "https" else None
    else:
        if isinstance(literal, ipaddress.IPv6Address):
            host = "[{0}]".format(literal)
        else:
            host = str(literal)
        sni_hostname = None

    default_port = 443 if scheme == "https" else 80
    if port == default_port:
        host_header = host
    else:
        host_header = "{0}:{1}".format(host, port)

    return PinnedConnection(
        ip=ip,
        host_header=host_header,
        sni_hostname=sni_hostname,
    )
