"""LangChain tools for interacting with the DomiNode proxy service.

Each tool subclasses ``BaseTool`` from ``langchain_core.tools`` and wraps
the DomiNode Python SDK to expose proxy, wallet, usage, and configuration
operations to LangChain agents.

Security:
    - The ``DominusNodeProxiedFetchTool`` validates URLs to prevent SSRF
      attacks.  Private IP ranges, localhost, and non-HTTP(S) schemes are
      blocked.
    - Response bodies are truncated to 4 000 characters to avoid context
      window overflow.
    - API keys and proxy credentials are never exposed in tool outputs.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any, Optional, Type
from urllib.parse import quote, urlparse

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from dominusnode import AsyncDominusNodeClient, DominusNodeClient
from dominusnode.types import ProxyUrlOptions

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_RESPONSE_CHARS = 4_000
"""Maximum characters returned from a proxied fetch to prevent LLM context overflow."""

_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
"""HTTP methods permitted through the proxied fetch tool.  Mutating methods
(POST, PUT, DELETE, PATCH) are intentionally excluded to prevent the LLM
from performing destructive actions through the proxy."""

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Pre-compiled for hostname validation
_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")

# OFAC sanctioned countries
_SANCTIONED_COUNTRIES = frozenset({"CU", "IR", "KP", "RU", "SY"})

# Credential scrubbing pattern
_CREDENTIAL_RE = re.compile(r"dn_(live|test)_[a-zA-Z0-9]+|eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")

# Maximum response body size (10 MB)
_MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024


def _sanitize_error(message: str) -> str:
    """Remove DomiNode API key patterns from error messages."""
    return _CREDENTIAL_RE.sub("***", message)


# ──────────────────────────────────────────────────────────────────────
# Prototype pollution prevention
# ──────────────────────────────────────────────────────────────────────

_DANGEROUS_KEYS = frozenset({"__proto__", "constructor", "prototype"})


def _strip_dangerous_keys(obj: Any, depth: int = 0) -> None:
    """Recursively remove prototype pollution keys from parsed JSON."""
    if depth > 50 or obj is None:
        return
    if isinstance(obj, list):
        for item in obj:
            _strip_dangerous_keys(item, depth + 1)
    elif isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in _DANGEROUS_KEYS:
                del obj[key]
            else:
                _strip_dangerous_keys(obj[key], depth + 1)


# ──────────────────────────────────────────────────────────────────────
# SSRF validation
# ──────────────────────────────────────────────────────────────────────

# Private / reserved IPv4 networks
_BLOCKED_IPV4_NETWORKS = [
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("100.64.0.0/10"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.0.0.0/24"),
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("198.18.0.0/15"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv4Network("240.0.0.0/4"),
    ipaddress.IPv4Network("255.255.255.255/32"),
]

# Private / reserved IPv6 networks
_BLOCKED_IPV6_NETWORKS = [
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("::/128"),
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("100::/64"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("fc00::/7"),  # includes fd00::/8
    ipaddress.IPv6Network("ff00::/8"),   # multicast
    ipaddress.IPv6Network("2001::/32"),  # Teredo tunneling
    ipaddress.IPv6Network("2002::/16"),  # 6to4 tunneling
]


def _is_private_ip(hostname: str) -> bool:
    """Check whether *hostname* resolves to a private/reserved IP address.

    Handles raw IPv4, raw IPv6 (with bracket stripping and zone-ID removal),
    and hex/octal/decimal encoded IP forms.
    """
    # Strip IPv6 brackets
    clean = hostname.strip("[]")
    # Strip IPv6 zone ID (e.g. %25eth0)
    if "%" in clean:
        clean = clean.split("%")[0]

    try:
        addr = ipaddress.ip_address(clean)
    except ValueError:
        # Not a literal IP -- could still be a DNS name that resolves to
        # a private IP but we cannot do DNS resolution safely here.
        # Block well-known aliases.
        lower = hostname.lower()
        if lower in ("localhost", "localhost.localdomain"):
            return True
        if lower.endswith(".localhost"):
            return True
        # Block hex-encoded IPv4 (e.g. 0x7f000001)
        if lower.startswith("0x"):
            try:
                num = int(lower, 16)
                if 0 <= num <= 0xFFFFFFFF:
                    addr = ipaddress.IPv4Address(num)
                    return any(addr in net for net in _BLOCKED_IPV4_NETWORKS)
            except (ValueError, ipaddress.AddressValueError):
                pass
        # Block decimal-encoded IPv4 (e.g. 2130706433)
        if lower.isdigit():
            try:
                num = int(lower)
                if 0 <= num <= 0xFFFFFFFF:
                    addr = ipaddress.IPv4Address(num)
                    return any(addr in net for net in _BLOCKED_IPV4_NETWORKS)
            except (ValueError, ipaddress.AddressValueError):
                pass
        return False

    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in net for net in _BLOCKED_IPV4_NETWORKS)
    if isinstance(addr, ipaddress.IPv6Address):
        if any(addr in net for net in _BLOCKED_IPV6_NETWORKS):
            return True
        # IPv4-compatible IPv6 (::x.x.x.x or hex form ::7f00:1)
        packed = addr.packed
        if all(b == 0 for b in packed[:12]):
            embedded = ipaddress.IPv4Address(packed[12:16])
            if any(embedded in net for net in _BLOCKED_IPV4_NETWORKS):
                return True
        # IPv4-mapped check via ipaddress
        mapped = addr.ipv4_mapped
        if mapped is not None:
            if any(mapped in net for net in _BLOCKED_IPV4_NETWORKS):
                return True
        return False
    return False


def _validate_url(url: str) -> str:
    """Validate and sanitize *url*, returning the cleaned URL.

    Raises ``ValueError`` on any SSRF-risky or malformed input.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    # Limit URL length to prevent abuse
    if len(url) > 2048:
        raise ValueError("URL exceeds maximum length of 2048 characters")

    parsed = urlparse(url)

    # Scheme check
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Only {', '.join(sorted(_ALLOWED_SCHEMES))} are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname")

    # Block credentials in URL
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed")

    # Block private/reserved IPs
    if _is_private_ip(hostname):
        raise ValueError(
            "URLs targeting private, reserved, or loopback addresses are blocked"
        )

    # DNS rebinding protection: resolve hostname and check all IPs
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        # It is a hostname, not a raw IP -- try resolving
        try:
            infos = socket.getaddrinfo(
                hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
            for _family, _type, _proto, _canonname, sockaddr in infos:
                addr_str = sockaddr[0]
                if "%" in addr_str:
                    addr_str = addr_str.split("%")[0]
                if _is_private_ip(addr_str):
                    raise ValueError(
                        f"Hostname {hostname!r} resolves to private IP {addr_str}"
                    )
        except socket.gaierror:
            raise ValueError(f"Could not resolve hostname: {hostname!r}")

    return url


# ──────────────────────────────────────────────────────────────────────
# Pydantic input schemas
# ──────────────────────────────────────────────────────────────────────


class ProxiedFetchInput(BaseModel):
    """Input schema for the DomiNode proxied fetch tool."""

    url: str = Field(description="The URL to fetch through the DomiNode proxy.")
    method: str = Field(
        default="GET",
        description="HTTP method (GET, HEAD, or OPTIONS). Default: GET.",
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code for geo-targeting (e.g. 'US', 'GB', 'DE').",
    )
    proxy_type: str = Field(
        default="dc",
        description="Proxy type: 'dc' for datacenter ($3/GB) or 'residential' ($5/GB). Default: dc.",
    )


class EmptyInput(BaseModel):
    """Empty input schema for tools that require no parameters."""

    pass


# ──────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────


class DominusNodeProxiedFetchTool(BaseTool):
    """Make HTTP requests through the DomiNode rotating proxy network.

    Fetches the given URL through a DomiNode proxy IP, optionally targeting a
    specific country.  Returns the HTTP status code and the first 4 000
    characters of the response body.

    Supports both datacenter (cheaper, $3/GB) and residential (premium, $5/GB)
    proxy types.  Only read-only HTTP methods (GET, HEAD, OPTIONS) are allowed.
    """

    name: str = "dominusnode_proxied_fetch"
    description: str = (
        "Fetch a URL through the DomiNode rotating proxy network. "
        "Useful for accessing geo-restricted content or browsing anonymously. "
        "Supports country targeting and both datacenter and residential proxies. "
        "Input: url (required), method (GET/HEAD/OPTIONS), country (e.g. US), "
        "proxy_type (dc or residential)."
    )
    args_schema: Type[BaseModel] = ProxiedFetchInput

    # Client instances -- set after construction via toolkit
    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        url: str,
        method: str = "GET",
        country: Optional[str] = None,
        proxy_type: str = "dc",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool synchronously."""
        try:
            validated_url = _validate_url(url)
        except ValueError as exc:
            return f"Error: {exc}"

        method_upper = method.upper()
        if method_upper not in _ALLOWED_METHODS:
            return (
                f"Error: HTTP method '{method_upper}' is not allowed. "
                f"Permitted methods: {', '.join(sorted(_ALLOWED_METHODS))}."
            )

        # OFAC sanctioned country check
        if country and country.upper() in _SANCTIONED_COUNTRIES:
            return f"Error: Country '{country.upper()}' is blocked (OFAC sanctioned)"

        if proxy_type not in ("dc", "residential"):
            return "Error: proxy_type must be 'dc' or 'residential'."

        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."

        try:
            options = ProxyUrlOptions(country=country, proxy_type=proxy_type)
            proxy_url = self.sync_client.proxy.build_url(options)

            with httpx.Client(
                proxy=proxy_url,
                timeout=30.0,
                follow_redirects=False,
            ) as http_client:
                response = http_client.request(method_upper, validated_url)

            # Response body size cap (H-9)
            content_length = response.headers.get("content-length", "0")
            try:
                if int(content_length) > _MAX_RESPONSE_BODY_BYTES:
                    return "Error: Response too large (exceeds 10MB limit)"
            except ValueError:
                pass
            if len(response.content) > _MAX_RESPONSE_BODY_BYTES:
                return "Error: Response too large (exceeds 10MB limit)"

            body = response.text[:MAX_RESPONSE_CHARS]
            truncated = " [truncated]" if len(response.text) > MAX_RESPONSE_CHARS else ""

            return (
                f"Status: {response.status_code}\n"
                f"Content-Type: {response.headers.get('content-type', 'unknown')}\n"
                f"Body:\n{body}{truncated}"
            )

        except httpx.HTTPError as exc:
            return f"Error: HTTP request failed: {_sanitize_error(str(exc))}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        url: str,
        method: str = "GET",
        country: Optional[str] = None,
        proxy_type: str = "dc",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool asynchronously."""
        try:
            validated_url = _validate_url(url)
        except ValueError as exc:
            return f"Error: {exc}"

        method_upper = method.upper()
        if method_upper not in _ALLOWED_METHODS:
            return (
                f"Error: HTTP method '{method_upper}' is not allowed. "
                f"Permitted methods: {', '.join(sorted(_ALLOWED_METHODS))}."
            )

        # OFAC sanctioned country check
        if country and country.upper() in _SANCTIONED_COUNTRIES:
            return f"Error: Country '{country.upper()}' is blocked (OFAC sanctioned)"

        if proxy_type not in ("dc", "residential"):
            return "Error: proxy_type must be 'dc' or 'residential'."

        if self.async_client is None and self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."

        try:
            # build_url is synchronous even on the async client
            client_for_url = self.async_client or self.sync_client
            options = ProxyUrlOptions(country=country, proxy_type=proxy_type)
            proxy_url = client_for_url.proxy.build_url(options)

            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=30.0,
                follow_redirects=False,
            ) as http_client:
                response = await http_client.request(method_upper, validated_url)

            # Response body size cap (H-9)
            content_length = response.headers.get("content-length", "0")
            try:
                if int(content_length) > _MAX_RESPONSE_BODY_BYTES:
                    return "Error: Response too large (exceeds 10MB limit)"
            except ValueError:
                pass
            if len(response.content) > _MAX_RESPONSE_BODY_BYTES:
                return "Error: Response too large (exceeds 10MB limit)"

            body = response.text[:MAX_RESPONSE_CHARS]
            truncated = " [truncated]" if len(response.text) > MAX_RESPONSE_CHARS else ""

            return (
                f"Status: {response.status_code}\n"
                f"Content-Type: {response.headers.get('content-type', 'unknown')}\n"
                f"Body:\n{body}{truncated}"
            )

        except httpx.HTTPError as exc:
            return f"Error: HTTP request failed: {_sanitize_error(str(exc))}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeBalanceTool(BaseTool):
    """Check the current DomiNode wallet balance.

    Returns the wallet balance in both USD and cents, useful for monitoring
    spend before making proxied requests.
    """

    name: str = "dominusnode_balance"
    description: str = (
        "Check your DomiNode wallet balance. Returns the current balance "
        "in dollars and cents. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."
        try:
            wallet = self.sync_client.wallet.get_balance()
            return (
                f"Balance: ${wallet.balance_usd:.2f} ({wallet.balance_cents} cents)\n"
                f"Currency: {wallet.currency}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_client is not None:
            try:
                wallet = await self.async_client.wallet.get_balance()
                return (
                    f"Balance: ${wallet.balance_usd:.2f} ({wallet.balance_cents} cents)\n"
                    f"Currency: {wallet.currency}"
                )
            except Exception as exc:
                return f"Error: {_sanitize_error(str(exc))}"

        if self.sync_client is not None:
            return self._run()

        return "Error: No DomiNode client configured. Initialize the toolkit first."


class DominusNodeUsageTool(BaseTool):
    """Check DomiNode proxy usage statistics.

    Returns a summary of bandwidth usage including total bytes transferred,
    cost, and request count.
    """

    name: str = "dominusnode_usage"
    description: str = (
        "Check your DomiNode proxy usage statistics. Returns total bandwidth "
        "used (in GB), total cost, and request count. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."
        try:
            usage_resp = self.sync_client.usage.get()
            s = usage_resp.summary
            return (
                f"Usage Summary:\n"
                f"  Total Data: {s.total_gb:.4f} GB ({s.total_bytes:,} bytes)\n"
                f"  Total Cost: ${s.total_cost_usd:.2f} ({s.total_cost_cents} cents)\n"
                f"  Requests: {s.request_count:,}\n"
                f"Period: {usage_resp.period.since} to {usage_resp.period.until}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_client is not None:
            try:
                usage_resp = await self.async_client.usage.get()
                s = usage_resp.summary
                return (
                    f"Usage Summary:\n"
                    f"  Total Data: {s.total_gb:.4f} GB ({s.total_bytes:,} bytes)\n"
                    f"  Total Cost: ${s.total_cost_usd:.2f} ({s.total_cost_cents} cents)\n"
                    f"  Requests: {s.request_count:,}\n"
                    f"Period: {usage_resp.period.since} to {usage_resp.period.until}"
                )
            except Exception as exc:
                return f"Error: {_sanitize_error(str(exc))}"

        if self.sync_client is not None:
            return self._run()

        return "Error: No DomiNode client configured. Initialize the toolkit first."


class TopupPaypalInput(BaseModel):
    """Input schema for the DomiNode PayPal top-up tool."""

    amount_cents: int = Field(
        description="Amount in cents to top up via PayPal (min 500 = $5, max 100000 = $1,000).",
    )


class DominusNodeTopupPaypalTool(BaseTool):
    """Top up your DomiNode wallet balance via PayPal.

    Creates a PayPal order and returns an approval URL to complete payment.
    Minimum $5 (500 cents), maximum $1,000 (100,000 cents).
    """

    name: str = "dominusnode_topup_paypal"
    description: str = (
        "Top up your DomiNode wallet balance via PayPal. "
        "Creates a PayPal order and returns an approval URL to complete payment. "
        "Input: amount_cents (integer, min 500 = $5, max 100000 = $1,000)."
    )
    args_schema: Type[BaseModel] = TopupPaypalInput

    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        amount_cents: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."

        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."

        try:
            result = self.sync_client.wallet.topup_paypal(amount_cents=amount_cents)
            return (
                f"PayPal Top-Up Order Created\n"
                f"Order ID: {result.order_id}\n"
                f"Amount: ${amount_cents / 100:.2f}\n"
                f"Approval URL: {result.approval_url}\n\n"
                f"Open the approval URL in a browser to complete payment."
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        amount_cents: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."

        if self.async_client is not None:
            try:
                result = await self.async_client.wallet.topup_paypal(amount_cents=amount_cents)
                return (
                    f"PayPal Top-Up Order Created\n"
                    f"Order ID: {result.order_id}\n"
                    f"Amount: ${amount_cents / 100:.2f}\n"
                    f"Approval URL: {result.approval_url}\n\n"
                    f"Open the approval URL in a browser to complete payment."
                )
            except Exception as exc:
                return f"Error: {_sanitize_error(str(exc))}"

        if self.sync_client is not None:
            return self._run(amount_cents=amount_cents)

        return "Error: No DomiNode client configured. Initialize the toolkit first."


class DominusNodeProxyConfigTool(BaseTool):
    """Get DomiNode proxy configuration and supported countries.

    Returns the proxy endpoints, supported countries for geo-targeting,
    and available geo-targeting features.
    """

    name: str = "dominusnode_proxy_config"
    description: str = (
        "Get the DomiNode proxy configuration including supported countries, "
        "proxy endpoints, and geo-targeting capabilities. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."
        try:
            config = self.sync_client.proxy.get_config()
            countries = ", ".join(config.supported_countries) if config.supported_countries else "none listed"
            gt = config.geo_targeting
            return (
                f"Proxy Configuration:\n"
                f"  HTTP Proxy: {config.http_proxy.host}:{config.http_proxy.port}\n"
                f"  SOCKS5 Proxy: {config.socks5_proxy.host}:{config.socks5_proxy.port}\n"
                f"  Supported Countries: {countries}\n"
                f"  Blocked Countries: {', '.join(config.blocked_countries) if config.blocked_countries else 'none'}\n"
                f"  Rotation Interval: {config.min_rotation_interval_minutes}-{config.max_rotation_interval_minutes} minutes\n"
                f"Geo-Targeting Features:\n"
                f"  State targeting: {'yes' if gt.state_support else 'no'}\n"
                f"  City targeting: {'yes' if gt.city_support else 'no'}\n"
                f"  ASN targeting: {'yes' if gt.asn_support else 'no'}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_client is not None:
            try:
                config = await self.async_client.proxy.get_config()
                countries = ", ".join(config.supported_countries) if config.supported_countries else "none listed"
                gt = config.geo_targeting
                return (
                    f"Proxy Configuration:\n"
                    f"  HTTP Proxy: {config.http_proxy.host}:{config.http_proxy.port}\n"
                    f"  SOCKS5 Proxy: {config.socks5_proxy.host}:{config.socks5_proxy.port}\n"
                    f"  Supported Countries: {countries}\n"
                    f"  Blocked Countries: {', '.join(config.blocked_countries) if config.blocked_countries else 'none'}\n"
                    f"  Rotation Interval: {config.min_rotation_interval_minutes}-{config.max_rotation_interval_minutes} minutes\n"
                    f"Geo-Targeting Features:\n"
                    f"  State targeting: {'yes' if gt.state_support else 'no'}\n"
                    f"  City targeting: {'yes' if gt.city_support else 'no'}\n"
                    f"  ASN targeting: {'yes' if gt.asn_support else 'no'}"
                )
            except Exception as exc:
                return f"Error: {_sanitize_error(str(exc))}"

        if self.sync_client is not None:
            return self._run()

        return "Error: No DomiNode client configured. Initialize the toolkit first."


class DominusNodeX402InfoTool(BaseTool):
    """Get x402 micropayment protocol information.

    Returns details about x402 HTTP 402 Payment Required protocol support
    including facilitators, pricing, supported currencies, and payment options
    for AI agent micropayments.
    """

    name: str = "dominusnode_x402_info"
    description: str = (
        "Get x402 micropayment protocol information including supported "
        "facilitators, pricing, and payment options. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    sync_client: Optional[Any] = None
    async_client: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_client is None:
            return "Error: No DomiNode client configured. Initialize the toolkit first."
        try:
            info = self.sync_client.x402.get_info()
            return (
                f"x402 Protocol Information:\n"
                f"  Supported: {info.supported}\n"
                f"  Enabled: {info.enabled}\n"
                f"  Protocol: {info.protocol}\n"
                f"  Version: {info.version}\n"
                f"  Currencies: {', '.join(info.currencies)}\n"
                f"  Wallet Type: {info.wallet_type}\n"
                f"  Agentic Wallets: {info.agentic_wallets}\n"
                f"  Pricing: {info.pricing.per_request_cents} cents/request, "
                f"{info.pricing.per_gb_cents} cents/GB ({info.pricing.currency})"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_client is not None:
            try:
                info = await self.async_client.x402.get_info()
                return (
                    f"x402 Protocol Information:\n"
                    f"  Supported: {info.supported}\n"
                    f"  Enabled: {info.enabled}\n"
                    f"  Protocol: {info.protocol}\n"
                    f"  Version: {info.version}\n"
                    f"  Currencies: {', '.join(info.currencies)}\n"
                    f"  Wallet Type: {info.wallet_type}\n"
                    f"  Agentic Wallets: {info.agentic_wallets}\n"
                    f"  Pricing: {info.pricing.per_request_cents} cents/request, "
                    f"{info.pricing.per_gb_cents} cents/GB ({info.pricing.currency})"
                )
            except Exception as exc:
                return f"Error: {_sanitize_error(str(exc))}"

        if self.sync_client is not None:
            return self._run()

        return "Error: No DomiNode client configured. Initialize the toolkit first."


# ──────────────────────────────────────────────────────────────────────
# Agentic Wallet validation helpers
# ──────────────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_DOMAIN_RE = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$"
)


def _validate_label(label: Any) -> Optional[str]:
    """Validate a label string; returns an error message or ``None``."""
    if not label or not isinstance(label, str):
        return "label is required and must be a non-empty string"
    if len(label) > 100:
        return "label must be 100 characters or fewer"
    if any(0 <= ord(c) <= 0x1F or ord(c) == 0x7F for c in label):
        return "label contains invalid control characters"
    return None


def _validate_wallet_id(wallet_id: Any) -> Optional[str]:
    """Validate a wallet ID; returns an error message or ``None``."""
    if not wallet_id or not isinstance(wallet_id, str):
        return "wallet_id is required and must be a string"
    if not _UUID_RE.match(wallet_id):
        return "wallet_id must be a valid UUID"
    return None


def _validate_domains(domains: Any) -> Optional[str]:
    """Validate an ``allowed_domains`` list; returns error message or ``None``."""
    if not isinstance(domains, list):
        return "allowed_domains must be a list of strings"
    if len(domains) > 100:
        return "allowed_domains must have at most 100 entries"
    for i, d in enumerate(domains):
        if not isinstance(d, str) or not d:
            return f"allowed_domains[{i}] must be a non-empty string"
        if len(d) > 253:
            return f"allowed_domains[{i}] exceeds 253 characters"
        if not _DOMAIN_RE.match(d):
            return f"allowed_domains[{i}] is not a valid domain"
    return None


def _api_request_sync(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> dict:
    """Make a synchronous authenticated REST API request.

    Raises ``RuntimeError`` on non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DominusNode-Agent": "mcp",
    }
    with httpx.Client(timeout=30.0, follow_redirects=False, max_redirects=0) as client:
        resp = client.request(method, url, headers=headers, json=body)
    if len(resp.content) > _MAX_RESPONSE_BODY_BYTES:
        raise RuntimeError("Response body exceeds 10 MB size limit")
    if resp.status_code >= 400:
        raise RuntimeError(f"API error {resp.status_code}: {_sanitize_error(resp.text[:200])}")
    data = resp.json()
    _strip_dangerous_keys(data)
    return data


async def _api_request_async(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> dict:
    """Make an asynchronous authenticated REST API request.

    Raises ``RuntimeError`` on non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DominusNode-Agent": "mcp",
    }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, max_redirects=0) as client:
        resp = await client.request(method, url, headers=headers, json=body)
    if len(resp.content) > _MAX_RESPONSE_BODY_BYTES:
        raise RuntimeError("Response body exceeds 10 MB size limit")
    if resp.status_code >= 400:
        raise RuntimeError(f"API error {resp.status_code}: {_sanitize_error(resp.text[:200])}")
    data = resp.json()
    _strip_dangerous_keys(data)
    return data


# ──────────────────────────────────────────────────────────────────────
# Agentic Wallet Pydantic input schemas
# ──────────────────────────────────────────────────────────────────────


class CreateAgenticWalletInput(BaseModel):
    """Input schema for creating an agentic sub-wallet."""

    label: str = Field(description="Human-readable label for the wallet (max 100 chars).")
    spending_limit_cents: int = Field(
        description="Per-transaction spending limit in cents (positive integer)."
    )
    daily_limit_cents: Optional[int] = Field(
        default=None,
        description="Optional daily spending cap in cents (1 - 1,000,000).",
    )
    allowed_domains: Optional[list] = Field(
        default=None,
        description="Optional list of allowed target domains for this wallet.",
    )


class FundAgenticWalletInput(BaseModel):
    """Input schema for funding an agentic sub-wallet."""

    wallet_id: str = Field(description="UUID of the agentic wallet to fund.")
    amount_cents: int = Field(description="Amount in cents to transfer (positive integer).")


class WalletIdInput(BaseModel):
    """Input schema for operations that require only a wallet ID."""

    wallet_id: str = Field(description="UUID of the agentic wallet.")


class AgenticTransactionsInput(BaseModel):
    """Input schema for listing agentic wallet transactions."""

    wallet_id: str = Field(description="UUID of the agentic wallet.")
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of transactions to return (1-100).",
    )


class UpdateWalletPolicyInput(BaseModel):
    """Input schema for updating an agentic wallet's policy."""

    wallet_id: str = Field(description="UUID of the agentic wallet.")
    daily_limit_cents: Optional[int] = Field(
        default=None,
        description="New daily spending cap in cents (1 - 1,000,000).",
    )
    allowed_domains: Optional[list] = Field(
        default=None,
        description="New list of allowed target domains.",
    )


# ──────────────────────────────────────────────────────────────────────
# Agentic Wallet Tools
# ──────────────────────────────────────────────────────────────────────


class DominusNodeCreateAgenticWalletTool(BaseTool):
    """Create a new agentic sub-wallet with a spending limit.

    Agentic wallets are server-side custodial sub-wallets that AI agents
    can use for pay-per-request or metered billing without exposing the
    main wallet credentials.
    """

    name: str = "dominusnode_create_agentic_wallet"
    description: str = (
        "Create a new agentic sub-wallet with a spending limit. "
        "Input: label (string), spending_limit_cents (int), "
        "optional daily_limit_cents (int), optional allowed_domains (list)."
    )
    args_schema: Type[BaseModel] = CreateAgenticWalletInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        label: str = "",
        spending_limit_cents: int = 0,
        daily_limit_cents: Optional[int] = None,
        allowed_domains: Optional[list] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_label(label)
        if err:
            return f"Error: {err}"
        if not isinstance(spending_limit_cents, int) or isinstance(spending_limit_cents, bool):
            return "Error: spending_limit_cents must be an integer"
        if spending_limit_cents < 1 or spending_limit_cents > 2_147_483_647:
            return "Error: spending_limit_cents must be between 1 and 2,147,483,647"
        if daily_limit_cents is not None:
            if not isinstance(daily_limit_cents, int) or isinstance(daily_limit_cents, bool):
                return "Error: daily_limit_cents must be an integer"
            if daily_limit_cents < 1 or daily_limit_cents > 1_000_000:
                return "Error: daily_limit_cents must be between 1 and 1,000,000"
        if allowed_domains is not None:
            err = _validate_domains(allowed_domains)
            if err:
                return f"Error: {err}"

        body: dict = {
            "label": label,
            "spendingLimitCents": spending_limit_cents,
        }
        if daily_limit_cents is not None:
            body["dailyLimitCents"] = daily_limit_cents
        if allowed_domains is not None:
            body["allowedDomains"] = allowed_domains

        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/agent-wallet", body)
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Created\n"
                f"  ID: {w.get('id', 'unknown')}\n"
                f"  Label: {w.get('label', label)}\n"
                f"  Balance: {w.get('balanceCents', 0)} cents\n"
                f"  Spending Limit: {w.get('spendingLimitCents', spending_limit_cents)} cents\n"
                f"  Status: {w.get('status', 'active')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        label: str = "",
        spending_limit_cents: int = 0,
        daily_limit_cents: Optional[int] = None,
        allowed_domains: Optional[list] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_label(label)
        if err:
            return f"Error: {err}"
        if not isinstance(spending_limit_cents, int) or isinstance(spending_limit_cents, bool):
            return "Error: spending_limit_cents must be an integer"
        if spending_limit_cents < 1 or spending_limit_cents > 2_147_483_647:
            return "Error: spending_limit_cents must be between 1 and 2,147,483,647"
        if daily_limit_cents is not None:
            if not isinstance(daily_limit_cents, int) or isinstance(daily_limit_cents, bool):
                return "Error: daily_limit_cents must be an integer"
            if daily_limit_cents < 1 or daily_limit_cents > 1_000_000:
                return "Error: daily_limit_cents must be between 1 and 1,000,000"
        if allowed_domains is not None:
            err = _validate_domains(allowed_domains)
            if err:
                return f"Error: {err}"

        body: dict = {
            "label": label,
            "spendingLimitCents": spending_limit_cents,
        }
        if daily_limit_cents is not None:
            body["dailyLimitCents"] = daily_limit_cents
        if allowed_domains is not None:
            body["allowedDomains"] = allowed_domains

        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/agent-wallet", body)
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Created\n"
                f"  ID: {w.get('id', 'unknown')}\n"
                f"  Label: {w.get('label', label)}\n"
                f"  Balance: {w.get('balanceCents', 0)} cents\n"
                f"  Spending Limit: {w.get('spendingLimitCents', spending_limit_cents)} cents\n"
                f"  Status: {w.get('status', 'active')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeFundAgenticWalletTool(BaseTool):
    """Fund an agentic sub-wallet by transferring from the main wallet."""

    name: str = "dominusnode_fund_agentic_wallet"
    description: str = (
        "Fund an agentic sub-wallet by transferring cents from your main wallet. "
        "Input: wallet_id (UUID), amount_cents (positive integer)."
    )
    args_schema: Type[BaseModel] = FundAgenticWalletInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        amount_cents: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool):
            return "Error: amount_cents must be an integer"
        if amount_cents < 1 or amount_cents > 2_147_483_647:
            return "Error: amount_cents must be between 1 and 2,147,483,647"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/fund",
                {"amountCents": amount_cents},
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Funded\n"
                f"  ID: {wallet_id}\n"
                f"  Amount Added: {amount_cents} cents (${amount_cents / 100:.2f})\n"
                f"  New Balance: {w.get('balanceCents', 'unknown')} cents"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        amount_cents: int = 0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool):
            return "Error: amount_cents must be an integer"
        if amount_cents < 1 or amount_cents > 2_147_483_647:
            return "Error: amount_cents must be between 1 and 2,147,483,647"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/fund",
                {"amountCents": amount_cents},
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Funded\n"
                f"  ID: {wallet_id}\n"
                f"  Amount Added: {amount_cents} cents (${amount_cents / 100:.2f})\n"
                f"  New Balance: {w.get('balanceCents', 'unknown')} cents"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeAgenticWalletBalanceTool(BaseTool):
    """Check the balance and details of an agentic sub-wallet."""

    name: str = "dominusnode_agentic_wallet_balance"
    description: str = (
        "Get the balance and details of a specific agentic sub-wallet. "
        "Input: wallet_id (UUID)."
    )
    args_schema: Type[BaseModel] = WalletIdInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "GET",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Details\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Label: {w.get('label', 'unknown')}\n"
                f"  Balance: {w.get('balanceCents', 0)} cents "
                f"(${w.get('balanceCents', 0) / 100:.2f})\n"
                f"  Spending Limit: {w.get('spendingLimitCents', 'N/A')} cents\n"
                f"  Status: {w.get('status', 'unknown')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "GET",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Details\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Label: {w.get('label', 'unknown')}\n"
                f"  Balance: {w.get('balanceCents', 0)} cents "
                f"(${w.get('balanceCents', 0) / 100:.2f})\n"
                f"  Spending Limit: {w.get('spendingLimitCents', 'N/A')} cents\n"
                f"  Status: {w.get('status', 'unknown')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeListAgenticWalletsTool(BaseTool):
    """List all agentic sub-wallets for the current user."""

    name: str = "dominusnode_list_agentic_wallets"
    description: str = (
        "List all agentic sub-wallets for your account. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "GET", "/api/agent-wallet",
            )
            wallets = data.get("wallets", [])
            if not wallets:
                return "No agentic wallets found."

            lines = [f"Agentic Wallets ({len(wallets)}):"]
            for w in wallets:
                lines.append(
                    f"  - {w.get('id', '?')}: {w.get('label', 'unlabeled')} "
                    f"| {w.get('balanceCents', 0)} cents "
                    f"| limit {w.get('spendingLimitCents', 'N/A')} cents "
                    f"| {w.get('status', '?')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "GET", "/api/agent-wallet",
            )
            wallets = data.get("wallets", [])
            if not wallets:
                return "No agentic wallets found."

            lines = [f"Agentic Wallets ({len(wallets)}):"]
            for w in wallets:
                lines.append(
                    f"  - {w.get('id', '?')}: {w.get('label', 'unlabeled')} "
                    f"| {w.get('balanceCents', 0)} cents "
                    f"| limit {w.get('spendingLimitCents', 'N/A')} cents "
                    f"| {w.get('status', '?')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeAgenticTransactionsTool(BaseTool):
    """List transactions for a specific agentic sub-wallet."""

    name: str = "dominusnode_agentic_transactions"
    description: str = (
        "List recent transactions for an agentic sub-wallet. "
        "Input: wallet_id (UUID), optional limit (1-100)."
    )
    args_schema: Type[BaseModel] = AgenticTransactionsInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        limit: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                return "Error: limit must be an integer"
            if limit < 1 or limit > 100:
                return "Error: limit must be between 1 and 100"

        path = f"/api/agent-wallet/{quote(wallet_id, safe='')}/transactions"
        if limit is not None:
            path += f"?limit={limit}"

        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", path)
            txns = data.get("transactions", [])
            if not txns:
                return f"No transactions found for wallet {wallet_id}."

            lines = [f"Transactions for {wallet_id} ({len(txns)}):"]
            for tx in txns:
                lines.append(
                    f"  - {tx.get('type', '?')}: {tx.get('amountCents', 0)} cents "
                    f"| {tx.get('description', '')} "
                    f"| {tx.get('createdAt', '?')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        limit: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                return "Error: limit must be an integer"
            if limit < 1 or limit > 100:
                return "Error: limit must be between 1 and 100"

        path = f"/api/agent-wallet/{quote(wallet_id, safe='')}/transactions"
        if limit is not None:
            path += f"?limit={limit}"

        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", path)
            txns = data.get("transactions", [])
            if not txns:
                return f"No transactions found for wallet {wallet_id}."

            lines = [f"Transactions for {wallet_id} ({len(txns)}):"]
            for tx in txns:
                lines.append(
                    f"  - {tx.get('type', '?')}: {tx.get('amountCents', 0)} cents "
                    f"| {tx.get('description', '')} "
                    f"| {tx.get('createdAt', '?')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeFreezeAgenticWalletTool(BaseTool):
    """Freeze an agentic sub-wallet to prevent spending."""

    name: str = "dominusnode_freeze_agentic_wallet"
    description: str = (
        "Freeze an agentic sub-wallet to temporarily stop all spending. "
        "Input: wallet_id (UUID)."
    )
    args_schema: Type[BaseModel] = WalletIdInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/freeze",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Frozen\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Status: {w.get('status', 'frozen')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/freeze",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Frozen\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Status: {w.get('status', 'frozen')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUnfreezeAgenticWalletTool(BaseTool):
    """Unfreeze a previously frozen agentic sub-wallet."""

    name: str = "dominusnode_unfreeze_agentic_wallet"
    description: str = (
        "Unfreeze a previously frozen agentic sub-wallet to resume spending. "
        "Input: wallet_id (UUID)."
    )
    args_schema: Type[BaseModel] = WalletIdInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/unfreeze",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Unfrozen\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Status: {w.get('status', 'active')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/unfreeze",
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Unfrozen\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Status: {w.get('status', 'active')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeDeleteAgenticWalletTool(BaseTool):
    """Delete an agentic sub-wallet and refund remaining balance."""

    name: str = "dominusnode_delete_agentic_wallet"
    description: str = (
        "Delete an agentic sub-wallet. Any remaining balance is returned "
        "to the main wallet. Input: wallet_id (UUID)."
    )
    args_schema: Type[BaseModel] = WalletIdInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "DELETE",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
            )
            return (
                f"Agentic Wallet Deleted\n"
                f"  ID: {wallet_id}\n"
                f"  Refunded: {data.get('refundedCents', 0)} cents"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "DELETE",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
            )
            return (
                f"Agentic Wallet Deleted\n"
                f"  ID: {wallet_id}\n"
                f"  Refunded: {data.get('refundedCents', 0)} cents"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUpdateWalletPolicyTool(BaseTool):
    """Update the policy (daily limit, allowed domains) of an agentic sub-wallet."""

    name: str = "dominusnode_update_wallet_policy"
    description: str = (
        "Update the policy of an agentic sub-wallet. "
        "Input: wallet_id (UUID), optional daily_limit_cents (int 1-1,000,000), "
        "optional allowed_domains (list of domain strings)."
    )
    args_schema: Type[BaseModel] = UpdateWalletPolicyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        daily_limit_cents: Optional[int] = None,
        allowed_domains: Optional[list] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        body: dict = {}
        if daily_limit_cents is not None:
            if not isinstance(daily_limit_cents, int) or isinstance(daily_limit_cents, bool):
                return "Error: daily_limit_cents must be an integer"
            if daily_limit_cents < 1 or daily_limit_cents > 1_000_000:
                return "Error: daily_limit_cents must be between 1 and 1,000,000"
            body["dailyLimitCents"] = daily_limit_cents
        if allowed_domains is not None:
            err = _validate_domains(allowed_domains)
            if err:
                return f"Error: {err}"
            body["allowedDomains"] = allowed_domains

        if not body:
            return "Error: At least one of daily_limit_cents or allowed_domains must be provided."

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "PATCH",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/policy",
                body,
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Policy Updated\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Daily Limit: {w.get('dailyLimitCents', 'N/A')} cents\n"
                f"  Status: {w.get('status', 'unknown')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        wallet_id: str = "",
        daily_limit_cents: Optional[int] = None,
        allowed_domains: Optional[list] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No DomiNode API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        body: dict = {}
        if daily_limit_cents is not None:
            if not isinstance(daily_limit_cents, int) or isinstance(daily_limit_cents, bool):
                return "Error: daily_limit_cents must be an integer"
            if daily_limit_cents < 1 or daily_limit_cents > 1_000_000:
                return "Error: daily_limit_cents must be between 1 and 1,000,000"
            body["dailyLimitCents"] = daily_limit_cents
        if allowed_domains is not None:
            err = _validate_domains(allowed_domains)
            if err:
                return f"Error: {err}"
            body["allowedDomains"] = allowed_domains

        if not body:
            return "Error: At least one of daily_limit_cents or allowed_domains must be provided."

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "PATCH",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/policy",
                body,
            )
            w = data.get("wallet", data)
            return (
                f"Agentic Wallet Policy Updated\n"
                f"  ID: {w.get('id', wallet_id)}\n"
                f"  Daily Limit: {w.get('dailyLimitCents', 'N/A')} cents\n"
                f"  Status: {w.get('status', 'unknown')}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"
