"""LangChain tools for interacting with the Dominus Node proxy service.

Provides 53 tools covering proxy, wallet, usage, account, API keys, plans,
agentic wallets, and teams — each subclassing ``BaseTool`` from
``langchain_core.tools`` and wrapping the Dominus Node REST API.

Security:
    - The ``DominusNodeProxiedFetchTool`` validates URLs to prevent SSRF
      attacks.  Private IP ranges, localhost, and non-HTTP(S) schemes are
      blocked.
    - Response bodies are truncated to 4 000 characters to avoid context
      window overflow.
    - API keys and proxy credentials are never exposed in tool outputs.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
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
    """Remove Dominus Node API key patterns from error messages."""
    return _CREDENTIAL_RE.sub("***", message)


def _count_leading_zero_bits(data: bytes) -> int:
    """Count leading zero bits in a byte array."""
    count = 0
    for byte in data:
        if byte == 0:
            count += 8
        else:
            mask = 0x80
            while mask and not (byte & mask):
                count += 1
                mask >>= 1
            break
    return count


def _solve_pow(base_url: str) -> Optional[dict]:
    """Solve a Proof-of-Work challenge for CAPTCHA-free registration."""
    try:
        pow_url = f"{base_url.rstrip('/')}/api/auth/pow/challenge"
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            resp = client.post(pow_url, headers={"Content-Type": "application/json"})
            if resp.status_code >= 400:
                return None
            challenge = resp.json()
        prefix = challenge.get("prefix", "")
        difficulty = challenge.get("difficulty", 20)
        challenge_id = challenge.get("challengeId", "")
        if not prefix or not challenge_id:
            return None
        nonce = 0
        while nonce < 100_000_000:
            h = hashlib.sha256((prefix + str(nonce)).encode()).digest()
            if _count_leading_zero_bits(h) >= difficulty:
                return {"challengeId": challenge_id, "nonce": str(nonce)}
            nonce += 1
        return None
    except Exception:
        return None


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
    """Input schema for the Dominus Node proxied fetch tool."""

    url: str = Field(description="The URL to fetch through the Dominus Node proxy.")
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
    """Make HTTP requests through the Dominus Node rotating proxy network.

    Fetches the given URL through a Dominus Node proxy IP, optionally targeting a
    specific country.  Returns the HTTP status code and the first 4 000
    characters of the response body.

    Supports both datacenter (cheaper, $3/GB) and residential (premium, $5/GB)
    proxy types.  Only read-only HTTP methods (GET, HEAD, OPTIONS) are allowed.
    """

    name: str = "dominusnode_proxied_fetch"
    description: str = (
        "Fetch a URL through the Dominus Node rotating proxy network. "
        "Useful for accessing geo-restricted content or browsing anonymously. "
        "Supports country targeting and both datacenter and residential proxies. "
        "Input: url (required), method (GET/HEAD/OPTIONS), country (e.g. US), "
        "proxy_type (dc or residential)."
    )
    args_schema: Type[BaseModel] = ProxiedFetchInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    proxy_host: Optional[str] = None
    proxy_port: int = 8080
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _build_proxy_url(self, country: Optional[str], proxy_type: str) -> str:
        parts = []
        if proxy_type and proxy_type != "auto":
            parts.append(proxy_type)
        if country:
            parts.append(f"country-{country.upper()}")
        username = "-".join(parts) if parts else "auto"
        host = self.proxy_host or os.environ.get("DOMINUSNODE_PROXY_HOST", "localhost")
        port = self.proxy_port
        return f"http://{username}:{self.api_key}@{host}:{port}"

    def _run(
        self,
        url: str,
        method: str = "GET",
        country: Optional[str] = None,
        proxy_type: str = "dc",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
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

        if country and country.upper() in _SANCTIONED_COUNTRIES:
            return f"Error: Country '{country.upper()}' is blocked (OFAC sanctioned)"

        if proxy_type not in ("dc", "residential"):
            return "Error: proxy_type must be 'dc' or 'residential'."

        if not self.api_key:
            return "Error: No Dominus Node API key configured."

        try:
            proxy_url = self._build_proxy_url(country, proxy_type)

            with httpx.Client(
                proxy=proxy_url,
                timeout=30.0,
                follow_redirects=False,
            ) as http_client:
                response = http_client.request(method_upper, validated_url)

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

        if country and country.upper() in _SANCTIONED_COUNTRIES:
            return f"Error: Country '{country.upper()}' is blocked (OFAC sanctioned)"

        if proxy_type not in ("dc", "residential"):
            return "Error: proxy_type must be 'dc' or 'residential'."

        if not self.api_key:
            return "Error: No Dominus Node API key configured."

        try:
            proxy_url = self._build_proxy_url(country, proxy_type)

            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=30.0,
                follow_redirects=False,
            ) as http_client:
                response = await http_client.request(method_upper, validated_url)

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
    """Check the current Dominus Node wallet balance.

    Returns the wallet balance in both USD and cents, useful for monitoring
    spend before making proxied requests.
    """

    name: str = "dominusnode_check_balance"
    description: str = (
        "Check your Dominus Node wallet balance. Returns the current balance "
        "in dollars and cents. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/wallet/balance", agent_secret=self.agent_secret)
            cents = data.get("balanceCents", data.get("balance_cents", 0))
            usd = cents / 100
            return (
                f"Balance: ${usd:.2f} ({cents} cents)\n"
                f"Currency: USD"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/wallet/balance", agent_secret=self.agent_secret)
            cents = data.get("balanceCents", data.get("balance_cents", 0))
            usd = cents / 100
            return (
                f"Balance: ${usd:.2f} ({cents} cents)\n"
                f"Currency: USD"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUsageTool(BaseTool):
    """Check Dominus Node proxy usage statistics.

    Returns a summary of bandwidth usage including total bytes transferred,
    cost, and request count.
    """

    name: str = "dominusnode_check_usage"
    description: str = (
        "Check your Dominus Node proxy usage statistics. Returns total bandwidth "
        "used (in GB), total cost, and request count. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/usage", agent_secret=self.agent_secret)
            s = data.get("summary", data)
            total_bytes = s.get("totalBytes", s.get("total_bytes", 0))
            total_gb = total_bytes / (1024 ** 3)
            cost_cents = s.get("totalCostCents", s.get("total_cost_cents", 0))
            requests = s.get("requestCount", s.get("request_count", 0))
            return (
                f"Usage Summary:\n"
                f"  Total Data: {total_gb:.4f} GB ({total_bytes:,} bytes)\n"
                f"  Total Cost: ${cost_cents / 100:.2f} ({cost_cents} cents)\n"
                f"  Requests: {requests:,}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/usage", agent_secret=self.agent_secret)
            s = data.get("summary", data)
            total_bytes = s.get("totalBytes", s.get("total_bytes", 0))
            total_gb = total_bytes / (1024 ** 3)
            cost_cents = s.get("totalCostCents", s.get("total_cost_cents", 0))
            requests = s.get("requestCount", s.get("request_count", 0))
            return (
                f"Usage Summary:\n"
                f"  Total Data: {total_gb:.4f} GB ({total_bytes:,} bytes)\n"
                f"  Total Cost: ${cost_cents / 100:.2f} ({cost_cents} cents)\n"
                f"  Requests: {requests:,}"
            )
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class TopupPaypalInput(BaseModel):
    """Input schema for the Dominus Node PayPal top-up tool."""

    amount_cents: int = Field(
        description="Amount in cents to top up via PayPal (min 500 = $5, max 100000 = $1,000).",
    )


class DominusNodeTopupPaypalTool(BaseTool):
    """Top up your Dominus Node wallet balance via PayPal.

    Creates a PayPal order and returns an approval URL to complete payment.
    Minimum $5 (500 cents), maximum $1,000 (100,000 cents).
    """

    name: str = "dominusnode_topup_paypal"
    description: str = (
        "Top up your Dominus Node wallet balance via PayPal. "
        "Creates a PayPal order and returns an approval URL to complete payment. "
        "Input: amount_cents (integer, min 500 = $5, max 100000 = $1,000)."
    )
    args_schema: Type[BaseModel] = TopupPaypalInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, amount_cents: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/billing/paypal/create-order", body={"amountCents": amount_cents}, agent_secret=self.agent_secret)
            return f"PayPal Top-Up Order Created\nOrder ID: {data.get('orderId', '?')}\nAmount: ${amount_cents / 100:.2f}\nApproval URL: {data.get('approvalUrl', '?')}\n\nOpen the approval URL in a browser to complete payment."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, amount_cents: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/billing/paypal/create-order", body={"amountCents": amount_cents}, agent_secret=self.agent_secret)
            return f"PayPal Top-Up Order Created\nOrder ID: {data.get('orderId', '?')}\nAmount: ${amount_cents / 100:.2f}\nApproval URL: {data.get('approvalUrl', '?')}\n\nOpen the approval URL in a browser to complete payment."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class TopupStripeInput(BaseModel):
    """Input schema for the Dominus Node Stripe top-up tool."""

    amount_cents: int = Field(
        description="Amount in cents to top up via Stripe (min 500 = $5, max 100000 = $1,000).",
    )


class DominusNodeTopupStripeTool(BaseTool):
    """Top up your Dominus Node wallet balance via Stripe.

    Creates a Stripe checkout session and returns a URL to complete payment.
    Supports credit/debit card, Apple Pay, Google Pay, and Link.
    Minimum $5 (500 cents), maximum $1,000 (100,000 cents).
    """

    name: str = "dominusnode_topup_stripe"
    description: str = (
        "Top up your Dominus Node wallet balance via Stripe (credit/debit card, "
        "Apple Pay, Google Pay, Link). Creates a Stripe checkout session and returns "
        "a URL to complete payment. Input: amount_cents (integer, min 500 = $5, "
        "max 100000 = $1,000)."
    )
    args_schema: Type[BaseModel] = TopupStripeInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, amount_cents: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/billing/stripe/create-session", body={"amountCents": amount_cents}, agent_secret=self.agent_secret)
            return f"Stripe Checkout Session Created\nSession ID: {data.get('sessionId', '?')}\nAmount: ${amount_cents / 100:.2f}\nCheckout URL: {data.get('url', '?')}\n\nOpen the checkout URL in a browser to complete payment."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, amount_cents: int, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(amount_cents, int) or amount_cents < 500 or amount_cents > 100000:
            return "Error: amount_cents must be an integer between 500 ($5) and 100000 ($1,000)."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/billing/stripe/create-session", body={"amountCents": amount_cents}, agent_secret=self.agent_secret)
            return f"Stripe Checkout Session Created\nSession ID: {data.get('sessionId', '?')}\nAmount: ${amount_cents / 100:.2f}\nCheckout URL: {data.get('url', '?')}\n\nOpen the checkout URL in a browser to complete payment."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class TopupCryptoInput(BaseModel):
    """Input schema for the Dominus Node crypto top-up tool."""

    amount_usd: float = Field(
        description="Amount in USD to top up with cryptocurrency (min 5, max 1000).",
    )
    currency: str = Field(
        description=(
            "Cryptocurrency code: BTC, ETH, LTC, XMR, ZEC, USDC, SOL, USDT, DAI, BNB, LINK. "
            "Privacy coins (XMR, ZEC) provide anonymous billing."
        ),
    )


class DominusNodeTopupCryptoTool(BaseTool):
    """Top up your Dominus Node wallet with cryptocurrency.

    Creates a crypto invoice via NOWPayments and returns a payment URL.
    Supports BTC, ETH, LTC, XMR, ZEC, USDC, SOL, USDT, DAI, BNB, LINK.
    Privacy coins (XMR, ZEC) provide anonymous billing.
    Minimum $5, maximum $1,000.
    """

    name: str = "dominusnode_topup_crypto"
    description: str = (
        "Top up your Dominus Node wallet with cryptocurrency. Supports BTC, ETH, "
        "LTC, XMR, ZEC, USDC, SOL, USDT, DAI, BNB, LINK. Privacy coins (XMR, ZEC) "
        "provide anonymous billing. Input: amount_usd (number, min 5, max 1000), "
        "currency (string)."
    )
    args_schema: Type[BaseModel] = TopupCryptoInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    _VALID_CURRENCIES: frozenset = frozenset({
        "btc", "eth", "ltc", "xmr", "zec", "usdc", "sol", "usdt", "dai", "bnb", "link",
    })

    def _validate_crypto_input(self, amount_usd: float, currency: str) -> Optional[str]:
        if not isinstance(amount_usd, (int, float)) or isinstance(amount_usd, bool):
            return "Error: amount_usd must be a number between 5 ($5) and 1000 ($1,000)."
        if not math.isfinite(amount_usd) or amount_usd < 5 or amount_usd > 1000:
            return "Error: amount_usd must be a number between 5 ($5) and 1000 ($1,000)."
        if not isinstance(currency, str) or currency.lower() not in self._VALID_CURRENCIES:
            return f"Error: currency must be one of: {', '.join(sorted(self._VALID_CURRENCIES)).upper()}."
        return None

    def _format_crypto_result(self, data: dict, amount_usd: float) -> str:
        return (f"Crypto Invoice Created\nInvoice ID: {data.get('invoiceId', data.get('invoice_id', '?'))}\n"
                f"Amount: ${amount_usd:.2f}\nCurrency: {data.get('payCurrency', data.get('pay_currency', '?')).upper()}\n"
                f"Payment URL: {data.get('invoiceUrl', data.get('invoice_url', '?'))}\n\nOpen the payment URL in a browser to complete payment.")

    def _run(self, amount_usd: float, currency: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        err = self._validate_crypto_input(amount_usd, currency)
        if err:
            return err
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/billing/crypto/create-invoice", body={"amountUsd": amount_usd, "currency": currency.lower()}, agent_secret=self.agent_secret)
            return self._format_crypto_result(data, amount_usd)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, amount_usd: float, currency: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        err = self._validate_crypto_input(amount_usd, currency)
        if err:
            return err
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/billing/crypto/create-invoice", body={"amountUsd": amount_usd, "currency": currency.lower()}, agent_secret=self.agent_secret)
            return self._format_crypto_result(data, amount_usd)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeProxyConfigTool(BaseTool):
    """Get Dominus Node proxy configuration and supported countries.

    Returns the proxy endpoints, supported countries for geo-targeting,
    and available geo-targeting features.
    """

    name: str = "dominusnode_get_proxy_config"
    description: str = (
        "Get the Dominus Node proxy configuration including supported countries, "
        "proxy endpoints, and geo-targeting capabilities. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/proxy/config", agent_secret=self.agent_secret)
            import json as _json
            return f"Proxy Configuration:\n{_json.dumps(data, indent=2)[:MAX_RESPONSE_CHARS]}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/proxy/config", agent_secret=self.agent_secret)
            import json as _json
            return f"Proxy Configuration:\n{_json.dumps(data, indent=2)[:MAX_RESPONSE_CHARS]}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeX402InfoTool(BaseTool):
    """Get x402 micropayment protocol information."""

    name: str = "dominusnode_x402_info"
    description: str = (
        "Get x402 micropayment protocol information including supported "
        "facilitators, pricing, and payment options. No input required."
    )
    args_schema: Type[BaseModel] = EmptyInput

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/x402/info", agent_secret=self.agent_secret)
            import json as _json
            return f"x402 Protocol Information:\n{_json.dumps(data, indent=2)[:MAX_RESPONSE_CHARS]}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/x402/info", agent_secret=self.agent_secret)
            import json as _json
            return f"x402 Protocol Information:\n{_json.dumps(data, indent=2)[:MAX_RESPONSE_CHARS]}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


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
    agent_secret: Optional[str] = None,
) -> dict:
    """Make a synchronous authenticated REST API request.

    Raises ``RuntimeError`` on non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    secret = agent_secret or os.environ.get("DOMINUSNODE_AGENT_SECRET")
    if secret:
        headers["X-DominusNode-Agent"] = "mcp"
        headers["X-DominusNode-Agent-Secret"] = secret
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
    agent_secret: Optional[str] = None,
) -> dict:
    """Make an asynchronous authenticated REST API request.

    Raises ``RuntimeError`` on non-2xx responses.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    secret = agent_secret or os.environ.get("DOMINUSNODE_AGENT_SECRET")
    if secret:
        headers["X-DominusNode-Agent"] = "mcp"
        headers["X-DominusNode-Agent-Secret"] = secret
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
    agent_secret: Optional[str] = None

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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/agent-wallet", body, agent_secret=self.agent_secret)
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/agent-wallet", body, agent_secret=self.agent_secret)
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        amount_cents: int = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "GET",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "GET",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "GET", "/api/agent-wallet",
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "GET", "/api/agent-wallet",
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        limit: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
            data = _api_request_sync(self.base_url, self.api_key, "GET", path, agent_secret=self.agent_secret)
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
            data = await _api_request_async(self.base_url, self.api_key, "GET", path, agent_secret=self.agent_secret)
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/freeze",
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/freeze",
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/unfreeze",
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "POST",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}/unfreeze",
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = _api_request_sync(
                self.base_url, self.api_key, "DELETE",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

        err = _validate_wallet_id(wallet_id)
        if err:
            return f"Error: {err}"

        try:
            data = await _api_request_async(
                self.base_url, self.api_key, "DELETE",
                f"/api/agent-wallet/{quote(wallet_id, safe='')}",
                agent_secret=self.agent_secret,
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
    agent_secret: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def _run(
        self,
        wallet_id: str = "",
        daily_limit_cents: Optional[int] = None,
        allowed_domains: Optional[list] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
                agent_secret=self.agent_secret,
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
            return "Error: No Dominus Node API credentials configured. Initialize the toolkit first."

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
                agent_secret=self.agent_secret,
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


# ──────────────────────────────────────────────────────────────────────
# NEW TOOLS: Proxy status, Sessions, Wallet extras, Usage extras,
#            Account, API Keys, Plans, Teams (36 new tools)
# ──────────────────────────────────────────────────────────────────────

# --- Additional input schemas ---


class GetProxyStatusInput(BaseModel):
    """Empty input for proxy status."""
    pass


class ListSessionsInput(BaseModel):
    """Empty input for listing sessions."""
    pass


class GetTransactionsInput(BaseModel):
    """Input schema for wallet transaction history."""
    page: int = Field(default=1, description="Page number (starting at 1).")
    limit: int = Field(default=20, description="Items per page (1-100).")


class GetForecastInput(BaseModel):
    """Empty input for spending forecast."""
    pass


class CheckPaymentInput(BaseModel):
    """Input schema for checking crypto payment status."""
    invoice_id: str = Field(description="Invoice UUID from a crypto top-up.")


class GetDailyUsageInput(BaseModel):
    """Input schema for daily usage breakdown."""
    days: int = Field(default=7, description="Number of days to look back (1-90).")


class GetTopHostsInput(BaseModel):
    """Input schema for top hosts by bandwidth."""
    limit: int = Field(default=10, description="Number of top hosts to return (1-50).")
    days: int = Field(default=30, description="Number of days to look back (1-365).")


class RegisterInput(BaseModel):
    """Input schema for account registration."""
    email: str = Field(description="Email address for the new account.")
    password: str = Field(description="Password (min 8 characters).")


class LoginInput(BaseModel):
    """Input schema for login."""
    email: str = Field(description="Account email address.")
    password: str = Field(description="Account password.")


class VerifyEmailInput(BaseModel):
    """Input schema for email verification."""
    token: str = Field(description="Email verification token from registration.")


class UpdatePasswordInput(BaseModel):
    """Input schema for password change."""
    current_password: str = Field(description="Current account password.")
    new_password: str = Field(description="New password (min 8 characters).")


class CreateKeyInput(BaseModel):
    """Input schema for creating an API key."""
    label: str = Field(description="Descriptive label for the key (max 100 chars).")


class RevokeKeyInput(BaseModel):
    """Input schema for revoking an API key."""
    key_id: str = Field(description="UUID of the API key to revoke.")


class ChangePlanInput(BaseModel):
    """Input schema for changing plan."""
    plan_id: str = Field(description="Plan ID to switch to (e.g. free-dc, payg, agent).")


class CreateTeamInput(BaseModel):
    """Input schema for creating a team."""
    name: str = Field(description="Team name (max 100 chars).")
    max_members: Optional[int] = Field(default=None, description="Maximum members (1-100).")


class TeamIdInput(BaseModel):
    """Input schema for operations requiring only a team ID."""
    team_id: str = Field(description="UUID of the team.")


class UpdateTeamInput(BaseModel):
    """Input schema for updating a team."""
    team_id: str = Field(description="UUID of the team.")
    name: Optional[str] = Field(default=None, description="New team name.")
    max_members: Optional[int] = Field(default=None, description="New max members (1-100).")


class TeamFundInput(BaseModel):
    """Input schema for funding a team wallet."""
    team_id: str = Field(description="UUID of the team.")
    amount_cents: int = Field(description="Amount in cents to transfer (min 100, max 1000000).")


class TeamCreateKeyInput(BaseModel):
    """Input schema for creating a team API key."""
    team_id: str = Field(description="UUID of the team.")
    label: str = Field(description="Label for the API key (max 100 chars).")


class TeamRevokeKeyInput(BaseModel):
    """Input schema for revoking a team API key."""
    team_id: str = Field(description="UUID of the team.")
    key_id: str = Field(description="UUID of the API key to revoke.")


class TeamUsageInput(BaseModel):
    """Input schema for team usage/transactions."""
    team_id: str = Field(description="UUID of the team.")
    limit: int = Field(default=20, description="Number of transactions to return (1-100).")


class TeamAddMemberInput(BaseModel):
    """Input schema for adding a team member."""
    team_id: str = Field(description="UUID of the team.")
    email: str = Field(description="Email of the user to add.")
    role: Optional[str] = Field(default=None, description="Role: 'member' or 'admin'. Default: member.")


class TeamRemoveMemberInput(BaseModel):
    """Input schema for removing a team member."""
    team_id: str = Field(description="UUID of the team.")
    user_id: str = Field(description="UUID of the user to remove.")


class UpdateTeamMemberRoleInput(BaseModel):
    """Input schema for updating a team member's role."""
    team_id: str = Field(description="UUID of the team.")
    user_id: str = Field(description="UUID of the member.")
    role: str = Field(description="New role: 'member' or 'admin'.")


class TeamInviteMemberInput(BaseModel):
    """Input schema for inviting a member to a team."""
    team_id: str = Field(description="UUID of the team.")
    email: str = Field(description="Email address to invite.")
    role: str = Field(default="member", description="Role: 'member' or 'admin'.")


class TeamCancelInviteInput(BaseModel):
    """Input schema for cancelling a team invite."""
    team_id: str = Field(description="UUID of the team.")
    invite_id: str = Field(description="UUID of the invite to cancel.")


# --- Unauthenticated API helpers ---

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_VALID_ROLES = frozenset({"member", "admin"})


def _api_request_unauth_sync(
    base_url: str, method: str, path: str,
    body: Optional[dict] = None, agent_secret: Optional[str] = None,
) -> dict:
    """Synchronous unauthenticated REST request."""
    url = f"{base_url.rstrip('/')}{path}"
    headers: dict = {"Content-Type": "application/json"}
    secret = agent_secret or os.environ.get("DOMINUSNODE_AGENT_SECRET")
    if secret:
        headers["X-DominusNode-Agent"] = "mcp"
        headers["X-DominusNode-Agent-Secret"] = secret
    with httpx.Client(timeout=30.0, follow_redirects=False, max_redirects=0) as client:
        resp = client.request(method, url, headers=headers, json=body)
    if len(resp.content) > _MAX_RESPONSE_BODY_BYTES:
        raise RuntimeError("Response body exceeds 10 MB size limit")
    if resp.status_code >= 400:
        raise RuntimeError(f"API error {resp.status_code}: {_sanitize_error(resp.text[:200])}")
    data = resp.json()
    _strip_dangerous_keys(data)
    return data


async def _api_request_unauth_async(
    base_url: str, method: str, path: str,
    body: Optional[dict] = None, agent_secret: Optional[str] = None,
) -> dict:
    """Asynchronous unauthenticated REST request."""
    url = f"{base_url.rstrip('/')}{path}"
    headers: dict = {"Content-Type": "application/json"}
    secret = agent_secret or os.environ.get("DOMINUSNODE_AGENT_SECRET")
    if secret:
        headers["X-DominusNode-Agent"] = "mcp"
        headers["X-DominusNode-Agent-Secret"] = secret
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, max_redirects=0) as client:
        resp = await client.request(method, url, headers=headers, json=body)
    if len(resp.content) > _MAX_RESPONSE_BODY_BYTES:
        raise RuntimeError("Response body exceeds 10 MB size limit")
    if resp.status_code >= 400:
        raise RuntimeError(f"API error {resp.status_code}: {_sanitize_error(resp.text[:200])}")
    data = resp.json()
    _strip_dangerous_keys(data)
    return data


def _validate_team_id(team_id: Any) -> Optional[str]:
    """Validate a team ID; returns an error message or ``None``."""
    if not team_id or not isinstance(team_id, str):
        return "team_id is required and must be a string"
    if not _UUID_RE.match(team_id):
        return "team_id must be a valid UUID"
    return None


# ──────────────────────────────────────────────────────────────────────
# Proxy: getProxyStatus
# ──────────────────────────────────────────────────────────────────────


class DominusNodeProxyStatusTool(BaseTool):
    """Get live proxy network status."""

    name: str = "dominusnode_get_proxy_status"
    description: str = (
        "Get live proxy network status including latency, active session count, "
        "and uptime. No input required."
    )
    args_schema: Type[BaseModel] = GetProxyStatusInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/proxy/status", agent_secret=self.agent_secret)
            return (f"Proxy Status:\n  Status: {data.get('status', '?')}\n"
                    f"  Latency: {data.get('avgLatencyMs', data.get('avg_latency_ms', 0))}ms\n"
                    f"  Active Sessions: {data.get('activeSessions', data.get('active_sessions', 0))}\n"
                    f"  Uptime: {data.get('uptimeSeconds', data.get('uptime_seconds', 0))}s")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/proxy/status", agent_secret=self.agent_secret)
            return (f"Proxy Status:\n  Status: {data.get('status', '?')}\n"
                    f"  Latency: {data.get('avgLatencyMs', data.get('avg_latency_ms', 0))}ms\n"
                    f"  Active Sessions: {data.get('activeSessions', data.get('active_sessions', 0))}\n"
                    f"  Uptime: {data.get('uptimeSeconds', data.get('uptime_seconds', 0))}s")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Sessions: listSessions
# ──────────────────────────────────────────────────────────────────────


class DominusNodeListSessionsTool(BaseTool):
    """List all active proxy sessions."""

    name: str = "dominusnode_list_sessions"
    description: str = "List all active proxy sessions. No input required."
    args_schema: Type[BaseModel] = ListSessionsInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/sessions/active", agent_secret=self.agent_secret)
            sessions = data.get("sessions", data if isinstance(data, list) else [])
            if not sessions:
                return "No active proxy sessions."
            lines = [f"Active Sessions ({len(sessions)}):"]
            for s in sessions:
                sid = s.get("id", "?") if isinstance(s, dict) else str(s)
                st = s.get("status", "?") if isinstance(s, dict) else "?"
                lines.append(f"  {sid} | Status: {st}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/sessions/active", agent_secret=self.agent_secret)
            sessions = data.get("sessions", data if isinstance(data, list) else [])
            if not sessions:
                return "No active proxy sessions."
            lines = [f"Active Sessions ({len(sessions)}):"]
            for s in sessions:
                sid = s.get("id", "?") if isinstance(s, dict) else str(s)
                st = s.get("status", "?") if isinstance(s, dict) else "?"
                lines.append(f"  {sid} | Status: {st}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Wallet extras: getTransactions, getForecast, checkPayment
# ──────────────────────────────────────────────────────────────────────


class DominusNodeGetTransactionsTool(BaseTool):
    """Get wallet transaction history."""

    name: str = "dominusnode_get_transactions"
    description: str = "Get wallet transaction history. Input: optional page (int), limit (int 1-100)."
    args_schema: Type[BaseModel] = GetTransactionsInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, page: int = 1, limit: int = 20, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(page, int) or page < 1:
            return "Error: page must be a positive integer"
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return "Error: limit must be between 1 and 100"
        offset = (page - 1) * limit
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/wallet/transactions?offset={offset}&limit={limit}", agent_secret=self.agent_secret)
            txns = data.get("transactions", [])
            if not txns:
                return "No transactions found."
            lines = [f"Transactions (page {page}):"]
            for t in txns:
                lines.append(f"  {t.get('createdAt', '?')} | {t.get('type', '?'):10s} | ${t.get('amountCents', 0) / 100:.2f} | {t.get('description', '')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, page: int = 1, limit: int = 20, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(page, int) or page < 1:
            return "Error: page must be a positive integer"
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            return "Error: limit must be between 1 and 100"
        offset = (page - 1) * limit
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/wallet/transactions?offset={offset}&limit={limit}", agent_secret=self.agent_secret)
            txns = data.get("transactions", [])
            if not txns:
                return "No transactions found."
            lines = [f"Transactions (page {page}):"]
            for t in txns:
                lines.append(f"  {t.get('createdAt', '?')} | {t.get('type', '?'):10s} | ${t.get('amountCents', 0) / 100:.2f} | {t.get('description', '')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeGetForecastTool(BaseTool):
    """Get spending forecast."""

    name: str = "dominusnode_get_forecast"
    description: str = "Get spending forecast: daily average, days remaining, trend. No input required."
    args_schema: Type[BaseModel] = GetForecastInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/wallet/forecast", agent_secret=self.agent_secret)
            return (f"Spending Forecast:\n  Daily Average: ${data.get('dailyAvgCents', 0) / 100:.2f}\n"
                    f"  Days Remaining: {data.get('daysRemaining', 'unlimited')}\n"
                    f"  Trend: {data.get('trend', '?')} ({data.get('trendPct', 0)}%)")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/wallet/forecast", agent_secret=self.agent_secret)
            return (f"Spending Forecast:\n  Daily Average: ${data.get('dailyAvgCents', 0) / 100:.2f}\n"
                    f"  Days Remaining: {data.get('daysRemaining', 'unlimited')}\n"
                    f"  Trend: {data.get('trend', '?')} ({data.get('trendPct', 0)}%)")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeCheckPaymentTool(BaseTool):
    """Check crypto payment invoice status."""

    name: str = "dominusnode_check_payment"
    description: str = "Check the status of a cryptocurrency payment invoice. Input: invoice_id (UUID)."
    args_schema: Type[BaseModel] = CheckPaymentInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, invoice_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not invoice_id or not _UUID_RE.match(invoice_id):
            return "Error: invoice_id must be a valid UUID"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/wallet/crypto/status/{quote(invoice_id, safe='')}", agent_secret=self.agent_secret)
            return (f"Payment Status:\n  Invoice ID: {data.get('invoiceId', invoice_id)}\n"
                    f"  Status: {data.get('status', '?')}\n  Amount: ${data.get('amountCents', 0) / 100:.2f}\n"
                    f"  Provider: {data.get('provider', '?')}\n  Created: {data.get('createdAt', '?')}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, invoice_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not invoice_id or not _UUID_RE.match(invoice_id):
            return "Error: invoice_id must be a valid UUID"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/wallet/crypto/status/{quote(invoice_id, safe='')}", agent_secret=self.agent_secret)
            return (f"Payment Status:\n  Invoice ID: {data.get('invoiceId', invoice_id)}\n"
                    f"  Status: {data.get('status', '?')}\n  Amount: ${data.get('amountCents', 0) / 100:.2f}\n"
                    f"  Provider: {data.get('provider', '?')}\n  Created: {data.get('createdAt', '?')}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Usage extras: getDailyUsage, getTopHosts
# ──────────────────────────────────────────────────────────────────────


class DominusNodeDailyUsageTool(BaseTool):
    """Get daily bandwidth breakdown."""

    name: str = "dominusnode_get_daily_usage"
    description: str = "Get daily bandwidth breakdown. Input: optional days (1-90, default 7)."
    args_schema: Type[BaseModel] = GetDailyUsageInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, days: int = 7, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(days, int) or days < 1 or days > 90:
            return "Error: days must be between 1 and 90"
        try:
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            until = datetime.now(timezone.utc).isoformat()
            data = _api_request_sync(self.base_url, self.api_key, "GET",
                f"/api/usage/daily?since={quote(since, safe='')}&until={quote(until, safe='')}", agent_secret=self.agent_secret)
            day_list = data.get("days", [])
            if not day_list:
                return "No usage data for this period."
            lines = ["Date       | Bandwidth      | Cost    | Requests"]
            for d in day_list:
                bw = d.get("totalBytes", 0)
                bw_s = f"{bw / (1024**3):.3f} GB" if bw >= 1024**3 else f"{bw / (1024**2):.2f} MB"
                lines.append(f"{d.get('date', '?')} | {bw_s:14s} | ${d.get('totalCostUsd', 0):5.2f} | {d.get('requestCount', 0)}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, days: int = 7, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(days, int) or days < 1 or days > 90:
            return "Error: days must be between 1 and 90"
        try:
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            until = datetime.now(timezone.utc).isoformat()
            data = await _api_request_async(self.base_url, self.api_key, "GET",
                f"/api/usage/daily?since={quote(since, safe='')}&until={quote(until, safe='')}", agent_secret=self.agent_secret)
            day_list = data.get("days", [])
            if not day_list:
                return "No usage data for this period."
            lines = ["Date       | Bandwidth      | Cost    | Requests"]
            for d in day_list:
                bw = d.get("totalBytes", 0)
                bw_s = f"{bw / (1024**3):.3f} GB" if bw >= 1024**3 else f"{bw / (1024**2):.2f} MB"
                lines.append(f"{d.get('date', '?')} | {bw_s:14s} | ${d.get('totalCostUsd', 0):5.2f} | {d.get('requestCount', 0)}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTopHostsTool(BaseTool):
    """Get top target hosts by bandwidth."""

    name: str = "dominusnode_get_top_hosts"
    description: str = "Get top target hosts by bandwidth usage. Input: optional limit (1-50), days (1-365)."
    args_schema: Type[BaseModel] = GetTopHostsInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, limit: int = 10, days: int = 30, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            return "Error: limit must be between 1 and 50"
        if not isinstance(days, int) or days < 1 or days > 365:
            return "Error: days must be between 1 and 365"
        try:
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            until = datetime.now(timezone.utc).isoformat()
            data = _api_request_sync(self.base_url, self.api_key, "GET",
                f"/api/usage/top-hosts?limit={limit}&since={quote(since, safe='')}&until={quote(until, safe='')}", agent_secret=self.agent_secret)
            hosts = data.get("hosts", [])
            if not hosts:
                return "No host data for this period."
            lines = ["Host                         | Bandwidth      | Requests"]
            for h in hosts:
                bw = h.get("totalBytes", 0)
                bw_s = f"{bw / (1024**3):.3f} GB" if bw >= 1024**3 else f"{bw / (1024**2):.2f} MB"
                lines.append(f"{h.get('targetHost', '?'):28s} | {bw_s:14s} | {h.get('requestCount', 0)}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, limit: int = 10, days: int = 30, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            return "Error: limit must be between 1 and 50"
        if not isinstance(days, int) or days < 1 or days > 365:
            return "Error: days must be between 1 and 365"
        try:
            from datetime import datetime, timezone, timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            until = datetime.now(timezone.utc).isoformat()
            data = await _api_request_async(self.base_url, self.api_key, "GET",
                f"/api/usage/top-hosts?limit={limit}&since={quote(since, safe='')}&until={quote(until, safe='')}", agent_secret=self.agent_secret)
            hosts = data.get("hosts", [])
            if not hosts:
                return "No host data for this period."
            lines = ["Host                         | Bandwidth      | Requests"]
            for h in hosts:
                bw = h.get("totalBytes", 0)
                bw_s = f"{bw / (1024**3):.3f} GB" if bw >= 1024**3 else f"{bw / (1024**2):.2f} MB"
                lines.append(f"{h.get('targetHost', '?'):28s} | {bw_s:14s} | {h.get('requestCount', 0)}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Account (6): register, login, getAccountInfo, verifyEmail,
#              resendVerification, updatePassword
# ──────────────────────────────────────────────────────────────────────


class DominusNodeRegisterTool(BaseTool):
    """Register a new Dominus Node account (unauthenticated)."""
    name: str = "dominusnode_register"
    description: str = "Register a new Dominus Node account. Input: email, password (min 8 chars)."
    args_schema: Type[BaseModel] = RegisterInput
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, email: str = "", password: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not email or not _EMAIL_RE.match(email):
            return "Error: A valid email address is required."
        if not password or len(password) < 8 or len(password) > 128:
            return "Error: Password must be between 8 and 128 characters."
        try:
            body: dict = {"email": email, "password": password}
            pow_result = _solve_pow(self.base_url)
            if pow_result:
                body["pow"] = pow_result
            data = _api_request_unauth_sync(self.base_url, "POST", "/api/auth/register", body, agent_secret=self.agent_secret)
            user = data.get("user", {})
            pow_msg = "Email auto-verified via Proof-of-Work." if pow_result else "Email auto-verified (MCP agent)"
            return f"Account Created\n  Email: {user.get('email', email)}\n  User ID: {user.get('id', '?')}\n  {pow_msg}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, email: str = "", password: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not email or not _EMAIL_RE.match(email):
            return "Error: A valid email address is required."
        if not password or len(password) < 8 or len(password) > 128:
            return "Error: Password must be between 8 and 128 characters."
        try:
            body: dict = {"email": email, "password": password}
            pow_result = _solve_pow(self.base_url)
            if pow_result:
                body["pow"] = pow_result
            data = await _api_request_unauth_async(self.base_url, "POST", "/api/auth/register", body, agent_secret=self.agent_secret)
            user = data.get("user", {})
            pow_msg = "Email auto-verified via Proof-of-Work." if pow_result else "Email auto-verified (MCP agent)"
            return f"Account Created\n  Email: {user.get('email', email)}\n  User ID: {user.get('id', '?')}\n  {pow_msg}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeLoginTool(BaseTool):
    """Login to an existing Dominus Node account (unauthenticated)."""
    name: str = "dominusnode_login"
    description: str = "Login to a Dominus Node account. Input: email, password."
    args_schema: Type[BaseModel] = LoginInput
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, email: str = "", password: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not email or not _EMAIL_RE.match(email):
            return "Error: A valid email address is required."
        if not password or len(password) > 128:
            return "Error: Password is required (max 128 characters)."
        try:
            data = _api_request_unauth_sync(self.base_url, "POST", "/api/auth/login", {"email": email, "password": password}, agent_secret=self.agent_secret)
            user = data.get("user", {})
            return f"Logged In\n  Email: {user.get('email', email)}\n  User ID: {user.get('id', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, email: str = "", password: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not email or not _EMAIL_RE.match(email):
            return "Error: A valid email address is required."
        if not password or len(password) > 128:
            return "Error: Password is required (max 128 characters)."
        try:
            data = await _api_request_unauth_async(self.base_url, "POST", "/api/auth/login", {"email": email, "password": password}, agent_secret=self.agent_secret)
            user = data.get("user", {})
            return f"Logged In\n  Email: {user.get('email', email)}\n  User ID: {user.get('id', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeGetAccountInfoTool(BaseTool):
    """Get account details."""
    name: str = "dominusnode_get_account_info"
    description: str = "Get account details including email, verification, admin status. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/auth/me", agent_secret=self.agent_secret)
            user = data.get("user", data)
            return (f"Account Info:\n  User ID: {user.get('id', '?')}\n  Email: {user.get('email', '?')}\n"
                    f"  Email Verified: {'yes' if user.get('email_verified') else 'no'}\n"
                    f"  Admin: {'yes' if user.get('is_admin') else 'no'}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/auth/me", agent_secret=self.agent_secret)
            user = data.get("user", data)
            return (f"Account Info:\n  User ID: {user.get('id', '?')}\n  Email: {user.get('email', '?')}\n"
                    f"  Email Verified: {'yes' if user.get('email_verified') else 'no'}\n"
                    f"  Admin: {'yes' if user.get('is_admin') else 'no'}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeVerifyEmailTool(BaseTool):
    """Verify email address (unauthenticated)."""
    name: str = "dominusnode_verify_email"
    description: str = "Verify email using a verification token. Input: token (min 32 chars)."
    args_schema: Type[BaseModel] = VerifyEmailInput
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, token: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not token or len(token) < 32 or len(token) > 128:
            return "Error: token must be between 32 and 128 characters."
        try:
            _api_request_unauth_sync(self.base_url, "POST", "/api/auth/verify-email", {"token": token}, agent_secret=self.agent_secret)
            return "Email verified successfully."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, token: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.base_url:
            return "Error: No Dominus Node base URL configured."
        if not token or len(token) < 32 or len(token) > 128:
            return "Error: token must be between 32 and 128 characters."
        try:
            await _api_request_unauth_async(self.base_url, "POST", "/api/auth/verify-email", {"token": token}, agent_secret=self.agent_secret)
            return "Email verified successfully."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeResendVerificationTool(BaseTool):
    """Resend email verification link."""
    name: str = "dominusnode_resend_verification"
    description: str = "Resend the email verification link. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/auth/resend-verification", {}, agent_secret=self.agent_secret)
            return data.get("message", "Verification email resent.")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/auth/resend-verification", {}, agent_secret=self.agent_secret)
            return data.get("message", "Verification email resent.")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUpdatePasswordTool(BaseTool):
    """Change the account password."""
    name: str = "dominusnode_update_password"
    description: str = "Change account password. Input: current_password, new_password (min 8 chars)."
    args_schema: Type[BaseModel] = UpdatePasswordInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, current_password: str = "", new_password: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not current_password or len(current_password) > 128:
            return "Error: current_password is required (max 128 characters)."
        if not new_password or len(new_password) < 8 or len(new_password) > 128:
            return "Error: new_password must be between 8 and 128 characters."
        try:
            _api_request_sync(self.base_url, self.api_key, "POST", "/api/auth/change-password", {"currentPassword": current_password, "newPassword": new_password}, agent_secret=self.agent_secret)
            return "Password changed successfully. All API keys and tokens have been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, current_password: str = "", new_password: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not current_password or len(current_password) > 128:
            return "Error: current_password is required (max 128 characters)."
        if not new_password or len(new_password) < 8 or len(new_password) > 128:
            return "Error: new_password must be between 8 and 128 characters."
        try:
            await _api_request_async(self.base_url, self.api_key, "POST", "/api/auth/change-password", {"currentPassword": current_password, "newPassword": new_password}, agent_secret=self.agent_secret)
            return "Password changed successfully. All API keys and tokens have been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# API Keys (3): listKeys, createKey, revokeKey
# ──────────────────────────────────────────────────────────────────────


class DominusNodeListKeysTool(BaseTool):
    """List all API keys."""
    name: str = "dominusnode_list_keys"
    description: str = "List all API keys on this account. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/keys", agent_secret=self.agent_secret)
            keys = data.get("keys", [])
            if not keys:
                return "No API keys found."
            lines = [f"API Keys ({len(keys)}):"]
            for k in keys:
                lines.append(f"  {k.get('prefix', '?')}... | Label: {k.get('label', '(none)')} | Created: {k.get('createdAt', '?')} | Revoked: {k.get('revokedAt', 'no')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/keys", agent_secret=self.agent_secret)
            keys = data.get("keys", [])
            if not keys:
                return "No API keys found."
            lines = [f"API Keys ({len(keys)}):"]
            for k in keys:
                lines.append(f"  {k.get('prefix', '?')}... | Label: {k.get('label', '(none)')} | Created: {k.get('createdAt', '?')} | Revoked: {k.get('revokedAt', 'no')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeCreateKeyTool(BaseTool):
    """Create a new API key."""
    name: str = "dominusnode_create_key"
    description: str = "Create a new API key. The full key is shown only once. Input: label (max 100 chars)."
    args_schema: Type[BaseModel] = CreateKeyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, label: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        err = _validate_label(label)
        if err:
            return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/keys", {"label": label}, agent_secret=self.agent_secret)
            return f"API Key Created\n  Key: {data.get('key', '?')}\n  ID: {data.get('id', '?')}\n  Label: {data.get('label', label)}\n\nIMPORTANT: Save this key now -- it will not be shown again."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, label: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        err = _validate_label(label)
        if err:
            return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/keys", {"label": label}, agent_secret=self.agent_secret)
            return f"API Key Created\n  Key: {data.get('key', '?')}\n  ID: {data.get('id', '?')}\n  Label: {data.get('label', label)}\n\nIMPORTANT: Save this key now -- it will not be shown again."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeRevokeKeyTool(BaseTool):
    """Revoke an API key by ID."""
    name: str = "dominusnode_revoke_key"
    description: str = "Revoke an API key. Input: key_id (UUID)."
    args_schema: Type[BaseModel] = RevokeKeyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, key_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not key_id or not _UUID_RE.match(key_id):
            return "Error: key_id must be a valid UUID"
        try:
            _api_request_sync(self.base_url, self.api_key, "DELETE", f"/api/keys/{quote(key_id, safe='')}", agent_secret=self.agent_secret)
            return f"API key {key_id} has been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, key_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not key_id or not _UUID_RE.match(key_id):
            return "Error: key_id must be a valid UUID"
        try:
            await _api_request_async(self.base_url, self.api_key, "DELETE", f"/api/keys/{quote(key_id, safe='')}", agent_secret=self.agent_secret)
            return f"API key {key_id} has been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Plans (3): getPlan, listPlans, changePlan
# ──────────────────────────────────────────────────────────────────────


class DominusNodeGetPlanTool(BaseTool):
    """Get current plan details."""
    name: str = "dominusnode_get_plan"
    description: str = "Get current plan details including usage and limits. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/plans/user/plan", agent_secret=self.agent_secret)
            plan = data.get("plan", {})
            usage = data.get("usage", {})
            return (f"Current Plan:\n  Plan: {plan.get('name', '?')}\n  Price: ${plan.get('pricePerGbUsd', 0):.2f}/GB\n"
                    f"  Max Connections: {plan.get('maxConnections', '?')}\n"
                    f"  Monthly Usage: {usage.get('monthlyUsageBytes', 0) / (1024**3):.3f} GB\n"
                    f"  Percent Used: {usage.get('percentUsed', 0):.1f}%")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/plans/user/plan", agent_secret=self.agent_secret)
            plan = data.get("plan", {})
            usage = data.get("usage", {})
            return (f"Current Plan:\n  Plan: {plan.get('name', '?')}\n  Price: ${plan.get('pricePerGbUsd', 0):.2f}/GB\n"
                    f"  Max Connections: {plan.get('maxConnections', '?')}\n"
                    f"  Monthly Usage: {usage.get('monthlyUsageBytes', 0) / (1024**3):.3f} GB\n"
                    f"  Percent Used: {usage.get('percentUsed', 0):.1f}%")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeListPlansTool(BaseTool):
    """List all available pricing plans."""
    name: str = "dominusnode_list_plans"
    description: str = "List all available pricing plans. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/plans", agent_secret=self.agent_secret)
            plans = data.get("plans", [])
            if not plans:
                return "No plans available."
            lines = ["Available Plans:"]
            for p in plans:
                bw = p.get("monthlyBandwidthGB")
                bw_s = f"{bw} GB" if bw is not None else "unlimited"
                lines.append(f"  {p.get('name', '?')} -- ${p.get('pricePerGbUsd', 0):.2f}/GB | {bw_s} bandwidth | {p.get('maxConnections', '?')} connections")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/plans", agent_secret=self.agent_secret)
            plans = data.get("plans", [])
            if not plans:
                return "No plans available."
            lines = ["Available Plans:"]
            for p in plans:
                bw = p.get("monthlyBandwidthGB")
                bw_s = f"{bw} GB" if bw is not None else "unlimited"
                lines.append(f"  {p.get('name', '?')} -- ${p.get('pricePerGbUsd', 0):.2f}/GB | {bw_s} bandwidth | {p.get('maxConnections', '?')} connections")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeChangePlanTool(BaseTool):
    """Switch to a different pricing plan."""
    name: str = "dominusnode_change_plan"
    description: str = "Switch pricing plan. Input: plan_id (e.g. free-dc, payg, agent)."
    args_schema: Type[BaseModel] = ChangePlanInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, plan_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not plan_id or len(plan_id) > 50:
            return "Error: plan_id is required (max 50 characters)."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "PUT", "/api/plans/user/plan", {"planId": plan_id}, agent_secret=self.agent_secret)
            plan = data.get("plan", {})
            return f"Plan Changed\n  Plan: {plan.get('name', plan_id)}\n  Price: ${plan.get('pricePerGbUsd', 0):.2f}/GB\n  Max Connections: {plan.get('maxConnections', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, plan_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        if not self.api_key or not self.base_url:
            return "Error: No Dominus Node API credentials configured."
        if not plan_id or len(plan_id) > 50:
            return "Error: plan_id is required (max 50 characters)."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "PUT", "/api/plans/user/plan", {"planId": plan_id}, agent_secret=self.agent_secret)
            plan = data.get("plan", {})
            return f"Plan Changed\n  Plan: {plan.get('name', plan_id)}\n  Price: ${plan.get('pricePerGbUsd', 0):.2f}/GB\n  Max Connections: {plan.get('maxConnections', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


# ──────────────────────────────────────────────────────────────────────
# Teams (17 tools)
# ──────────────────────────────────────────────────────────────────────

def _team_tool_common(api_key: Any, base_url: Any) -> Optional[str]:
    if not api_key or not base_url:
        return "Error: No Dominus Node API credentials configured."
    return None


class DominusNodeCreateTeamTool(BaseTool):
    name: str = "dominusnode_create_team"
    description: str = "Create a new team. Input: name (max 100 chars), optional max_members (1-100)."
    args_schema: Type[BaseModel] = CreateTeamInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, name: str = "", max_members: Optional[int] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_label(name)
        if err: return f"Error: {err}"
        if max_members is not None and (not isinstance(max_members, int) or max_members < 1 or max_members > 100):
            return "Error: max_members must be between 1 and 100"
        body: dict = {"name": name}
        if max_members is not None: body["maxMembers"] = max_members
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", "/api/teams", body, agent_secret=self.agent_secret)
            return f"Team Created\n  ID: {data.get('id', '?')}\n  Name: {data.get('name', name)}\n  Max Members: {data.get('maxMembers', 'unlimited')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, name: str = "", max_members: Optional[int] = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_label(name)
        if err: return f"Error: {err}"
        if max_members is not None and (not isinstance(max_members, int) or max_members < 1 or max_members > 100):
            return "Error: max_members must be between 1 and 100"
        body: dict = {"name": name}
        if max_members is not None: body["maxMembers"] = max_members
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", "/api/teams", body, agent_secret=self.agent_secret)
            return f"Team Created\n  ID: {data.get('id', '?')}\n  Name: {data.get('name', name)}\n  Max Members: {data.get('maxMembers', 'unlimited')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeListTeamsTool(BaseTool):
    name: str = "dominusnode_list_teams"
    description: str = "List all teams you belong to. No input required."
    args_schema: Type[BaseModel] = EmptyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", "/api/teams", agent_secret=self.agent_secret)
            teams = data.get("teams", [])
            if not teams: return "No teams found."
            lines = [f"Teams ({len(teams)}):"]
            for t in teams:
                lines.append(f"  {t.get('name', '?')} ({t.get('id', '?')[:8]}...) | Role: {t.get('role', '?')} | Balance: ${t.get('balanceCents', 0) / 100:.2f}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", "/api/teams", agent_secret=self.agent_secret)
            teams = data.get("teams", [])
            if not teams: return "No teams found."
            lines = [f"Teams ({len(teams)}):"]
            for t in teams:
                lines.append(f"  {t.get('name', '?')} ({t.get('id', '?')[:8]}...) | Role: {t.get('role', '?')} | Balance: ${t.get('balanceCents', 0) / 100:.2f}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamDetailsTool(BaseTool):
    name: str = "dominusnode_team_details"
    description: str = "Get detailed info about a team. Input: team_id (UUID)."
    args_schema: Type[BaseModel] = TeamIdInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}", agent_secret=self.agent_secret)
            return (f"Team: {data.get('name', '?')}\n  ID: {data.get('id', team_id)}\n  Owner: {data.get('ownerId', '?')}\n"
                    f"  Status: {data.get('status', '?')}\n  Role: {data.get('role', '?')}\n  Balance: ${data.get('balanceCents', 0) / 100:.2f}\n"
                    f"  Max Members: {data.get('maxMembers', 'unlimited')}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}", agent_secret=self.agent_secret)
            return (f"Team: {data.get('name', '?')}\n  ID: {data.get('id', team_id)}\n  Owner: {data.get('ownerId', '?')}\n"
                    f"  Status: {data.get('status', '?')}\n  Role: {data.get('role', '?')}\n  Balance: ${data.get('balanceCents', 0) / 100:.2f}\n"
                    f"  Max Members: {data.get('maxMembers', 'unlimited')}")
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUpdateTeamTool(BaseTool):
    name: str = "dominusnode_update_team"
    description: str = "Update a team's name or max members. Input: team_id (UUID), optional name, optional max_members."
    args_schema: Type[BaseModel] = UpdateTeamInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", name: Optional[str] = None, max_members: Optional[int] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        body: dict = {}
        if name is not None: body["name"] = name
        if max_members is not None: body["maxMembers"] = max_members
        if not body: return "Error: Provide name or max_members to update."
        try:
            data = _api_request_sync(self.base_url, self.api_key, "PATCH", f"/api/teams/{quote(team_id, safe='')}", body, agent_secret=self.agent_secret)
            return f"Team Updated\n  ID: {data.get('id', team_id)}\n  Name: {data.get('name', '?')}\n  Max Members: {data.get('maxMembers', 'unlimited')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", name: Optional[str] = None, max_members: Optional[int] = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        body: dict = {}
        if name is not None: body["name"] = name
        if max_members is not None: body["maxMembers"] = max_members
        if not body: return "Error: Provide name or max_members to update."
        try:
            data = await _api_request_async(self.base_url, self.api_key, "PATCH", f"/api/teams/{quote(team_id, safe='')}", body, agent_secret=self.agent_secret)
            return f"Team Updated\n  ID: {data.get('id', team_id)}\n  Name: {data.get('name', '?')}\n  Max Members: {data.get('maxMembers', 'unlimited')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamDeleteTool(BaseTool):
    name: str = "dominusnode_team_delete"
    description: str = "Delete a team. Remaining balance refunded. Input: team_id (UUID)."
    args_schema: Type[BaseModel] = TeamIdInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}", agent_secret=self.agent_secret)
            return f"Team Deleted\n  Refunded: ${data.get('refundedCents', 0) / 100:.2f}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}", agent_secret=self.agent_secret)
            return f"Team Deleted\n  Refunded: ${data.get('refundedCents', 0) / 100:.2f}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamFundTool(BaseTool):
    name: str = "dominusnode_team_fund"
    description: str = "Fund a team wallet. Input: team_id (UUID), amount_cents (100-1000000)."
    args_schema: Type[BaseModel] = TeamFundInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", amount_cents: int = 0, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not isinstance(amount_cents, int) or amount_cents < 100 or amount_cents > 1_000_000:
            return "Error: amount_cents must be between 100 and 1,000,000"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/wallet/fund", {"amountCents": amount_cents}, agent_secret=self.agent_secret)
            tx = data.get("transaction", data)
            return f"Team Funded\n  Amount: ${amount_cents / 100:.2f}\n  Transaction ID: {tx.get('id', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", amount_cents: int = 0, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not isinstance(amount_cents, int) or amount_cents < 100 or amount_cents > 1_000_000:
            return "Error: amount_cents must be between 100 and 1,000,000"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/wallet/fund", {"amountCents": amount_cents}, agent_secret=self.agent_secret)
            tx = data.get("transaction", data)
            return f"Team Funded\n  Amount: ${amount_cents / 100:.2f}\n  Transaction ID: {tx.get('id', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamCreateKeyTool(BaseTool):
    name: str = "dominusnode_team_create_key"
    description: str = "Create a shared team API key. Input: team_id (UUID), label (max 100 chars)."
    args_schema: Type[BaseModel] = TeamCreateKeyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", label: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        lerr = _validate_label(label)
        if lerr: return f"Error: {lerr}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/keys", {"label": label}, agent_secret=self.agent_secret)
            return f"Team API Key Created\n  Key: {data.get('key', '?')}\n  ID: {data.get('id', '?')}\n  Label: {data.get('label', label)}\n\nSave this key now -- it will not be shown again."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", label: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        lerr = _validate_label(label)
        if lerr: return f"Error: {lerr}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/keys", {"label": label}, agent_secret=self.agent_secret)
            return f"Team API Key Created\n  Key: {data.get('key', '?')}\n  ID: {data.get('id', '?')}\n  Label: {data.get('label', label)}\n\nSave this key now -- it will not be shown again."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamRevokeKeyTool(BaseTool):
    name: str = "dominusnode_team_revoke_key"
    description: str = "Revoke a team API key. Input: team_id (UUID), key_id (UUID)."
    args_schema: Type[BaseModel] = TeamRevokeKeyInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", key_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not key_id or not _UUID_RE.match(key_id): return "Error: key_id must be a valid UUID"
        try:
            _api_request_sync(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/keys/{quote(key_id, safe='')}", agent_secret=self.agent_secret)
            return f"Team API key {key_id} has been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", key_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not key_id or not _UUID_RE.match(key_id): return "Error: key_id must be a valid UUID"
        try:
            await _api_request_async(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/keys/{quote(key_id, safe='')}", agent_secret=self.agent_secret)
            return f"Team API key {key_id} has been revoked."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamListKeysTool(BaseTool):
    name: str = "dominusnode_team_list_keys"
    description: str = "List all API keys for a team. Input: team_id (UUID)."
    args_schema: Type[BaseModel] = TeamIdInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/keys", agent_secret=self.agent_secret)
            keys = data.get("keys", [])
            if not keys: return "No API keys found for this team."
            lines = [f"Team API Keys ({len(keys)}):"]
            for k in keys:
                lines.append(f"  {k.get('prefix', '?')}... | {k.get('label', '')} | ID: {k.get('id', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/keys", agent_secret=self.agent_secret)
            keys = data.get("keys", [])
            if not keys: return "No API keys found for this team."
            lines = [f"Team API Keys ({len(keys)}):"]
            for k in keys:
                lines.append(f"  {k.get('prefix', '?')}... | {k.get('label', '')} | ID: {k.get('id', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamUsageTool(BaseTool):
    name: str = "dominusnode_team_usage"
    description: str = "Get team wallet transactions. Input: team_id (UUID), optional limit (1-100)."
    args_schema: Type[BaseModel] = TeamUsageInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", limit: int = 20, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not isinstance(limit, int) or limit < 1 or limit > 100: return "Error: limit must be 1-100"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/wallet/transactions?limit={limit}", agent_secret=self.agent_secret)
            txns = data.get("transactions", [])
            if not txns: return "No transactions found for this team."
            lines = [f"Team Transactions ({len(txns)}):"]
            for tx in txns:
                sign = "+" if tx.get("type") in ("fund", "refund") else "-"
                lines.append(f"  {sign}${tx.get('amountCents', 0) / 100:.2f} [{tx.get('type', '?')}] {tx.get('description', '')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", limit: int = 20, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not isinstance(limit, int) or limit < 1 or limit > 100: return "Error: limit must be 1-100"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/wallet/transactions?limit={limit}", agent_secret=self.agent_secret)
            txns = data.get("transactions", [])
            if not txns: return "No transactions found for this team."
            lines = [f"Team Transactions ({len(txns)}):"]
            for tx in txns:
                sign = "+" if tx.get("type") in ("fund", "refund") else "-"
                lines.append(f"  {sign}${tx.get('amountCents', 0) / 100:.2f} [{tx.get('type', '?')}] {tx.get('description', '')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamListMembersTool(BaseTool):
    name: str = "dominusnode_team_list_members"
    description: str = "List all members of a team. Input: team_id (UUID)."
    args_schema: Type[BaseModel] = TeamIdInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/members", agent_secret=self.agent_secret)
            members = data.get("members", [])
            if not members: return "No members found."
            lines = [f"Team Members ({len(members)}):"]
            for m in members:
                lines.append(f"  {m.get('email', '?')} | Role: {m.get('role', '?')} | Joined: {m.get('joinedAt', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/members", agent_secret=self.agent_secret)
            members = data.get("members", [])
            if not members: return "No members found."
            lines = [f"Team Members ({len(members)}):"]
            for m in members:
                lines.append(f"  {m.get('email', '?')} | Role: {m.get('role', '?')} | Joined: {m.get('joinedAt', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamAddMemberTool(BaseTool):
    name: str = "dominusnode_team_add_member"
    description: str = "Add a member to a team by email. Input: team_id (UUID), email, optional role (member/admin)."
    args_schema: Type[BaseModel] = TeamAddMemberInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", email: str = "", role: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not email or not _EMAIL_RE.match(email): return "Error: valid email required"
        if role is not None and role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        body: dict = {"email": email}
        if role is not None: body["role"] = role
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/members", body, agent_secret=self.agent_secret)
            return f"Member added\n  User ID: {data.get('userId', '?')}\n  Role: {data.get('role', 'member')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", email: str = "", role: Optional[str] = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not email or not _EMAIL_RE.match(email): return "Error: valid email required"
        if role is not None and role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        body: dict = {"email": email}
        if role is not None: body["role"] = role
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/members", body, agent_secret=self.agent_secret)
            return f"Member added\n  User ID: {data.get('userId', '?')}\n  Role: {data.get('role', 'member')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamRemoveMemberTool(BaseTool):
    name: str = "dominusnode_team_remove_member"
    description: str = "Remove a member from a team. Input: team_id (UUID), user_id (UUID)."
    args_schema: Type[BaseModel] = TeamRemoveMemberInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", user_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not user_id or not _UUID_RE.match(user_id): return "Error: user_id must be a valid UUID"
        try:
            _api_request_sync(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/members/{quote(user_id, safe='')}", agent_secret=self.agent_secret)
            return f"Member {user_id} removed from team {team_id}."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", user_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not user_id or not _UUID_RE.match(user_id): return "Error: user_id must be a valid UUID"
        try:
            await _api_request_async(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/members/{quote(user_id, safe='')}", agent_secret=self.agent_secret)
            return f"Member {user_id} removed from team {team_id}."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeUpdateTeamMemberRoleTool(BaseTool):
    name: str = "dominusnode_update_team_member_role"
    description: str = "Change a team member's role. Input: team_id, user_id (UUIDs), role (member/admin)."
    args_schema: Type[BaseModel] = UpdateTeamMemberRoleInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", user_id: str = "", role: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not user_id or not _UUID_RE.match(user_id): return "Error: user_id must be a valid UUID"
        if role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "PATCH", f"/api/teams/{quote(team_id, safe='')}/members/{quote(user_id, safe='')}", {"role": role}, agent_secret=self.agent_secret)
            return f"Member {user_id} role updated to '{data.get('role', role)}'."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", user_id: str = "", role: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not user_id or not _UUID_RE.match(user_id): return "Error: user_id must be a valid UUID"
        if role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "PATCH", f"/api/teams/{quote(team_id, safe='')}/members/{quote(user_id, safe='')}", {"role": role}, agent_secret=self.agent_secret)
            return f"Member {user_id} role updated to '{data.get('role', role)}'."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamInviteMemberTool(BaseTool):
    name: str = "dominusnode_team_invite_member"
    description: str = "Send an email invite to join a team. Input: team_id (UUID), email, role (member/admin)."
    args_schema: Type[BaseModel] = TeamInviteMemberInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", email: str = "", role: str = "member", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not email or not _EMAIL_RE.match(email): return "Error: valid email required"
        if role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/invites", {"email": email, "role": role}, agent_secret=self.agent_secret)
            return f"Invite Sent\n  Invite ID: {data.get('id', '?')}\n  Email: {data.get('email', email)}\n  Role: {data.get('role', role)}\n  Expires: {data.get('expiresAt', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", email: str = "", role: str = "member", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not email or not _EMAIL_RE.match(email): return "Error: valid email required"
        if role not in _VALID_ROLES: return "Error: role must be 'member' or 'admin'"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "POST", f"/api/teams/{quote(team_id, safe='')}/invites", {"email": email, "role": role}, agent_secret=self.agent_secret)
            return f"Invite Sent\n  Invite ID: {data.get('id', '?')}\n  Email: {data.get('email', email)}\n  Role: {data.get('role', role)}\n  Expires: {data.get('expiresAt', '?')}"
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamListInvitesTool(BaseTool):
    name: str = "dominusnode_team_list_invites"
    description: str = "List pending invitations for a team. Input: team_id (UUID)."
    args_schema: Type[BaseModel] = TeamIdInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = _api_request_sync(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/invites", agent_secret=self.agent_secret)
            invites = data.get("invites", [])
            if not invites: return "No pending invites."
            lines = [f"Pending Invites ({len(invites)}):"]
            for inv in invites:
                lines.append(f"  {inv.get('email', '?')} -- {inv.get('role', '?')} | ID: {inv.get('id', '?')} | Expires: {inv.get('expiresAt', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        try:
            data = await _api_request_async(self.base_url, self.api_key, "GET", f"/api/teams/{quote(team_id, safe='')}/invites", agent_secret=self.agent_secret)
            invites = data.get("invites", [])
            if not invites: return "No pending invites."
            lines = [f"Pending Invites ({len(invites)}):"]
            for inv in invites:
                lines.append(f"  {inv.get('email', '?')} -- {inv.get('role', '?')} | ID: {inv.get('id', '?')} | Expires: {inv.get('expiresAt', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"


class DominusNodeTeamCancelInviteTool(BaseTool):
    name: str = "dominusnode_team_cancel_invite"
    description: str = "Cancel a pending team invitation. Input: team_id (UUID), invite_id (UUID)."
    args_schema: Type[BaseModel] = TeamCancelInviteInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    agent_secret: Optional[str] = None
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, team_id: str = "", invite_id: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not invite_id or not _UUID_RE.match(invite_id): return "Error: invite_id must be a valid UUID"
        try:
            _api_request_sync(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/invites/{quote(invite_id, safe='')}", agent_secret=self.agent_secret)
            return f"Invite {invite_id} cancelled."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"

    async def _arun(self, team_id: str = "", invite_id: str = "", run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        e = _team_tool_common(self.api_key, self.base_url)
        if e: return e
        err = _validate_team_id(team_id)
        if err: return f"Error: {err}"
        if not invite_id or not _UUID_RE.match(invite_id): return "Error: invite_id must be a valid UUID"
        try:
            await _api_request_async(self.base_url, self.api_key, "DELETE", f"/api/teams/{quote(team_id, safe='')}/invites/{quote(invite_id, safe='')}", agent_secret=self.agent_secret)
            return f"Invite {invite_id} cancelled."
        except Exception as exc:
            return f"Error: {_sanitize_error(str(exc))}"
