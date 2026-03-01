"""Tests for DomiNode LangChain tools.

All tests mock the DomiNode SDK client to avoid network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dominusnode.types import (
    GeoTargeting,
    ProxyConfig,
    ProxyEndpointConfig,
    UsagePagination,
    UsagePeriod,
    UsageResponse,
    UsageSummary,
    Wallet,
)

from dominusnode_langchain.tools import (
    DominusNodeAgenticTransactionsTool,
    DominusNodeAgenticWalletBalanceTool,
    DominusNodeBalanceTool,
    DominusNodeCreateAgenticWalletTool,
    DominusNodeDeleteAgenticWalletTool,
    DominusNodeFreezeAgenticWalletTool,
    DominusNodeFundAgenticWalletTool,
    DominusNodeListAgenticWalletsTool,
    DominusNodeProxiedFetchTool,
    DominusNodeProxyConfigTool,
    DominusNodeUnfreezeAgenticWalletTool,
    DominusNodeUpdateWalletPolicyTool,
    DominusNodeUsageTool,
    _is_private_ip,
    _validate_url,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_sync_client():
    """Create a mock DominusNodeClient with all needed resources."""
    client = MagicMock()

    # proxy.build_url returns a proxy URL string
    client.proxy.build_url.return_value = "http://user:dn_live_test@proxy.dominusnode.com:8080"

    # wallet.get_balance returns a Wallet dataclass
    client.wallet.get_balance.return_value = Wallet(
        balance_cents=5000,
        balance_usd=50.00,
        currency="usd",
        last_topped_up="2026-01-15T12:00:00Z",
    )

    # usage.get returns a UsageResponse
    client.usage.get.return_value = UsageResponse(
        summary=UsageSummary(
            total_bytes=1_073_741_824,
            total_cost_cents=300,
            request_count=150,
            total_gb=1.0,
            total_cost_usd=3.00,
        ),
        records=[],
        pagination=UsagePagination(limit=200, offset=0, total=150),
        period=UsagePeriod(since="2026-01-01", until="2026-02-01"),
    )

    # proxy.get_config returns a ProxyConfig
    client.proxy.get_config.return_value = ProxyConfig(
        http_proxy=ProxyEndpointConfig(host="proxy.dominusnode.com", port=8080),
        socks5_proxy=ProxyEndpointConfig(host="proxy.dominusnode.com", port=1080),
        supported_countries=["US", "GB", "DE", "JP"],
        blocked_countries=["CU", "IR", "KP", "RU", "SY"],
        max_rotation_interval_minutes=60,
        min_rotation_interval_minutes=1,
        geo_targeting=GeoTargeting(
            state_support=True,
            city_support=True,
            asn_support=True,
            us_states=["CA", "NY", "TX"],
            major_us_cities=["Los Angeles", "New York", "Houston"],
        ),
    )

    return client


@pytest.fixture()
def mock_async_client():
    """Create a mock AsyncDominusNodeClient."""
    client = MagicMock()

    client.proxy.build_url.return_value = "http://user:dn_live_test@proxy.dominusnode.com:8080"

    client.wallet.get_balance = AsyncMock(
        return_value=Wallet(
            balance_cents=5000,
            balance_usd=50.00,
            currency="usd",
            last_topped_up="2026-01-15T12:00:00Z",
        )
    )

    client.usage.get = AsyncMock(
        return_value=UsageResponse(
            summary=UsageSummary(
                total_bytes=1_073_741_824,
                total_cost_cents=300,
                request_count=150,
                total_gb=1.0,
                total_cost_usd=3.00,
            ),
            records=[],
            pagination=UsagePagination(limit=200, offset=0, total=150),
            period=UsagePeriod(since="2026-01-01", until="2026-02-01"),
        )
    )

    client.proxy.get_config = AsyncMock(
        return_value=ProxyConfig(
            http_proxy=ProxyEndpointConfig(host="proxy.dominusnode.com", port=8080),
            socks5_proxy=ProxyEndpointConfig(host="proxy.dominusnode.com", port=1080),
            supported_countries=["US", "GB", "DE", "JP"],
            blocked_countries=["CU", "IR", "KP", "RU", "SY"],
            max_rotation_interval_minutes=60,
            min_rotation_interval_minutes=1,
            geo_targeting=GeoTargeting(
                state_support=True,
                city_support=True,
                asn_support=True,
                us_states=["CA", "NY", "TX"],
                major_us_cities=["Los Angeles", "New York", "Houston"],
            ),
        )
    )

    return client


# ──────────────────────────────────────────────────────────────────────
# URL Validation / SSRF Prevention
# ──────────────────────────────────────────────────────────────────────


class TestUrlValidation:
    """Tests for SSRF prevention in _validate_url."""

    def test_valid_http_url(self):
        assert _validate_url("http://example.com") == "http://example.com"

    def test_valid_https_url(self):
        assert _validate_url("https://example.com/path?q=1") == "https://example.com/path?q=1"

    def test_rejects_empty_url(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_url("")

    def test_rejects_none_url(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_url(None)  # type: ignore[arg-type]

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="scheme.*not allowed"):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="scheme.*not allowed"):
            _validate_url("ftp://example.com/file")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(ValueError, match="scheme.*not allowed"):
            _validate_url("javascript:alert(1)")

    def test_rejects_localhost(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://localhost/admin")

    def test_rejects_127_0_0_1(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://127.0.0.1/admin")

    def test_rejects_10_x_private(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://10.0.0.1/internal")

    def test_rejects_172_16_private(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://172.16.0.1/internal")

    def test_rejects_192_168_private(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://192.168.1.1/router")

    def test_rejects_ipv6_loopback(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://[::1]/admin")

    def test_rejects_ipv6_private_fd(self):
        with pytest.raises(ValueError, match="private.*reserved.*loopback"):
            _validate_url("http://[fd00::1]/internal")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            _validate_url("http://user:pass@example.com")

    def test_rejects_long_url(self):
        with pytest.raises(ValueError, match="maximum length"):
            _validate_url("http://example.com/" + "a" * 2048)

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="valid hostname"):
            _validate_url("http://")

    def test_rejects_hex_encoded_loopback(self):
        # 0x7f000001 = 127.0.0.1
        assert _is_private_ip("0x7f000001") is True

    def test_rejects_decimal_encoded_loopback(self):
        # 2130706433 = 127.0.0.1
        assert _is_private_ip("2130706433") is True

    def test_allows_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_allows_public_hostname(self):
        assert _is_private_ip("example.com") is False


# ──────────────────────────────────────────────────────────────────────
# DominusNodeProxiedFetchTool
# ──────────────────────────────────────────────────────────────────────


class TestProxiedFetchTool:
    """Tests for the proxied HTTP fetch tool."""

    def test_ssrf_blocked(self, mock_sync_client):
        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="http://127.0.0.1/admin")
        assert "Error" in result
        assert "private" in result.lower() or "reserved" in result.lower()

    def test_disallowed_method(self, mock_sync_client):
        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="http://example.com", method="POST")
        assert "Error" in result
        assert "not allowed" in result

    def test_invalid_proxy_type(self, mock_sync_client):
        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="http://example.com", proxy_type="mobile")
        assert "Error" in result
        assert "dc" in result or "residential" in result

    def test_no_client_configured(self):
        tool = DominusNodeProxiedFetchTool()
        result = tool._run(url="http://example.com")
        assert "Error" in result
        assert "No DomiNode client" in result

    @patch("dominusnode_langchain.tools.httpx.Client")
    def test_successful_fetch(self, mock_httpx_client_cls, mock_sync_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Hello, World!"
        mock_response.headers = {"content-type": "text/html"}

        mock_httpx_instance = MagicMock()
        mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx_instance.request.return_value = mock_response
        mock_httpx_client_cls.return_value = mock_httpx_instance

        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="https://example.com")

        assert "Status: 200" in result
        assert "Hello, World!" in result
        mock_sync_client.proxy.build_url.assert_called_once()

    @patch("dominusnode_langchain.tools.httpx.Client")
    def test_response_truncation(self, mock_httpx_client_cls, mock_sync_client):
        long_body = "x" * 5000
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = long_body
        mock_response.headers = {"content-type": "text/plain"}

        mock_httpx_instance = MagicMock()
        mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx_instance.request.return_value = mock_response
        mock_httpx_client_cls.return_value = mock_httpx_instance

        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="https://example.com")

        assert "[truncated]" in result
        # Body portion should be at most MAX_RESPONSE_CHARS (4000)
        body_line_start = result.index("Body:\n") + len("Body:\n")
        body_content = result[body_line_start:]
        # Account for " [truncated]" suffix
        assert len(body_content) <= 4000 + len(" [truncated]")

    def test_file_scheme_blocked(self, mock_sync_client):
        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        result = tool._run(url="file:///etc/passwd")
        assert "Error" in result
        assert "scheme" in result.lower()

    def test_head_method_allowed(self, mock_sync_client):
        """HEAD is an allowed read-only method."""
        # This test just verifies it does not return a method-not-allowed error;
        # the actual HTTP call would fail without a real proxy.
        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        # Will fail on the httpx call (no real proxy), but should NOT fail on method validation
        result = tool._run(url="https://example.com", method="HEAD")
        # Should not contain the "method not allowed" error
        assert "not allowed" not in result or "HTTP method" not in result

    @patch("dominusnode_langchain.tools.httpx.Client")
    def test_country_targeting(self, mock_httpx_client_cls, mock_sync_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.headers = {"content-type": "text/plain"}

        mock_httpx_instance = MagicMock()
        mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__exit__ = MagicMock(return_value=False)
        mock_httpx_instance.request.return_value = mock_response
        mock_httpx_client_cls.return_value = mock_httpx_instance

        tool = DominusNodeProxiedFetchTool(sync_client=mock_sync_client)
        tool._run(url="https://example.com", country="DE")

        # Verify build_url was called with a ProxyUrlOptions containing country
        call_args = mock_sync_client.proxy.build_url.call_args
        options = call_args[0][0]
        assert options.country == "DE"


# ──────────────────────────────────────────────────────────────────────
# DominusNodeBalanceTool
# ──────────────────────────────────────────────────────────────────────


class TestBalanceTool:
    """Tests for the balance check tool."""

    def test_returns_balance(self, mock_sync_client):
        tool = DominusNodeBalanceTool(sync_client=mock_sync_client)
        result = tool._run()

        assert "$50.00" in result
        assert "5000 cents" in result
        assert "usd" in result
        mock_sync_client.wallet.get_balance.assert_called_once()

    def test_no_client(self):
        tool = DominusNodeBalanceTool()
        result = tool._run()
        assert "Error" in result
        assert "No DomiNode client" in result

    def test_sdk_error_handled(self, mock_sync_client):
        mock_sync_client.wallet.get_balance.side_effect = Exception("Connection refused")
        tool = DominusNodeBalanceTool(sync_client=mock_sync_client)
        result = tool._run()
        assert "Error" in result
        assert "Connection refused" in result

    @pytest.mark.asyncio
    async def test_async_returns_balance(self, mock_async_client):
        tool = DominusNodeBalanceTool(async_client=mock_async_client)
        result = await tool._arun()

        assert "$50.00" in result
        assert "5000 cents" in result
        mock_async_client.wallet.get_balance.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# DominusNodeUsageTool
# ──────────────────────────────────────────────────────────────────────


class TestUsageTool:
    """Tests for the usage statistics tool."""

    def test_returns_usage(self, mock_sync_client):
        tool = DominusNodeUsageTool(sync_client=mock_sync_client)
        result = tool._run()

        assert "1.0000 GB" in result
        assert "$3.00" in result
        assert "150" in result
        assert "2026-01-01" in result
        mock_sync_client.usage.get.assert_called_once()

    def test_no_client(self):
        tool = DominusNodeUsageTool()
        result = tool._run()
        assert "Error" in result

    def test_sdk_error_handled(self, mock_sync_client):
        mock_sync_client.usage.get.side_effect = Exception("Timeout")
        tool = DominusNodeUsageTool(sync_client=mock_sync_client)
        result = tool._run()
        assert "Error" in result
        assert "Timeout" in result

    @pytest.mark.asyncio
    async def test_async_returns_usage(self, mock_async_client):
        tool = DominusNodeUsageTool(async_client=mock_async_client)
        result = await tool._arun()

        assert "1.0000 GB" in result
        assert "$3.00" in result
        mock_async_client.usage.get.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# DominusNodeProxyConfigTool
# ──────────────────────────────────────────────────────────────────────


class TestProxyConfigTool:
    """Tests for the proxy configuration tool."""

    def test_returns_config(self, mock_sync_client):
        tool = DominusNodeProxyConfigTool(sync_client=mock_sync_client)
        result = tool._run()

        assert "proxy.dominusnode.com:8080" in result
        assert "proxy.dominusnode.com:1080" in result
        assert "US" in result
        assert "GB" in result
        assert "DE" in result
        assert "State targeting: yes" in result
        assert "City targeting: yes" in result
        assert "CU" in result  # blocked country
        mock_sync_client.proxy.get_config.assert_called_once()

    def test_no_client(self):
        tool = DominusNodeProxyConfigTool()
        result = tool._run()
        assert "Error" in result

    def test_sdk_error_handled(self, mock_sync_client):
        mock_sync_client.proxy.get_config.side_effect = Exception("Auth expired")
        tool = DominusNodeProxyConfigTool(sync_client=mock_sync_client)
        result = tool._run()
        assert "Error" in result
        assert "Auth expired" in result

    @pytest.mark.asyncio
    async def test_async_returns_config(self, mock_async_client):
        tool = DominusNodeProxyConfigTool(async_client=mock_async_client)
        result = await tool._arun()

        assert "proxy.dominusnode.com:8080" in result
        assert "US" in result
        mock_async_client.proxy.get_config.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# Async ProxiedFetchTool
# ──────────────────────────────────────────────────────────────────────


class TestProxiedFetchToolAsync:
    """Async-specific tests for the proxied fetch tool."""

    @pytest.mark.asyncio
    async def test_ssrf_blocked_async(self, mock_async_client):
        tool = DominusNodeProxiedFetchTool(async_client=mock_async_client)
        result = await tool._arun(url="http://10.0.0.1/secret")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_no_client_async(self):
        tool = DominusNodeProxiedFetchTool()
        result = await tool._arun(url="http://example.com")
        assert "Error" in result
        assert "No DomiNode client" in result

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools.httpx.AsyncClient")
    async def test_successful_async_fetch(self, mock_httpx_cls, mock_async_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Async OK"
        mock_response.headers = {"content-type": "text/html"}

        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.__aenter__ = AsyncMock(return_value=mock_httpx_instance)
        mock_httpx_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_instance.request = AsyncMock(return_value=mock_response)
        mock_httpx_cls.return_value = mock_httpx_instance

        tool = DominusNodeProxiedFetchTool(async_client=mock_async_client)
        result = await tool._arun(url="https://example.com")

        assert "Status: 200" in result
        assert "Async OK" in result

    @pytest.mark.asyncio
    async def test_disallowed_method_async(self, mock_async_client):
        tool = DominusNodeProxiedFetchTool(async_client=mock_async_client)
        result = await tool._arun(url="http://example.com", method="DELETE")
        assert "Error" in result
        assert "not allowed" in result


# ──────────────────────────────────────────────────────────────────────
# Agentic Wallet Tools -- helpers
# ──────────────────────────────────────────────────────────────────────

_VALID_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_INVALID_UUID = "not-a-uuid"
_API_KEY = "dn_live_testkey123"
_BASE_URL = "http://localhost:3000"


def _make_tool(cls, **kwargs):
    """Instantiate an agentic wallet tool with default credentials."""
    return cls(api_key=_API_KEY, base_url=_BASE_URL, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# TestAgenticWalletValidation
# ──────────────────────────────────────────────────────────────────────


class TestAgenticWalletValidation:
    """Input validation tests for all 9 agentic wallet tools."""

    # -- CreateAgenticWalletTool --

    def test_create_empty_label(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="", spending_limit_cents=100)
        assert "Error" in result
        assert "label" in result.lower()

    def test_create_label_too_long(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="x" * 101, spending_limit_cents=100)
        assert "Error" in result
        assert "100 characters" in result

    def test_create_label_control_chars(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="bad\x00label", spending_limit_cents=100)
        assert "Error" in result
        assert "control characters" in result

    def test_create_spending_limit_zero(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=0)
        assert "Error" in result
        assert "spending_limit_cents" in result

    def test_create_spending_limit_negative(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=-5)
        assert "Error" in result
        assert "spending_limit_cents" in result

    def test_create_spending_limit_overflow(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=2_147_483_648)
        assert "Error" in result
        assert "2,147,483,647" in result

    def test_create_spending_limit_bool_rejected(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=True)
        assert "Error" in result
        assert "integer" in result.lower()

    def test_create_daily_limit_zero(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=100, daily_limit_cents=0)
        assert "Error" in result
        assert "daily_limit_cents" in result

    def test_create_daily_limit_too_high(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=100, daily_limit_cents=1_000_001)
        assert "Error" in result
        assert "1,000,000" in result

    def test_create_invalid_domains(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=100, allowed_domains=["bad domain!"])
        assert "Error" in result
        assert "domain" in result.lower()

    def test_create_domains_too_many(self):
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=100, allowed_domains=["a.com"] * 101)
        assert "Error" in result
        assert "100" in result

    def test_create_no_credentials(self):
        tool = DominusNodeCreateAgenticWalletTool()
        result = tool._run(label="test", spending_limit_cents=100)
        assert "Error" in result
        assert "credentials" in result.lower() or "configured" in result.lower()

    # -- FundAgenticWalletTool --

    def test_fund_invalid_wallet_id(self):
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id=_INVALID_UUID, amount_cents=100)
        assert "Error" in result
        assert "UUID" in result

    def test_fund_empty_wallet_id(self):
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id="", amount_cents=100)
        assert "Error" in result
        assert "wallet_id" in result

    def test_fund_amount_zero(self):
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID, amount_cents=0)
        assert "Error" in result
        assert "amount_cents" in result

    def test_fund_amount_overflow(self):
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID, amount_cents=2_147_483_648)
        assert "Error" in result
        assert "2,147,483,647" in result

    # -- AgenticWalletBalanceTool --

    def test_balance_invalid_wallet_id(self):
        tool = _make_tool(DominusNodeAgenticWalletBalanceTool)
        result = tool._run(wallet_id=_INVALID_UUID)
        assert "Error" in result
        assert "UUID" in result

    # -- AgenticTransactionsTool --

    def test_transactions_invalid_wallet_id(self):
        tool = _make_tool(DominusNodeAgenticTransactionsTool)
        result = tool._run(wallet_id=_INVALID_UUID)
        assert "Error" in result
        assert "UUID" in result

    def test_transactions_limit_zero(self):
        tool = _make_tool(DominusNodeAgenticTransactionsTool)
        result = tool._run(wallet_id=_VALID_UUID, limit=0)
        assert "Error" in result
        assert "limit" in result.lower()

    def test_transactions_limit_too_high(self):
        tool = _make_tool(DominusNodeAgenticTransactionsTool)
        result = tool._run(wallet_id=_VALID_UUID, limit=101)
        assert "Error" in result
        assert "100" in result

    # -- UpdateWalletPolicyTool --

    def test_update_policy_no_fields(self):
        tool = _make_tool(DominusNodeUpdateWalletPolicyTool)
        result = tool._run(wallet_id=_VALID_UUID)
        assert "Error" in result
        assert "At least one" in result


# ──────────────────────────────────────────────────────────────────────
# TestAgenticWalletSync
# ──────────────────────────────────────────────────────────────────────


class TestAgenticWalletSync:
    """Mocked API success cases for synchronous _run() calls."""

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_create_wallet_success(self, mock_req):
        mock_req.return_value = {
            "wallet": {
                "id": _VALID_UUID,
                "label": "Agent Alpha",
                "balanceCents": 0,
                "spendingLimitCents": 500,
                "status": "active",
            }
        }
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="Agent Alpha", spending_limit_cents=500)
        assert "Agentic Wallet Created" in result
        assert _VALID_UUID in result
        assert "Agent Alpha" in result
        assert "500" in result
        mock_req.assert_called_once()

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_fund_wallet_success(self, mock_req):
        mock_req.return_value = {"wallet": {"balanceCents": 1500}}
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID, amount_cents=1000)
        assert "Agentic Wallet Funded" in result
        assert "1000 cents" in result
        assert "$10.00" in result
        assert "1500" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_wallet_balance_success(self, mock_req):
        mock_req.return_value = {
            "wallet": {
                "id": _VALID_UUID,
                "label": "Test",
                "balanceCents": 2500,
                "spendingLimitCents": 1000,
                "status": "active",
            }
        }
        tool = _make_tool(DominusNodeAgenticWalletBalanceTool)
        result = tool._run(wallet_id=_VALID_UUID)
        assert "Agentic Wallet Details" in result
        assert "2500 cents" in result
        assert "$25.00" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_list_wallets_success(self, mock_req):
        mock_req.return_value = {
            "wallets": [
                {"id": _VALID_UUID, "label": "W1", "balanceCents": 100, "spendingLimitCents": 50, "status": "active"},
                {"id": "b2c3d4e5-f6a7-8901-bcde-f12345678901", "label": "W2", "balanceCents": 200, "spendingLimitCents": 100, "status": "frozen"},
            ]
        }
        tool = _make_tool(DominusNodeListAgenticWalletsTool)
        result = tool._run()
        assert "Agentic Wallets (2)" in result
        assert "W1" in result
        assert "W2" in result
        assert "frozen" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_list_wallets_empty(self, mock_req):
        mock_req.return_value = {"wallets": []}
        tool = _make_tool(DominusNodeListAgenticWalletsTool)
        result = tool._run()
        assert "No agentic wallets found" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_transactions_success(self, mock_req):
        mock_req.return_value = {
            "transactions": [
                {"type": "fund", "amountCents": 500, "description": "Initial funding", "createdAt": "2026-01-15T12:00:00Z"},
                {"type": "spend", "amountCents": -50, "description": "Proxy request", "createdAt": "2026-01-15T13:00:00Z"},
            ]
        }
        tool = _make_tool(DominusNodeAgenticTransactionsTool)
        result = tool._run(wallet_id=_VALID_UUID, limit=10)
        assert "Transactions for" in result
        assert "fund" in result
        assert "spend" in result
        assert "Initial funding" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_freeze_wallet_success(self, mock_req):
        mock_req.return_value = {"wallet": {"id": _VALID_UUID, "status": "frozen"}}
        tool = _make_tool(DominusNodeFreezeAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID)
        assert "Agentic Wallet Frozen" in result
        assert "frozen" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_unfreeze_wallet_success(self, mock_req):
        mock_req.return_value = {"wallet": {"id": _VALID_UUID, "status": "active"}}
        tool = _make_tool(DominusNodeUnfreezeAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID)
        assert "Agentic Wallet Unfrozen" in result
        assert "active" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_delete_wallet_success(self, mock_req):
        mock_req.return_value = {"refundedCents": 750}
        tool = _make_tool(DominusNodeDeleteAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID)
        assert "Agentic Wallet Deleted" in result
        assert "750" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_update_policy_success(self, mock_req):
        mock_req.return_value = {
            "wallet": {
                "id": _VALID_UUID,
                "dailyLimitCents": 5000,
                "status": "active",
            }
        }
        tool = _make_tool(DominusNodeUpdateWalletPolicyTool)
        result = tool._run(wallet_id=_VALID_UUID, daily_limit_cents=5000)
        assert "Agentic Wallet Policy Updated" in result
        assert "5000" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_create_wallet_with_domains(self, mock_req):
        mock_req.return_value = {
            "wallet": {
                "id": _VALID_UUID,
                "label": "Restricted",
                "balanceCents": 0,
                "spendingLimitCents": 100,
                "status": "active",
            }
        }
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(
            label="Restricted",
            spending_limit_cents=100,
            daily_limit_cents=500,
            allowed_domains=["api.example.com", "data.example.com"],
        )
        assert "Agentic Wallet Created" in result
        call_body = mock_req.call_args[0][4]  # body argument
        assert call_body["dailyLimitCents"] == 500
        assert call_body["allowedDomains"] == ["api.example.com", "data.example.com"]


# ──────────────────────────────────────────────────────────────────────
# TestAgenticWalletAsync
# ──────────────────────────────────────────────────────────────────────


class TestAgenticWalletAsync:
    """Mocked async _arun() tests."""

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools._api_request_async")
    async def test_async_create_wallet(self, mock_req):
        mock_req.return_value = {
            "wallet": {
                "id": _VALID_UUID,
                "label": "Async Agent",
                "balanceCents": 0,
                "spendingLimitCents": 200,
                "status": "active",
            }
        }
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = await tool._arun(label="Async Agent", spending_limit_cents=200)
        assert "Agentic Wallet Created" in result
        assert "Async Agent" in result
        mock_req.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools._api_request_async")
    async def test_async_list_wallets(self, mock_req):
        mock_req.return_value = {
            "wallets": [
                {"id": _VALID_UUID, "label": "AW1", "balanceCents": 300, "spendingLimitCents": 100, "status": "active"},
            ]
        }
        tool = _make_tool(DominusNodeListAgenticWalletsTool)
        result = await tool._arun()
        assert "Agentic Wallets (1)" in result
        assert "AW1" in result
        mock_req.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools._api_request_async")
    async def test_async_fund_wallet(self, mock_req):
        mock_req.return_value = {"wallet": {"balanceCents": 600}}
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = await tool._arun(wallet_id=_VALID_UUID, amount_cents=600)
        assert "Agentic Wallet Funded" in result
        assert "600" in result
        mock_req.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools._api_request_async")
    async def test_async_freeze_wallet(self, mock_req):
        mock_req.return_value = {"wallet": {"id": _VALID_UUID, "status": "frozen"}}
        tool = _make_tool(DominusNodeFreezeAgenticWalletTool)
        result = await tool._arun(wallet_id=_VALID_UUID)
        assert "Frozen" in result
        mock_req.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("dominusnode_langchain.tools._api_request_async")
    async def test_async_delete_wallet(self, mock_req):
        mock_req.return_value = {"refundedCents": 100}
        tool = _make_tool(DominusNodeDeleteAgenticWalletTool)
        result = await tool._arun(wallet_id=_VALID_UUID)
        assert "Deleted" in result
        assert "100" in result
        mock_req.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# TestAgenticWalletErrors
# ──────────────────────────────────────────────────────────────────────


class TestAgenticWalletErrors:
    """Error handling and credential scrubbing tests."""

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_api_error_handled(self, mock_req):
        mock_req.side_effect = RuntimeError("API error 403: Forbidden")
        tool = _make_tool(DominusNodeCreateAgenticWalletTool)
        result = tool._run(label="test", spending_limit_cents=100)
        assert "Error" in result
        assert "403" in result

    @patch("dominusnode_langchain.tools._api_request_sync")
    def test_credential_scrubbing(self, mock_req):
        mock_req.side_effect = RuntimeError(
            "API error 401: invalid key dn_live_secret123abc"
        )
        tool = _make_tool(DominusNodeFundAgenticWalletTool)
        result = tool._run(wallet_id=_VALID_UUID, amount_cents=100)
        assert "Error" in result
        assert "dn_live_secret123abc" not in result
        assert "***" in result

    @patch("dominusnode_langchain.tools._api_request_async")
    @pytest.mark.asyncio
    async def test_async_credential_scrubbing(self, mock_req):
        mock_req.side_effect = RuntimeError(
            "API error 401: dn_test_secretXYZ leaked"
        )
        tool = _make_tool(DominusNodeAgenticWalletBalanceTool)
        result = await tool._arun(wallet_id=_VALID_UUID)
        assert "Error" in result
        assert "dn_test_secretXYZ" not in result
        assert "***" in result
