"""Dominus Node LangChain Toolkit -- bundles all 53 Dominus Node tools for agent use.

The :class:`DominusNodeToolkit` injects API credentials into each tool so that
a single authenticated configuration is shared across all tools in a LangChain
agent.  All tools use direct REST calls (no SDK dependency).

Example::

    from dominusnode_langchain import DominusNodeToolkit

    toolkit = DominusNodeToolkit(
        api_key="dn_live_abc123",
        base_url="http://localhost:3000",
    )
    tools = toolkit.get_tools()
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.tools import BaseTool, BaseToolkit

from .tools import (
    # Proxy (3)
    DominusNodeProxiedFetchTool,
    DominusNodeProxyConfigTool,
    DominusNodeProxyStatusTool,
    # Sessions (1)
    DominusNodeListSessionsTool,
    # Wallet (8)
    DominusNodeBalanceTool,
    DominusNodeGetTransactionsTool,
    DominusNodeGetForecastTool,
    DominusNodeTopupPaypalTool,
    DominusNodeTopupStripeTool,
    DominusNodeTopupCryptoTool,
    DominusNodeCheckPaymentTool,
    DominusNodeX402InfoTool,
    # Usage (3)
    DominusNodeUsageTool,
    DominusNodeDailyUsageTool,
    DominusNodeTopHostsTool,
    # Account (6)
    DominusNodeRegisterTool,
    DominusNodeLoginTool,
    DominusNodeGetAccountInfoTool,
    DominusNodeVerifyEmailTool,
    DominusNodeResendVerificationTool,
    DominusNodeUpdatePasswordTool,
    # API Keys (3)
    DominusNodeListKeysTool,
    DominusNodeCreateKeyTool,
    DominusNodeRevokeKeyTool,
    # Plans (3)
    DominusNodeGetPlanTool,
    DominusNodeListPlansTool,
    DominusNodeChangePlanTool,
    # Agentic Wallets (9)
    DominusNodeCreateAgenticWalletTool,
    DominusNodeFundAgenticWalletTool,
    DominusNodeAgenticWalletBalanceTool,
    DominusNodeListAgenticWalletsTool,
    DominusNodeAgenticTransactionsTool,
    DominusNodeFreezeAgenticWalletTool,
    DominusNodeUnfreezeAgenticWalletTool,
    DominusNodeDeleteAgenticWalletTool,
    DominusNodeUpdateWalletPolicyTool,
    # Teams (17)
    DominusNodeCreateTeamTool,
    DominusNodeListTeamsTool,
    DominusNodeTeamDetailsTool,
    DominusNodeUpdateTeamTool,
    DominusNodeTeamDeleteTool,
    DominusNodeTeamFundTool,
    DominusNodeTeamCreateKeyTool,
    DominusNodeTeamRevokeKeyTool,
    DominusNodeTeamListKeysTool,
    DominusNodeTeamUsageTool,
    DominusNodeTeamListMembersTool,
    DominusNodeTeamAddMemberTool,
    DominusNodeTeamRemoveMemberTool,
    DominusNodeUpdateTeamMemberRoleTool,
    DominusNodeTeamInviteMemberTool,
    DominusNodeTeamListInvitesTool,
    DominusNodeTeamCancelInviteTool,
)


class DominusNodeToolkit(BaseToolkit):
    """LangChain toolkit that provides all 53 Dominus Node proxy service tools.

    Injects API key and base URL into each tool for direct REST API calls.

    Args:
        api_key: Dominus Node API key (``dn_live_...``).  Falls back to the
            ``DOMINUSNODE_API_KEY`` environment variable if not provided.
        base_url: Base URL for the Dominus Node REST API.  Falls back to
            ``DOMINUSNODE_BASE_URL`` or the SDK default.
        proxy_host: Hostname of the Dominus Node proxy server.  Falls back to
            ``DOMINUSNODE_PROXY_HOST`` or the SDK default.
        http_proxy_port: HTTP proxy port (default 8080).
        socks5_proxy_port: SOCKS5 proxy port (default 1080).

    Raises:
        ValueError: If no API key is provided or found in environment.
    """

    _tools: Optional[List[BaseTool]] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        proxy_host: Optional[str] = None,
        http_proxy_port: int = 8080,
        socks5_proxy_port: int = 1080,
        agent_secret: Optional[str] = None,
    ) -> None:
        super().__init__()

        resolved_key = api_key or os.environ.get("DOMINUSNODE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Dominus Node API key is required.  Pass api_key= or set the "
                "DOMINUSNODE_API_KEY environment variable."
            )

        resolved_base = base_url or os.environ.get(
            "DOMINUSNODE_BASE_URL", "https://api.dominusnode.com"
        )
        resolved_proxy_host = proxy_host or os.environ.get(
            "DOMINUSNODE_PROXY_HOST", "proxy.dominusnode.com"
        )

        self._api_key = resolved_key
        self._base_url = resolved_base
        self._agent_secret = agent_secret or os.environ.get("DOMINUSNODE_AGENT_SECRET")

        # Store proxy host/port for env-based tools
        os.environ.setdefault("DOMINUSNODE_PROXY_HOST", resolved_proxy_host)
        os.environ.setdefault("DOMINUSNODE_PROXY_PORT", str(http_proxy_port))

        # Build tools once
        self._tools = self._build_tools()

    def _build_tools(self) -> List[BaseTool]:
        """Construct all 53 tool instances with injected credentials."""
        _ak = self._api_key
        _bu = self._base_url
        _as = self._agent_secret
        _ph = os.environ.get("DOMINUSNODE_PROXY_HOST", "localhost")
        _pp = int(os.environ.get("DOMINUSNODE_PROXY_PORT", "8080"))

        # Helper for REST-API-based tools (authenticated)
        def _rest(cls: type) -> BaseTool:
            return cls(api_key=_ak, base_url=_bu, agent_secret=_as)

        # Helper for proxy tools that need proxy host/port
        def _proxy(cls: type) -> BaseTool:
            return cls(api_key=_ak, base_url=_bu, agent_secret=_as,
                       proxy_host=_ph, proxy_port=_pp)

        # Helper for unauthenticated tools
        def _unauth(cls: type) -> BaseTool:
            return cls(base_url=_bu, agent_secret=_as)

        return [
            # Proxy (3)
            _proxy(DominusNodeProxiedFetchTool),
            _rest(DominusNodeProxyConfigTool),
            _rest(DominusNodeProxyStatusTool),
            # Sessions (1)
            _rest(DominusNodeListSessionsTool),
            # Wallet (8)
            _rest(DominusNodeBalanceTool),
            _rest(DominusNodeGetTransactionsTool),
            _rest(DominusNodeGetForecastTool),
            _rest(DominusNodeTopupPaypalTool),
            _rest(DominusNodeTopupStripeTool),
            _rest(DominusNodeTopupCryptoTool),
            _rest(DominusNodeCheckPaymentTool),
            _rest(DominusNodeX402InfoTool),
            # Usage (3)
            _rest(DominusNodeUsageTool),
            _rest(DominusNodeDailyUsageTool),
            _rest(DominusNodeTopHostsTool),
            # Account (6) -- register, login, verify_email are unauthenticated
            _unauth(DominusNodeRegisterTool),
            _unauth(DominusNodeLoginTool),
            _rest(DominusNodeGetAccountInfoTool),
            _unauth(DominusNodeVerifyEmailTool),
            _rest(DominusNodeResendVerificationTool),
            _rest(DominusNodeUpdatePasswordTool),
            # API Keys (3)
            _rest(DominusNodeListKeysTool),
            _rest(DominusNodeCreateKeyTool),
            _rest(DominusNodeRevokeKeyTool),
            # Plans (3)
            _rest(DominusNodeGetPlanTool),
            _rest(DominusNodeListPlansTool),
            _rest(DominusNodeChangePlanTool),
            # Agentic Wallets (9)
            _rest(DominusNodeCreateAgenticWalletTool),
            _rest(DominusNodeFundAgenticWalletTool),
            _rest(DominusNodeAgenticWalletBalanceTool),
            _rest(DominusNodeListAgenticWalletsTool),
            _rest(DominusNodeAgenticTransactionsTool),
            _rest(DominusNodeFreezeAgenticWalletTool),
            _rest(DominusNodeUnfreezeAgenticWalletTool),
            _rest(DominusNodeDeleteAgenticWalletTool),
            _rest(DominusNodeUpdateWalletPolicyTool),
            # Teams (17)
            _rest(DominusNodeCreateTeamTool),
            _rest(DominusNodeListTeamsTool),
            _rest(DominusNodeTeamDetailsTool),
            _rest(DominusNodeUpdateTeamTool),
            _rest(DominusNodeTeamDeleteTool),
            _rest(DominusNodeTeamFundTool),
            _rest(DominusNodeTeamCreateKeyTool),
            _rest(DominusNodeTeamRevokeKeyTool),
            _rest(DominusNodeTeamListKeysTool),
            _rest(DominusNodeTeamUsageTool),
            _rest(DominusNodeTeamListMembersTool),
            _rest(DominusNodeTeamAddMemberTool),
            _rest(DominusNodeTeamRemoveMemberTool),
            _rest(DominusNodeUpdateTeamMemberRoleTool),
            _rest(DominusNodeTeamInviteMemberTool),
            _rest(DominusNodeTeamListInvitesTool),
            _rest(DominusNodeTeamCancelInviteTool),
        ]

    def get_tools(self) -> List[BaseTool]:
        """Return the list of 53 Dominus Node LangChain tools."""
        if self._tools is None:
            self._tools = self._build_tools()
        return list(self._tools)

    def close(self) -> None:
        """Release resources."""
        self._tools = None
