"""DomiNode LangChain Toolkit -- bundles all DomiNode tools for agent use.

The :class:`DominusNodeToolkit` creates a :class:`DominusNodeClient` internally
and injects it into each tool so that a single authenticated session is shared
across all tools in a LangChain agent.

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

from dominusnode import AsyncDominusNodeClient, DominusNodeClient

from .tools import (
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
    DominusNodeTopupPaypalTool,
    DominusNodeUnfreezeAgenticWalletTool,
    DominusNodeUpdateWalletPolicyTool,
    DominusNodeUsageTool,
    DominusNodeX402InfoTool,
)


class DominusNodeToolkit(BaseToolkit):
    """LangChain toolkit that provides all DomiNode proxy service tools.

    Creates and manages a :class:`DominusNodeClient` (sync) and optionally
    an :class:`AsyncDominusNodeClient` (async) for use by LangChain agents.

    Args:
        api_key: DomiNode API key (``dn_live_...``).  Falls back to the
            ``DOMINUSNODE_API_KEY`` environment variable if not provided.
        base_url: Base URL for the DomiNode REST API.  Falls back to
            ``DOMINUSNODE_BASE_URL`` or the SDK default.
        proxy_host: Hostname of the DomiNode proxy server.  Falls back to
            ``DOMINUSNODE_PROXY_HOST`` or the SDK default.
        http_proxy_port: HTTP proxy port (default 8080).
        socks5_proxy_port: SOCKS5 proxy port (default 1080).

    Raises:
        ValueError: If no API key is provided or found in environment.
    """

    # Stored so tools can reference them; not exposed to LLM
    _sync_client: Optional[DominusNodeClient] = None
    _async_client: Optional[AsyncDominusNodeClient] = None
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
    ) -> None:
        super().__init__()

        resolved_key = api_key or os.environ.get("DOMINUSNODE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A DomiNode API key is required.  Pass api_key= or set the "
                "DOMINUSNODE_API_KEY environment variable."
            )

        resolved_base = base_url or os.environ.get(
            "DOMINUSNODE_BASE_URL", "https://api.dominusnode.com"
        )
        resolved_proxy_host = proxy_host or os.environ.get(
            "DOMINUSNODE_PROXY_HOST", "proxy.dominusnode.com"
        )

        # Store credentials for agentic wallet tools (direct REST API calls)
        self._api_key = resolved_key
        self._base_url = resolved_base

        # Create sync client
        self._sync_client = DominusNodeClient(
            base_url=resolved_base,
            api_key=resolved_key,
            proxy_host=resolved_proxy_host,
            http_proxy_port=http_proxy_port,
            socks5_proxy_port=socks5_proxy_port,
        )

        # Create async client (deferred connect -- will connect in __aenter__)
        self._async_client = AsyncDominusNodeClient(
            base_url=resolved_base,
            api_key=resolved_key,
            proxy_host=resolved_proxy_host,
            http_proxy_port=http_proxy_port,
            socks5_proxy_port=socks5_proxy_port,
        )

        # Build tools once
        self._tools = self._build_tools()

    def _build_tools(self) -> List[BaseTool]:
        """Construct tool instances with injected clients."""
        fetch_tool = DominusNodeProxiedFetchTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )
        balance_tool = DominusNodeBalanceTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )
        usage_tool = DominusNodeUsageTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )
        config_tool = DominusNodeProxyConfigTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )
        topup_paypal_tool = DominusNodeTopupPaypalTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )
        x402_info_tool = DominusNodeX402InfoTool(
            sync_client=self._sync_client,
            async_client=self._async_client,
        )

        # Agentic wallet tools use direct REST API calls
        _ak = self._api_key
        _bu = self._base_url
        create_wallet_tool = DominusNodeCreateAgenticWalletTool(api_key=_ak, base_url=_bu)
        fund_wallet_tool = DominusNodeFundAgenticWalletTool(api_key=_ak, base_url=_bu)
        wallet_balance_tool = DominusNodeAgenticWalletBalanceTool(api_key=_ak, base_url=_bu)
        list_wallets_tool = DominusNodeListAgenticWalletsTool(api_key=_ak, base_url=_bu)
        transactions_tool = DominusNodeAgenticTransactionsTool(api_key=_ak, base_url=_bu)
        freeze_tool = DominusNodeFreezeAgenticWalletTool(api_key=_ak, base_url=_bu)
        unfreeze_tool = DominusNodeUnfreezeAgenticWalletTool(api_key=_ak, base_url=_bu)
        delete_wallet_tool = DominusNodeDeleteAgenticWalletTool(api_key=_ak, base_url=_bu)
        update_policy_tool = DominusNodeUpdateWalletPolicyTool(api_key=_ak, base_url=_bu)

        return [
            fetch_tool,
            balance_tool,
            usage_tool,
            config_tool,
            topup_paypal_tool,
            x402_info_tool,
            create_wallet_tool,
            fund_wallet_tool,
            wallet_balance_tool,
            list_wallets_tool,
            transactions_tool,
            freeze_tool,
            unfreeze_tool,
            delete_wallet_tool,
            update_policy_tool,
        ]

    def get_tools(self) -> List[BaseTool]:
        """Return the list of DomiNode LangChain tools."""
        if self._tools is None:
            self._tools = self._build_tools()
        return list(self._tools)

    def close(self) -> None:
        """Close the underlying SDK clients and release resources."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            self._async_client.close()
            self._async_client = None
        self._tools = None
