"""LangChain tools for DomiNode rotating proxy service.

Provides LangChain-compatible tools for making proxied HTTP requests,
checking wallet balance, viewing usage statistics, and querying proxy
configuration through the DomiNode platform.

Example::

    from dominusnode_langchain import DominusNodeToolkit

    toolkit = DominusNodeToolkit(api_key="dn_live_...", base_url="http://localhost:3000")
    tools = toolkit.get_tools()

    # Use with a LangChain agent
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    agent = create_tool_calling_agent(llm, tools, prompt)
"""

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
from .toolkit import DominusNodeToolkit

__all__ = [
    "DominusNodeToolkit",
    "DominusNodeProxiedFetchTool",
    "DominusNodeBalanceTool",
    "DominusNodeUsageTool",
    "DominusNodeProxyConfigTool",
    "DominusNodeTopupPaypalTool",
    "DominusNodeX402InfoTool",
    "DominusNodeCreateAgenticWalletTool",
    "DominusNodeFundAgenticWalletTool",
    "DominusNodeAgenticWalletBalanceTool",
    "DominusNodeListAgenticWalletsTool",
    "DominusNodeAgenticTransactionsTool",
    "DominusNodeFreezeAgenticWalletTool",
    "DominusNodeUnfreezeAgenticWalletTool",
    "DominusNodeDeleteAgenticWalletTool",
    "DominusNodeUpdateWalletPolicyTool",
]
