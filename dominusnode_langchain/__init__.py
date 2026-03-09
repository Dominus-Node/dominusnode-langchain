"""LangChain tools for Dominus Node rotating proxy service.

Provides 53 LangChain-compatible tools covering proxy, wallet, usage, account,
API keys, plans, agentic wallets, and teams through the Dominus Node platform.

Example::

    from dominusnode_langchain import DominusNodeToolkit

    toolkit = DominusNodeToolkit(api_key="dn_live_...", base_url="http://localhost:3000")
    tools = toolkit.get_tools()

    # Use with a LangChain agent
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    agent = create_tool_calling_agent(llm, tools, prompt)
"""

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
from .toolkit import DominusNodeToolkit

__all__ = [
    "DominusNodeToolkit",
    # Proxy (3)
    "DominusNodeProxiedFetchTool",
    "DominusNodeProxyConfigTool",
    "DominusNodeProxyStatusTool",
    # Sessions (1)
    "DominusNodeListSessionsTool",
    # Wallet (8)
    "DominusNodeBalanceTool",
    "DominusNodeGetTransactionsTool",
    "DominusNodeGetForecastTool",
    "DominusNodeTopupPaypalTool",
    "DominusNodeTopupStripeTool",
    "DominusNodeTopupCryptoTool",
    "DominusNodeCheckPaymentTool",
    "DominusNodeX402InfoTool",
    # Usage (3)
    "DominusNodeUsageTool",
    "DominusNodeDailyUsageTool",
    "DominusNodeTopHostsTool",
    # Account (6)
    "DominusNodeRegisterTool",
    "DominusNodeLoginTool",
    "DominusNodeGetAccountInfoTool",
    "DominusNodeVerifyEmailTool",
    "DominusNodeResendVerificationTool",
    "DominusNodeUpdatePasswordTool",
    # API Keys (3)
    "DominusNodeListKeysTool",
    "DominusNodeCreateKeyTool",
    "DominusNodeRevokeKeyTool",
    # Plans (3)
    "DominusNodeGetPlanTool",
    "DominusNodeListPlansTool",
    "DominusNodeChangePlanTool",
    # Agentic Wallets (9)
    "DominusNodeCreateAgenticWalletTool",
    "DominusNodeFundAgenticWalletTool",
    "DominusNodeAgenticWalletBalanceTool",
    "DominusNodeListAgenticWalletsTool",
    "DominusNodeAgenticTransactionsTool",
    "DominusNodeFreezeAgenticWalletTool",
    "DominusNodeUnfreezeAgenticWalletTool",
    "DominusNodeDeleteAgenticWalletTool",
    "DominusNodeUpdateWalletPolicyTool",
    # Teams (17)
    "DominusNodeCreateTeamTool",
    "DominusNodeListTeamsTool",
    "DominusNodeTeamDetailsTool",
    "DominusNodeUpdateTeamTool",
    "DominusNodeTeamDeleteTool",
    "DominusNodeTeamFundTool",
    "DominusNodeTeamCreateKeyTool",
    "DominusNodeTeamRevokeKeyTool",
    "DominusNodeTeamListKeysTool",
    "DominusNodeTeamUsageTool",
    "DominusNodeTeamListMembersTool",
    "DominusNodeTeamAddMemberTool",
    "DominusNodeTeamRemoveMemberTool",
    "DominusNodeUpdateTeamMemberRoleTool",
    "DominusNodeTeamInviteMemberTool",
    "DominusNodeTeamListInvitesTool",
    "DominusNodeTeamCancelInviteTool",
]
