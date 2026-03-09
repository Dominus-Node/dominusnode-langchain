"""Mock dominusnode SDK and langchain_core modules before any test imports."""
import sys
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict


# ── Flexible mock types that accept any kwargs ──────────────────────────

class _FlexType:
    """Base that accepts any keyword arguments."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Wallet(_FlexType):
    pass


class GeoTargeting(_FlexType):
    pass


class ProxyEndpointConfig(_FlexType):
    pass


class ProxyConfig(_FlexType):
    pass


class UsageSummary(_FlexType):
    pass


class UsagePagination(_FlexType):
    pass


class UsagePeriod(_FlexType):
    pass


class UsageResponse(_FlexType):
    pass


class ProxyUrlOptions(_FlexType):
    pass


class StripeCheckout(_FlexType):
    pass


class CryptoInvoice(_FlexType):
    pass


# Create mock dominusnode module hierarchy
dominusnode_mock = MagicMock()
dominusnode_types_mock = MagicMock()

# Wire up types
dominusnode_types_mock.Wallet = Wallet
dominusnode_types_mock.GeoTargeting = GeoTargeting
dominusnode_types_mock.ProxyConfig = ProxyConfig
dominusnode_types_mock.ProxyEndpointConfig = ProxyEndpointConfig
dominusnode_types_mock.UsageSummary = UsageSummary
dominusnode_types_mock.UsagePagination = UsagePagination
dominusnode_types_mock.UsagePeriod = UsagePeriod
dominusnode_types_mock.UsageResponse = UsageResponse
dominusnode_types_mock.ProxyUrlOptions = ProxyUrlOptions
dominusnode_types_mock.StripeCheckout = StripeCheckout
dominusnode_types_mock.CryptoInvoice = CryptoInvoice

# Wire DominusNodeClient and AsyncDominusNodeClient as MagicMock classes
dominusnode_mock.DominusNodeClient = MagicMock
dominusnode_mock.AsyncDominusNodeClient = MagicMock
dominusnode_mock.types = dominusnode_types_mock

sys.modules.setdefault("dominusnode", dominusnode_mock)
sys.modules.setdefault("dominusnode.types", dominusnode_types_mock)


# ── Mock langchain_core ─────────────────────────────────────────────────

lc_mock = MagicMock()
lc_tools_mock = MagicMock()
lc_callbacks_mock = MagicMock()


class BaseTool(BaseModel):
    """Mock langchain_core BaseTool using Pydantic (like real LangChain)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    description: str = ""
    args_schema: Any = None


lc_tools_mock.BaseTool = BaseTool
lc_callbacks_mock.CallbackManagerForToolRun = MagicMock
lc_callbacks_mock.AsyncCallbackManagerForToolRun = MagicMock

lc_mock.tools = lc_tools_mock
lc_mock.callbacks = lc_callbacks_mock

sys.modules.setdefault("langchain_core", lc_mock)
sys.modules.setdefault("langchain_core.tools", lc_tools_mock)
sys.modules.setdefault("langchain_core.callbacks", lc_callbacks_mock)
