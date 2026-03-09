# Dominus Node LangChain Integration

LangChain tools for the [Dominus Node](https://dominusnode.com) rotating proxy-as-a-service platform. Enables LangChain agents to make proxied HTTP requests, check wallet balances, monitor usage, and query proxy configuration through the Dominus Node network.

## Installation

```bash
pip install dominusnode-langchain
```

Or install from source:

```bash
cd integrations/langchain
pip install -e ".[dev]"
```

### Dependencies

- `langchain-core >= 0.2.0`
- `dominusnode >= 0.1.0` (Dominus Node Python SDK)
- `httpx >= 0.24.0`

## Quick Start

### Environment Variables

Set up your Dominus Node credentials:

```bash
export DOMINUSNODE_API_KEY="dn_live_your_api_key_here"
export DOMINUSNODE_BASE_URL="https://api.dominusnode.com"   # optional, this is the default
export DOMINUSNODE_PROXY_HOST="proxy.dominusnode.com"       # optional, this is the default
```

### Basic Usage with a LangChain Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from dominusnode_langchain import DominusNodeToolkit

# Initialize the toolkit (reads DOMINUSNODE_API_KEY from env)
toolkit = DominusNodeToolkit()
tools = toolkit.get_tools()

# Set up a LangChain agent
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to a proxy network."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = executor.invoke({
    "input": "Check my proxy balance, then fetch https://httpbin.org/ip through a US proxy"
})
print(result["output"])
```

### Direct Tool Usage

You can also use the tools directly without an agent:

```python
from dominusnode_langchain import (
    DominusNodeToolkit,
    DominusNodeProxiedFetchTool,
    DominusNodeBalanceTool,
)

toolkit = DominusNodeToolkit(api_key="dn_live_your_key")
tools = toolkit.get_tools()

# Get tools by name
fetch_tool = next(t for t in tools if t.name == "dominusnode_proxied_fetch")
balance_tool = next(t for t in tools if t.name == "dominusnode_check_balance")

# Check balance
print(balance_tool.run({}))
# Output: Balance: $50.00 (5000 cents)
#         Currency: usd

# Fetch a URL through a US proxy
print(fetch_tool.run({
    "url": "https://httpbin.org/ip",
    "country": "US",
    "proxy_type": "dc",
}))
# Output: Status: 200
#         Content-Type: application/json
#         Body:
#         { "origin": "203.0.113.42" }
```

## Tools

### `dominusnode_proxied_fetch`

Makes HTTP requests through the Dominus Node rotating proxy network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | *required* | The URL to fetch |
| `method` | str | `"GET"` | HTTP method (`GET`, `HEAD`, or `OPTIONS`) |
| `country` | str | `None` | ISO 3166-1 alpha-2 country code (e.g. `"US"`, `"GB"`, `"DE"`) |
| `proxy_type` | str | `"dc"` | `"dc"` for datacenter ($3/GB) or `"residential"` ($5/GB) |

**Security:**
- Only read-only HTTP methods (GET, HEAD, OPTIONS) are allowed
- SSRF protection blocks private IPs, localhost, and reserved ranges
- Response bodies are truncated to 4,000 characters
- `file://`, `ftp://`, and other non-HTTP schemes are rejected
- URLs with embedded credentials are rejected

### `dominusnode_check_balance`

Check your Dominus Node wallet balance. No input required.

Returns the balance in both USD and cents.

### `dominusnode_check_usage`

Check your Dominus Node proxy usage statistics. No input required.

Returns total bandwidth used (GB), total cost, and request count.

### `dominusnode_get_proxy_config`

Get the Dominus Node proxy configuration. No input required.

Returns proxy endpoints, supported countries for geo-targeting, blocked countries, rotation intervals, and available geo-targeting features (state, city, ASN).

## Async Usage

All tools support async execution for use with async LangChain agents:

```python
import asyncio
from dominusnode_langchain import DominusNodeToolkit

async def main():
    toolkit = DominusNodeToolkit(api_key="dn_live_your_key")
    tools = toolkit.get_tools()

    balance_tool = next(t for t in tools if t.name == "dominusnode_check_balance")
    fetch_tool = next(t for t in tools if t.name == "dominusnode_proxied_fetch")

    # Async balance check
    balance = await balance_tool.arun({})
    print(balance)

    # Async proxied fetch
    result = await fetch_tool.arun({
        "url": "https://httpbin.org/headers",
        "country": "GB",
    })
    print(result)

    toolkit.close()

asyncio.run(main())
```

## Explicit Credentials

If you prefer not to use environment variables:

```python
from dominusnode_langchain import DominusNodeToolkit

toolkit = DominusNodeToolkit(
    api_key="dn_live_abc123",
    base_url="http://localhost:3000",
    proxy_host="localhost",
    http_proxy_port=8080,
    socks5_proxy_port=1080,
)
tools = toolkit.get_tools()
```

## Running Tests

```bash
cd integrations/langchain
pip install -e ".[dev]"
pytest tests/ -v
```

## Pricing

| Proxy Type | Price |
|-----------|-------|
| Datacenter | $3/GB |
| Residential | $5/GB |

## License

MIT
