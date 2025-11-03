from typing import Optional

import requests

MCP_BASE_URL = "http://localhost:8089/mcp"


def call(tool_name: str, arguments: Optional[dict] = None,
             base_url: str = MCP_BASE_URL, timeout: float = 10.0) -> dict:
    """Deterministic MCP tool call."""
    url = f"{base_url}/call"
    payload = {"name": tool_name, "arguments": arguments or {}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"MCP error: {data.get('error', 'unknown error')}")
    return data["result"]
