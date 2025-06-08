import asyncio
from typing import Any, Dict, List

import fastmcp


async def build_tools(mcp: fastmcp.Client) -> List[dict]:
    """Return tool definitions compatible with the OpenAI Responses API."""
    tools: List[dict] = []
    for t in await mcp.list_tools():
        schema = dict(t.inputSchema or {})
        # ensure minimal JSON Schema validity for OpenAI
        schema.setdefault("type", "object")
        
        # Fix required field validation - ensure all properties are in required array
        properties = schema.get("properties", {})
        existing_required = schema.get("required", [])
        all_properties = list(properties.keys())
        
        # Use all properties as required if existing required is incomplete
        schema["required"] = all_properties if all_properties else existing_required
        
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                }
            }
        )
    return tools


async def fetch_response(
    client,
    cfg,
    messages: List[Any],
    tools: List[dict],
) -> Any:
    delay = 1.0
    for attempt in range(3):
        try:
            return await client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=tools,
                stream=False,
            )
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")