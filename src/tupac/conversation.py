import json
from typing import Any, List

import fastmcp
from rich.console import Console

from .resource_cache import ResourceCache, _process_tool_result
from .tool_processing import fetch_response


console = Console()


async def conversation_loop(
    client,
    mcp: fastmcp.Client,
    cfg,
    messages: List[Any],
    tools: List[dict],
    cache: ResourceCache,
    verbose: bool = False,
) -> None:
    while True:
        for block in cache.consume_changed_blocks():
            messages.append({"role": "user", "content": block})

        resp = await fetch_response(client, cfg, messages, tools)

        response_message = resp.choices[0].message
        messages.append(response_message)
        
        # Show reasoning if available
        if hasattr(response_message, 'reasoning') and response_message.reasoning:
            console.print(response_message.reasoning, style="grey42")
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                console.print(f"Tool call: {tool_call.function.name}({tool_call.function.arguments})", style="yellow")
                try:
                    result = await mcp.call_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )
                    if verbose:
                        console.print(str(result), style="magenta")
                    
                    # Handle tool result - could be single result or array
                    tool_content = _process_tool_result(result, cache)
                    
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content,
                        }
                    )
                except fastmcp.exceptions.ClientError as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(exc),
                        }
                    )
        else:
            console.print(response_message.content, style="cyan")
            return