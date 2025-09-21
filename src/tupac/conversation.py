import json
from typing import Any, Awaitable, Callable, List, Protocol

import fastmcp
import fastmcp.exceptions
from rich.console import Console

from .resource_cache import ResourceCache, _process_tool_result
from .tool_processing import AssistantTurn


class MCPClientProtocol(Protocol):
    async def call_tool(self, name: str, args: dict) -> Any:
        ...


console = Console()


async def conversation_loop(
    responder: Callable[..., Awaitable[AssistantTurn]],
    mcp: MCPClientProtocol,
    cfg: Any,
    messages: List[Any],
    tools: List[dict],
    cache: ResourceCache,
    verbose: bool = False,
) -> None:
    async def handle_event(event_type: str, payload: Any) -> None:
        if event_type == "reasoning" and payload:
            console.print(payload, style="grey42")
        elif event_type == "tool_call":
            call = payload
            console.print(
                f"Tool call: {call.function.name}({call.function.arguments})",
                style="yellow",
            )

    while True:
        for block in cache.consume_changed_blocks():
            messages.append({"role": "user", "content": block})

        assistant_turn = await responder(cfg, messages, tools, on_event=handle_event)

        messages.append(assistant_turn.to_message_dict())

        if assistant_turn.tool_calls:
            for tool_call in assistant_turn.tool_calls:
                try:
                    result = await mcp.call_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )
                    if verbose:
                        console.print(str(result), style="magenta")

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
            if assistant_turn.content:
                console.print(assistant_turn.content, style="cyan")
            return
