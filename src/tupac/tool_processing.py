"""Tool helpers and LiteLLM response handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Protocol

import fastmcp
import litellm
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)


EventCallback = Callable[[str, Any], Awaitable[None] | None]


@dataclass(slots=True)
class ToolCallFunction:
    name: str
    arguments: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(slots=True)
class ToolCall:
    id: str
    function: ToolCallFunction

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": self.function.to_dict()}


@dataclass(slots=True)
class AssistantTurn:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    raw: Response | None = None

    def to_message_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant"}
        if self.content:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return message

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class MCPClientProtocol(Protocol):
    async def list_tools(self) -> List[Any]:
        ...


async def build_tools(mcp: MCPClientProtocol) -> List[dict]:
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


def _message_text(message: ResponseOutputMessage) -> List[str]:
    texts: List[str] = []
    for item in message.content or []:
        text = getattr(item, "text", None)
        if text:
            texts.append(text)
        elif hasattr(item, "refusal"):
            texts.append(getattr(item, "refusal"))
    return texts


def _reasoning_text(item: ResponseReasoningItem) -> List[str]:
    texts: List[str] = []
    for summary in item.summary or []:
        text = getattr(summary, "text", None)
        if text:
            texts.append(text)
    for content in item.content or []:
        text = getattr(content, "text", None)
        if text:
            texts.append(text)
    return texts


def _tool_call(call: ResponseFunctionToolCall) -> ToolCall:
    call_id = call.id or call.call_id
    return ToolCall(
        id=str(call_id),
        function=ToolCallFunction(name=call.name, arguments=call.arguments or ""),
    )


def _normalise_response(response: Response) -> AssistantTurn:
    texts: List[str] = []
    reasoning: List[str] = []
    tool_calls: List[ToolCall] = []

    for item in response.output or []:
        item_type = getattr(item, "type", None)
        if item_type == "message" and isinstance(item, ResponseOutputMessage):
            texts.extend(_message_text(item))
        elif item_type == "function_call" and isinstance(item, ResponseFunctionToolCall):
            tool_calls.append(_tool_call(item))
        elif item_type == "reasoning" and isinstance(item, ResponseReasoningItem):
            reasoning.extend(_reasoning_text(item))

    content = "\n".join(texts) if texts else None
    return AssistantTurn(content=content, tool_calls=tool_calls, reasoning=reasoning, raw=response)


async def _maybe_call(callback: Optional[EventCallback], event_type: str, payload: Any) -> None:
    if callback is None:
        return
    result = callback(event_type, payload)
    if asyncio.iscoroutine(result):
        await result


async def _emit_events(turn: AssistantTurn, callback: Optional[EventCallback]) -> None:
    if callback is None:
        return
    for chunk in turn.reasoning:
        if chunk:
            await _maybe_call(callback, "reasoning", chunk)
    for call in turn.tool_calls:
        await _maybe_call(callback, "tool_call", call)
    if turn.content:
        await _maybe_call(callback, "message", turn.content)


async def fetch_response(
    cfg,
    messages: List[Any],
    tools: List[dict],
    *,
    on_event: Optional[EventCallback] = None,
) -> AssistantTurn:
    model = getattr(cfg, "model", None) or "gpt-4o"
    provider = (getattr(cfg, "provider", "openai") or "openai").lower()
    params = dict(getattr(cfg, "provider_options", {}))
    if provider and provider != "openai":
        params.setdefault("custom_llm_provider", provider)

    backoff = 1.0
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = await litellm.aresponses(
                input=list(messages),
                model=model,
                tools=list(tools) if tools else None,
                **params,
            )
            turn = _normalise_response(response)
            await _emit_events(turn, on_event)
            return turn
        except Exception as exc:  # pragma: no cover - exercised via retry logic in tests
            last_err = exc
            if attempt == 2:
                break
            await asyncio.sleep(backoff)
            backoff *= 2
    if last_err:
        raise last_err
    raise RuntimeError("unreachable")
