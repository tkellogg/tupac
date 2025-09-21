import inspect
import os
from pathlib import Path

import fastmcp
import pytest

from tupac.cli import Config
from tupac.conversation import conversation_loop
from tupac.resource_cache import ResourceCache
from tupac.tool_processing import AssistantTurn, ToolCall, ToolCallFunction, build_tools, fetch_response

import dotenv

dotenv.load_dotenv()


class DummyMCP:
    async def call_tool(self, name: str, args: dict) -> dict:
        raise RuntimeError("tool call not expected")


class SuccessMCP:
    async def call_tool(self, name: str, args: dict) -> dict:
        return {
            "uri": "tool://result",
            "title": "Result",
            "type": "text",
            "text": f"echo {args.get('text', '')}",
        }


class ErrorMCP:
    async def call_tool(self, name: str, args: dict) -> dict:
        from fastmcp.exceptions import ClientError

        raise ClientError("boom")


async def _maybe_emit(callback, event_type: str, payload):
    if callback is None:
        return
    result = callback(event_type, payload)
    if inspect.isawaitable(result):
        await result


class DummyResponder:
    def __init__(self, turns: list[AssistantTurn]):
        self._turns = turns
        self._index = 0

    async def __call__(self, cfg, messages, tools, *, on_event=None):
        if self._index < len(self._turns):
            turn = self._turns[self._index]
            self._index += 1
        else:
            turn = AssistantTurn(content="done")

        if on_event:
            if turn.reasoning:
                for chunk in turn.reasoning:
                    await _maybe_emit(on_event, "reasoning", chunk)
            for call in turn.tool_calls:
                await _maybe_emit(on_event, "tool_call", call)
            if turn.content:
                await _maybe_emit(on_event, "message", turn.content)

        return turn


@pytest.mark.asyncio
async def test_conversation_simple():
    client = DummyResponder([AssistantTurn(content="hi")])
    cfg = Config(system_prompt="You are a helpful assistant.", mcp_servers={})
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": "Hello"},
    ]
    await conversation_loop(
        client,
        DummyMCP(),
        cfg,
        messages,
        [],
        ResourceCache(),
    )
    assert len(messages) > 2


@pytest.mark.asyncio
async def test_tool_success():
    tool_call = ToolCall("1", ToolCallFunction("echo", '{"text": "hi"}'))
    client = DummyResponder(
        [
            AssistantTurn(content=None, tool_calls=[tool_call]),
            AssistantTurn(content="done"),
        ]
    )
    cfg = Config(system_prompt="you", mcp_servers={})
    messages = [
        {"role": "system", "content": "you"},
        {"role": "user", "content": "call"},
    ]
    await conversation_loop(
        client,
        SuccessMCP(),
        cfg,
        messages,
        [],
        ResourceCache(),
    )
    assert any(
        isinstance(m, dict)
        and m.get("role") == "tool"
        for m in messages
    )


@pytest.mark.asyncio
async def test_tool_error():
    tool_call = ToolCall("1", ToolCallFunction("echo", '{"text": "hi"}'))
    client = DummyResponder(
        [
            AssistantTurn(content=None, tool_calls=[tool_call]),
            AssistantTurn(content="done"),
        ]
    )
    cfg = Config(system_prompt="you", mcp_servers={})
    messages = [
        {"role": "system", "content": "you"},
        {"role": "user", "content": "call"},
    ]
    await conversation_loop(
        client,
        ErrorMCP(),
        cfg,
        messages,
        [],
        ResourceCache(),
    )
    assert any(
        isinstance(m, dict)
        and m.get("role") == "tool"
        and "boom" in m.get("content", "")
        for m in messages
    )


def test_config_env(tmp_path, monkeypatch):
    data = '{"instructions": "${SYS}", "mcpServers": {}, "model": "${MOD}"}'
    path = tmp_path / "cfg.json"
    path.write_text(data)
    monkeypatch.setenv("SYS", "sys")
    monkeypatch.setenv("MOD", "test-model")
    cfg = Config.load(path)
    assert cfg.system_prompt == "sys"
    assert cfg.model == "test-model"


def test_load_example_config():
    cfg = Config.load(Path("configs/web-search.json"))
    if cfg.to_fastmcp():
        fastmcp.Client({"mcpServers": cfg.to_fastmcp()})
    assert cfg.system_prompt
    assert cfg.mcp_servers


@pytest.mark.asyncio
async def test_build_tools_missing_required(monkeypatch):
    from mcp.types import Tool

    class MCPT:
        async def list_tools(self):
            return [
                Tool(
                    name="t",
                    description="d",
                    inputSchema={"properties": {"a": {"type": "string"}}},
                )
            ]

    tools = await build_tools(MCPT())
    params = tools[0]["function"]["parameters"]
    assert params["required"] == ["a"]
    assert params["type"] == "object"


@pytest.mark.asyncio
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="needs API key")
async def test_openai_integration():
    cfg = Config(system_prompt="Say hi", mcp_servers={})
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": "Hello"},
    ]
    await conversation_loop(
        fetch_response,
        DummyMCP(),
        cfg,
        messages,
        [],
        ResourceCache(),
    )
    assert len(messages) > 2
