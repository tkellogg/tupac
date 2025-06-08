import asyncio
from pathlib import Path
import pytest
import os
import openai

from tupac.cli import Config
from tupac.conversation import conversation_loop
from tupac.resource_cache import ResourceCache
from tupac.tool_processing import build_tools
import fastmcp


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


@pytest.mark.asyncio
async def test_conversation_simple():
    items = [DummyItem(content="hi", tool_calls=None)]
    client = DummyClient(items)
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


class DummyToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = DummyFunction(name, arguments)

class DummyFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class DummyItem:
    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
        # Ensure tool_calls exists
        if not hasattr(self, 'tool_calls'):
            self.tool_calls = None


class DummyResponse:
    def __init__(self, items):
        self.output = items


class DummyChoice:
    def __init__(self, item):
        self.message = item

class DummyResponse:
    def __init__(self, item):
        self.choices = [DummyChoice(item)]

class DummyCompletions:
    def __init__(self, items):
        self._items = items
        self._index = 0

    async def create(self, *args, **kwargs):
        if self._index < len(self._items):
            item = self._items[self._index]
            self._index += 1
            return DummyResponse(item)
        return DummyResponse(DummyItem(content="done", tool_calls=None))

class DummyChat:
    def __init__(self, items):
        self.completions = DummyCompletions(items)

class DummyClient:
    def __init__(self, items):
        self._items = items
        self.chat = DummyChat(items)


@pytest.mark.asyncio
async def test_tool_success():
    tool_call = DummyToolCall("1", "echo", '{"text": "hi"}')
    items = [
        DummyItem(content=None, tool_calls=[tool_call]),
        DummyItem(content="done", tool_calls=None),
    ]
    client = DummyClient(items)
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
    tool_call = DummyToolCall("1", "echo", '{"text": "hi"}')
    items = [
        DummyItem(content=None, tool_calls=[tool_call]),
        DummyItem(content="done", tool_calls=None),
    ]
    client = DummyClient(items)
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
    client = openai.AsyncOpenAI()
    await conversation_loop(
        client,
        DummyMCP(),
        cfg,
        messages,
        [],
        ResourceCache(),
    )
    assert len(messages) > 2
