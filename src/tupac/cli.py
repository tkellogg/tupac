import json
import time
import mimetypes
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import re

import openai
import fastmcp
import asyncio
import typer
from rich.console import Console


console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BINARY_TYPES = {"image", "pdf", "audio", "video", "blob"}


@dataclass
class Config:
    system_prompt: str
    mcp_servers: Dict[str, Dict[str, Any]]
    model: str = "gpt-4o"

    @classmethod
    def load(cls, path: Path) -> "Config":
        text = path.read_text()
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
        text = pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)
        data = json.loads(text)
        return cls(
            system_prompt=data.get("system_prompt") or data.get("instructions"),
            mcp_servers=data.get("mcp_servers") or data.get("mcpServers") or {},
            model=data.get("model", "gpt-4o"),
        )

    def to_fastmcp(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, server in self.mcp_servers.items():
            if server.get("type") == "url":
                headers = {}
                token = server.get("authorization_token")
                if token:
                    headers["authorization"] = token
                entry = {"url": server["url"]}
                if headers:
                    entry["headers"] = headers
                if transport := server.get("transport"):
                    entry["transport"] = transport
                out[name] = entry
            else:
                out[name] = server
        return out


class ResourceCache:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
        self._changed = False

    def contains(self, uri: str) -> bool:
        return uri in self.cache

    def add(self, uri: str, title: str, type_: str, text: str) -> None:
        if uri in self.cache:
            self.cache.move_to_end(uri)
            return  # Already cached, no change needed
        self.cache[uri] = {"title": title, "type": type_, "text": text}
        self._changed = True
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def xml_blocks(self) -> tuple[str, str]:
        resources = "\n".join(
            f'<resource uri="{uri}" title="{v["title"]}" type="{v["type"]}"/>'
            for uri, v in self.cache.items()
        )
        details = "\n".join(
            f'<resource uri="{uri}">{v["text"]}</resource>'
            for uri, v in self.cache.items()
        )
        return (
            f"<resources>{resources}</resources>",
            f"<resource_details>{details}</resource_details>",
        )

    def consume_changed_blocks(self) -> list[str]:
        if not self._changed:
            return []
        self._changed = False
        return list(self.xml_blocks())


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
    client: openai.AsyncOpenAI,
    cfg: Config,
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


def save_blob(data: bytes, mime_type: str) -> Path:
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    path = OUTPUT_DIR / f"out_{ts}{ext}"
    path.write_bytes(data)
    return path


def _process_tool_result(result: Any, cache: ResourceCache) -> str:
    """Process tool result, handling resources with caching."""
    # FastMCP returns results in content array format
    if hasattr(result, 'content') and result.content:
        content_items = result.content
    elif isinstance(result, list):
        content_items = result
    else:
        # Single result, wrap in list
        content_items = [result]
    
    processed_content = []
    
    for item in content_items:
        # Handle FastMCP TextContent or similar objects
        if hasattr(item, 'text'):
            text_content = item.text
            
            # Try to parse as JSON to see if it contains resource info
            try:
                import json
                parsed = json.loads(text_content)
                
                # Check if parsed content has resources
                if isinstance(parsed, dict) and "results" in parsed:
                    for res in parsed.get("results", []):
                        if isinstance(res, dict) and all(k in res for k in ["id", "title", "text"]):
                            uri = res["id"]
                            title = res["title"]
                            type_ = res.get("type", "text/plain")
                            text = res["text"]
                            
                            # Always add resource reference
                            resource_ref = f'<resource uri="{uri}" title="{title}" type="{type_}"/>'
                            processed_content.append(resource_ref)
                            
                            # Only include content if not already cached
                            if not cache.contains(uri):
                                cache.add(uri, title, type_, text)
                                processed_content.append(f'<resource_content uri="{uri}">{text}</resource_content>')
                else:
                    # Regular content, add as-is
                    processed_content.append(text_content)
            except (json.JSONDecodeError, AttributeError):
                # Not JSON or doesn't have expected structure, add as-is
                processed_content.append(text_content)
        elif isinstance(item, dict):
            # Direct dict with resource fields
            if "uri" in item and "text" in item:
                uri = item["uri"]
                title = item.get("title", item.get("name", "Unknown"))
                type_ = item.get("type", item.get("mimeType", "text/plain"))
                text = item["text"]
                
                # Always add resource reference
                resource_ref = f'<resource uri="{uri}" title="{title}" type="{type_}"/>'
                processed_content.append(resource_ref)
                
                # Only include content if not already cached
                if not cache.contains(uri):
                    cache.add(uri, title, type_, text)
                    processed_content.append(f'<resource_content uri="{uri}">{text}</resource_content>')
            else:
                # Regular dict result, just stringify
                processed_content.append(str(item))
        else:
            # Other types, just stringify
            processed_content.append(str(item))
    
    return "\n".join(processed_content)


async def conversation_loop(
    client: openai.AsyncOpenAI,
    mcp: fastmcp.Client,
    cfg: Config,
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


async def cli(config_path: Path, prompt: str, verbose: bool = False) -> None:
    """Run tupac with CONFIG_PATH and PROMPT."""
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    cfg = Config.load(config_path)
    client = openai.AsyncOpenAI()
    
    # Handle configs without MCP servers
    if not cfg.mcp_servers:
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = await fetch_response(client, cfg, messages, [])
        console.print(resp.choices[0].message.content, style="cyan")
        return
    
    mcp = fastmcp.Client({"mcpServers": cfg.to_fastmcp()})
    async with mcp:
        tools = await build_tools(mcp)

        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]

        await conversation_loop(client, mcp, cfg, messages, tools, ResourceCache(), verbose)


def main() -> None:
    app = typer.Typer(pretty_exceptions_enable=False)

    @app.command()
    def _run(
        config_path: Path, 
        prompt: str,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool call results")
    ) -> None:
        asyncio.run(cli(config_path, prompt, verbose))

    app()


if __name__ == "__main__":
    main()
