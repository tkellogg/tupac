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
    mcp_servers: List[Dict[str, Any]]
    model: str = "gpt-4o"

    @classmethod
    def load(cls, path: Path) -> "Config":
        text = path.read_text()
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
        text = pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)
        data = json.loads(text)
        return cls(
            system_prompt=data["system_prompt"],
            mcp_servers=data["mcp_servers"],
            model=data.get("model", "gpt-4o"),
        )


class ResourceCache:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
        self._changed = False

    def add(self, uri: str, title: str, type_: str, text: str) -> None:
        if uri in self.cache:
            self.cache.move_to_end(uri)
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


async def fetch_response(
    client: openai.AsyncOpenAI, cfg: Config, messages: List[Any]
) -> Any:
    delay = 1.0
    for attempt in range(3):
        try:
            return await client.responses.create(
                model=cfg.model,
                input=messages,
                mcp_servers=cfg.mcp_servers,
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


async def conversation_loop(
    client: openai.AsyncOpenAI,
    mcp: fastmcp.Client,
    cfg: Config,
    messages: List[Any],
    cache: ResourceCache,
) -> None:
    while True:
        for block in cache.consume_changed_blocks():
            messages.append({"role": "user", "content": block})

        resp = await fetch_response(client, cfg, messages)

        tool_called = False
        for item in resp.output:
            messages.append(item)
            if getattr(item, "type", None) == "function_call":
                tool_called = True
                console.print(f"Tool call: {item.name}", style="yellow")
                try:
                    result = await mcp.call_tool(
                        item.name,
                        json.loads(getattr(item, "arguments", "{}")),
                    )
                    console.print(str(result), style="magenta")
                    messages.append(
                        {
                            "type": "function_call_output",
                            "tool_use_id": item.id,
                            "is_error": False,
                            "content": result,
                        }
                    )
                    if isinstance(result, dict) and {
                        "uri",
                        "title",
                        "type",
                        "text",
                    }.issubset(result):
                        cache.add(
                            result["uri"],
                            result["title"],
                            result["type"],
                            result["text"],
                        )
                except fastmcp.exceptions.ClientError as exc:
                    messages.append(
                        {
                            "type": "function_call_output",
                            "tool_use_id": item.id,
                            "is_error": True,
                            "content": str(exc),
                        }
                    )
            elif getattr(item, "type", None) == "reasoning":
                text = " ".join(s.text for s in getattr(item, "summary", []))
                console.print(text, style="grey42")
            elif getattr(item, "type", None) == "message":
                parts: List[str] = []
                for c in getattr(item, "content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") in BINARY_TYPES:
                        data = c.get("data", b"")
                        if isinstance(data, str):
                            data = data.encode()
                        path = save_blob(
                            data, c.get("mime_type", "application/octet-stream")
                        )
                        parts.append(f"[binary saved to {path.name}]")
                console.print("".join(parts), style="cyan")
                return
            else:
                console.print(str(item), style="magenta")
                return

        if not tool_called:
            return


async def cli(config_path: Path, prompt: str) -> None:
    """Run tupac with CONFIG_PATH and PROMPT."""
    cfg = Config.load(config_path)
    client = openai.AsyncOpenAI()
    mcp = fastmcp.Client({"mcpServers": cfg.mcp_servers})

    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": prompt},
    ]

    await conversation_loop(client, mcp, cfg, messages, ResourceCache())


def main() -> None:
    def _run(config_path: Path, prompt: str) -> None:
        asyncio.run(cli(config_path, prompt))

    typer.run(_run)


if __name__ == "__main__":
    main()
