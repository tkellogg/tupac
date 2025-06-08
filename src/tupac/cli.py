import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import re

import openai
import fastmcp
import asyncio
import typer
from rich.console import Console

from .resource_cache import ResourceCache
from .tool_processing import build_tools, fetch_response
from .conversation import conversation_loop


console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



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
