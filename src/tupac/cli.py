import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict
import os
import re

import asyncio
import fastmcp
import typer
from rich.console import Console

from .conversation import conversation_loop
from .resource_cache import ResourceCache
from .tool_processing import build_tools, fetch_response


console = Console()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



@dataclass
class Config:
    system_prompt: str
    mcp_servers: Dict[str, Dict[str, Any]]
    model: str = "gpt-4o"
    provider: str = "openai"
    provider_mode: str = "chat"
    provider_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        text = path.read_text()
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
        text = pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)
        data = json.loads(text)
        provider_name = (
            data.get("provider_name")
            or data.get("providerName")
            or data.get("provider")
            if isinstance(data.get("provider"), str)
            else None
        )
        provider_mode = data.get("provider_mode") or data.get("providerMode")
        provider_options: Dict[str, Any] = {}

        provider_field = data.get("provider")
        if isinstance(provider_field, dict):
            provider_name = provider_field.get("name") or provider_field.get("provider") or provider_name
            provider_mode = (
                provider_field.get("mode")
                or provider_field.get("variant")
                or provider_field.get("api")
                or provider_mode
            )
            provider_options.update(
                {
                    k: v
                    for k, v in provider_field.items()
                    if k
                    not in {
                        "name",
                        "provider",
                        "mode",
                        "variant",
                        "api",
                        "params",
                        "options",
                    }
                }
            )
            params = provider_field.get("params") or provider_field.get("options")
            if isinstance(params, dict):
                provider_options.update(params)

        for key in ("provider_options", "providerOptions", "client_options", "clientOptions"):
            extra = data.get(key)
            if isinstance(extra, dict):
                provider_options.update(extra)

        return cls(
            system_prompt=data.get("system_prompt") or data.get("instructions"),
            mcp_servers=data.get("mcp_servers") or data.get("mcpServers") or {},
            model=data.get("model", "gpt-4o"),
            provider=provider_name or "openai",
            provider_mode=(provider_mode or "chat").lower(),
            provider_options=provider_options,
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
    # Handle configs without MCP servers
    if not cfg.mcp_servers:
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async def handle_event(event_type: str, payload: Any) -> None:
            if event_type == "reasoning" and payload:
                console.print(payload, style="grey42")

        assistant_turn = await fetch_response(cfg, messages, [], on_event=handle_event)
        if assistant_turn.content:
            console.print(assistant_turn.content, style="cyan")
        return
    
    mcp = fastmcp.Client({"mcpServers": cfg.to_fastmcp()})
    async with mcp:
        tools = await build_tools(mcp)

        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]

        await conversation_loop(fetch_response, mcp, cfg, messages, tools, ResourceCache(), verbose)


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
