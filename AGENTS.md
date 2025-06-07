**Spec for “tupac” — a CLI MCP ↔ OpenAI Responses bridge**
(latest deps, quoting your brief)

---

> *“i want to build a CLI MCP client app. python, uv, FastMCP, typer, on the openai responses API”*

* Name the package **tupac**.
* Latest libs (all installed with `uv add`):

  * `openai` @ latest (o-series ready)
  * `fastmcp` @ latest — v 2.4.0 adds one-line **Client(config)** multi-server support ([gofastmcp.com][1])
  * `typer` @ latest (>= 0.12)
  * `rich` @ latest (for colour) ([typer.tiangolo.com][2])
  * Python ≥ 3.11

---

> *“the CLI takes a JSON file and a prompt. the JSON configures MCP servers and a system prompt. the app is a simple loop that…”*

CLI (`tupac run cfg.json "prompt"`):

1. **Load config** (`system_prompt`, `mcp_servers`, model, etc.).
2. Construct **`mcp_servers`** array **exactly** as in Claude’s MCP connector docs ([docs.anthropic.com][3]).
3. Instantiate `fastmcp.Client(config)` — passing the parsed MCP-server list straight in satisfies the new constructor signature; no manual tool mapping required.
4. Seed `messages = [{"role":"system","content":system_prompt}, {"role":"user","content":prompt}]`.

Loop:

```python
while True:
    resp = client.responses.create(
        model=config.model,
        input=messages,
        mcp_servers=config.mcp_servers,  # Claude-format JSON
        stream=False
    )
    out = resp.output[0]
    if out.type == "mcp_tool_use":
        try:
            result = await mcp.call_tool(out)      # fastmcp handles dispatch
            messages += [
                out,
                {"type":"mcp_tool_result",
                 "tool_use_id": out.id,
                 "is_error": False,
                 "content": result}
            ]
        except Exception as exc:
            messages += [
                out,
                {"type":"mcp_tool_result",
                 "tool_use_id": out.id,
                 "is_error": True,
                 "content": str(exc)}
            ]
        continue          # keep cycling
    break                 # plain assistant text → done
```

Tool errors are surfaced to the model ( `is_error=True` ) so it can recover automatically.

---

> *“maps MCP tools to responses API … parameters are the complicated part”*

FastMCP now exposes `Tool.model_json_schema()`; embed that JSON Schema untouched inside each tool definition so the Responses API can validate arguments itself ([gofastmcp.com][1]).

---

> *“handle all MCP data types properly…”*

| MCP content                                                                     | How **tupac** returns it                                                                                                                            |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TextContent`                                                                   | inline text                                                                                                                                         |
| `ImageContent`, `PdfContent`, `AudioContent`, `VideoContent`, any `BlobContent` | **print only the generated file-name** (e.g. `out_2025-06-07T12-00-01.png`) and tell the human “open this file to view”; save bytes to `./outputs/` |

The canonical list of binary types (images, PDFs, **audio, video**) is in the MCP *Resources* spec ([modelcontextprotocol.io][4]).

---

> *“for resources, maintain a cache… list of compact representations … cache misses include the full text”*

Use an LRU keyed by `uri`.
Send two XML blocks (why XML? Anthropic’s prompt-engineering guide emphasises XML tags for structured context) ([docs.anthropic.com][5]):

```xml
<resources>
  <resource uri="https://foo" title="Foo doc" type="text"/>
  …
</resources>
<resource_details>
  <resource uri="https://foo"><![CDATA[full text]]></resource>
</resource_details>
```

---

> *“i also used some fancy lib to print text in colors…”*

Rich is the library Typer piggy-backs for colour ([typer.tiangolo.com][2]).
Colour scheme:

* **grey42** for model “reasoning” lines (exists in Rich palette ([rich.readthedocs.io][6]))
* green = user, cyan = assistant, yellow = tool-call, magenta = MCP response.

---

### Error handling

* Network / OpenAI 5xx / rate-limit → exponential back-off (3 tries).
* Any `fastmcp.ClientError` caught and echoed as an `"is_error": true` tool result (loop continues).

---

### Packaging & publish pipeline

* Create `pyproject.toml`; dependences managed by **`uv add`** (see uv docs) ([docs.astral.sh][7]).
* Manual release:

```bash
uv build           # build dist/*
uv publish         # uploads via PEP 741 trusted-publisher if token present
```

`uv publish` is the officially supported manual path to PyPI ([docs.astral.sh][8]).

---

### Example integration — Exa web search

Add this to `mcp_servers`:

```json
{
  "type": "url",
  "url": "https://api.exa.ai/mcp/sse",
  "name": "exa-search",
  "authorization_token": "EXA_KEY"
}
```

The search tool then mirrors the POST `/search` endpoint documented here ([docs.exa.ai][9]).

---

> *“keeps cycling until there’s no tool calls left”* — tupac follows exactly that loop, with robust error propagation, grey chain-of-thought, and latest libraries throughout.

[1]: https://gofastmcp.com/clients/client "Client Overview - FastMCP"
[2]: https://typer.tiangolo.com/tutorial/printing/?utm_source=chatgpt.com "Printing and Colors - Typer"
[3]: https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector "MCP connector - Anthropic"
[4]: https://modelcontextprotocol.io/docs/concepts/resources "Resources - Model Context Protocol"
[5]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags?utm_source=chatgpt.com "Use XML tags to structure your prompts - Anthropic API"
[6]: https://rich.readthedocs.io/en/stable/appendix/colors.html?utm_source=chatgpt.com "Standard Colors — Rich 13.6.0 documentation"
[7]: https://docs.astral.sh/uv/guides/projects/?utm_source=chatgpt.com "Working on projects | uv - Astral Docs"
[8]: https://docs.astral.sh/uv/guides/package/ "Building and publishing a package | uv"
[9]: https://docs.exa.ai/reference/search "Search - Exa"
