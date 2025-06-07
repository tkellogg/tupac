# tupac
A GPT wrapper with MCP support. It's a thin layer around the OpenAI [responses][api] API
with functions being specified as MCP config.

You can write a simple "LLM app" very quickly, by specifying MCP config and a system prompt.

MCP functionality supported:
* ✅ tools
* ✅ resources — but only as far as they're being returned from tools. No listing or fetching.

Nothing else. It's what I consider to be an absolute [bare-bones][blog] MCP app.

## Usage: LLM app

```bash
uvx tupac configs/web-search.json "When are we getting to Mars?"
```

Configuration files may contain `${VARNAME}` placeholders which are expanded
from the environment before parsing. Environment variables can also be loaded
from a `.env` file via `python-dotenv`. See `configs/web-search.json` for an
example using `${EXA_API_KEY}`.

Configuration format follows the standard MCP schema:

```json
{
  "instructions": "Use search to answer questions.",
  "model": "o3",
  "mcpServers": {
    "exa": {
      "type": "url",
      "url": "https://mcp.exa.ai/mcp?exaApiKey=${EXA_API_KEY}"
    }
  }
}
```

You can use that `${EXA_API_KEY}` syntax to reference environment variables. It
does load [`.env` files][env].

 [api]: https://platform.openai.com/docs/api-reference/responses
 [env]: https://pypi.org/project/python-dotenv/
 [blog]: https://timkellogg.me/blog/2025/06/05/mcp-resources
