# tupac
A terminal MCP client based on the OpenAI responses API.

## Usage

```bash
tupac config.json "Hello"
```

Configuration files may contain `${VARNAME}` placeholders which are expanded
from the environment before parsing. See `configs/web-search.json` for an
example using `${EXA_API_KEY}`.

Configuration format follows the standard MCP schema:

```json
{
  "instructions": "You are a helpful assistant.",
  "model": "gpt-4o",
  "mcpServers": {
    "exa": {
      "type": "streamable-http",
      "url": "https://mcp.exa.ai/mcp?exaApiKey=${EXA_API_KEY}",
      "note": "For Streamable HTTP connections, add this URL directly in your MCP Client"
    }
  }
}
```
