# tupac
A terminal MCP client based on the OpenAI responses API.

## Usage

```bash
tupac config.json "Hello"
```

Configuration files may contain `${VARNAME}` placeholders which are expanded
from the environment before parsing. Environment variables can also be loaded
from a `.env` file via `python-dotenv`. See `configs/web-search.json` for an
example using `${EXA_API_KEY}`.

Configuration format follows the standard MCP schema:

```json
{
  "instructions": "You are a helpful assistant.",
  "model": "gpt-4o",
  "mcpServers": {
    "exa": {
      "type": "url",
      "url": "https://api.exa.ai/mcp/sse",
      "name": "exa-search",
      "authorization_token": "${EXA_API_KEY}"
    }
  }
}
```
