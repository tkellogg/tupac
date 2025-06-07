# tupac
A terminal MCP client based on the OpenAI responses API.

## Usage

```bash
tupac config.json "Hello"
```

Configuration files may contain `${VARNAME}` placeholders which are expanded
from the environment before parsing. See `configs/web-search.json` for an
example using `${EXA_KEY}`.
