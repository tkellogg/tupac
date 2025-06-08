#!/bin/bash
set -eou pipefail

uvx ty check
uv run pytest
uv run tupac configs/web-search.json "why are horses always green?"
