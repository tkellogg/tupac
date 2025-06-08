import json
from collections import OrderedDict
from typing import Any, Dict


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


    def consume_changed_blocks(self) -> list[str]:
        if not self._changed:
            return []
        self._changed = False
        resources = "\n".join(
            f'<resource uri="{uri}" title="{v["title"]}" type="{v["type"]}"/>'
            for uri, v in self.cache.items()
        )
        details = "\n".join(
            f'<resource uri="{uri}">{v["text"]}</resource>'
            for uri, v in self.cache.items()
        )
        return [
            f"<resources>{resources}</resources>",
            f"<resource_details>{details}</resource_details>",
        ]


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