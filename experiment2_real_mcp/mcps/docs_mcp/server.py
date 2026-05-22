#!/usr/bin/env python3
import json
from mcp.server.fastmcp import FastMCP

server = FastMCP("Docs MCP")
_memory: dict[str, str] = {}

@server.tool()
def fetch_url(url: str) -> str:
    import httpx
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        from html.parser import HTMLParser
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._text = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip:
                    self._text.append(data.strip())
            def get_text(self):
                return "\n".join(t for t in self._text if t)
        extractor = TextExtractor()
        extractor.feed(text)
        return extractor.get_text()[:8000]
    except Exception as e:
        return f"Error fetching {url}: {e}"

@server.tool()
def store_memory(key: str, value: str) -> str:
    _memory[key] = value
    return f"Stored: {key}"

@server.tool()
def recall_memory(key: str) -> str:
    return _memory.get(key, f"Key not found: {key}")

@server.tool()
def search_memory(query: str) -> str:
    results = {k: v for k, v in _memory.items() if query.lower() in k.lower() or query.lower() in v.lower()}
    if not results:
        return "No results found"
    return json.dumps(results, indent=2)

@server.tool()
def list_memory() -> str:
    if not _memory:
        return "No stored items"
    return "\n".join(f"{k}: {v[:100]}..." for k, v in _memory.items())

if __name__ == "__main__":
    server.run(transport="stdio")
