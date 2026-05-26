#!/usr/bin/env python3
"""Quick smoke test: connect to filesystem MCP, list tools, call read_file."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_client import connect_mcp, close_all

async def main():
    print("Smoke test: connecting to filesystem MCP via npx...")
    conn = await connect_mcp("filesystem")
    print(f"  Connected: {conn.server_name}")
    print(f"  Tools ({len(conn.tools)}):")
    for t in conn.tools:
        print(f"    - {t.name}: {t.description}")
    
    # Try calling read_file on test.txt
    tool = next((t for t in conn.tools if t.name == "read_file"), None)
    if tool:
        print("\n  Calling read_file(test.txt)...")
        result = await conn.call_tool("read_file", {"path": "/test.txt"})
        content = result.get("content", [{}])
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        print(f"  Result: {text[:200]}")
        print("\n[green]SMOKE TEST PASSED: MCP server connection + tool execution work[/green]")
    else:
        print("\n[yellow]read_file not found — smoke test partial (connection OK)[/yellow]")
    
    await close_all({"filesystem": conn})

if __name__ == "__main__":
    asyncio.run(main())
