#!/usr/bin/env python3
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP("Dev MCP")

@server.tool()
def read_file(path: str) -> str:
    p = Path(path).resolve()
    if not p.exists():
        return f"Error: {path} not found"
    if not p.is_file():
        return f"Error: {path} is not a file"
    return p.read_text(encoding="utf-8", errors="replace")

@server.tool()
def search_files(pattern: str, path: str = ".") -> str:
    matches = list(Path(path).rglob(pattern))
    if not matches:
        return "No matches found"
    return "\n".join(str(m) for m in matches[:50])

@server.tool()
def grep_files(pattern: str, path: str = ".", glob: str = "*") -> str:
    import re
    results = []
    for f in Path(path).rglob(glob):
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(f"{f}:{i}: {line[:200]}")
        except Exception:
            pass
        if len(results) >= 50:
            break
    return "\n".join(results) if results else "No matches found"

@server.tool()
def git_log(limit: int = 10, path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={limit}", "--oneline", "--decorate"],
            capture_output=True, text=True, cwd=path, timeout=10
        )
        return result.stdout or result.stderr
    except Exception as e:
        return f"Error: {e}"

@server.tool()
def git_diff(revision: str = "HEAD~1..HEAD", path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "diff", revision],
            capture_output=True, text=True, cwd=path, timeout=10
        )
        return result.stdout[:5000] or result.stderr
    except Exception as e:
        return f"Error: {e}"

@server.tool()
def git_status(path: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, cwd=path, timeout=10
        )
        return result.stdout or "clean"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    server.run(transport="stdio")
