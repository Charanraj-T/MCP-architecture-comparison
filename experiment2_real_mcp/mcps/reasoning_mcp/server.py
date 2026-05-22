#!/usr/bin/env python3
import json
from mcp.server.fastmcp import FastMCP

server = FastMCP("Reasoning MCP")

@server.tool()
def sequential_think(thought: str, context: str = "") -> str:
    steps = [s.strip() for s in context.split("\n") if s.strip().startswith("STEP")]
    step_num = len(steps) + 1
    entry = f"STEP {step_num}: {thought}"
    return entry

@server.tool()
def decompose_task(task: str) -> str:
    prompt = (
        f"Decompose this task into 2-4 subtasks:\n{task}\n\n"
        "Output as a JSON array: [{\"id\": 1, \"description\": \"...\"}]"
    )
    import httpx
    try:
        resp = httpx.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen/qwen3-8b",
                "messages": [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        import re
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return json.dumps(parsed, indent=2)
        return content
    except Exception as e:
        return f"Decomposition error: {e}"

@server.tool()
def summarize(text: str, max_words: int = 100) -> str:
    prompt = f"Summarize in at most {max_words} words:\n\n{text[:4000]}"
    import httpx
    try:
        resp = httpx.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen/qwen3-8b",
                "messages": [
                    {"role": "system", "content": "/no_think"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Summarization error: {e}"

if __name__ == "__main__":
    server.run(transport="stdio")
