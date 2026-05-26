import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SANDBOX = Path(__file__).resolve().parent / "sandbox"


@dataclass
class MCPServerDef:
    key: str
    name: str
    description: str
    command: str
    args: list[str]


MCP_DEFINITIONS: dict[str, MCPServerDef] = {
    "filesystem": MCPServerDef(
        key="filesystem",
        name="Filesystem MCP",
        description="File system operations (read, write, search, list)",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(_SANDBOX)],
    ),
    "git": MCPServerDef(
        key="git",
        name="Git MCP",
        description="Git operations (log, diff, status, blame)",
        command="uvx",
        args=["mcp-server-git", "--repository", str(_SANDBOX / "repo")],
    ),
    "fetch": MCPServerDef(
        key="fetch",
        name="Fetch MCP",
        description="URL fetching and web content retrieval",
        command="npx",
        args=["-y", "mcp-server-fetch-typescript"],
    ),
    "memory": MCPServerDef(
        key="memory",
        name="Memory MCP",
        description="Key-value memory storage and retrieval",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
    ),
    "data": MCPServerDef(
        key="data",
        name="Data MCP",
        description="SQLite database queries, schema inspection, data analysis",
        command="uvx",
        args=["mcp-server-sqlite", "--db", str(_SANDBOX / "data" / "test.db")],
    ),
    "reasoning": MCPServerDef(
        key="reasoning",
        name="Reasoning MCP",
        description="Sequential thinking, structured reasoning, step-by-step analysis",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    ),
}


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict
    server_key: str


class MCPConnection:
    def __init__(self, server_key: str, server_name: str, process):
        self.server_key = server_key
        self.server_name = server_name
        self.process = process
        self.tools: list[MCPTool] = []
        self._id = 0

    async def _request(self, method: str, params: dict = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode())
        await self.process.stdin.drain()

        while True:
            resp_line = await self.process.stdout.readline()
            if not resp_line:
                raise ConnectionError(f"MCP {self.server_key} closed connection")
            decoded = resp_line.decode().strip()
            if not decoded:
                continue
            data = json.loads(decoded)
            if "id" in data and data["id"] == self._id:
                if "error" in data:
                    raise Exception(f"MCP error: {data['error']}")
                return data.get("result", {})
            if "id" in data:
                continue
            if "method" in data:
                continue

    async def initialize(self):
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-benchmark", "version": "1.0"},
        })
        notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self.process.stdin.write(notif.encode())
        await self.process.stdin.drain()
        return result

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list")
        self.tools = [
            MCPTool(name=t["name"], description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}), server_key=self.server_key)
            for t in result.get("tools", [])
        ]
        return self.tools

    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        return await self._request("tools/call", {"name": name, "arguments": arguments or {}})

    async def close(self):
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                self.process.kill()
            except Exception:
                pass


def _resolve_path() -> str:
    extra = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
    ]
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        extra.insert(0, os.path.join(venv, "bin"))
    current = os.environ.get("PATH", "")
    parts = current.split(":")
    for p in extra:
        if p and p not in parts:
            parts.insert(0, p)
    return ":".join(parts)


async def connect_mcp(server_key: str, env: dict = None) -> MCPConnection:
    definition = MCP_DEFINITIONS[server_key]
    merged_env = {**os.environ, **(env or {})}
    merged_env["PATH"] = _resolve_path()

    process = await asyncio.create_subprocess_exec(
        definition.command, *definition.args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=merged_env,
    )

    conn = MCPConnection(server_key=server_key, server_name=definition.name, process=process)
    await conn.initialize()
    await conn.list_tools()
    return conn


async def connect_all_mcps(keys: list[str] = None, env: dict = None) -> dict[str, MCPConnection]:
    if keys is None:
        keys = list(MCP_DEFINITIONS.keys())
    tasks = {k: connect_mcp(k, env=env) for k in keys}
    results = {}
    for k, task in tasks.items():
        try:
            results[k] = await task
        except Exception as e:
            print(f"  Failed to connect {k} MCP: {e}")
    return results


async def close_all(connections: dict[str, MCPConnection]):
    for conn in connections.values():
        try:
            await conn.close()
        except Exception:
            pass
