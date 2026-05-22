import asyncio
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass

MCP_DEFINITIONS = {
    "dev": {
        "path": str(Path(__file__).resolve().parent / "mcps" / "dev_mcp" / "server.py"),
        "name": "Dev MCP",
        "description": "File system and git operations",
    },
    "docs": {
        "path": str(Path(__file__).resolve().parent / "mcps" / "docs_mcp" / "server.py"),
        "name": "Docs MCP",
        "description": "URL fetching and memory store",
    },
    "data": {
        "path": str(Path(__file__).resolve().parent / "mcps" / "data_mcp" / "server.py"),
        "name": "Data MCP",
        "description": "SQLite database queries",
    },
    "reasoning": {
        "path": str(Path(__file__).resolve().parent / "mcps" / "reasoning_mcp" / "server.py"),
        "name": "Reasoning MCP",
        "description": "Structured reasoning and summarization",
    },
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
            "clientInfo": {"name": "experiment3", "version": "1.0"},
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


async def connect_mcp(server_key: str) -> MCPConnection:
    definition = MCP_DEFINITIONS[server_key]
    server_path = definition["path"]
    venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
    python_exe = str(venv_python) if venv_python.exists() else "python3"

    process = await asyncio.create_subprocess_exec(
        python_exe, server_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    conn = MCPConnection(server_key=server_key, server_name=definition["name"], process=process)
    await conn.initialize()
    await conn.list_tools()
    return conn


async def connect_all_mcps(keys: list[str] = None) -> dict[str, MCPConnection]:
    if keys is None:
        keys = list(MCP_DEFINITIONS.keys())
    tasks = {k: connect_mcp(k) for k in keys}
    results = {}
    for k, task in tasks.items():
        try:
            results[k] = await task
        except Exception as e:
            print(f"  [red]Failed to connect {k} MCP: {e}[/red]")
    return results


async def close_all(connections: dict[str, MCPConnection]):
    for conn in connections.values():
        try:
            await conn.close()
        except Exception:
            pass
