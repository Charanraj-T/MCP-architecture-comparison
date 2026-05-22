import asyncio
import json
import subprocess
import time
import shutil
import tiktoken
import httpx
from pathlib import Path
from common import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")
_SANDBOX = Path(__file__).resolve().parent.parent / "sandbox"
_BIFROST_PORT = 8080

_BIFROST_CLIENTS = [
    {"name": "filesystem", "connection_type": "stdio", "stdio_config": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", str(_SANDBOX)]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
    {"name": "git", "connection_type": "stdio", "stdio_config": {"command": "uvx", "args": ["mcp-server-git", "--repository", str(_SANDBOX / "repo")]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
    {"name": "fetch", "connection_type": "stdio", "stdio_config": {"command": "npx", "args": ["-y", "mcp-server-fetch-typescript"]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
    {"name": "memory", "connection_type": "stdio", "stdio_config": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
    {"name": "sqlite", "connection_type": "stdio", "stdio_config": {"command": "uvx", "args": ["mcp-server-sqlite", "--db", str(_SANDBOX / "data" / "test.db")]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
    {"name": "reasoning", "connection_type": "stdio", "stdio_config": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]}, "tools_to_execute": ["*"], "is_code_mode_client": True},
]

CODE_MODE_TOOLS = [
    {"type": "function", "function": {"name": "listToolFiles", "description": "List available MCP server .pyi stub files for code mode servers", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "readToolFile", "description": "Read a virtual .pyi file to get compact Python function signatures for a server's tools", "parameters": {"type": "object", "required": ["fileName"], "properties": {"fileName": {"type": "string"}, "startLine": {"type": "integer"}, "endLine": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "getToolDocs", "description": "Get detailed documentation for a specific tool when compact signatures are insufficient", "parameters": {"type": "object", "required": ["server", "tool"], "properties": {"server": {"type": "string"}, "tool": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "executeToolCode", "description": "Execute Python code in a sandboxed Starlark interpreter with access to all code mode server tools. Use keyword arguments: server.tool(param=value). Assign result to 'result' variable.", "parameters": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}}}},
]

_TOKEN_BUDGET = 3500


class FederatedOrchestrator:
    def __init__(self, client: LMStudioClient):
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._llm = client
        self._bifrost_proc = None
        self._bf_client = None
        self._started = False
        self._mcp_id = 0

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _schema_tokens(self) -> int:
        return self._count_tokens(json.dumps(CODE_MODE_TOOLS, indent=2))

    async def _ensure_bifrost(self):
        if self._started:
            return True

        bifrost_dir = Path(__file__).resolve().parent.parent / ".bifrost"
        if bifrost_dir.exists():
            shutil.rmtree(bifrost_dir)
        bifrost_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "mcp": {"client_configs": _BIFROST_CLIENTS, "tool_manager_config": {"code_mode_binding_level": "server"}},
            "config_store": {"enabled": True, "type": "sqlite", "config": {"path": "./config.db"}},
        }
        with open(bifrost_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self._bifrost_proc = await asyncio.create_subprocess_exec(
            "npx", "-y", "@maximhq/bifrost",
            "--app-dir", str(bifrost_dir),
            "--port", str(_BIFROST_PORT),
            "--log-level", "error",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._bf_client = httpx.AsyncClient(base_url=f"http://localhost:{_BIFROST_PORT}", timeout=60)
        for i in range(30):
            try:
                r = await self._bf_client.get("/health")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            return False

        for i in range(20):
            try:
                r = await self._bf_client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                if r.status_code == 200 and r.json().get("result", {}).get("tools"):
                    self._started = True
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
        return False

    async def _call_tool(self, name: str, args: dict) -> str:
        self._mcp_id += 1
        try:
            r = await self._bf_client.post("/mcp", json={
                "jsonrpc": "2.0", "id": self._mcp_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }, timeout=60)
            data = r.json()
            if "error" in data:
                return f"Error: {data['error']['message']}"
            parts = data.get("result", {}).get("content", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
            return text
        except Exception as e:
            return f"Error: {e}"

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Federated (Bifrost)")

        if self._schema_tokens() > _TOKEN_BUDGET:
            metrics.tools_truncated = -1
            return metrics
        metrics.tool_schema_tokens = self._schema_tokens()
        metrics.tools_exposed = len(CODE_MODE_TOOLS)
        metrics.mcp_servers_connected = 6
        metrics.mcps_activated = 6

        ok = await self._ensure_bifrost()
        if not ok:
            print("  Bifrost failed to start or MCP tools not ready")
            metrics.tools_truncated = -1
            return metrics

        orchestration_header = (
            "You are an AI assistant with access to tools via Bifrost Code Mode.\n"
            "Only 4 meta-tools are available. Use them to discover and orchestrate real MCP tools:\n\n"
            "1. listToolFiles() - discover available server .pyi stub files\n"
            "2. readToolFile(fileName) - load compact Python signatures\n"
            "3. getToolDocs(server, tool) - get detailed documentation\n"
            "4. executeToolCode(code) - run Python code that calls MCP tools\n\n"
            "For executeToolCode: use server.tool(param=value), assign final result to 'result' variable. "
            "All servers are global objects: filesystem, git, fetch, memory, sqlite, reasoning.\n"
        )
        metrics.orchestration_prompt_tokens = self._count_tokens(orchestration_header)

        messages = [
            {"role": "system", "content": orchestration_header},
            {"role": "user", "content": workflow["prompt"]},
        ]
        tools_used = set()
        total_latency = 0.0
        real_calls = 0

        for turn in range(8):
            turn_start = time.monotonic()
            response = await self._llm.chat(
                messages, temperature=0.1, max_tokens=4096,
                tools=CODE_MODE_TOOLS, tool_choice="auto",
            )
            turn_elapsed = (time.monotonic() - turn_start) * 1000

            if response.error:
                print(f"  LLM error: {response.error}")
                metrics.tools_truncated = -1
                break

            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.agent_hops += 1
            total_latency += turn_elapsed

            content = response.content
            raw = response.raw or {}
            msg = (raw.get("choices") or [{}])[0].get("message", {})
            tool_calls = msg.get("tool_calls", [])

            self.trace.record("llm_call", f"Turn {turn+1}", {
                "tokens": response.usage.total_tokens,
                "content_preview": content[:200],
                "tool_calls": len(tool_calls),
                "latency_ms": round(turn_elapsed, 1),
            })

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                break

            results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                exec_start = time.monotonic()
                result_text = await self._call_tool(name, args)
                exec_elapsed = (time.monotonic() - exec_start) * 1000

                results.append({name: str(result_text)[:1000]})
                real_calls += 1
                tools_used.add(name)
                self.trace.record("tool_call", name, {
                    "latency_ms": round(exec_elapsed, 1), "args": args,
                    "result_preview": str(result_text)[:200],
                })

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            messages.append({
                "role": "user",
                "content": f"Tool execution results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
            })

        metrics.tools_used = len(tools_used)
        metrics.real_tool_calls = real_calls
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(wf_name, "Federated (Bifrost)", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/federated_{wf_name.replace(' ', '_')}.json")
        return metrics

    async def close(self):
        if self._bf_client:
            await self._bf_client.aclose()
            self._bf_client = None
        if self._bifrost_proc:
            try:
                self._bifrost_proc.terminate()
                await asyncio.wait_for(self._bifrost_proc.wait(), timeout=5)
            except Exception:
                try:
                    self._bifrost_proc.kill()
                except Exception:
                    pass
            self._bifrost_proc = None
        self._started = False
