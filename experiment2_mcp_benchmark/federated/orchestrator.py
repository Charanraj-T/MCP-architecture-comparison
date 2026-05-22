import asyncio
import json
import subprocess
import tiktoken
from pathlib import Path
from common import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector
from mcp_client import MCPConnection

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")
_SANDBOX = Path(__file__).resolve().parent.parent / "sandbox"


def _build_smartmcp_config() -> dict:
    return {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(_SANDBOX)],
            },
            "git": {
                "command": "uvx",
                "args": ["mcp-server-git", "--repository", str(_SANDBOX / "repo")],
            },
            "fetch": {
                "command": "npx",
                "args": ["-y", "mcp-server-fetch-typescript"],
            },
            "memory": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
            },
            "data": {
                "command": "uvx",
                "args": ["mcp-server-sqlite", "--db", str(_SANDBOX / "data" / "test.db")],
            },
            "reasoning": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            },
        },
        "top_k": 5,
    }


class FederatedOrchestrator:
    def __init__(self, client: LMStudioClient):
        self.client = client
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tool_schema_block(self, tools: list) -> str:
        lines = []
        for t in tools:
            lines.append(f"## {t.name}")
            lines.append(f"Description: {t.description}")
            lines.append(f"Schema: {json.dumps(t.input_schema, indent=2)}")
            lines.append("")
        return "\n".join(lines)

    def _schema_tokens(self, tools: list) -> int:
        return self._count_tokens(self._tool_schema_block(tools))

    def _parse_tool_calls(self, text: str) -> list[dict]:
        calls = []
        i = 0
        while True:
            idx = text.find("TOOL_CALL:", i)
            if idx == -1:
                break
            json_start = text.find("{", idx)
            if json_start == -1:
                i = idx + 10
                continue
            depth = 0
            json_end = -1
            for j in range(json_start, len(text)):
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_end = j + 1
                        break
            if depth == 0 and json_end > json_start:
                raw = text[json_start:json_end]
                try:
                    calls.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
            i = json_end if json_end > 0 else idx + 10
        return calls

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Federated (smartmcp)")

        config = _build_smartmcp_config()
        config_path = Path(__file__).resolve().parent.parent / "smartmcp_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        process = await asyncio.create_subprocess_exec(
            "smartmcp", "--config", str(config_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        conn = MCPConnection(server_key="smartmcp", server_name="SmartMCP Router", process=process)
        await conn.initialize()
        await conn.list_tools()

        metrics.mcp_servers_connected = 1
        metrics.tools_exposed = len(conn.tools)
        metrics.tool_schema_tokens = self._schema_tokens(conn.tools)
        metrics.mcps_activated = 6

        orchestration_header = (
            "You are an AI assistant with access to tools via SmartMCP semantic router.\n"
            "You do NOT have all tools pre-loaded. Instead, discover tools dynamically:\n\n"
            "1. Call search_tools to find relevant tools for your task\n"
            "2. Each result includes the target identifier and full input schema\n"
            "3. Build arguments matching the schema, then call call_discovered_tool\n\n"
            "Call tools using:\n"
            'TOOL_CALL: {"name": "<tool_name>", "arguments": {...}}\n\n'
            "Available tools:\n"
        )
        metrics.orchestration_prompt_tokens = self._count_tokens(orchestration_header)

        system_prompt = orchestration_header + self._tool_schema_block(conn.tools)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": workflow["prompt"]},
        ]

        tools_used = set()
        total_latency = 0.0
        real_calls = 0

        for turn in range(8):
            response = await self.client.chat(messages, temperature=0.1)
            if response.error:
                metrics.tools_truncated = -1
                return metrics
            total_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.agent_hops += 1

            content = response.content or ""
            self.trace.record("llm_call", f"Turn {turn+1}", {
                "tokens": response.usage.total_tokens,
                "content_preview": content[:200],
            })

            tool_calls = self._parse_tool_calls(content)
            if tool_calls:
                results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    matched = [t for t in conn.tools if t.name == name]
                    if matched:
                        result = await conn.call_tool(name, args)
                        tools_used.add(name)
                        real_calls += 1
                        result_content = result.get("content", [{}])
                        text = ""
                        for c in result_content:
                            if isinstance(c, dict):
                                text += c.get("text", json.dumps(c))
                            else:
                                text += str(c)
                        results.append({name: text})
                        self.trace.record("tool_call", name, {
                            "mcp": "smartmcp", "args": args, "result_preview": text[:200],
                        })
                    else:
                        results.append({name: f"Error: tool '{name}' not found"})

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool execution results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                break

        metrics.tools_used = len(tools_used)
        metrics.real_tool_calls = real_calls
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(wf_name, "Federated (smartmcp)", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/federated_{wf_name.replace(' ', '_')}.json")

        try:
            await conn.close()
        except Exception:
            pass
        try:
            config_path.unlink()
        except Exception:
            pass
        return metrics
