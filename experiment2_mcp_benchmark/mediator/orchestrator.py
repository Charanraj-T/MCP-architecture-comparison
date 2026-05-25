import asyncio
import json
import re
import time
import tiktoken
from pathlib import Path
from common import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector
from mcp_client import MCPConnection, connect_all_mcps, close_all

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

PLAN_PROMPT = (
    "You are a planning agent. Given the available tools and the user request, "
    "generate a step-by-step execution plan as a JSON array of tool calls.\n\n"
    'Respond with ONLY a JSON array:\n'
    '[\n'
    '  {"name": "<tool_name>", "arguments": {<args>}},\n'
    '  ...\n'
    ']\n\n'
    "Rules:\n"
    "- Each step must use a tool from the available tools list\n"
    "- Use exact tool names as listed\n"
    "- Do NOT execute the tools yourself\n"
    "- Output ONLY the JSON array\n\n"
    "Available tools:\n"
)

COMPILE_PROMPT = (
    "You are a compiler. Given the original task and execution results, "
    "produce a coherent final answer.\n\n"
    "Original task:\n{task}\n\n"
    "Execution results:\n{results}\n\n"
    "Final answer:\n"
)


class MediatorOrchestrator:
    def __init__(self, client: LMStudioClient):
        self.client = client
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._connections: dict[str, MCPConnection] = {}

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

    async def close(self):
        await close_all(self._connections)
        self._connections = {}

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="MCP Mediator")

        self._connections = await connect_all_mcps()
        if not self._connections:
            metrics.tools_truncated = -1
            return metrics

        all_tools = []
        for conn in self._connections.values():
            all_tools.extend(conn.tools)

        metrics.mcp_servers_connected = len(self._connections)
        metrics.mcps_activated = len(self._connections)
        metrics.tools_exposed = len(all_tools)
        metrics.tool_schema_tokens = self._count_tokens(self._tool_schema_block(all_tools))

        plan_system = PLAN_PROMPT + self._tool_schema_block(all_tools)
        messages = [
            {"role": "system", "content": plan_system},
            {"role": "user", "content": workflow["prompt"]},
        ]

        plan_start = time.monotonic()
        plan_response = await self.client.chat(messages, temperature=0.1)
        plan_elapsed = (time.monotonic() - plan_start) * 1000

        if plan_response.error:
            metrics.tools_truncated = -1
            self.trace.record("plan_error", "plan_failed", {"error": plan_response.error})
            await self.close()
            return metrics

        metrics.prompt_tokens += plan_response.usage.prompt_tokens
        metrics.completion_tokens += plan_response.usage.completion_tokens
        metrics.reasoning_tokens += plan_response.usage.reasoning_tokens
        metrics.total_tokens += plan_response.usage.total_tokens
        metrics.agent_hops += 1
        total_latency = plan_elapsed

        plan = self._parse_plan(plan_response.content or "")
        if not plan:
            metrics.tools_truncated = -1
            self.trace.record("plan_error", "parse_failed", {"content": (plan_response.content or "")[:500]})
            await self.close()
            return metrics

        self.trace.record("plan", "generated", {"steps": len(plan), "plan": plan})

        tools_used = set()
        exec_results = []
        for step in plan:
            name = step.get("name", "")
            args = step.get("arguments", {})
            if not name:
                continue

            server_key = self._find_server_for_tool(name)
            if not server_key:
                exec_results.append({name: f"Error: tool '{name}' not found on any connected server"})
                continue

            exec_start = time.monotonic()
            try:
                result = await self._connections[server_key].call_tool(name, args)
                result_content = result.get("content", [{}])
                text = "".join(c.get("text", json.dumps(c)) for c in result_content if isinstance(c, dict))
                exec_results.append({name: text})
            except Exception as e:
                exec_results.append({name: f"Error: {e}"})
            exec_elapsed = (time.monotonic() - exec_start) * 1000
            total_latency += exec_elapsed

            tools_used.add(name)
            self.trace.record("tool_exec", name, {
                "latency_ms": round(exec_elapsed, 1),
                "result_preview": str(exec_results[-1].get(name, ""))[:200],
            })

        metrics.tools_used = len(tools_used)
        metrics.real_tool_calls = len(exec_results)

        compile_prompt = COMPILE_PROMPT.format(
            task=workflow["prompt"],
            results=json.dumps(exec_results, indent=2),
        )

        compile_start = time.monotonic()
        compile_response = await self.client.chat(
            [{"role": "user", "content": compile_prompt}],
            temperature=0.1,
        )
        compile_elapsed = (time.monotonic() - compile_start) * 1000

        if not compile_response.error:
            metrics.prompt_tokens += compile_response.usage.prompt_tokens
            metrics.completion_tokens += compile_response.usage.completion_tokens
            metrics.reasoning_tokens += compile_response.usage.reasoning_tokens
            metrics.total_tokens += compile_response.usage.total_tokens
            metrics.agent_hops += 1
            total_latency += compile_elapsed

        metrics.latency_ms = round(total_latency, 1)

        metrics.explanation = (
            "Single LLM call generates a plan with all tool schemas loaded once. "
            "Deterministic executor runs all real MCP tool calls without further LLM cost. "
            "Compile hop produces final answer. Only 2 LLM hops — zero token cost during execution."
        )

        self.trace.record("done", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/mediator_{wf_name.replace(' ', '_')}.json")
        self.logger.log_event(wf_name, "MCP Mediator", "completed", metrics.snapshot())

        await self.close()
        return metrics

    def _parse_plan(self, text: str) -> list[dict]:
        try:
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                if isinstance(plan, list):
                    return plan
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _find_server_for_tool(self, tool_name: str):
        for key, conn in self._connections.items():
            for t in conn.tools:
                if t.name == tool_name:
                    return key
        return None
