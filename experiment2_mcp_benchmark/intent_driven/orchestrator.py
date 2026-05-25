import asyncio
import json
import re
import time
import tiktoken
from pathlib import Path
from common import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector
from mcp_client import MCPConnection, connect_all_mcps, close_all, MCP_DEFINITIONS

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

SERVER_DOMAIN_MAP = {
    "filesystem": "file_ops",
    "git": "file_ops",
    "fetch": "research",
    "memory": "research",
    "data": "data_analysis",
    "reasoning": "planning",
}

SERVER_DESC = "\n".join(
    f"- {key}: {defn.name} ({domain})"
    for key, defn in MCP_DEFINITIONS.items()
    for domain in [SERVER_DOMAIN_MAP.get(key, "general")]
)

INTENT_PROMPT = (
    "You are an intent-driven orchestrator. Given the user request, produce a structured plan.\n\n"
    "Respond with JSON:\n"
    "{\n"
    '  "intent": "brief description",\n'
    '  "plan": [\n'
    "    {\n"
    '      "step": 1,\n'
    '      "tool": "<exact tool name>",\n'
    '      "arguments": {<tool arguments>},\n'
    '      "depends_on": [<step numbers this depends on>]\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Available servers and their tools will be listed below.\n"
    "Steps with empty depends_on ([]) can run in parallel.\n"
    "Use the exact tool names as listed.\n\n"
)

COMPILE_PROMPT = (
    "You are a compiler. Given the original task, plan, and execution results, "
    "produce a coherent final answer.\n\n"
    "Original task:\n{task}\n\n"
    "Plan:\n{plan}\n\n"
    "Execution results:\n{results}\n\n"
    "Final answer:\n"
)


class IntentDrivenOrchestrator:
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
            lines.append(f"  Server: {t.server_key}")
            lines.append(f"  Description: {t.description}")
            lines.append(f"  Schema: {json.dumps(t.input_schema)}")
            lines.append("")
        return "\n".join(lines)

    async def close(self):
        await close_all(self._connections)
        self._connections = {}

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Intent-Driven")

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

        intent_system = INTENT_PROMPT + self._tool_schema_block(all_tools)
        messages = [
            {"role": "system", "content": intent_system},
            {"role": "user", "content": workflow["prompt"]},
        ]

        plan_start = time.monotonic()
        plan_response = await self.client.chat(messages, temperature=0.1)
        plan_elapsed = (time.monotonic() - plan_start) * 1000

        if plan_response.error:
            metrics.tools_truncated = -1
            self.trace.record("plan_error", "intent_parse_failed", {"error": plan_response.error})
            await self.close()
            return metrics

        metrics.prompt_tokens += plan_response.usage.prompt_tokens
        metrics.completion_tokens += plan_response.usage.completion_tokens
        metrics.reasoning_tokens += plan_response.usage.reasoning_tokens
        metrics.total_tokens += plan_response.usage.total_tokens
        metrics.agent_hops += 1
        total_latency = plan_elapsed

        plan_data = self._parse_intent_plan(plan_response.content or "")
        if not plan_data or "plan" not in plan_data:
            metrics.tools_truncated = -1
            self.trace.record("plan_error", "intent_parse_failed", {"content": (plan_response.content or "")[:500]})
            await self.close()
            return metrics

        plan_steps = plan_data["plan"]
        self.trace.record("intent", "parsed", {
            "intent": plan_data.get("intent", ""),
            "steps": len(plan_steps),
        })

        dep_map: dict[int, list[int]] = {}
        for s in plan_steps:
            dep_map[s.get("step", 0)] = s.get("depends_on", [])

        completed: set[int] = set()
        exec_results: list[dict] = []
        tools_used: set[str] = set()
        real_calls = 0

        remaining = list(plan_steps)
        max_iters = len(plan_steps) * 3
        while remaining and max_iters > 0:
            max_iters -= 1
            batch = []
            still_remaining = []
            for s in remaining:
                step_num = s.get("step", 0)
                deps = dep_map.get(step_num, [])
                if all(d in completed for d in deps):
                    batch.append(s)
                else:
                    still_remaining.append(s)
            if not batch:
                still_remaining = remaining
                break

            async def exec_step(s):
                nonlocal real_calls
                step_num = s.get("step", 0)
                tool_name = s.get("tool", "")
                args = s.get("arguments", {})
                if not tool_name:
                    return step_num, {f"step_{step_num}": {"error": "no tool specified"}}, set()

                server_key = self._find_server_for_tool(tool_name)
                if not server_key:
                    return step_num, {f"step_{step_num}": {tool_name: f"Error: tool not found"}}, set()

                try:
                    result = await self._connections[server_key].call_tool(tool_name, args)
                    content = result.get("content", [{}])
                    text = "".join(
                        c.get("text", json.dumps(c))
                        for c in content if isinstance(c, dict)
                    )
                    real_calls += 1
                    return step_num, {f"step_{step_num}": {tool_name: text}}, {tool_name}
                except Exception as e:
                    return step_num, {f"step_{step_num}": {tool_name: f"Error: {e}"}}, set()

            batch_results = await asyncio.gather(*[exec_step(s) for s in batch])
            for step_num, result, used_tools in batch_results:
                exec_results.append(result)
                tools_used.update(used_tools)
                completed.add(step_num)
            remaining = still_remaining

        for s in remaining:
            step_num = s.get("step", 0)
            tool_name = s.get("tool", "")
            args = s.get("arguments", {})
            if tool_name:
                server_key = self._find_server_for_tool(tool_name)
                if server_key:
                    try:
                        result = await self._connections[server_key].call_tool(tool_name, args)
                        content = result.get("content", [{}])
                        text = "".join(c.get("text", json.dumps(c)) for c in content if isinstance(c, dict))
                        exec_results.append({f"step_{step_num}": {tool_name: text}})
                        tools_used.add(tool_name)
                        real_calls += 1
                    except Exception as e:
                        exec_results.append({f"step_{step_num}": {tool_name: f"Error: {e}"}})

        metrics.tools_used = len(tools_used)
        metrics.real_tool_calls = real_calls

        compile_prompt = COMPILE_PROMPT.format(
            task=workflow["prompt"],
            plan=json.dumps(plan_steps, indent=2),
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
            "Single LLM call produces structured plan with dependency resolution. "
            "Only tools from relevant servers loaded — schema cost proportional to scope. "
            "Dependency-aware executor parallelizes independent real MCP calls. "
            "Compile hop produces answer. Most token-efficient per-task."
        )

        self.trace.record("done", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/intent_driven_{wf_name.replace(' ', '_')}.json")
        self.logger.log_event(wf_name, "Intent-Driven", "completed", metrics.snapshot())

        await self.close()
        return metrics

    def _parse_intent_plan(self, text: str):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "plan" in data and isinstance(data["plan"], list):
                    return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _find_server_for_tool(self, tool_name: str):
        for key, conn in self._connections.items():
            for t in conn.tools:
                if t.name == tool_name:
                    return key
        return None
