import asyncio
import json
import re
import warnings
from pathlib import Path

import tiktoken
from common import LMStudioClient
from common_tools import ToolRegistry
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")
_CALL_TIMEOUT = 120

PLAN_PROMPT = (
    "You are a planning agent. Your job is to generate a step-by-step execution plan "
    "as a JSON array of tool calls.\n\n"
    'Respond with ONLY a valid JSON array:\n'
    '[\n'
    '  {"name": "<tool_name>", "arguments": {<args>}},\n'
    '  ...\n'
    ']\n\n'
    "Do NOT execute the tools. Do NOT include explanations. Output ONLY the JSON array.\n\n"
    "Available tools:\n"
)

COMPILE_PROMPT = (
    "You are a compiler. Given the original task and tool execution results, "
    "produce a final coherent answer.\n\n"
    "Original task:\n{task}\n\n"
    "Execution results:\n{results}\n\n"
    "Final answer:\n"
)

_RETRY_PROMPT = (
    "\n\nThe previous response was not a valid JSON plan. "
    "Respond with ONLY a JSON array of tool calls. No other text."
)


class MediatorOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry, mcps: list[dict], total_token_budget: int = 38000, mcp_count: int = 0):
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.total_token_budget = total_token_budget
        self.mcp_count = mcp_count
        self.logger = JSONLogger(_TRACE_DIR)
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def _chat_with_timeout(self, messages, **kwargs):
        try:
            return await asyncio.wait_for(
                self.client.chat(messages, **kwargs),
                timeout=_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            from common import LLMResponse
            return LLMResponse(content="", error="Timeout after 120s")

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(workflow_name=workflow["name"], architecture_name="MCP Mediator")
        metrics.mcps_total = self.mcp_count

        all_tool_names = self.registry.get_tool_names()
        total_schema_tokens = self.registry.schema_tokens()
        metrics.tool_schema_tokens = total_schema_tokens
        metrics.tools_exposed = len(all_tool_names)
        metrics.mcps_activated = len(self.mcps)

        tool_text = self.registry.tool_schema_text(all_tool_names)
        plan_system = PLAN_PROMPT + tool_text

        plan_response = await self._chat_with_timeout(
            [{"role": "system", "content": plan_system}, {"role": "user", "content": workflow["prompt"]}],
            temperature=0.1, max_tokens=2048,
        )
        if plan_response.error:
            metrics.tools_truncated = -1
            warnings.warn(f"MCP Mediator SKIP {workflow['name']}: plan failed ({plan_response.error})", ResourceWarning)
            return metrics

        plan_latency = plan_response.latency_ms
        metrics.prompt_tokens += plan_response.usage.prompt_tokens
        metrics.completion_tokens += plan_response.usage.completion_tokens
        metrics.reasoning_tokens += plan_response.usage.reasoning_tokens
        metrics.total_tokens += plan_response.usage.total_tokens
        metrics.agent_hops += 1

        plan = self._parse_plan(plan_response.content or "")
        if not plan:
            retry = await self._chat_with_timeout(
                [{"role": "user", "content": plan_system + workflow["prompt"] + _RETRY_PROMPT}],
                temperature=0.1, max_tokens=2048,
            )
            if retry.error:
                metrics.tools_truncated = -1
                warnings.warn(f"MCP Mediator SKIP {workflow['name']}: plan parse failed after retry", ResourceWarning)
                return metrics
            plan_latency += retry.latency_ms
            metrics.prompt_tokens += retry.usage.prompt_tokens
            metrics.completion_tokens += retry.usage.completion_tokens
            metrics.reasoning_tokens += retry.usage.reasoning_tokens
            metrics.total_tokens += retry.usage.total_tokens
            metrics.agent_hops += 1
            plan = self._parse_plan(retry.content or "")
            if not plan:
                metrics.tools_truncated = -1
                warnings.warn(f"MCP Mediator SKIP {workflow['name']}: plan parse failed after retry", ResourceWarning)
                return metrics

        tools_used = set()
        exec_results = []
        for step in plan:
            name = step.get("name", "")
            args = step.get("arguments", {})
            if not name:
                continue
            result = self.registry.mock_call(name, **args)
            tools_used.add(name)
            exec_results.append({name: result})

        metrics.tools_used = len(tools_used)

        compile_prompt = COMPILE_PROMPT.format(task=workflow["prompt"], results=json.dumps(exec_results, indent=2))
        compile_response = await self._chat_with_timeout(
            [{"role": "user", "content": compile_prompt}], temperature=0.1,
        )
        if not compile_response.error:
            metrics.prompt_tokens += compile_response.usage.prompt_tokens
            metrics.completion_tokens += compile_response.usage.completion_tokens
            metrics.reasoning_tokens += compile_response.usage.reasoning_tokens
            metrics.total_tokens += compile_response.usage.total_tokens
            metrics.agent_hops += 1
            plan_latency += compile_response.latency_ms

        metrics.latency_ms = round(plan_latency, 1)
        metrics.explanation = (
            "Single LLM call generates a deterministic execution plan with all tool schemas loaded once. "
            "A lightweight executor runs all tool calls without further LLM cost. "
            "A final compile call produces the answer. Only 2 LLM hops total — "
            "dramatically fewer tokens than architectures that loop tool calls. "
            "Schema cost is paid once regardless of tool count."
        )
        self.logger.log_event(workflow["name"], "MCP Mediator", "done", metrics.snapshot())
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
        calls = parse_tool_calls(text)
        if calls:
            return calls
        return []
