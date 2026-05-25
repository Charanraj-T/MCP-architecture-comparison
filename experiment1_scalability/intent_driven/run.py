import asyncio
import json
import re
import warnings
from pathlib import Path

import tiktoken
from common import LMStudioClient
from common_tools import ToolRegistry
from common_tools.factory import DOMAIN_TYPES
from metrics import WorkflowMetrics, JSONLogger

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")
_CALL_TIMEOUT = 120

DOMAIN_DESC = "\n".join(f"- {d}: handles {d} tasks" for d in DOMAIN_TYPES)

INTENT_PROMPT = (
    "You are an intent-driven orchestrator. Given a user request, produce a structured execution plan "
    "in JSON format:\n\n"
    '{\n'
    '  "intent": "high-level description",\n'
    '  "plan": [\n'
    '    {\n'
    '      "step": 1,\n'
    '      "domain": "<domain_name>",\n'
    '      "action": "description",\n'
    '      "tool": "<tool_name>",\n'
    '      "arguments": {<args>},\n'
    '      "depends_on": [<step_numbers>]\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "Available domains:\n" + DOMAIN_DESC + "\n\n"
    "Rules:\n"
    "1. Only use relevant domains\n"
    "2. Steps with no dependencies can run in parallel (depends_on: [])\n"
    "3. Each step calls exactly one tool\n"
    "4. Use concrete argument values\n\n"
    "Request:\n"
)

COMPILE_PROMPT = (
    "You are a compiler. Given the original intent, plan, and execution results, "
    "produce a final coherent answer.\n\n"
    "Original intent:\n{intent}\n\n"
    "Plan:\n{plan}\n\n"
    "Execution results:\n{results}\n\n"
    "Final answer:\n"
)

_RETRY_PROMPT = (
    "\n\nThe previous response was not valid JSON. "
    "Output ONLY valid JSON. No other text."
)


class IntentDrivenOrchestrator:
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

    def _get_tools_for_domains(self, domains: set[str]) -> list[str]:
        names = []
        for mcp in self.mcps:
            if mcp["domain"] in domains:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(workflow_name=workflow["name"], architecture_name="Intent-Driven")
        metrics.mcps_total = self.mcp_count

        intent_prompt = INTENT_PROMPT + workflow["prompt"]

        plan_response = await self._chat_with_timeout(
            [{"role": "user", "content": intent_prompt}], temperature=0.1, max_tokens=2048,
        )
        if plan_response.error:
            metrics.tools_truncated = -1
            warnings.warn(f"Intent-Driven SKIP {workflow['name']}: plan failed ({plan_response.error})", ResourceWarning)
            return metrics

        total_latency = plan_response.latency_ms
        metrics.prompt_tokens += plan_response.usage.prompt_tokens
        metrics.completion_tokens += plan_response.usage.completion_tokens
        metrics.reasoning_tokens += plan_response.usage.reasoning_tokens
        metrics.total_tokens += plan_response.usage.total_tokens
        metrics.agent_hops += 1

        plan_data = self._parse_intent_plan(plan_response.content or "")
        if not plan_data or "plan" not in plan_data:
            retry = await self._chat_with_timeout(
                [{"role": "user", "content": intent_prompt + _RETRY_PROMPT}],
                temperature=0.1, max_tokens=2048,
            )
            if retry.error:
                metrics.tools_truncated = -1
                warnings.warn(f"Intent-Driven SKIP {workflow['name']}: plan parse failed", ResourceWarning)
                return metrics
            total_latency += retry.latency_ms
            metrics.prompt_tokens += retry.usage.prompt_tokens
            metrics.completion_tokens += retry.usage.completion_tokens
            metrics.reasoning_tokens += retry.usage.reasoning_tokens
            metrics.total_tokens += retry.usage.total_tokens
            metrics.agent_hops += 1
            plan_data = self._parse_intent_plan(retry.content or "")
            if not plan_data or "plan" not in plan_data:
                metrics.tools_truncated = -1
                warnings.warn(f"Intent-Driven SKIP {workflow['name']}: plan parse failed after retry", ResourceWarning)
                return metrics

        plan_steps = plan_data["plan"]
        domains_used = set(s.get("domain", "") for s in plan_steps)
        active_tool_names = self._get_tools_for_domains(domains_used)
        metrics.tool_schema_tokens = self.registry.schema_tokens(active_tool_names)
        metrics.tools_exposed = len(active_tool_names)
        metrics.mcps_activated = sum(1 for m in self.mcps if m["domain"] in domains_used)

        dep_map = {s.get("step", 0): s.get("depends_on", []) for s in plan_steps}

        completed = set()
        exec_results = []
        tools_used = set()
        remaining = list(plan_steps)
        max_iters = len(plan_steps) * 3

        while remaining and max_iters > 0:
            max_iters -= 1
            batch = []
            still_remaining = []
            for s in remaining:
                deps = dep_map.get(s.get("step", 0), [])
                if all(d in completed for d in deps):
                    batch.append(s)
                else:
                    still_remaining.append(s)
            if not batch:
                still_remaining = remaining
                break
            for s in batch:
                tool_name = s.get("tool", "")
                args = s.get("arguments", {})
                if tool_name:
                    result = self.registry.mock_call(tool_name, **args)
                    tools_used.add(tool_name)
                    exec_results.append({f"step_{s['step']}": {tool_name: result}})
                completed.add(s["step"])
            remaining = still_remaining

        for s in remaining:
            tool_name = s.get("tool", "")
            args = s.get("arguments", {})
            if tool_name:
                result = self.registry.mock_call(tool_name, **args)
                tools_used.add(tool_name)
                exec_results.append({f"step_{s['step']}": {tool_name: result}})

        metrics.tools_used = len(tools_used)

        compile_prompt = COMPILE_PROMPT.format(
            intent=plan_data.get("intent", workflow["name"]),
            plan=json.dumps(plan_steps, indent=2),
            results=json.dumps(exec_results, indent=2),
        )

        compile_response = await self._chat_with_timeout(
            [{"role": "user", "content": compile_prompt}], temperature=0.1,
        )
        if not compile_response.error:
            metrics.prompt_tokens += compile_response.usage.prompt_tokens
            metrics.completion_tokens += compile_response.usage.completion_tokens
            metrics.reasoning_tokens += compile_response.usage.reasoning_tokens
            metrics.total_tokens += compile_response.usage.total_tokens
            metrics.agent_hops += 1
            total_latency += compile_response.latency_ms

        metrics.latency_ms = round(total_latency, 1)
        metrics.explanation = (
            "Single LLM call produces a structured intent plan (JSON with dependencies). "
            "Only tools from domains in the plan are loaded — schema cost is proportional to task scope. "
            "A dependency-aware executor runs tool calls, parallelizing independent steps. "
            "A compile hop produces the answer. Only 2 LLM hops total. "
            "Most token-efficient per-task but requires LLM to output valid structured JSON."
        )
        self.logger.log_event(workflow["name"], "Intent-Driven", "done", metrics.snapshot())
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
