import json
import warnings
from pathlib import Path

import tiktoken
from common import LMStudioClient
from common_tools import ToolRegistry
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")



class CentralizedOrchestrator:
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

    def _select_tools_for_budget(self, tool_names: list[str], fixed_overhead: int) -> list[str]:
        system_prefix = (
            "You are an AI assistant with access to tools.\n"
            'Output: TOOL_CALL: {"name": "...", "arguments": {...}}\n'
            "After all calls: FINAL_ANSWER: <answer>\n\n"
            "Available tools:\n"
        )
        prompt_tokens = self._count_tokens(system_prefix)
        usable_budget = self.total_token_budget - fixed_overhead - prompt_tokens - 500
        if usable_budget <= 0:
            return []
        selected = []
        running_total = 0
        for name in tool_names:
            tool = self.registry.get_tool(name)
            if tool is None:
                continue
            text = f"## {tool['name']}\nDescription: {tool['description']}\nSchema: {json.dumps(tool['input_schema'])}\n\n"
            tokens = self._count_tokens(text)
            if running_total + tokens <= usable_budget:
                selected.append(name)
                running_total += tokens
        return selected

    def _estimate_next_request_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self._count_tokens(m.get("content", ""))
        return total + 500

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Centralized MCP",
        )
        metrics.mcps_total = self.mcp_count

        all_tool_names = self.registry.get_tool_names()
        total_schema_tokens = self.registry.schema_tokens()

        prompt_overhead = self._count_tokens(workflow["prompt"]) + 200
        selected_names = self._select_tools_for_budget(all_tool_names, prompt_overhead)

        if not selected_names:
            metrics.mcps_activated = 0
            metrics.tool_schema_tokens = total_schema_tokens
            metrics.tools_exposed = len(all_tool_names)
            metrics.tools_truncated = -1
            warnings.warn(
                f"Centralized MCP SKIPPED for {workflow['name']}: "
                f"too many tools ({len(all_tool_names)}) to fit in budget of {self.total_token_budget}.",
                ResourceWarning,
            )
            return metrics

        schema_slice_tokens = self.registry.schema_tokens(selected_names)
        metrics.tool_schema_tokens = schema_slice_tokens
        metrics.tools_exposed = len(selected_names)
        metrics.tools_truncated = len(all_tool_names) - len(selected_names)
        metrics.mcps_activated = len(selected_names) // 3 + (1 if len(selected_names) % 3 else 0)

        metrics.agent_hops = 0
        tool_text = self.registry.tool_schema_text(selected_names)

        system = (
            "You are an AI assistant with access to tools.\n"
            'Output: TOOL_CALL: {"name": "...", "arguments": {...}}\n'
            "After all calls: FINAL_ANSWER: <answer>\n\n"
            "Available tools:\n" + tool_text
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": workflow["prompt"]},
        ]

        tools_used = set()
        total_latency = 0.0

        for turn in range(6):
            next_est = self._estimate_next_request_tokens(messages)
            if next_est > self.total_token_budget:
                warnings.warn(
                    f"Centralized MCP stopping early for {workflow['name']}: "
                    f"estimated next request {next_est} tokens exceeds budget.",
                    ResourceWarning,
                )
                break

            response = await self.client.chat(messages, temperature=0.1)
            if response.error:
                warnings.warn(f"Centralized MCP: model error at turn {turn} ({response.error})", ResourceWarning)
                break
            total_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.agent_hops += 1

            content = response.content or ""
            calls = parse_tool_calls(content)

            if calls:
                results = []
                for tc in calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    result = self.registry.mock_call(name, **args)
                    tools_used.add(name)
                    results.append({name: result})
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER."})
            else:
                messages.append({"role": "assistant", "content": content})
                break

        metrics.tools_used = len(tools_used)
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(workflow["name"], "Centralized MCP", "done", metrics.snapshot())
        return metrics
