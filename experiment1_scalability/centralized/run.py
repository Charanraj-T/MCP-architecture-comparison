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
    def __init__(self, client: LMStudioClient, registry: ToolRegistry, mcps: list[dict], total_token_budget: int = 38000):
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.total_token_budget = total_token_budget
        self.logger = JSONLogger(_TRACE_DIR)
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Centralized MCP",
        )
        metrics.mcps_activated = len(self.mcps)

        all_tool_names = self.registry.get_tool_names()
        total_schema_tokens = self.registry.schema_tokens()
        metrics.tool_schema_tokens = total_schema_tokens
        metrics.tools_exposed = len(all_tool_names)

        estimated_total = total_schema_tokens + self._count_tokens(workflow["prompt"]) + 200
        if estimated_total > self.total_token_budget:
            metrics.tools_truncated = -1
            warnings.warn(
                f"Centralized MCP SKIPPED for {workflow['name']}: "
                f"estimated {estimated_total} tokens exceeds budget of {self.total_token_budget} "
                f"({len(all_tool_names)} tools across {len(self.mcps)} MCPs).",
                ResourceWarning,
            )
            return metrics

        metrics.agent_hops = 0
        tool_text = self.registry.tool_schema_text()

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
            response = await self.client.chat(messages, temperature=0.1)
            total_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.agent_hops += 1

            if metrics.total_tokens > self.total_token_budget:
                metrics.tools_truncated = -1
                warnings.warn(
                    f"Centralized MCP SKIPPED for {workflow['name']}: "
                    f"{metrics.total_tokens} total tokens exceeded budget of {self.total_token_budget}.",
                    ResourceWarning,
                )
                return metrics

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
                break

        metrics.tools_used = len(tools_used)
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(workflow["name"], "Centralized MCP", "done", metrics.snapshot())
        return metrics
