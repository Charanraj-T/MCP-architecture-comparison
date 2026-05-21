import json
import warnings
from pathlib import Path

import tiktoken
from lmstudio_client import LMStudioClient
from common_tools import ToolRegistry
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

# Safety limit: total tool schema tokens allowed in the system prompt.
# When exceeded, the run is skipped entirely (tools_truncated = -1)
# rather than truncating tools, preserving clean experimental comparison.
TOOL_SCHEMA_TOKEN_BUDGET = 3500


class CentralizedOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry, mcps: list[dict]):
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
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

        if total_schema_tokens > TOOL_SCHEMA_TOKEN_BUDGET:
            metrics.tools_truncated = -1
            warnings.warn(
                f"Centralized MCP SKIPPED for {workflow['name']}: "
                f"{total_schema_tokens} schema tokens exceeds budget of {TOOL_SCHEMA_TOKEN_BUDGET} "
                f"({len(all_tool_names)} tools across {len(self.mcps)} MCPs). "
                f"Centralized architecture does not scale to this configuration.",
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
