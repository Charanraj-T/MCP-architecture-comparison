"""Centralized MCP orchestrator — all tools in one system prompt.

All available MCP tools are injected into the system prompt.
Skipped when schema tokens exceed budget.
"""
import json
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_tools.architecture_base import BaseOrchestrator
from common_tools.parser import parse_tool_calls
from metrics.token_tracker import WorkflowMetrics


class CentralizedOrchestrator(BaseOrchestrator):

    def _select_tools_for_budget(self, tool_names: list[str], fixed_overhead: int = 500) -> list[str]:
        """Select as many tools as fit within the token budget."""
        selected = []
        running = fixed_overhead
        for name in tool_names:
            tool = self.registry.get_tool(name)
            if not tool:
                continue
            est = self._count_tokens(
                f"Tool: {tool['name']}\nDescription: {tool['description']}\nSchema: {json.dumps(tool['input_schema'])}\n\n"
            )
            if running + est > self.total_token_budget:
                break
            selected.append(name)
            running += est
        return selected

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Centralized MCP",
            mcps_total=self.mcp_count,
        )

        all_tool_names = self.registry.get_tool_names()
        metrics.tools_exposed = len(all_tool_names)
        metrics.tool_schema_tokens = self.registry.schema_tokens()
        metrics.mcps_activated = self.mcp_count

        if metrics.tool_schema_tokens > self.total_token_budget:
            warnings.warn(f"Schema tokens {metrics.tool_schema_tokens} > budget {self.total_token_budget}, skipping")
            metrics.tools_truncated = -1
            return metrics

        selected = self._select_tools_for_budget(all_tool_names)
        metrics.tools_exposed = len(selected)
        schema_text = self.registry.tool_schema_text(selected)

        system = (
            f"You are an AI assistant with access to the following tools:\n\n{schema_text}\n\n"
            "To use a tool, respond with: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When done, respond with: FINAL_ANSWER: <your answer>"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {workflow['prompt']}"},
        ]

        await self._execute_tool_loop(messages, metrics, max_turns=6)
        return metrics
