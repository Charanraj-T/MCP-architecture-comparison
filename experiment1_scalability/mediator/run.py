"""MCP Mediator orchestrator — plan-then-execute with zero-cost tool execution."""
import json
import re
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_tools.architecture_base import BaseOrchestrator, _CALL_TIMEOUT
from common_tools.parser import parse_tool_calls
from metrics.token_tracker import WorkflowMetrics


def _parse_plan(text: str) -> list[dict]:
    """Parse execution plan from LLM output."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            return obj.get("steps", [])
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


class MediatorOrchestrator(BaseOrchestrator):

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="MCP Mediator",
            mcps_total=self.mcp_count,
        )

        all_tool_names = self.registry.get_tool_names()
        metrics.tools_exposed = len(all_tool_names)
        metrics.tool_schema_tokens = self.registry.schema_tokens()
        metrics.mcps_activated = self.mcp_count

        # Planning step (pays schema cost once)
        schema_text = self.registry.tool_schema_text()
        plan_system = (
            f"You have these tools available:\n\n{schema_text}\n\n"
            "Given a task, output a JSON execution plan: "
            '{"steps": [{"tool": "tool_name", "args": {...}}, ...]}\n'
            "Output ONLY valid JSON."
        )
        plan_messages = [
            {"role": "system", "content": plan_system},
            {"role": "user", "content": f"Task: {workflow['prompt']}"},
        ]

        plan_response = await self._chat_with_timeout(plan_messages, temperature=0.1)
        if plan_response.error:
            warnings.warn(f"Plan error: {plan_response.error}")
            metrics.tools_truncated = -1
            return metrics
        self._accumulate_usage(metrics, plan_response)

        steps = _parse_plan(plan_response.content or "")

        # Execute plan with zero LLM cost (direct mock calls)
        tools_used = set()
        for step in steps:
            name = step.get("tool", "")
            args = step.get("args", {})
            if name:
                self.registry.mock_call(name, **args)
                tools_used.add(name)

        metrics.tools_used = len(tools_used)
        metrics.latency_ms = plan_response.latency_ms
        return metrics
