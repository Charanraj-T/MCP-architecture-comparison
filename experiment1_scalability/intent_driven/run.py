"""Intent-Driven orchestrator — domain descriptions only, no tool schemas."""
import json
import re
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_tools.architecture_base import BaseOrchestrator, _CALL_TIMEOUT
from common_tools.parser import parse_tool_calls
from metrics.token_tracker import WorkflowMetrics


DOMAIN_DESCRIPTIONS = {
    "code_search": "Search codebase for functions, classes, patterns, and usages",
    "log_analysis": "Search and analyze application logs for errors and patterns",
    "deployment": "Manage Kubernetes deployments, pods, services, and rollbacks",
    "monitoring": "Query metrics, dashboards, alerts, and health checks",
    "database": "Query and manage database schemas, data, and migrations",
    "networking": "Manage DNS, firewalls, load balancers, and certificates",
    "security": "Vulnerability scans, access controls, secrets management",
    "documentation": "Search and manage technical documentation and wikis",
    "project_mgmt": "Manage tasks, sprints, issues, and project tracking",
    "notifications": "Send alerts via Slack, email, PagerDuty, webhooks",
}


def _parse_intent_plan(text: str) -> list[dict]:
    """Parse structured intent plan from LLM output."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            return obj.get("steps", [])
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


class IntentDrivenOrchestrator(BaseOrchestrator):

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Intent-Driven",
            mcps_total=self.mcp_count,
        )

        # Build domain description list (only ~75 tokens, not full schemas)
        desc_text = "\n".join(f"- {d}: {desc}" for d, desc in DOMAIN_DESCRIPTIONS.items())
        metrics.tool_schema_tokens = self._count_tokens(desc_text)
        metrics.tools_exposed = len(DOMAIN_DESCRIPTIONS)

        # Planning step
        plan_system = (
            f"You have access to these domains:\n{desc_text}\n\n"
            "Given a task, output a JSON plan with steps. Each step: "
            '{"domain": "<domain>", "action": "<what to do>"}\n'
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

        steps = _parse_intent_plan(plan_response.content or "")
        if not steps:
            steps = [{"domain": "code_search", "action": workflow["prompt"]}] if workflow.get("prompt") else []

        # Load only tools from planned domains
        planned_domains = set(s.get("domain", "") for s in steps)
        domain_tools = self._get_tools_for_domains(planned_domains)
        metrics.mcps_activated = len(planned_domains)
        metrics.tools_exposed = len(domain_tools)
        schema_text = self.registry.tool_schema_text(domain_tools) if domain_tools else "No tools loaded."

        # Execution step
        exec_system = (
            f"You have these tools available:\n\n{schema_text}\n\n"
            "To use a tool, respond with: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When done, respond with: FINAL_ANSWER: <your answer>"
        )
        exec_messages = [
            {"role": "system", "content": exec_system},
            {"role": "user", "content": f"Execute this plan: {json.dumps(steps, indent=2)}"},
        ]

        await self._execute_tool_loop(exec_messages, metrics, max_turns=len(steps) * 3 or 6)
        return metrics
