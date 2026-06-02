"""Federated MCP orchestrator — router filters relevant domains first."""
import json
import re
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_tools.architecture_base import BaseOrchestrator
from common_tools.parser import parse_tool_calls
from metrics.token_tracker import WorkflowMetrics


def _extract_domains(text: str) -> list[str]:
    """Extract domain names from LLM router response."""
    domains = re.findall(r'(?:code_search|log_analysis|deployment|monitoring|database|networking|security|documentation|project_mgmt|notifications)', text)
    return list(set(domains))


class FederatedOrchestrator(BaseOrchestrator):

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Federated MCP",
            mcps_total=self.mcp_count,
        )

        all_tool_names = self.registry.get_tool_names()
        metrics.tools_exposed = len(all_tool_names)
        metrics.tool_schema_tokens = self.registry.schema_tokens()

        # Router call
        router_system = (
            "You are a router. Given a task, list the relevant domains.\n"
            "Domains: code_search, log_analysis, deployment, monitoring, database, "
            "networking, security, documentation, project_mgmt, notifications\n"
            "Respond with ONLY the domain names, comma-separated."
        )
        router_messages = [
            {"role": "system", "content": router_system},
            {"role": "user", "content": f"Task: {workflow['prompt']}"},
        ]

        router_response = await self.client.chat(router_messages, temperature=0.1)
        if router_response.error:
            warnings.warn(f"Router error: {router_response.error}")
            metrics.tools_truncated = -1
            return metrics

        self._accumulate_usage(metrics, router_response)
        domains = _extract_domains(router_response.content or "")
        if not domains:
            domains = list(set(m["domain"] for m in self.mcps))

        domain_tools = self._get_tools_for_domains(domains)
        metrics.mcps_activated = len(set(
            mcp["domain"] for mcp in self.mcps if mcp["domain"] in domains
        ))
        metrics.tools_exposed = len(domain_tools)
        schema_text = self.registry.tool_schema_text(domain_tools)

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
