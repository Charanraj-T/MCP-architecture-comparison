import json
import re
import warnings
from pathlib import Path

import tiktoken
from common import LMStudioClient
from common_tools import ToolRegistry
from common_tools.factory import DOMAIN_TYPES
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")



def _extract_domains(text: str) -> list[str]:
    try:
        match = re.search(r'\{.*?"domains".*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            domains = data.get("domains", [])
            if isinstance(domains, list) and all(d in DOMAIN_TYPES for d in domains):
                return domains
    except (json.JSONDecodeError, KeyError):
        pass
    return list(DOMAIN_TYPES)


class FederatedOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry, mcps: list[dict], total_token_budget: int = 38000):
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.total_token_budget = total_token_budget
        self.logger = JSONLogger(_TRACE_DIR)
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _get_tools_for_domains(self, domains: list[str]) -> list[str]:
        names = []
        for mcp in self.mcps:
            if mcp["domain"] in domains:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Federated MCP",
        )

        domain_desc = "\n".join(f"- {d}" for d in DOMAIN_TYPES)
        router_prompt = (
            "Classify this request into relevant domains. "
            "Only select absolutely necessary domains.\n\n"
            f"Available domains:\n{domain_desc}\n\n"
            f'Respond: {{"domains": ["domain1", "domain2"]}}\n\nRequest: {workflow["prompt"]}'
        )

        router_response = await self.client.chat([{"role": "user", "content": router_prompt}], temperature=0.1)
        metrics.prompt_tokens += router_response.usage.prompt_tokens
        metrics.completion_tokens += router_response.usage.completion_tokens
        metrics.reasoning_tokens += router_response.usage.reasoning_tokens
        metrics.total_tokens += router_response.usage.total_tokens
        total_latency = router_response.latency_ms

        domains = _extract_domains(router_response.content or "")

        active_tool_names = self._get_tools_for_domains(domains)
        mcp_count = sum(1 for m in self.mcps if m["domain"] in domains)

        tool_text = self.registry.tool_schema_text(active_tool_names)
        schema_tokens = self.registry.schema_tokens(active_tool_names)

        estimated = schema_tokens + self._count_tokens(router_prompt) + 200
        if estimated > self.total_token_budget:
            metrics.tool_schema_tokens = schema_tokens
            metrics.tools_exposed = len(active_tool_names)
            metrics.mcps_activated = mcp_count
            metrics.tools_truncated = -1
            warnings.warn(
                f"Federated MCP SKIPPED for {workflow['name']}: "
                f"estimated {estimated} tokens exceeds budget of {self.total_token_budget}.",
                ResourceWarning,
            )
            return metrics

        metrics.tool_schema_tokens = schema_tokens
        metrics.tools_exposed = len(active_tool_names)
        metrics.mcps_activated = mcp_count
        metrics.agent_hops = 0

        system = (
            f"You are an AI assistant with access to domain MCPs: {', '.join(domains)}.\n"
            'Output: TOOL_CALL: {"name": "...", "arguments": {...}}\n'
            "After all calls: FINAL_ANSWER: <answer>\n\n"
            "Available tools:\n" + tool_text
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": workflow["prompt"]},
        ]

        tools_used = set()

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
                    f"Federated MCP SKIPPED for {workflow['name']}: "
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

        self.logger.log_event(workflow["name"], "Federated MCP", "done", metrics.snapshot())
        return metrics
