import json
import re
import warnings
from typing import Optional
from pathlib import Path

import tiktoken
from common import LMStudioClient
from common_tools import ToolRegistry
from common_tools.factory import DOMAIN_TYPES
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")



def _extract_domains(text: str) -> Optional[list[str]]:
    try:
        match = re.search(r'\{.*?"domains".*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            domains = data.get("domains", [])
            if isinstance(domains, list) and all(d in DOMAIN_TYPES for d in domains):
                return domains
    except (json.JSONDecodeError, KeyError):
        pass
    return None


class FederatedOrchestrator:
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

    def _get_tools_for_domains(self, domains: list[str]) -> list[str]:
        names = []
        for mcp in self.mcps:
            if mcp["domain"] in domains:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    def _estimate_next_request_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self._count_tokens(m.get("content", ""))
        return total + 500

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Federated MCP",
        )
        metrics.mcps_total = self.mcp_count

        domain_desc = "\n".join(f"- {d}" for d in DOMAIN_TYPES)
        router_prompt = (
            "Classify this user request into the most relevant domains from the list below.\n"
            "Choose ONLY domains that are strictly necessary for the task.\n\n"
            f"Available domains:\n{domain_desc}\n\n"
            "Example response format:\n"
            '{"domains": ["code_search", "database"]}\n\n'
            "Never include domains that are not needed. If no specific domain matches, return empty.\n"
            f'User request: {workflow["prompt"]}\n\n'
            'Respond with JSON only: {"domains": [...]}'
        )

        domains = None
        router_latency = 0.0
        for attempt in range(2):
            router_response = await self.client.chat([{"role": "user", "content": router_prompt}], temperature=0.1)
            if router_response.error:
                metrics.tools_truncated = -1
                warnings.warn(f"Federated MCP SKIPPED for {workflow['name']}: router crashed ({router_response.error})", ResourceWarning)
                return metrics
            router_latency += router_response.latency_ms
            metrics.prompt_tokens += router_response.usage.prompt_tokens
            metrics.completion_tokens += router_response.usage.completion_tokens
            metrics.reasoning_tokens += router_response.usage.reasoning_tokens
            metrics.total_tokens += router_response.usage.total_tokens
            metrics.agent_hops += 1

            domains = _extract_domains(router_response.content or "")
            if domains is not None:
                break
            if attempt == 0:
                router_prompt += "\n\nInvalid format. Respond with ONLY valid JSON: {\"domains\": [...]}"

        if domains is None:
            metrics.tools_truncated = -1
            warnings.warn(f"Federated MCP SKIPPED for {workflow['name']}: router returned invalid JSON after retry", ResourceWarning)
            return metrics

        total_latency = router_latency

        active_tool_names = self._get_tools_for_domains(domains)
        active_mcp_count = sum(1 for m in self.mcps if m["domain"] in domains)

        schema_tokens = self.registry.schema_tokens(active_tool_names)
        prompt_overhead = self._count_tokens(workflow["prompt"]) + 200

        system_overhead = self._count_tokens(
            f"You are an AI assistant with access to domain MCPs: {', '.join(domains)}.\n\n"
            "To call a tool, output a JSON code block:\n"
            '```json\n{"name": "tool_name", "arguments": {"arg1": "value1"}}\n```\n\n'
            "After all tool calls, output: FINAL_ANSWER: your final answer\n\n"
            "Available tools:\n"
        )

        usable_budget = self.total_token_budget - prompt_overhead - system_overhead - 500
        if usable_budget <= 0:
            metrics.tool_schema_tokens = schema_tokens
            metrics.tools_exposed = len(active_tool_names)
            metrics.mcps_activated = active_mcp_count
            metrics.tools_truncated = -1
            warnings.warn(
                f"Federated MCP SKIPPED for {workflow['name']}: "
                f"no budget for tools after overhead.",
                ResourceWarning,
            )
            return metrics

        selected_names = []
        running_total = 0
        for name in active_tool_names:
            tool = self.registry.get_tool(name)
            if tool is None:
                continue
            text = f"## {tool['name']}\nDescription: {tool['description']}\nSchema: {json.dumps(tool['input_schema'])}\n\n"
            tokens = self._count_tokens(text)
            if running_total + tokens <= usable_budget:
                selected_names.append(name)
                running_total += tokens

        if not selected_names:
            metrics.tool_schema_tokens = schema_tokens
            metrics.tools_exposed = len(active_tool_names)
            metrics.mcps_activated = active_mcp_count
            metrics.tools_truncated = -1
            warnings.warn(
                f"Federated MCP SKIPPED for {workflow['name']}: "
                f"tool schemas too large for budget.",
                ResourceWarning,
            )
            return metrics

        tool_text = self.registry.tool_schema_text(selected_names)
        schema_slice_tokens = self.registry.schema_tokens(selected_names)

        metrics.tool_schema_tokens = schema_slice_tokens
        metrics.tools_exposed = len(selected_names)
        metrics.tools_truncated = len(active_tool_names) - len(selected_names)
        metrics.mcps_activated = active_mcp_count
        # agent_hops accumulates router (already counted) + execution turns

        system = (
            f"You are an AI assistant with access to domain MCPs: {', '.join(domains)}.\n\n"
            "To call a tool, output a JSON code block:\n"
            '```json\n{"name": "tool_name", "arguments": {"arg1": "value1"}}\n```\n\n'
            "After all tool calls, output: FINAL_ANSWER: your final answer\n\n"
            "Available tools:\n" + tool_text
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": workflow["prompt"]},
        ]

        tools_used = set()

        for turn in range(6):
            next_est = self._estimate_next_request_tokens(messages)
            if next_est > self.total_token_budget:
                warnings.warn(
                    f"Federated MCP stopping early for {workflow['name']}: "
                    f"estimated next request {next_est} tokens exceeds budget.",
                    ResourceWarning,
                )
                break

            response = await self.client.chat(messages, temperature=0.1)
            if response.error:
                warnings.warn(f"Federated MCP: model error at turn {turn} ({response.error})", ResourceWarning)
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
        metrics.explanation = (
            "Router LLM call first filters to relevant domains, injecting only their tools. "
            "Extra hop for routing but schema tokens are lower than Centralized when filtering works. "
            "Router degrades at high MCP counts — may return all domains, eliminating savings."
        )

        self.logger.log_event(workflow["name"], "Federated MCP", "done", metrics.snapshot())
        return metrics
