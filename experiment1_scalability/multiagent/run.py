"""Multi-Agent orchestrator — supervisor decomposes, domain workers run in parallel."""
import asyncio
import json
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_tools.architecture_base import BaseOrchestrator
from common_tools.parser import parse_tool_calls
from metrics.token_tracker import WorkflowMetrics


DECOMPOSE_PROMPT = (
    "You are a supervisor. Decompose the task into domain-specific subtasks.\n"
    "Domains: code_search, log_analysis, deployment, monitoring, database, "
    "networking, security, documentation, project_mgmt, notifications\n"
    'Output JSON: {"subtasks": [{"domain": "...", "task": "..."}]}'
)

COMPILE_PROMPT = (
    "You are a compiler. Given results from multiple domain workers, "
    "compile them into a single coherent answer.\n"
    "When done, respond with: FINAL_ANSWER: <your answer>"
)


class MultiAgentOrchestrator(BaseOrchestrator):

    def _get_tools_for_agent(self, agent_domain: str) -> list[str]:
        """Get tool names for a single agent's domain."""
        names = []
        for mcp in self.mcps:
            if mcp["domain"] == agent_domain:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    def _empty_usage(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "latency_ms": 0.0}

    def _add_usage(self, acc: dict, resp) -> dict:
        if resp.usage:
            acc["prompt_tokens"] += getattr(resp.usage, 'prompt_tokens', 0) or 0
            acc["completion_tokens"] += getattr(resp.usage, 'completion_tokens', 0) or 0
            acc["reasoning_tokens"] += getattr(resp.usage, 'reasoning_tokens', 0) or 0
            acc["total_tokens"] += getattr(resp.usage, 'total_tokens', 0) or 0
        acc["latency_ms"] += resp.latency_ms
        return acc

    def _decompose(self, metrics: WorkflowMetrics, usage: dict) -> None:
        """Merge accumulated usage into metrics."""
        metrics.prompt_tokens += usage["prompt_tokens"]
        metrics.completion_tokens += usage["completion_tokens"]
        metrics.reasoning_tokens += usage["reasoning_tokens"]
        metrics.total_tokens += usage["total_tokens"]

    async def _run_worker(self, domain: str, task: str, metrics: WorkflowMetrics) -> tuple[str, set]:
        """Run a single domain worker and return (answer, tools_used)."""
        tools_used = set()
        agent_tools = self._get_tools_for_agent(domain)
        if not agent_tools:
            return f"No tools available for {domain}", tools_used

        schema_text = self.registry.tool_schema_text(agent_tools)
        system = (
            f"You are a {domain} specialist.\n\nTools:\n{schema_text}\n\n"
            "To use a tool: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When done: FINAL_ANSWER: <your answer>"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        for _turn in range(4):
            response = await self.client.chat(messages, temperature=0.1)
            if response.error:
                break
            self._accumulate_usage(metrics, response)

            content = response.content or ""
            calls = parse_tool_calls(content)

            if calls:
                results = []
                for tc in calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if not name:
                        continue
                    result = self.registry.mock_call(name, **args)
                    tools_used.add(name)
                    results.append({name: result})
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                return content, tools_used

        return "", tools_used

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Multi-Agent",
            mcps_total=self.mcp_count,
        )

        all_tool_names = self.registry.get_tool_names()
        metrics.tools_exposed = len(all_tool_names)
        metrics.tool_schema_tokens = self.registry.schema_tokens()

        # Supervisor decomposes
        decomp_messages = [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Task: {workflow['prompt']}"},
        ]
        decomp_response = await self.client.chat(decomp_messages, temperature=0.1)
        if decomp_response.error:
            warnings.warn(f"Decompose error: {decomp_response.error}")
            metrics.tools_truncated = -1
            return metrics
        self._accumulate_usage(metrics, decomp_response)

        # Parse subtasks
        try:
            match = re.search(r'\{.*\}', decomp_response.content or "", re.DOTALL)
            subtasks = json.loads(match.group()).get("subtasks", []) if match else []
        except (json.JSONDecodeError, AttributeError):
            subtasks = [{"domain": "code_search", "task": workflow["prompt"]}]

        # Run workers in parallel
        all_tools_used = set()
        worker_results = []
        domains_used = set()

        async def _worker(domain, task):
            answer, tools = await self._run_worker(domain, task, metrics)
            return domain, answer, tools

        tasks = [_worker(st["domain"], st["task"]) for st in subtasks if "domain" in st and "task" in st]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    continue
                domain, answer, tools = r
                all_tools_used.update(tools)
                domains_used.add(domain)
                worker_results.append(f"[{domain}]: {answer}")

        metrics.tools_used = len(all_tools_used)
        metrics.mcps_activated = len(domains_used)

        # Compile results
        if worker_results:
            compile_messages = [
                {"role": "system", "content": COMPILE_PROMPT},
                {"role": "user", "content": "\n\n".join(worker_results)},
            ]
            compile_response = await self.client.chat(compile_messages, temperature=0.1)
            if not compile_response.error:
                self._accumulate_usage(metrics, compile_response)

        return metrics
