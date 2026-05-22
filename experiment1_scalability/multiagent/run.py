import json
import re
from pathlib import Path

import tiktoken
from lmstudio_client import LMStudioClient
from common_tools import ToolRegistry
from common_tools.factory import DOMAIN_TYPES
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

SUPERVISOR_DECOMPOSE = (
    "You are a supervisor. Decompose this task into subtasks for these specialized agents:\n"
    + "\n".join(f"- {d}: handles {d} tasks" for d in DOMAIN_TYPES)
    + "\n\nOutput JSON array:\n"
    '[{"agent": "code_search", "task": "..."}, ...]\n\n'
    "Task:\n"
)

SUPERVISOR_COMPILE = (
    "You are a supervisor. Compile the following worker agent results into a coherent final answer.\n\n"
    "Worker Results:\n"
)

DOMAIN_AGENT_MAP = {d: d for d in DOMAIN_TYPES}


class MultiAgentOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry, mcps: list[dict]):
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _get_tools_for_agent(self, agent_domain: str) -> list[str]:
        names = []
        for mcp in self.mcps:
            if mcp["domain"] == agent_domain:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    def _empty_usage(self):
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "latency_ms": 0.0}

    def _add_usage(self, acc: dict, resp) -> dict:
        acc["prompt_tokens"] += resp.usage.prompt_tokens
        acc["completion_tokens"] += resp.usage.completion_tokens
        acc["reasoning_tokens"] += resp.usage.reasoning_tokens
        acc["total_tokens"] += resp.usage.total_tokens
        acc["latency_ms"] += resp.latency_ms
        return acc

    async def _decompose(self, prompt: str) -> tuple[list[dict], dict]:
        resp = await self.client.chat([{"role": "user", "content": SUPERVISOR_DECOMPOSE + prompt}], temperature=0.1)
        content = resp.content or ""
        usage = self._add_usage(self._empty_usage(), resp)
        try:
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                sub = json.loads(match.group())
                if isinstance(sub, list) and all(s.get("agent") in DOMAIN_TYPES for s in sub):
                    return sub, usage
        except (json.JSONDecodeError, KeyError):
            pass
        domain = DOMAIN_TYPES[0]
        return [{"agent": domain, "task": prompt}], usage

    async def _run_worker(self, agent_domain: str, task: str, results: list):
        tool_names = self._get_tools_for_agent(agent_domain)
        if not tool_names:
            results.append((agent_domain, "No tools available for this domain.", set(), self._empty_usage()))
            return

        tool_text = self.registry.tool_schema_text(tool_names)
        system = (
            f"You are a {agent_domain} specialist.\n"
            'Output: TOOL_CALL: {"name": "...", "arguments": {...}}\n'
            "After all calls: FINAL_ANSWER: <answer>\n\n"
            f"Available tools:\n{tool_text}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        final_content = ""
        tools_used = set()
        usage = self._empty_usage()

        for turn in range(6):
            resp = await self.client.chat(messages, temperature=0.1)
            usage = self._add_usage(usage, resp)
            content = resp.content or ""
            calls = parse_tool_calls(content)

            if calls:
                tool_results = []
                for tc in calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    tr = self.registry.mock_call(name, **args)
                    tools_used.add(name)
                    tool_results.append({name: tr})
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Tool results:\n{json.dumps(tool_results, indent=2)}\n\nContinue or provide FINAL_ANSWER."})
            else:
                final_content = content
                break

        match = re.search(r'FINAL_ANSWER:\s*(.*)', final_content, re.DOTALL)
        answer = match.group(1).strip() if match else final_content
        results.append((agent_domain, answer, tools_used, usage))

    async def run(self, workflow: dict) -> WorkflowMetrics:
        metrics = WorkflowMetrics(
            workflow_name=workflow["name"],
            architecture_name="Multi-Agent",
        )
        metrics.mcps_activated = 0
        metrics.tools_exposed = 0
        metrics.tools_used = 0
        metrics.agent_hops = 0

        subtasks, decompose_usage = await self._decompose(workflow["prompt"])
        metrics.agent_hops += 1
        metrics.prompt_tokens += decompose_usage["prompt_tokens"]
        metrics.completion_tokens += decompose_usage["completion_tokens"]
        metrics.reasoning_tokens += decompose_usage["reasoning_tokens"]
        metrics.total_tokens += decompose_usage["total_tokens"]

        worker_domains = list(set(s.get("agent", DOMAIN_TYPES[0]) for s in subtasks))
        mcp_activated_count = sum(1 for m in self.mcps if m["domain"] in worker_domains)
        metrics.mcps_activated = mcp_activated_count

        worker_results = []
        all_tools_used = set()
        for st in subtasks:
            agent_domain = st.get("agent", DOMAIN_TYPES[0])
            task = st.get("task", workflow["prompt"])
            await self._run_worker(agent_domain, task, worker_results)
            metrics.agent_hops += 1
            worker_tools = self._get_tools_for_agent(agent_domain)
            metrics.tools_exposed += len(worker_tools)

        for domain, answer, tools_used, usage in worker_results:
            all_tools_used.update(tools_used)
            metrics.prompt_tokens += usage["prompt_tokens"]
            metrics.completion_tokens += usage["completion_tokens"]
            metrics.reasoning_tokens += usage["reasoning_tokens"]
            metrics.total_tokens += usage["total_tokens"]

        metrics.tools_used = len(all_tools_used)

        compile_prompt = SUPERVISOR_COMPILE
        for domain, answer, tools_used, usage in worker_results:
            compile_prompt += f"<{domain}>:\n{answer}\n\n"

        final_resp = await self.client.chat([{"role": "user", "content": compile_prompt}], temperature=0.3)
        metrics.agent_hops += 1
        metrics.prompt_tokens += final_resp.usage.prompt_tokens
        metrics.completion_tokens += final_resp.usage.completion_tokens
        metrics.reasoning_tokens += final_resp.usage.reasoning_tokens
        metrics.total_tokens += final_resp.usage.total_tokens

        total_latency = decompose_usage["latency_ms"] + sum(u["latency_ms"] for _, _, _, u in worker_results) + final_resp.latency_ms
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(workflow["name"], "Multi-Agent", "done", metrics.snapshot())
        return metrics
