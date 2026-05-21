import json
import re
import tiktoken
from pathlib import Path
from lmstudio_client import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector
from multiagent.supervisor import DECOMPOSE_PROMPT, COMPILE_PROMPT
from multiagent.workers.dev_worker import DevWorker
from multiagent.workers.docs_worker import DocsWorker
from multiagent.workers.data_worker import DataWorker
from multiagent.workers.planning_worker import PlanningWorker

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

AGENT_MAP = {
    "dev_agent": DevWorker,
    "docs_agent": DocsWorker,
    "data_agent": DataWorker,
    "planning_agent": PlanningWorker,
}


class MultiAgentOrchestrator:
    def __init__(self, client: LMStudioClient):
        self.client = client
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def _decompose(self, user_prompt: str) -> list[dict]:
        response = await self.client.chat(
            [{"role": "user", "content": DECOMPOSE_PROMPT + user_prompt}],
            temperature=0.1, max_tokens=512,
        )
        content = response.content or ""
        self.trace.record("supervisor", "decompose", {
            "tokens": response.usage.total_tokens,
            "response_preview": content[:300],
        })
        try:
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                subtasks = json.loads(match.group())
                if isinstance(subtasks, list) and all(s.get("agent") in AGENT_MAP for s in subtasks):
                    return subtasks, response
            return [{"agent": "dev_agent", "task": user_prompt}], response
        except (json.JSONDecodeError, KeyError):
            return [{"agent": "dev_agent", "task": user_prompt}], response

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Multi-Agent")
        metrics.user_prompt_tokens = self._count_tokens(workflow["prompt"])

        decompose_tokens = self._count_tokens(DECOMPOSE_PROMPT)
        compile_tokens = self._count_tokens(COMPILE_PROMPT)
        metrics.orchestration_prompt_tokens = decompose_tokens + compile_tokens

        subtasks, decompose_resp = await self._decompose(workflow["prompt"])
        metrics.prompt_tokens += decompose_resp.usage.prompt_tokens
        metrics.completion_tokens += decompose_resp.usage.completion_tokens
        metrics.reasoning_tokens += decompose_resp.usage.reasoning_tokens
        metrics.total_tokens += decompose_resp.usage.total_tokens
        metrics.agent_hops += 1

        self.trace.record("supervisor", "subtasks", {"subtasks": subtasks})

        worker_results = []
        active_agents = set()

        async def run_worker(st: dict):
            agent = st.get("agent", "dev_agent")
            task = st.get("task", workflow["prompt"])
            active_agents.add(agent)
            worker_class = AGENT_MAP[agent]
            worker = worker_class(self.client)
            result = await worker.run(task)
            metrics.mcps_activated += 1
            self.trace.record("worker", agent, {
                "task_preview": task[:100],
                "usage": result["usage"],
            })
            return result

        import asyncio
        worker_results = await asyncio.gather(*[run_worker(st) for st in subtasks])

        all_tools_used = set()
        for result in worker_results:
            metrics.prompt_tokens += result["usage"]["prompt_tokens"]
            metrics.completion_tokens += result["usage"]["completion_tokens"]
            metrics.total_tokens += result["usage"]["total_tokens"]
            metrics.latency_ms += result["usage"]["latency_ms"]
            metrics.agent_hops += result["usage"]["hops"]
            metrics.real_tool_calls += result["usage"]["tool_calls"]
            metrics.tools_exposed += result["usage"]["tool_calls"]
            all_tools_used.update(result.get("tools_used", []))

        metrics.tools_used = len(all_tools_used)
        metrics.tools_exposed = metrics.tools_exposed
        metrics.mcp_servers_connected = len(active_agents)

        inter_agent_text = json.dumps([{
            "agent": r["agent"],
            "answer_preview": r["answer"][:200],
        } for r in worker_results], indent=2)
        metrics.inter_agent_tokens = self._count_tokens(inter_agent_text)

        compile_prompt = COMPILE_PROMPT
        for r in worker_results:
            compile_prompt += f"<{r['agent']}>:\n{r['answer']}\n\n"

        compile_resp = await self.client.chat(
            [{"role": "user", "content": compile_prompt}],
            temperature=0.3, max_tokens=1024,
        )
        metrics.prompt_tokens += compile_resp.usage.prompt_tokens
        metrics.completion_tokens += compile_resp.usage.completion_tokens
        metrics.reasoning_tokens += compile_resp.usage.reasoning_tokens
        metrics.total_tokens += compile_resp.usage.total_tokens
        metrics.latency_ms += compile_resp.latency_ms
        metrics.agent_hops += 1

        self.trace.record("supervisor", "compile", {
            "tokens": compile_resp.usage.total_tokens,
        })

        metrics.latency_ms = round(metrics.latency_ms, 1)

        self.logger.log_event(wf_name, "Multi-Agent", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/multiagent_{wf_name.replace(' ', '_')}.json")

        return metrics
