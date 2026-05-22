import asyncio
import json
import tiktoken
from pathlib import Path
from common import LMStudioClient
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

    async def _decompose(self, user_prompt: str) -> tuple[list[dict], object]:
        response = await self.client.chat(
            [{"role": "user", "content": DECOMPOSE_PROMPT + user_prompt}],
            temperature=0.1, max_tokens=512,
        )
        if response.error:
            return [{"agent": "dev_agent", "task": user_prompt}], response
        content = response.content or ""
        try:
            start = content.index("[")
            depth = 0
            in_string = False
            for i in range(start, len(content)):
                ch = content[i]
                if ch == '"' and (i == 0 or content[i-1] != '\\'):
                    in_string = not in_string
                elif in_string:
                    continue
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        subtasks = json.loads(content[start:i+1])
                        if isinstance(subtasks, list) and all(s.get("agent") in AGENT_MAP for s in subtasks):
                            return subtasks, response
                        return [{"agent": "dev_agent", "task": user_prompt}], response
            return [{"agent": "dev_agent", "task": user_prompt}], response
        except (ValueError, json.JSONDecodeError, KeyError):
            return [{"agent": "dev_agent", "task": user_prompt}], response

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Multi-Agent")

        subtasks, decompose_resp = await self._decompose(workflow["prompt"])
        metrics.orchestration_prompt_tokens = self._count_tokens(DECOMPOSE_PROMPT + workflow["prompt"])
        self.trace.record("decompose", f"{len(subtasks)} subtasks", {
            "subtasks": subtasks, "tokens": decompose_resp.usage.total_tokens,
        })
        metrics.prompt_tokens += decompose_resp.usage.prompt_tokens
        metrics.completion_tokens += decompose_resp.usage.completion_tokens
        metrics.reasoning_tokens += decompose_resp.usage.reasoning_tokens
        metrics.total_tokens += decompose_resp.usage.total_tokens
        metrics.latency_ms += decompose_resp.latency_ms
        metrics.agent_hops += 1

        active_agents = set()

        async def run_worker(st: dict):
            agent = st.get("agent", "dev_agent")
            task = st.get("task", workflow["prompt"])
            active_agents.add(agent)
            worker_class = AGENT_MAP[agent]
            worker = worker_class(self.client)
            result = await worker.run(task)
            return result

        worker_results = await asyncio.gather(*[run_worker(st) for st in subtasks], return_exceptions=True)

        all_tools_used = set()
        seen_infra_agents = set()
        for result in worker_results:
            if isinstance(result, Exception):
                self.trace.record("worker_error", str(type(result).__name__), {"error": str(result)})
                continue
            metrics.prompt_tokens += result["usage"]["prompt_tokens"]
            metrics.completion_tokens += result["usage"]["completion_tokens"]
            metrics.reasoning_tokens += result["usage"]["reasoning_tokens"]
            metrics.total_tokens += result["usage"]["total_tokens"]
            metrics.latency_ms += result["usage"]["latency_ms"]
            metrics.agent_hops += result["usage"]["hops"]
            metrics.real_tool_calls += result["usage"]["tool_calls"]
            if result["agent"] not in seen_infra_agents:
                seen_infra_agents.add(result["agent"])
                metrics.mcps_activated += result.get("mcps_activated", 0)
                metrics.tool_schema_tokens += result.get("tool_schema_tokens", 0)
                metrics.tools_exposed += result.get("tools_exposed", 0)
            all_tools_used.update(result.get("tools_used", []))

        metrics.tools_used = len(all_tools_used)
        metrics.mcp_servers_connected = len(active_agents)

        successful_results = [r for r in worker_results if not isinstance(r, Exception)]
        for r in successful_results:
            self.trace.record("worker_result", r["agent"], {
                "tokens": r["usage"]["total_tokens"],
                "tools_used": r.get("tools_used", []),
                "answer_preview": r["answer"][:200],
            })
        inter_agent_text = json.dumps([{
            "agent": r["agent"],
            "answer_preview": r["answer"][:200],
        } for r in successful_results], indent=2)
        metrics.inter_agent_tokens = self._count_tokens(inter_agent_text)

        compile_prompt = COMPILE_PROMPT
        for r in successful_results:
            compile_prompt += f"<{r['agent']}>:\n{r['answer']}\n\n"

        self.trace.record("compile", f"{len(successful_results)} results", {
            "input_tokens": self._count_tokens(compile_prompt),
        })
        compile_resp = await self.client.chat(
            [{"role": "user", "content": compile_prompt}],
            temperature=0.3, max_tokens=1024,
        )
        if compile_resp.error:
            metrics.latency_ms = round(metrics.latency_ms, 1)
            self.trace.record("compile_error", "compile_failed", {"error": compile_resp.error})
            self.trace.flush(f"{_TRACE_DIR}/multiagent_{wf_name.replace(' ', '_')}.json")
            return metrics
        metrics.prompt_tokens += compile_resp.usage.prompt_tokens
        metrics.completion_tokens += compile_resp.usage.completion_tokens
        metrics.reasoning_tokens += compile_resp.usage.reasoning_tokens
        metrics.total_tokens += compile_resp.usage.total_tokens
        metrics.latency_ms += compile_resp.latency_ms
        metrics.agent_hops += 1

        metrics.latency_ms = round(metrics.latency_ms, 1)

        self.trace.record("done", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/multiagent_{wf_name.replace(' ', '_')}.json")

        self.logger.log_event(wf_name, "Multi-Agent", "completed", metrics.snapshot())

        return metrics
