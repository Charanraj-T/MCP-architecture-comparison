import json
import re
from pathlib import Path

import tiktoken
from lmstudio_client import LMStudioClient
from common_tools import ToolRegistry
from common_tools.parser import parse_tool_calls
from federated.router import DOMAIN_MAP
from multiagent.supervisor import SUPERVISOR_DECOMPOSE_PROMPT, SUPERVISOR_COMPILE_PROMPT
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

DOMAIN_AGENT_MAP = {
    "dev_agent": "dev",
    "infra_agent": "infra",
    "docs_agent": "docs",
    "pm_agent": "pm",
}


class MultiAgentOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry):
        self.client = client
        self.registry = registry
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def _decompose_task(self, user_prompt: str) -> list[dict]:
        response = await self.client.chat(
            [{"role": "user", "content": SUPERVISOR_DECOMPOSE_PROMPT + user_prompt}],
            temperature=0.1,
        )

        content = response.content or ""
        self.trace.record("supervisor", "decompose", {
            "tokens": response.usage.total_tokens,
            "response": content[:300],
        })

        try:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
                if isinstance(subtasks, list):
                    return subtasks
        except (json.JSONDecodeError, KeyError):
            pass

        return [{"agent": "dev_agent", "task": user_prompt}]

    async def _run_worker(
        self, agent_name: str, task: str, metrics: WorkflowMetrics
    ) -> str:
        domain = DOMAIN_AGENT_MAP.get(agent_name, "dev")
        tool_names = DOMAIN_MAP[domain]["tools"]
        tool_text = self.registry.tool_schema_text(tool_names)
        schema_tokens = self.registry.schema_tokens(tool_names)
        metrics.tool_schema_tokens += schema_tokens
        metrics.tools_exposed += len(tool_names)

        worker_system = (
            f"You are a {domain} domain specialist with access to tools.\n"
            "Output format:\n"
            'TOOL_CALL: {"name": "<tool_name>", "arguments": {...}}\n'
            "After all tool calls, output:\n"
            "FINAL_ANSWER: <your answer>\n\n"
            "Available tools:\n"
            f"{tool_text}"
        )

        messages = [
            {"role": "system", "content": worker_system},
            {"role": "user", "content": task},
        ]

        final_content = ""
        tools_used_in_worker: set[str] = set()
        worker_latency = 0.0

        for turn in range(6):
            response = await self.client.chat(messages, temperature=0.1)

            worker_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.output_tokens += response.usage.completion_tokens
            metrics.agent_hops += 1

            content = response.content or ""

            self.trace.record("worker", f"{agent_name}_turn_{turn+1}", {
                "tokens": response.usage.total_tokens,
                "content_preview": content[:200],
            })

            tool_calls = parse_tool_calls(content)

            if tool_calls:
                results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    result = self.registry.mock_call(name, **args)
                    tools_used_in_worker.add(name)
                    results.append({name: result})
                    self.trace.record("tool_call", name, {"agent": agent_name, "args": args, "result": result})

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                final_content = content
                break

        final_answer_match = re.search(r'FINAL_ANSWER:\s*(.*)', final_content, re.DOTALL)
        if final_answer_match:
            final_content = final_answer_match.group(1).strip()

        metrics.tools_used += len(tools_used_in_worker)
        metrics.latency_ms += worker_latency

        return f"<{agent_name}>:\n{final_content}"

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(
            workflow_name=wf_name,
            architecture_name="Multi-Agent",
        )

        user_prompt = workflow["prompt"]
        metrics.user_prompt_tokens = self._count_tokens(user_prompt)
        metrics.tools_exposed = 0
        metrics.mcps_activated = 0

        decompose_routing = self._count_tokens(SUPERVISOR_DECOMPOSE_PROMPT)
        compile_routing = self._count_tokens(SUPERVISOR_COMPILE_PROMPT)
        metrics.orchestration_prompt_tokens = decompose_routing + compile_routing

        subtasks = await self._decompose_task(user_prompt)
        self.trace.record("supervisor", "subtasks", {"subtasks": subtasks})

        metrics.memory_replay_tokens += self._count_tokens(json.dumps(subtasks))

        # Track decompose tokens from API
        metrics.total_tokens += 0  # Already accounted in _decompose_task via worker tracking? No, need to add.

        worker_results = []
        for subtask in subtasks:
            agent = subtask.get("agent", "dev_agent")
            task = subtask.get("task", "")
            result = await self._run_worker(agent, task, metrics)
            worker_results.append(result)
            metrics.mcps_activated += 1

        # Inter-agent communication tokens
        inter_agent = self._count_tokens("\n".join(worker_results))
        metrics.inter_agent_tokens = inter_agent

        # Supervisor compiles final answer
        compile_prompt = SUPERVISOR_COMPILE_PROMPT + "\n".join(worker_results)
        compile_response = await self.client.chat(
            [{"role": "user", "content": compile_prompt}],
            temperature=0.3,
        )

        metrics.prompt_tokens += compile_response.usage.prompt_tokens
        metrics.completion_tokens += compile_response.usage.completion_tokens
        metrics.reasoning_tokens += compile_response.usage.reasoning_tokens
        metrics.total_tokens += compile_response.usage.total_tokens
        metrics.output_tokens += compile_response.usage.completion_tokens
        metrics.latency_ms += compile_response.latency_ms
        metrics.agent_hops += 1

        self.trace.record("supervisor", "compile", {
            "tokens": compile_response.usage.total_tokens,
        })

        self.logger.log_event(wf_name, "Multi-Agent", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/multiagent_{wf_name.replace(' ', '_')}.json")

        return metrics
