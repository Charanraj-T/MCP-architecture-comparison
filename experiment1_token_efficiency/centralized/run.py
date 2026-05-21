import json
from pathlib import Path

import tiktoken
from lmstudio_client import LMStudioClient
from common_tools import ToolRegistry
from common_tools.parser import parse_tool_calls
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")


class CentralizedOrchestrator:
    def __init__(self, client: LMStudioClient, registry: ToolRegistry):
        self.client = client
        self.registry = registry
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(
            workflow_name=wf_name,
            architecture_name="Centralized MCP",
        )

        tool_text = self.registry.tool_schema_text()
        schema_tokens = self.registry.schema_tokens()
        user_prompt = workflow["prompt"]
        user_tokens = self._count_tokens(user_prompt)

        orchestration_instructions = (
            "You are an AI assistant with access to the following tools.\n"
            "Analyze the request and call tools to gather information.\n\n"
            "Output format:\n"
            'TOOL_CALL: {"name": "<tool_name>", "arguments": {...}}\n'
            "Call one tool per line. After all tool calls, output:\n"
            "FINAL_ANSWER: <your complete answer>\n\n"
            "Available tools:\n"
        )
        orchestration_tokens = self._count_tokens(orchestration_instructions)

        system_prompt = orchestration_instructions + tool_text

        metrics.tool_schema_tokens = schema_tokens
        metrics.tools_exposed = self.registry.count_tools()
        metrics.user_prompt_tokens = user_tokens
        metrics.orchestration_prompt_tokens = orchestration_tokens
        metrics.mcps_activated = 1
        metrics.agent_hops = 1

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tools_used: set[str] = set()
        total_latency = 0.0
        turn_count = 0

        for turn in range(6):
            turn_count += 1
            response = await self.client.chat(messages, temperature=0.1)

            total_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.output_tokens += response.usage.completion_tokens

            content = response.content or ""

            self.trace.record("llm_call", f"Turn {turn_count}", {
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
                    tools_used.add(name)
                    results.append({name: result})
                    self.trace.record("tool_call", name, {"args": args, "result": result})

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                break

        metrics.tools_used = len(tools_used)
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(wf_name, "Centralized MCP", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/centralized_{wf_name.replace(' ', '_')}.json")

        return metrics
