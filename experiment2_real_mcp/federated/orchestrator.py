import json
import tiktoken
from pathlib import Path
from lmstudio_client import LMStudioClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector
from mcp_client import connect_all_mcps, close_all, MCP_DEFINITIONS
from federated.router import classify_mcps

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")


class FederatedOrchestrator:
    def __init__(self, client: LMStudioClient):
        self.client = client
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _tool_schema_block(self, tools: list) -> str:
        lines = []
        for t in tools:
            lines.append(f"## {t.name}")
            lines.append(f"Description: {t.description}")
            lines.append(f"Schema: {json.dumps(t.input_schema, indent=2)}")
            lines.append("")
        return "\n".join(lines)

    def _schema_tokens(self, tools: list) -> int:
        return self._count_tokens(self._tool_schema_block(tools))

    def _parse_tool_calls(self, text: str) -> list[dict]:
        calls = []
        i = 0
        while True:
            idx = text.find("TOOL_CALL:", i)
            if idx == -1:
                break
            json_start = text.find("{", idx)
            if json_start == -1:
                i = idx + 10
                continue
            depth = 0
            json_end = -1
            for j in range(json_start, len(text)):
                ch = text[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_end = j + 1
                        break
            if depth == 0 and json_end > json_start:
                raw = text[json_start:json_end]
                try:
                    calls.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
            i = json_end if json_end > 0 else idx + 10
        return calls

    async def run(self, workflow: dict) -> WorkflowMetrics:
        wf_name = workflow["name"]
        metrics = WorkflowMetrics(workflow_name=wf_name, architecture_name="Federated MCP")
        metrics.user_prompt_tokens = self._count_tokens(workflow["prompt"])

        selected_mcps, router_tokens_used = await classify_mcps(self.client, workflow["prompt"])
        metrics.router_tokens = router_tokens_used
        metrics.mcps_activated = len(selected_mcps)

        self.trace.record("router", "classification", {
            "selected_mcps": selected_mcps,
            "tokens": router_tokens_used,
        })

        connections = await connect_all_mcps(selected_mcps)
        metrics.mcp_servers_connected = len(connections)

        all_tools = []
        for key, conn in connections.items():
            all_tools.extend(conn.tools)

        metrics.tools_exposed = len(all_tools)
        metrics.tool_schema_tokens = self._schema_tokens(all_tools)

        orchestration_header = (
            f"You are an AI assistant with access to the following domain MCPs: {', '.join(selected_mcps)}.\n"
            "Call tools using:\n"
            'TOOL_CALL: {"name": "<tool_name>", "arguments": {...}}\n'
            "After all tool calls, output:\n"
            "FINAL_ANSWER: <your complete answer>\n\n"
            "Available tools:\n"
        )
        metrics.orchestration_prompt_tokens = self._count_tokens(orchestration_header)

        system_prompt = orchestration_header + self._tool_schema_block(all_tools)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": workflow["prompt"]},
        ]

        tools_used = set()
        total_latency = 0.0
        real_calls = 0

        for turn in range(8):
            response = await self.client.chat(messages, temperature=0.1)
            total_latency += response.latency_ms
            metrics.prompt_tokens += response.usage.prompt_tokens
            metrics.completion_tokens += response.usage.completion_tokens
            metrics.reasoning_tokens += response.usage.reasoning_tokens
            metrics.total_tokens += response.usage.total_tokens
            metrics.output_tokens += response.usage.completion_tokens
            metrics.agent_hops += 1

            content = response.content or ""
            self.trace.record("llm_call", f"Turn {turn+1}", {
                "tokens": response.usage.total_tokens,
                "content_preview": content[:200],
            })

            tool_calls = self._parse_tool_calls(content)
            if tool_calls:
                results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    found = False
                    for key, conn in connections.items():
                        for t in conn.tools:
                            if t.name == name:
                                result = await conn.call_tool(name, args)
                                tools_used.add(name)
                                real_calls += 1
                                result_content = result.get("content", [{}])
                                text = ""
                                for c in result_content:
                                    if isinstance(c, dict):
                                        text += c.get("text", json.dumps(c))
                                    else:
                                        text += str(c)
                                results.append({name: text})
                                self.trace.record("tool_call", name, {
                                    "mcp": key, "args": args, "result_preview": text[:200],
                                })
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        results.append({name: f"Error: tool '{name}' not found"})

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                break

        metrics.tools_used = len(tools_used)
        metrics.real_tool_calls = real_calls
        metrics.latency_ms = round(total_latency, 1)

        self.logger.log_event(wf_name, "Federated MCP", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/federated_{wf_name.replace(' ', '_')}.json")

        await close_all(connections)
        return metrics
