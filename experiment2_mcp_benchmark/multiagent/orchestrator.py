"""Multi-Agent orchestrator — supervisor decomposes, parallel workers execute."""
import asyncio
import json
import tiktoken
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.client import OpenAIClient
from metrics import WorkflowMetrics, JSONLogger, TraceCollector

_TRACE_DIR = str(Path(__file__).resolve().parent.parent / "traces")

DECOMPOSE_PROMPT = (
    "You are a supervisor. Decompose the task into domain-specific subtasks.\n"
    "Available agents: dev_agent (code/files), docs_agent (documentation/web), "
    "data_agent (database/SQL), planning_agent (reasoning/sequencing)\n"
    'Output JSON array: [{"agent": "agent_name", "task": "subtask description"}]'
)

COMPILE_PROMPT = (
    "You are a compiler. Given results from multiple specialist agents, "
    "compile them into a single coherent answer.\n"
    "Be concise and accurate. Combine insights from all agents."
)

AGENT_DOMAINS = {
    "dev_agent": ["filesystem", "git"],
    "docs_agent": ["fetch", "memory"],
    "data_agent": ["data"],
    "planning_agent": ["reasoning"],
}


class SimpleWorker:
    """A lightweight domain worker that connects to specific MCP servers."""

    def __init__(self, client, agent_name: str, mcp_connections: dict):
        self.client = client
        self.agent_name = agent_name
        self.connections = mcp_connections
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def run(self, task: str) -> dict:
        tools_used = set()
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "hops": 0, "tool_calls": 0}

        # Collect available tools from connected servers
        available_tools = []
        for server_key in AGENT_DOMAINS.get(self.agent_name, []):
            conn = self.connections.get(server_key)
            if conn and conn.tools:
                for tool in conn.tools:
                    available_tools.append(f"- {tool.name}: {tool.description}")

        tools_text = "\n".join(available_tools) if available_tools else "No tools available."
        system = (
            f"You are a specialist agent ({self.agent_name}).\n\n"
            f"Available tools:\n{tools_text}\n\n"
            "To use a tool, respond with: TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When done: FINAL_ANSWER: <your answer>"
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]

        for _turn in range(4):
            response = await self.client.chat(messages, temperature=0.1, max_tokens=1024)
            if response.error:
                break
            total_usage["prompt_tokens"] += response.usage.prompt_tokens
            total_usage["completion_tokens"] += response.usage.completion_tokens
            total_usage["reasoning_tokens"] += response.usage.reasoning_tokens
            total_usage["total_tokens"] += response.usage.total_tokens
            total_usage["latency_ms"] += response.latency_ms
            total_usage["hops"] += 1

            content = response.content or ""
            # Simple TOOL_CALL parsing
            if "TOOL_CALL:" in content:
                try:
                    start = content.index("TOOL_CALL:") + 10
                    json_str = content[start:].strip().split("\n")[0]
                    call = json.loads(json_str)
                    name = call.get("name", "")
                    # Execute via MCP connection
                    for server_key, conn in self.connections.items():
                        for tool in conn.tools:
                            if tool.name == name:
                                result = await conn.call_tool(name, call.get("args", {}))
                                tools_used.add(name)
                                total_usage["tool_calls"] += 1
                                messages.append({"role": "assistant", "content": content})
                                messages.append({"role": "user", "content": f"Result: {json.dumps(result)[:500]}\n\nContinue or FINAL_ANSWER:"})
                                break
                    else:
                        messages.append({"role": "assistant", "content": content})
                        break
                except (json.JSONDecodeError, ValueError):
                    messages.append({"role": "assistant", "content": content})
                    break
            else:
                return {"agent": self.agent_name, "answer": content, "usage": total_usage, "tools_used": list(tools_used)}

        return {"agent": self.agent_name, "answer": "", "usage": total_usage, "tools_used": list(tools_used)}


class MultiAgentOrchestrator:
    def __init__(self, client: OpenAIClient, mcp_connections: dict = None):
        self.client = client
        self.mcp_connections = mcp_connections or {}
        self.logger = JSONLogger(_TRACE_DIR)
        self.trace = TraceCollector()
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def _decompose(self, user_prompt: str) -> tuple[list[dict], object]:
        response = await self.client.chat(
            [{"role": "user", "content": DECOMPOSE_PROMPT + "\n\n" + user_prompt}],
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
                        valid_agents = set(AGENT_DOMAINS.keys())
                        if isinstance(subtasks, list) and all(s.get("agent") in valid_agents for s in subtasks):
                            return subtasks, response
                        return [{"agent": "dev_agent", "task": user_prompt}], response
            return [{"agent": "dev_agent", "task": user_prompt}], response
        except (ValueError, json.JSONDecodeError, KeyError):
            return [{"agent": "dev_agent", "task": user_prompt}], response

    async def close(self):
        for conn in self.mcp_connections.values():
            try:
                await conn.close()
            except Exception:
                pass

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
            worker = SimpleWorker(self.client, agent, self.mcp_connections)
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
        metrics.explanation = (
            "Supervisor LLM decomposes the task into subtasks. "
            "Parallel specialist workers (dev, docs, data, planning) each connect to their own MCP servers. "
            "Inter-agent serialization adds overhead but enables domain isolation. "
            "Compile hop merges all worker outputs into final answer."
        )

        self.trace.record("done", "completed", metrics.snapshot())
        self.trace.flush(f"{_TRACE_DIR}/multiagent_{wf_name.replace(' ', '_')}.json")

        self.logger.log_event(wf_name, "Multi-Agent", "completed", metrics.snapshot())

        return metrics
