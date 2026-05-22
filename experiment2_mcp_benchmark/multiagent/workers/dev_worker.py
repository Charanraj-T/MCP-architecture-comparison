import json
import re
from common import LMStudioClient
from mcp_client import connect_all_mcps, close_all


class DevWorker:
    def __init__(self, client: LMStudioClient):
        self.client = client

    def _tool_schema_block(self, tools: list) -> str:
        lines = []
        for t in tools:
            lines.append(f"## {t.name}")
            lines.append(f"Description: {t.description}")
            lines.append(f"Schema: {json.dumps(t.input_schema, indent=2)}")
            lines.append("")
        return "\n".join(lines)

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

    async def run(self, task: str) -> dict:
        connections = await connect_all_mcps(["filesystem", "git"])
        mcps_count = len(connections)
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "latency_ms": 0.0, "hops": 0, "tool_calls": 0}
        tools_used = set()

        all_tools = []
        for conn in connections.values():
            all_tools.extend(conn.tools)

        system = (
            "You are a Dev Agent specializing in file system and git operations.\n"
            'Call tools: TOOL_CALL: {"name": "...", "arguments": {...}}\n'
            "After all calls: FINAL_ANSWER: <answer>\n\n"
            "Available tools:\n" + self._tool_schema_block(all_tools)
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        for turn in range(6):
            resp = await self.client.chat(messages, temperature=0.1)
            if resp.error:
                await close_all(connections)
                return {"agent": "dev_agent", "answer": "", "usage": usage_data, "tools_used": list(tools_used), "mcps_activated": mcps_count}
            usage_data["prompt_tokens"] += resp.usage.prompt_tokens
            usage_data["completion_tokens"] += resp.usage.completion_tokens
            usage_data["reasoning_tokens"] += resp.usage.reasoning_tokens
            usage_data["total_tokens"] += resp.usage.total_tokens
            usage_data["latency_ms"] += resp.latency_ms
            usage_data["hops"] += 1

            content = resp.content or ""
            calls = self._parse_tool_calls(content)

            if calls:
                results = []
                for tc in calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    for conn in connections.values():
                        for t in conn.tools:
                            if t.name == name:
                                result = await conn.call_tool(name, args)
                                usage_data["tool_calls"] += 1
                                tools_used.add(name)
                                text = ""
                                for c in result.get("content", [{}]):
                                    if isinstance(c, dict):
                                        text += c.get("text", json.dumps(c))
                                    else:
                                        text += str(c)
                                results.append({name: text})
                                break
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or FINAL_ANSWER."})
            else:
                messages.append({"role": "assistant", "content": content})
                match = re.search(r'FINAL_ANSWER:\s*(.*)', content, re.DOTALL)
                answer = match.group(1).strip() if match else content
                await close_all(connections)
                return {
                    "agent": "dev_agent",
                    "answer": answer,
                    "usage": usage_data,
                    "tools_used": list(tools_used),
                    "mcps_activated": mcps_count,
                }

        await close_all(connections)
        return {"agent": "dev_agent", "answer": content, "usage": usage_data, "tools_used": list(tools_used), "mcps_activated": mcps_count}
