import json
import tiktoken


class ToolRegistry:
    def __init__(self):
        self._tools = []
        self._tools_by_name = {}
        self._enc = tiktoken.get_encoding("cl100k_base")

    def register_many(self, tool_defs: list[dict]):
        for t in tool_defs:
            self._tools.append(t)
            self._tools_by_name[t["name"]] = t

    def clear(self):
        self._tools.clear()
        self._tools_by_name.clear()

    @property
    def all_tools(self):
        return list(self._tools)

    def get_tool(self, name: str) -> dict | None:
        return self._tools_by_name.get(name)

    def get_tool_names(self) -> list[str]:
        return [t["name"] for t in self._tools]

    def count_tools(self) -> int:
        return len(self._tools)

    def schema_tokens(self, tool_names: list[str] = None) -> int:
        tools = self._tools
        if tool_names is not None:
            tools = [t for t in self._tools if t["name"] in tool_names]
        text = ""
        for t in tools:
            text += f"Tool: {t['name']}\nDescription: {t['description']}\nSchema: {json.dumps(t['input_schema'])}\n\n"
        return len(self._enc.encode(text))

    def tool_schema_text(self, tool_names: list[str] = None) -> str:
        tools = self._tools
        if tool_names is not None:
            tools = [t for t in self._tools if t["name"] in tool_names]
        lines = []
        for t in tools:
            lines.append(f"## {t['name']}")
            lines.append(f"Description: {t['description']}")
            lines.append(f"Schema: {json.dumps(t['input_schema'])}")
            lines.append("")
        return "\n".join(lines)

    def mock_call(self, tool_name: str, **kwargs) -> dict:
        tool = self.get_tool(tool_name)
        if tool is None:
            return {"error": f"unknown tool: {tool_name}"}
        return dict(tool["mock_response"])
