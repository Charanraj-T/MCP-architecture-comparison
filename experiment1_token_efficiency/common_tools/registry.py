import tiktoken
from typing import Callable


class ToolRegistry:
    def __init__(self):
        self._tools: list[dict] = []
        self._enc = tiktoken.get_encoding("cl100k_base")

    def register(self, tool_def: dict):
        self._tools.append(tool_def)

    def register_many(self, tool_defs: list[dict]):
        self._tools.extend(tool_defs)

    @property
    def all_tools(self) -> list[dict]:
        return list(self._tools)

    def get_tool(self, name: str) -> dict | None:
        for t in self._tools:
            if t["name"] == name:
                return t
        return None

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
            text += f"Tool: {t['name']}\n"
            text += f"Description: {t['description']}\n"
            text += f"Schema: {t['input_schema']}\n\n"
        return len(self._enc.encode(text))

    def build_fastmcp_server(self, tool_names: list[str] = None):
        from fastmcp import FastMCP

        tools = self._tools
        if tool_names is not None:
            tools = [t for t in self._tools if t["name"] in tool_names]

        name_hint = "_".join(sorted(set(t.get("name", "mcp").split("_")[0] for t in tools)))
        server = FastMCP(name_hint or "mcp")

        for tool_def in tools:
            name = tool_def["name"]
            description = tool_def["description"]
            schema = tool_def["input_schema"]
            mock = tool_def["mock_response"]
            params = {}
            for prop_name, prop_info in schema.get("properties", {}).items():
                params[prop_name] = (str, ...) if prop_name in schema.get("required", []) else (str, None)

            @server.tool(name=name, description=description)
            def dynamic_tool(**kwargs) -> dict:
                return mock

        return server

    def tool_schema_text(self, tool_names: list[str] = None) -> str:
        tools = self._tools
        if tool_names is not None:
            tools = [t for t in self._tools if t["name"] in tool_names]
        lines = []
        for t in tools:
            lines.append(f"## {t['name']}")
            lines.append(f"Description: {t['description']}")
            import json
            lines.append(f"Schema: {json.dumps(t['input_schema'], indent=2)}")
            lines.append("")
        return "\n".join(lines)

    def mock_call(self, tool_name: str, **kwargs) -> dict:
        tool = self.get_tool(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        return dict(tool["mock_response"])
