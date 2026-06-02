"""Tests for common_tools/factory.py — MCP generation and tool registry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment1_scalability"))

from common_tools.factory import generate_mcps, DOMAIN_TYPES
from common_tools.registry import ToolRegistry


class TestGenerateMcps:
    """Test MCP generation at various counts."""

    def test_generates_correct_count(self):
        mcps = generate_mcps(10)
        assert len(mcps) == 10

    def test_generates_20(self):
        mcps = generate_mcps(20)
        assert len(mcps) == 20

    def test_generates_50(self):
        mcps = generate_mcps(50)
        assert len(mcps) == 50

    def test_each_mcp_has_tools(self):
        mcps = generate_mcps(10)
        for mcp in mcps:
            assert "tools" in mcp
            assert len(mcp["tools"]) > 0

    def test_each_mcp_has_domain(self):
        mcps = generate_mcps(10)
        for mcp in mcps:
            assert "domain" in mcp

    def test_domains_are_distributed(self):
        mcps = generate_mcps(10)
        domains = [m["domain"] for m in mcps]
        # With 10 MCPs and 10 domains, each domain should appear once
        for domain in DOMAIN_TYPES:
            assert domain in domains

    def test_tools_have_required_fields(self):
        mcps = generate_mcps(10)
        for mcp in mcps:
            for tool in mcp["tools"]:
                assert "name" in tool
                assert "description" in tool
                assert "input_schema" in tool

    def test_invalid_count_raises(self):
        import pytest
        with pytest.raises(ValueError, match="must be divisible"):
            generate_mcps(7)  # Not divisible by 10 domains

    def test_tool_names_are_unique(self):
        mcps = generate_mcps(10)
        all_names = []
        for mcp in mcps:
            for tool in mcp["tools"]:
                all_names.append(tool["name"])
        assert len(all_names) == len(set(all_names))


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_register_and_retrieve(self):
        registry = ToolRegistry()
        tool = {"name": "test_tool", "description": "A test tool", "input_schema": {"type": "object"}}
        registry.register_many([tool])
        assert registry.get_tool("test_tool") is not None

    def test_register_multiple(self):
        registry = ToolRegistry()
        tools = [
            {"name": "tool_a", "description": "Tool A", "input_schema": {}},
            {"name": "tool_b", "description": "Tool B", "input_schema": {}},
        ]
        registry.register_many(tools)
        assert len(registry.all_tools) == 2

    def test_get_tool_names(self):
        registry = ToolRegistry()
        tools = [
            {"name": "tool_a", "description": "Tool A", "input_schema": {}},
            {"name": "tool_b", "description": "Tool B", "input_schema": {}},
        ]
        registry.register_many(tools)
        names = registry.get_tool_names()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_schema_tokens_positive(self):
        registry = ToolRegistry()
        tool = {"name": "test_tool", "description": "A test tool", "input_schema": {"type": "object"}}
        registry.register_many([tool])
        tokens = registry.schema_tokens()
        assert tokens > 0

    def test_mock_call(self):
        registry = ToolRegistry()
        tool = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {},
            "mock_response": {"result": "success"},
        }
        registry.register_many([tool])
        result = registry.mock_call("test_tool")
        assert result == {"result": "success"}

    def test_mock_call_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.mock_call("nonexistent_tool")
        assert "error" in result
