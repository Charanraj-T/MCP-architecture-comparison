"""Tests for common_tools/parser.py — tool call parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment1_scalability"))

from common_tools.parser import parse_tool_calls


class TestParseToolCalls:
    """Test the parse_tool_calls function with various LLM output formats."""

    def test_single_tool_call(self):
        text = 'TOOL_CALL: {"name": "code_search", "args": {"query": "hello"}}'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "code_search"
        assert result[0]["args"]["query"] == "hello"

    def test_multiple_tool_calls(self):
        text = (
            'TOOL_CALL: {"name": "code_search", "args": {"query": "hello"}}\n'
            'TOOL_CALL: {"name": "log_analysis", "args": {"pattern": "error"}}'
        )
        result = parse_tool_calls(text)
        assert len(result) == 2
        assert result[0]["name"] == "code_search"
        assert result[1]["name"] == "log_analysis"

    def test_tool_call_with_nested_args(self):
        text = 'TOOL_CALL: {"name": "deploy", "args": {"config": {"env": "prod", "replicas": 3}}}'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["args"]["config"]["env"] == "prod"
        assert result[0]["args"]["config"]["replicas"] == 3

    def test_no_tool_calls(self):
        text = "I'll help you search the codebase for that function."
        result = parse_tool_calls(text)
        assert len(result) == 0

    def test_empty_text(self):
        result = parse_tool_calls("")
        assert len(result) == 0

    def test_malformed_json(self):
        text = 'TOOL_CALL: {"name": "code_search", "args": {invalid}}'
        result = parse_tool_calls(text)
        # Should not crash — malformed JSON is skipped
        assert len(result) == 0

    def test_tool_call_without_prefix(self):
        text = '{"name": "code_search", "args": {"query": "hello"}}'
        result = parse_tool_calls(text)
        # May or may not parse depending on implementation — just verify no crash
        assert isinstance(result, list)

    def test_final_answer_marker(self):
        text = 'FINAL_ANSWER: The codebase has 42 files matching your query.'
        result = parse_tool_calls(text)
        assert len(result) == 0

    def test_tool_call_with_special_characters(self):
        text = 'TOOL_CALL: {"name": "code_search", "args": {"query": "function(x) => x * 2"}}'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert "function(x)" in result[0]["args"]["query"]

    def test_returns_list_type(self):
        result = parse_tool_calls("some text")
        assert isinstance(result, list)

    def test_code_block_json(self):
        text = '```json\n{"name": "test", "args": {"key": "val"}}\n```'
        result = parse_tool_calls(text)
        assert len(result) >= 1
        assert result[0]["name"] == "test"
