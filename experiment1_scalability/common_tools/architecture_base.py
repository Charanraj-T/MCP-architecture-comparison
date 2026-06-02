"""Base class for experiment 1 architecture orchestrators.

Eliminates ~70% code duplication across centralized, federated,
intent_driven, mediator, and multiagent implementations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import tiktoken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import LLMResponse

from metrics.token_tracker import WorkflowMetrics
from common_tools.parser import parse_tool_calls

if TYPE_CHECKING:
    from common.client import OpenAIClient
    from common_tools.registry import ToolRegistry

_CALL_TIMEOUT = 120

logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """Shared infrastructure for all architecture orchestrators.

    Subclasses must implement `run()` and may override `close()`.
    """

    def __init__(
        self,
        client: OpenAIClient,
        registry: ToolRegistry,
        mcps: list[dict],
        total_token_budget: int = 38000,
        mcp_count: int = 0,
    ) -> None:
        self.client = client
        self.registry = registry
        self.mcps = mcps
        self.total_token_budget = total_token_budget
        self.mcp_count = mcp_count
        self._enc = tiktoken.get_encoding("cl100k_base")
        self.logger = logging.getLogger(self.__class__.__name__)
        logger.info(
            "Initialized %s (budget=%d, mcps=%d)",
            self.__class__.__name__,
            total_token_budget,
            mcp_count,
        )

    # ── Token utilities ──────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken cl100k_base."""
        return len(self._enc.encode(text))

    def _estimate_next_request_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens for next LLM call including message overhead."""
        total = 0
        for m in messages:
            total += self._count_tokens(m.get("content", ""))
        return total + 500

    # ── Tool selection ───────────────────────────────────────────

    def _get_tools_for_domains(self, domains) -> list[str]:
        """Get all tool names belonging to the specified domains."""
        if isinstance(domains, set):
            domain_set = domains
        else:
            domain_set = set(domains)
        names = []
        for mcp in self.mcps:
            if mcp["domain"] in domain_set:
                for t in mcp["tools"]:
                    names.append(t["name"])
        return names

    # ── LLM interaction ──────────────────────────────────────────

    async def _chat_with_timeout(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Call LLM with a timeout, returning error response on timeout."""
        try:
            return await asyncio.wait_for(
                self.client.chat(messages, **kwargs),
                timeout=_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out after %ds", _CALL_TIMEOUT)
            return LLMResponse(content="", usage=None, error="Timeout after 120s")

    # ── Usage accumulation ───────────────────────────────────────

    @staticmethod
    def _accumulate_usage(metrics: WorkflowMetrics, response: LLMResponse) -> None:
        """Add response token usage to running metrics totals."""
        if response.usage:
            metrics.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0) or 0
            metrics.completion_tokens += getattr(response.usage, 'completion_tokens', 0) or 0
            metrics.reasoning_tokens += getattr(response.usage, 'reasoning_tokens', 0) or 0
            metrics.total_tokens += getattr(response.usage, 'total_tokens', 0) or 0
        metrics.agent_hops += 1

    # ── Tool execution loop ──────────────────────────────────────

    async def _execute_tool_loop(
        self,
        messages: list[dict],
        metrics: WorkflowMetrics,
        max_turns: int = 6,
    ) -> str:
        """Run the chat→parse→execute→respond loop. Returns final answer text."""
        tools_used: set[str] = set()
        total_latency = 0.0

        for turn in range(max_turns):
            next_est = self._estimate_next_request_tokens(messages)
            if next_est > self.total_token_budget:
                logger.warning(
                    "Token budget exceeded (%d > %d) at turn %d, stopping",
                    next_est, self.total_token_budget, turn,
                )
                warnings.warn(
                    f"Token budget exceeded ({next_est} > {self.total_token_budget}), stopping"
                )
                break

            response = await self.client.chat(messages, temperature=0.1)
            if response.error:
                logger.warning("LLM error at turn %d: %s", turn, response.error)
                warnings.warn(f"LLM error: {response.error}")
                break
            total_latency += response.latency_ms
            self._accumulate_usage(metrics, response)

            content = response.content or ""
            calls = parse_tool_calls(content)

            if calls:
                results = []
                for tc in calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if not name:
                        continue
                    result = self.registry.mock_call(name, **args)
                    tools_used.add(name)
                    results.append({name: result})
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{json.dumps(results, indent=2)}\n\nContinue or provide FINAL_ANSWER.",
                })
            else:
                messages.append({"role": "assistant", "content": content})
                break

        metrics.tools_used = len(tools_used)
        metrics.latency_ms = total_latency
        logger.debug(
            "Tool loop complete: %d tools used, %.1fms latency, %d hops",
            len(tools_used), total_latency, metrics.agent_hops,
        )
        return messages[-1].get("content", "") if messages else ""

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self) -> None:
        """Cleanup resources. Override in subclasses if needed."""

    @abstractmethod
    async def run(self, workflow: dict) -> WorkflowMetrics:
        """Execute a workflow and return metrics."""
        ...
