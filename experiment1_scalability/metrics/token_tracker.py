"""Token tracking and metrics collection for benchmark runs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkflowMetrics:
    """Metrics collected from a single workflow execution across one architecture."""
    workflow_name: str = ""
    architecture_name: str = ""
    model_name: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    tool_schema_tokens: int = 0

    tools_exposed: int = 0
    tools_used: int = 0
    tools_truncated: int = 0
    agent_hops: int = 0
    mcps_total: int = 0
    mcps_activated: int = 0

    latency_ms: float = 0.0
    explanation: str = ""

    # Experiment 2 specific
    inter_agent_tokens: int = 0
    orchestration_prompt_tokens: int = 0
    real_tool_calls: int = 0
    mcp_servers_connected: int = 0

    def snapshot(self) -> dict:
        """Return a JSON-serializable dict of all metrics."""
        return {
            "workflow": self.workflow_name,
            "architecture": self.architecture_name,
            "model": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "tools_exposed": self.tools_exposed,
            "tools_used": self.tools_used,
            "tools_truncated": self.tools_truncated,
            "agent_hops": self.agent_hops,
            "mcps_total": self.mcps_total,
            "mcps_activated": self.mcps_activated,
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
            "inter_agent_tokens": self.inter_agent_tokens,
            "orchestration_prompt_tokens": self.orchestration_prompt_tokens,
            "real_tool_calls": self.real_tool_calls,
            "mcp_servers_connected": self.mcp_servers_connected,
        }


class TokenTracker:
    """Collects WorkflowMetrics across multiple runs and exports results."""

    def __init__(self) -> None:
        self.metrics: list[WorkflowMetrics] = []
        logger.info("TokenTracker initialized")

    def append(self, metric: WorkflowMetrics) -> None:
        """Add a metric to the tracker."""
        self.metrics.append(metric)
        logger.debug(
            "Metric added: %s/%s — %d tokens, %d hops",
            metric.architecture_name,
            metric.workflow_name,
            metric.total_tokens,
            metric.agent_hops,
        )

    def get_results(self) -> list[dict]:
        """Return all metrics as a list of JSON-serializable dicts."""
        return [m.snapshot() for m in self.metrics]

    def summary(self) -> dict:
        """Return aggregate statistics across all collected metrics."""
        if not self.metrics:
            return {}
        totals = {"total_tokens": 0, "total_hops": 0, "total_latency_ms": 0.0, "count": len(self.metrics)}
        for m in self.metrics:
            totals["total_tokens"] += m.total_tokens
            totals["total_hops"] += m.agent_hops
            totals["total_latency_ms"] += m.latency_ms
        return totals
