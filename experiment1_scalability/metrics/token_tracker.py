from dataclasses import dataclass


class TokenTracker:
    def __init__(self):
        self.metrics: list[WorkflowMetrics] = []

    def get_results(self) -> list[dict]:
        return [m.snapshot() for m in self.metrics]


@dataclass
class WorkflowMetrics:
    workflow_name: str = ""
    architecture_name: str = ""

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

    def snapshot(self) -> dict:
        return {
            "workflow": self.workflow_name,
            "architecture": self.architecture_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "tools_exposed": self.tools_exposed,
            "tools_used": self.tools_used,
            "tools_truncated": self.tools_truncated,
            "agent_hops": self.agent_hops,
            "mcps_total": self.mcps_total,
            "mcps_activated": self.mcps_activated,
            "latency_ms": round(self.latency_ms, 1),
        }
