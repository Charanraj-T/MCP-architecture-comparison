from dataclasses import dataclass


@dataclass
class WorkflowMetrics:
    workflow_name: str = ""
    architecture_name: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    tool_schema_tokens: int = 0
    orchestration_prompt_tokens: int = 0
    inter_agent_tokens: int = 0
    router_tokens: int = 0

    tools_exposed: int = 0
    tools_used: int = 0
    tools_truncated: int = 0
    agent_hops: int = 0
    mcps_activated: int = 0
    mcp_servers_connected: int = 0
    real_tool_calls: int = 0

    latency_ms: float = 0.0
    explanation: str = ""

    def snapshot(self) -> dict:
        return {
            "workflow": self.workflow_name,
            "architecture": self.architecture_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "orchestration_prompt_tokens": self.orchestration_prompt_tokens,
            "inter_agent_tokens": self.inter_agent_tokens,
            "router_tokens": self.router_tokens,
            "tools_exposed": self.tools_exposed,
            "tools_used": self.tools_used,
            "tools_truncated": self.tools_truncated,
            "agent_hops": self.agent_hops,
            "mcps_activated": self.mcps_activated,
            "mcp_servers_connected": self.mcp_servers_connected,
            "real_tool_calls": self.real_tool_calls,
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
        }


class TokenTracker:
    def __init__(self):
        self.metrics: list[WorkflowMetrics] = []

    def get_results(self) -> list[dict]:
        return [m.snapshot() for m in self.metrics]

    def print_comparison(self, console: "Console" = None):
        from rich.console import Console as _Console
        from rich.table import Table

        if not self.metrics:
            print("No metrics collected.")
            return

        grouped: dict[str, list[WorkflowMetrics]] = {}
        for m in self.metrics:
            grouped.setdefault(m.workflow_name, []).append(m)

        for workflow, archs in grouped.items():
            table = Table(title=f"\nWorkflow: {workflow}")
            table.add_column("Metric", style="cyan")
            for a in archs:
                table.add_column(a.architecture_name, style="yellow")
            keys = [
                ("total_tokens", "Total Tokens"),
                ("prompt_tokens", "Input Tokens"),
                ("completion_tokens", "Output Tokens"),
                ("reasoning_tokens", "Reasoning Tokens"),
                ("tool_schema_tokens", "Tool Schema Tokens"),
                ("orchestration_prompt_tokens", "Orchestration Tokens"),
                ("router_tokens", "Router Tokens"),
                ("inter_agent_tokens", "Inter-Agent Tokens"),
                ("tools_exposed", "Tools Exposed"),
                ("tools_used", "Tools Used"),
                ("real_tool_calls", "Real Tool Calls"),
                ("agent_hops", "Agent Hops"),
                ("mcps_activated", "MCPs Activated"),
                ("mcp_servers_connected", "MCP Connections"),
                ("tools_truncated", "Tools Truncated"),
                ("latency_ms", "Latency (ms)"),
            ]
            for key, label in keys:
                row = [label]
                for a in archs:
                    val = getattr(a, key, "N/A")
                    if isinstance(val, float):
                        row.append(f"{val:,.1f}")
                    else:
                        row.append(f"{val:,}")
                table.add_row(*row)
            (console or _Console()).print(table)
