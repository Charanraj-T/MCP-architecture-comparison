#!/usr/bin/env python3
"""
MCP Architecture Comparison — Run all 3 architectures on all 3 workflows.
"""

import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from lmstudio_client import LMStudioClient
from common_tools import ToolRegistry, ALL_TOOLS
from metrics import TokenTracker, JSONLogger

from centralized.run import CentralizedOrchestrator
from federated.run import FederatedOrchestrator
from multiagent.run import MultiAgentOrchestrator

from workflows import WORKFLOWS


def print_separator(title: str):
    console = Console()
    width = 64
    console.print()
    console.print("=" * width, style="bold cyan")
    console.print(f"{title:^{width}}", style="bold cyan")
    console.print("=" * width, style="bold cyan")
    console.print()


def print_workflow_header(idx: int, name: str):
    console = Console()
    console.print(f"\n[bold yellow]▸ Workflow {idx}: {name}[/bold yellow]")
    console.print("─" * 64, style="dim")


def print_architecture_subheader(name: str):
    console = Console()
    console.print(f"\n  [bold magenta]── {name} ──[/bold magenta]")


def print_metrics_table(metrics_snapshot: dict):
    console = Console()
    arch = metrics_snapshot["architecture"]

    table = Table(
        title=f"Architecture: {arch}",
        box=box.ROUNDED,
        title_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", width=28)
    table.add_column("Value", style="yellow", justify="right")

    rows = [
        ("Input Tokens (prompt)", "prompt_tokens"),
        ("Output Tokens (completion)", "completion_tokens"),
        ("Reasoning Tokens", "reasoning_tokens"),
        ("Total Tokens", "total_tokens"),
        ("", None),
        ("User Prompt Tokens (estimated)", "user_prompt_tokens"),
        ("Tool Schema Tokens (estimated)", "tool_schema_tokens"),
        ("Orchestration Tokens (estimated)", "orchestration_prompt_tokens"),
        ("Memory/Replay Tokens (estimated)", "memory_replay_tokens"),
        ("Inter-Agent Tokens (estimated)", "inter_agent_tokens"),
        ("Output Tokens (estimated)", "output_tokens"),
        ("", None),
        ("Tools Exposed", "tools_exposed"),
        ("Tools Used", "tools_used"),
        ("Agent Hops", "agent_hops"),
        ("MCPs Activated", "mcps_activated"),
        ("", None),
        ("Latency", "latency_ms"),
    ]

    for label, key in rows:
        if key is None:
            table.add_row("", "", style="dim")
            continue
        val = metrics_snapshot.get(key, "N/A")
        if isinstance(val, float):
            table.add_row(label, f"{val:,.1f} ms")
        else:
            table.add_row(label, f"{val:,}")

    console.print(table)


async def run_single_architecture(
    client: LMStudioClient,
    registry: ToolRegistry,
    name: str,
    orchestrator_class,
    tracker: TokenTracker,
):
    orchestrator = orchestrator_class(client, registry)
    for i, wf in enumerate(WORKFLOWS, 1):
        print_workflow_header(i, wf["name"])
        print_architecture_subheader(name)
        metrics = await orchestrator.run(wf)
        tracker.metrics.append(metrics)
        print_metrics_table(metrics.snapshot())
    return tracker


async def main():
    console = Console()

    print_separator("MCP ARCHITECTURE COMPARISON POC")

    console.print("[bold]Model:[/bold] qwen/qwen3-8b via LM Studio")
    console.print("[bold]Endpoint:[/bold] http://localhost:1234/v1")
    console.print()

    client = LMStudioClient(base_url="http://localhost:1234/v1", model="qwen/qwen3-8b")

    # Verify connection
    try:
        console.print("[dim]Verifying LM Studio connection...[/dim]")
        test = await client.chat(
            [{"role": "user", "content": "Say 'ready' if you can hear me."}],
            temperature=0.1,
            max_tokens=20,
        )
        console.print(f"[green]✓[/green] LM Studio connected: {test.usage.total_tokens} tokens\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Cannot connect to LM Studio: {e}")
        console.print("[yellow]Make sure LM Studio is running with model loaded at http://localhost:1234/v1[/yellow]")
        sys.exit(1)

    # Build tool registry
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS)
    console.print(f"[dim]Loaded {registry.count_tools()} tools[/dim]\n")

    # Collect all metrics across runs
    all_metrics = TokenTracker()

    architectures = [
        ("Centralized MCP", CentralizedOrchestrator),
        ("Federated MCP", FederatedOrchestrator),
        ("Multi-Agent", MultiAgentOrchestrator),
    ]

    for name, orchestrator_class in architectures:
        print_separator(f"RUNNING: {name}")
        await run_single_architecture(client, registry, name, orchestrator_class, all_metrics)

    # === Final Comparison ===
    print_separator("FINAL COMPARISON")

    all_metrics.print_comparison()

    # === Summary Insights ===
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Expected Findings[/bold cyan]\n\n"
        "• [yellow]Centralized MCP[/yellow]: Largest prompts, highest tool schema injection,\n"
        "  highest total token usage, simplest implementation\n\n"
        "• [green]Federated MCP[/green]: Smallest active context, best token efficiency,\n"
        "  best scalability, best balance overall\n\n"
        "• [red]Multi-Agent[/red]: Highest orchestration overhead, repeated prompts,\n"
        "  duplicated context, inter-agent communication amplification",
        border_style="cyan",
    ))

    # Save results
    import json
    results = all_metrics.get_results()
    with open("results/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to results/comparison_results.json[/dim]")

    console.print()
    console.print("[bold green]Done![/bold green] See traces/ directory for detailed JSON trace logs.")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
