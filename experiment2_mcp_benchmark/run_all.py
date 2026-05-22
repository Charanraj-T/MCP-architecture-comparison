#!/usr/bin/env python3
import asyncio
import sys
import os
import json
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(record=True)

from common import LMStudioClient, select_provider
from metrics import TokenTracker

from centralized.orchestrator import CentralizedOrchestrator
from federated.orchestrator import FederatedOrchestrator
from multiagent.orchestrator import MultiAgentOrchestrator
from workflows import WORKFLOWS


def separator(title: str):
    console.print()
    console.print("=" * 66, style="bold cyan")
    console.print(f"{title:^66}", style="bold cyan")
    console.print("=" * 66, style="bold cyan")
    console.print()


def print_metrics(metrics):
    snap = metrics.snapshot()
    table = Table(title=f"Architecture: {snap['architecture']}", box=box.ROUNDED, title_style="bold cyan")
    table.add_column("Metric", style="cyan", width=28)
    table.add_column("Value", style="yellow", justify="right")
    rows = [
        ("Total Tokens", "total_tokens"),
        ("Input Tokens", "prompt_tokens"),
        ("Output Tokens", "completion_tokens"),
        ("Reasoning Tokens", "reasoning_tokens"),
        ("", None),
        ("Tool Schema Tokens", "tool_schema_tokens"),
        ("Orchestration Tokens", "orchestration_prompt_tokens"),
        ("Router Tokens", "router_tokens"),
        ("Inter-Agent Tokens", "inter_agent_tokens"),
        ("", None),
        ("Tools Exposed", "tools_exposed"),
        ("Tools Used", "tools_used"),
        ("Real Tool Calls", "real_tool_calls"),
        ("Agent Hops", "agent_hops"),
        ("MCPs Activated", "mcps_activated"),
        ("MCP Connections", "mcp_servers_connected"),
        ("", None),
        ("Latency", "latency_ms"),
    ]
    for label, key in rows:
        if key is None:
            table.add_row("", "", style="dim")
            continue
        val = snap.get(key, "N/A")
        if key == "latency_ms":
            table.add_row(label, f"{val:,.1f} ms")
        else:
            table.add_row(label, f"{val:,}")
    console.print(table)


async def main():
    separator("EXPERIMENT 2: REAL MCP ORCHESTRATION BENCHMARK")

    console.print("[bold]MCP Servers:[/bold] filesystem (npx), git (uvx), fetch (npx), memory (npx), SQLite (uvx), sequential-thinking (npx)")
    console.print("[bold]Architectures:[/bold] Centralized ([blue]@1mcp/agent[/blue]), Federated ([blue]smartmcp[/blue] router), Multi-Agent (direct)")
    console.print()

    client, _model_name = select_provider()

    try:
        console.print("[dim]Verifying connection...[/dim]")
        test = await client.chat([{"role": "user", "content": "Say 'ready' if you can hear me."}], temperature=0.1, max_tokens=20)
        console.print(f"[green]  Connected ({client.model})[/green] ({test.usage.total_tokens} tokens)\n")
    except Exception as e:
        console.print(f"[red]  Cannot connect to LM Studio: {e}[/red]")
        sys.exit(1)

    all_metrics = TokenTracker()

    architectures = [
        ("Centralized (1MCP)", CentralizedOrchestrator),
        ("Federated (smartmcp)", FederatedOrchestrator),
        ("Multi-Agent", MultiAgentOrchestrator),
    ]

    for name, ArchClass in architectures:
        separator(f"RUNNING: {name}")
        orchestrator = ArchClass(client)
        for i, wf in enumerate(WORKFLOWS, 1):
            console.print(f"\n[bold yellow]  Workflow {i}: {wf['name']}[/bold yellow]")
            console.print("  " + "─" * 60, style="dim")
            try:
                metrics = await orchestrator.run(wf)
                all_metrics.metrics.append(metrics)
                print_metrics(metrics)
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()

    separator("FINAL COMPARISON")
    all_metrics.print_comparison(console)

    os.makedirs("results", exist_ok=True)
    results = all_metrics.get_results()
    with open("results/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to results/comparison_results.json[/dim]")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{ts}.html", "w") as f:
        f.write(console.export_html())
    console.print(f"[dim]HTML report saved to reports/report_{ts}.html[/dim]")

    console.print("[dim]See traces/ directory for detailed JSON trace logs.[/dim]")
    console.print()
    console.print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
