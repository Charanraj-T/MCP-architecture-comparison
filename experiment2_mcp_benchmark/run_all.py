#!/usr/bin/env python3
import asyncio
import sys
import os
import json
import traceback
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

console = Console(record=True)

from common import select_provider
from metrics import TokenTracker

from centralized.orchestrator import CentralizedOrchestrator
from federated.orchestrator import FederatedOrchestrator
from multiagent.orchestrator import MultiAgentOrchestrator
from mediator.orchestrator import MediatorOrchestrator
from intent_driven.orchestrator import IntentDrivenOrchestrator
from workflows import WORKFLOWS

MCP_SERVERS = [
    ("filesystem", "npx", "@modelcontextprotocol/server-filesystem", "File read/write/search"),
    ("git", "uvx", "mcp-server-git", "Git log/diff/status"),
    ("fetch", "npx", "mcp-server-fetch-typescript", "HTTP fetch"),
    ("memory", "npx", "@modelcontextprotocol/server-memory", "Key-value store"),
    ("SQLite", "uvx", "mcp-server-sqlite", "SQL queries"),
    ("sequential-thinking", "npx", "@modelcontextprotocol/server-sequential-thinking", "Step-by-step reasoning"),
]

ARCHITECTURES = [
    ("Centralized (1MCP)", CentralizedOrchestrator, "All 6 servers behind @1mcp/agent aggregator — all tools in system prompt"),
    ("Federated (Bifrost)", FederatedOrchestrator, "All 6 servers behind Bifrost gateway — 4 meta-tools, on-demand discovery"),
]

COMPARISON_METRICS = [
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


def separator(title: str):
    console.print()
    console.print("=" * 66, style="bold cyan")
    console.print(f"{title:^66}", style="bold cyan")
    console.print("=" * 66, style="bold cyan")
    console.print()


def print_metadata(model_name: str, total_tokens: int):
    sep = Panel.fit(
        Text.from_markup(
            "[bold cyan]MCP Servers[/bold cyan]\n"
            + "\n".join(f"  • [yellow]{s[0]:24s}[/yellow]  [dim]({s[1]})[/dim]  {s[3]}" for s in MCP_SERVERS)
            + "\n\n"
            + "[bold cyan]Architectures[/bold cyan]\n"
            + "\n".join(f"  • [yellow]{a[0]:24s}[/yellow]  {a[2]}" for a in ARCHITECTURES)
            + "\n\n"
            + f"[bold cyan]Model[/bold cyan]\n  {model_name}  [dim]({total_tokens} tokens on connection)[/dim]"
        ),
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(sep)


def print_scenario_header(wf: dict):
    expected = ", ".join(f"[yellow]{m}[/yellow]" for m in wf["expected_mcps"])
    console.print()
    console.print("━" * 66, style="bold white")
    console.print(f"  [bold white]{wf['name']}[/bold white]")
    console.print(f"  [dim]{wf['description']}[/dim]")
    console.print(f"  [bold]Expected MCPs:[/bold] {expected}")
    console.print("━" * 66, style="bold white")
    console.print()


def print_comparison_table(metrics: list, wf_name: str):
    arch_metrics = [m for m in metrics if m.workflow_name == wf_name]
    if not arch_metrics:
        return

    table = Table(box=box.ROUNDED, title_style="bold cyan", expand=True)
    table.add_column("Metric", style="cyan", width=28)
    for m in arch_metrics:
        table.add_column(m.architecture_name, style="yellow", justify="right")

    for key, label in COMPARISON_METRICS:
        row = [label]
        for m in arch_metrics:
            if m.tools_truncated == -1:
                row.append("[red]SKIP[/red]")
                continue
            val = getattr(m, key, "N/A")
            if key == "latency_ms" and isinstance(val, (int, float)):
                row.append(f"{val:,.1f} ms")
            elif isinstance(val, float):
                row.append(f"{val:,.1f}")
            else:
                row.append(f"{val:,}")
        table.add_row(*row)

    console.print(table)


async def main():
    separator("EXPERIMENT 2: REAL MCP ORCHESTRATION BENCHMARK")

    console.print("[bold]MCP Servers:[/bold] filesystem (npx), git (uvx), fetch (npx), memory (npx), SQLite (uvx), sequential-thinking (npx)")
    console.print("[bold]Architectures:[/bold] Centralized ([blue]@1mcp/agent[/blue]), Federated ([blue]Bifrost[/blue] Code Mode), Multi-Agent (direct), MCP Mediator, Intent-Driven")
    console.print()

    client, model_name = select_provider()

    try:
        console.print("[dim]Verifying connection...[/dim]")
        test = await client.chat([{"role": "user", "content": "Say 'ready' if you can hear me."}], temperature=0.1, max_tokens=20)
        if test.error:
            console.print(f"[red]  Connection error: {test.error}[/red]")
            sys.exit(1)
        conn_tokens = test.usage.total_tokens
        console.print(f"[green]  Connected ({client.model})[/green] ({conn_tokens} tokens)\n")
    except Exception as e:
        console.print(f"[red]  Cannot connect to LM Studio: {e}[/red]")
        sys.exit(1)

    print_metadata(model_name, conn_tokens)

    all_metrics = TokenTracker()

    architectures = [
        ("Federated (Bifrost)", FederatedOrchestrator),
        ("Centralized (1MCP)", CentralizedOrchestrator),
        ("Multi-Agent", MultiAgentOrchestrator),
        ("MCP Mediator", MediatorOrchestrator),
        ("Intent-Driven", IntentDrivenOrchestrator),
    ]

    for name, ArchClass in architectures:
        separator(f"RUNNING: {name}")
        console.print(f"\n[bold yellow]Running {name}...[/bold yellow]")
        orchestrator = ArchClass(client)
        try:
            for i, wf in enumerate(WORKFLOWS, 1):
                console.print(f"  [{i}/{len(WORKFLOWS)}] {wf['name']} ... ", end="")
                try:
                    metrics = await orchestrator.run(wf)
                    all_metrics.metrics.append(metrics)
                    if metrics.tools_truncated == -1:
                        console.print("[red]SKIP (budget exceeded)[/red]")
                    else:
                        console.print(f"[green]done[/green] ({metrics.total_tokens:,} tokens, {metrics.latency_ms/1000:.1f}s)")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    traceback.print_exc()
        finally:
            if hasattr(orchestrator, 'close'):
                await orchestrator.close()

    separator("SCENARIO ANALYSIS")
    console.print("[dim][red]SKIP[/red] = tool schemas exceeded token budget; architecture did not run for that workflow[/dim]")
    console.print()

    for wf in WORKFLOWS:
        print_scenario_header(wf)
        print_comparison_table(all_metrics.metrics, wf["name"])

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
