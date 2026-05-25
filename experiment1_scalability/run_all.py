#!/usr/bin/env python3
import asyncio
import sys
import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
from pathlib import Path

console = Console(record=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import LMStudioClient, select_provider, get_token_budget
from common_tools import generate_mcps, ToolRegistry
from metrics import TokenTracker

from centralized.run import CentralizedOrchestrator
from federated.run import FederatedOrchestrator
from workflows import WORKFLOWS

ARCHITECTURES = [
    ("Centralized MCP", CentralizedOrchestrator),
    ("Federated MCP", FederatedOrchestrator),
]


def separator(title):
    console.print()
    console.print("=" * 66, style="bold cyan")
    console.print(f"{title:^66}", style="bold cyan")
    console.print("=" * 66, style="bold cyan")
    console.print()


def print_scalability_table(all_metrics: list, mcp_count: int):
    table = Table(
        title=f"MCP Count: {mcp_count}",
        box=box.ROUNDED,
        title_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", width=26)

    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count))

    for arch in arch_keys:
        for wf in wf_keys:
            table.add_column(f"{arch}\n{wf[:12]}", style="yellow", justify="right")

    keys = [
        ("Total Tokens", "total_tokens"),
        ("Tool Schema Tokens", "tool_schema_tokens"),
        ("Tools Exposed", "tools_exposed"),
        ("Tools Used", "tools_used"),
        ("Tools Truncated", "tools_truncated"),
        ("Agent Hops", "agent_hops"),
        ("MCPs Activated", "mcps_activated"),
        ("Latency (ms)", "latency_ms"),
    ]

    for label, key in keys:
        row = [label]
        for arch in arch_keys:
            for wf in wf_keys:
                matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
                if matches:
                    m = matches[0]
                    if m.tools_truncated == -1:
                        row.append("SKIP")
                    else:
                        val = getattr(m, key, "N/A")
                        if key == "latency_ms":
                            row.append(f"{val/1000:.1f}" if isinstance(val, (int, float)) else str(val))
                        elif isinstance(val, float):
                            row.append(f"{val:,.1f}")
                        else:
                            row.append(f"{val:,}")
                else:
                    row.append("—")
        table.add_row(*row)

    console.print(table)


def select_mcp_counts():
    from common_tools.factory import DOMAIN_TYPES
    n_domains = len(DOMAIN_TYPES)
    if len(sys.argv) > 1:
        try:
            counts = [int(x) for x in sys.argv[1].split(",")]
            counts = sorted(c for c in counts if c > 0 and c % n_domains == 0)
            if counts:
                return counts
            console.print(f"[red]All counts must be divisible by {n_domains}. Using defaults.[/red]")
        except ValueError:
            pass
    console.print("[bold]Select MCP counts to test:[/bold]")
    console.print(f"  (must be divisible by {n_domains})")
    console.print("  Default: 10, 20, 50, 100")
    raw = input("MCP counts (comma-separated, Enter for all): ").strip()
    if not raw:
        return [10, 20, 50, 100]
    try:
        counts = [int(x.strip()) for x in raw.split(",")]
        counts = sorted(c for c in counts if c > 0 and c % n_domains == 0)
        if counts:
            return counts
        console.print(f"[red]All counts must be divisible by {n_domains}. Using defaults: 10, 20, 50, 100[/red]")
        return [10, 20, 50, 100]
    except ValueError:
        console.print("[red]Invalid input, using defaults: 10, 20, 50, 100[/red]")
        return [10, 20, 50, 100]


async def main():
    separator("MCP SCALABILITY COMPARISON")

    client, model_name = select_provider()
    budget = get_token_budget(model_name)
    try:
        test = await client.chat([{"role": "user", "content": "Say ready"}], temperature=0.1, max_tokens=20)
        console.print(f"[green]  Connected ({client.model})[/green] ({test.usage.total_tokens} tokens)\n")
    except Exception as e:
        console.print(f"[red]  Cannot connect: {e}[/red]")
        sys.exit(1)

    mcp_counts = select_mcp_counts()
    console.print(f"  [dim]Testing: {mcp_counts}[/dim]\n")

    tracker = TokenTracker()

    for mcp_count in mcp_counts:
        separator(f"MCP COUNT: {mcp_count}")

        mcps = generate_mcps(mcp_count)
        total_tools = sum(len(m["tools"]) for m in mcps)
        console.print(f"  Generated {len(mcps)} MCPs, {total_tools} tools, {len(set(m['domain'] for m in mcps))} domains\n")

        registry = ToolRegistry()
        for mcp in mcps:
            registry.register_many(mcp["tools"])

        for arch_name, ArchClass in ARCHITECTURES:
            console.print(f"  [bold yellow]{arch_name}[/bold yellow]")
            orch = ArchClass(client, registry, mcps, total_token_budget=budget, mcp_count=mcp_count)

            for wf in WORKFLOWS:
                metrics = await orch.run(wf)
                tracker.metrics.append(metrics)

                if metrics.tools_truncated == -1:
                    console.print(
                        f"    {wf['name'][:30]:30s} "
                        f"[red]SKIP[/red]  schema_tokens={metrics.tool_schema_tokens:>6,}  "
                        f"exposed={metrics.tools_exposed:>4d}  "
                        f"mcps={metrics.mcps_activated:>3d}  "
                        f"(budget exceeded)"
                    )
                else:
                    console.print(
                        f"    {wf['name'][:30]:30s} "
                        f"tokens={metrics.total_tokens:>6,}  "
                        f"exposed={metrics.tools_exposed:>4d}  "
                        f"used={metrics.tools_used:>3d}  "
                        f"mcps={metrics.mcps_activated:>3d}  "
                        f"hops={metrics.agent_hops:>3d}  "
                        f"lat={metrics.latency_ms/1000:.1f}s"
                    )

    separator("SCALABILITY COMPARISON TABLE")
    for mcp_count in mcp_counts:
        print_scalability_table(tracker.metrics, mcp_count)

    os.makedirs("results", exist_ok=True)
    results = tracker.get_results()
    with open("results/scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to results/scalability_results.json[/dim]")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{ts}.html", "w") as f:
        f.write(console.export_html())
    console.print(f"[dim]HTML report saved to reports/report_{ts}.html[/dim]")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
