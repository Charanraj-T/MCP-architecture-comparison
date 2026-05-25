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
from multiagent.run import MultiAgentOrchestrator
from mediator.run import MediatorOrchestrator
from intent_driven.run import IntentDrivenOrchestrator

from workflows import WORKFLOWS

ARCHITECTURES = [
    ("Centralized MCP", CentralizedOrchestrator),
    ("Federated MCP", FederatedOrchestrator),
    ("MCP Mediator", MediatorOrchestrator),
    ("Intent-Driven", IntentDrivenOrchestrator),
]


def separator(title):
    console.print()
    console.print("=" * 66, style="bold cyan")
    console.print(f"{title:^66}", style="bold cyan")
    console.print("=" * 66, style="bold cyan")
    console.print()


def print_token_cost_methodology():
    separator("TOKEN COST METHODOLOGY")
    console.print(
        "  [bold]Token Calculation:[/bold]\n"
        "  • [green]Prompt Tokens[/green]: Every LLM call has a prompt. The prompt contains:\n"
        "    - System prompt (architecture instructions)\n"
        "    - Tool schemas (tool names + descriptions + JSON input schemas)\n"
        "    - Conversation history (user messages, assistant responses, tool results)\n"
        "  • [green]Completion Tokens[/green]: The model's response text or tool calls\n"
        "  • [green]Total Tokens[/green] = Sum of prompt + completion across all LLM calls\n"
        "  • [green]Schema Tokens[/green]: Computed via tiktoken (cl100k_base) on the tool schema text\n"
        "  • [green]Agent Hops[/green]: Count of sequential LLM invocations per workflow\n\n"
        "  [bold]Token Cost Drivers per Architecture:[/bold]\n"
        "  • [yellow]Centralized[/yellow]: Schema cost paid EVERY turn. Large prompts from full history.\n"
        "  • [yellow]Federated[/yellow]: Router call adds 1 hop but fewer schemas injected. Lower per-turn cost.\n"
        "  • [yellow]MCP Mediator[/yellow]: Schema loaded once for planning. Execution = zero LLM tokens.\n"
        "  • [yellow]Intent-Driven[/yellow]: Schemas loaded only for domains in the plan. Most selective.\n"
    )


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


def print_per_workflow_table(all_metrics, mcp_count, workflow_name):
    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    table = Table(
        title=f"{workflow_name}  |  MCP Count: {mcp_count}",
        box=box.SIMPLE,
        title_style="bold white",
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", width=24)
    for arch in arch_keys:
        table.add_column(arch, style="yellow", justify="right", width=16)

    rows = [
        ("Total Tokens", "total_tokens", "{:,}"),
        ("├─ Prompt Tokens", "prompt_tokens", "{:,}"),
        ("├─ Completion Tokens", "completion_tokens", "{:,}"),
        ("", None, None),
        ("Tool Schema Tokens", "tool_schema_tokens", "{:,}"),
        ("", None, None),
        ("Tools Exposed", "tools_exposed", "{:,}"),
        ("Tools Used", "tools_used", "{:,}"),
        ("Tools Truncated", "tools_truncated", "{:,}"),
        ("", None, None),
        ("Agent Hops", "agent_hops", "{:,}"),
        ("MCPs Activated", "mcps_activated", "{:,}"),
        ("", None, None),
        ("Latency", "latency_ms", "{:.1f}s"),
    ]

    for label, key, fmt in rows:
        if key is None:
            table.add_row("", *[""] * len(arch_keys))
            continue
        row = [label]
        for arch in arch_keys:
            matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == workflow_name]
            if matches:
                m = matches[0]
                if m.tools_truncated == -1:
                    row.append("[red]SKIP[/red]")
                else:
                    val = getattr(m, key, "N/A")
                    if key == "latency_ms":
                        row.append(f"{val/1000:.1f}s")
                    elif isinstance(val, float):
                        row.append(f"{val:,.1f}")
                    elif isinstance(val, int):
                        row.append(f"{val:,}")
                    else:
                        row.append(str(val))
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)
    console.print()


def print_summary_table(all_metrics, mcp_count):
    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count))

    table = Table(
        title=f"EXECUTIVE SUMMARY — MCP Count: {mcp_count} (Averaged Across All Workflows)",
        box=box.HEAVY_EDGE,
        title_style="bold white on blue",
        header_style="bold white",
    )
    table.add_column("Metric", style="cyan", width=22)
    for arch in arch_keys:
        table.add_column(arch, style="yellow", justify="right", width=16)
    table.add_column("Best", style="green", justify="right", width=14)

    rows_spec = [
        ("Avg Total Tokens", "total_tokens", "{:,.0f}"),
        ("Avg Schema Tokens", "tool_schema_tokens", "{:,.0f}"),
        ("Schema % of Total", None, None),
        ("Avg Tools Exposed", "tools_exposed", "{:,.0f}"),
        ("Avg Tools Used", "tools_used", "{:,.0f}"),
        ("Avg Agent Hops", "agent_hops", "{:,.1f}"),
        ("Avg MCPs Activated", "mcps_activated", "{:,.0f}"),
        ("Avg Latency", "latency_ms", "{:.1f}s"),
    ]

    for label, key, fmt in rows_spec:
        row = [label]
        vals = []

        if key is None:
            for i, arch in enumerate(arch_keys):
                total_vals = []
                schema_vals = []
                for wf in wf_keys:
                    matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
                    if matches and matches[0].tools_truncated != -1:
                        total_vals.append(matches[0].total_tokens)
                        schema_vals.append(matches[0].tool_schema_tokens)
                avg_total = sum(total_vals) / len(total_vals) if total_vals else 1
                avg_schema = sum(schema_vals) / len(schema_vals) if schema_vals else 0
                row.append(f"{avg_schema/avg_total*100:.0f}%")
            row.append("—")
            table.add_row(*row)
            continue

        best_idx = 0
        for arch in arch_keys:
            av = []
            for wf in wf_keys:
                matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
                if matches and matches[0].tools_truncated != -1:
                    av.append(getattr(matches[0], key, 0))
            avg = sum(av) / len(av) if av else 0
            vals.append(avg)
            if key == "latency_ms":
                row.append(f"{avg/1000:.1f}s")
            else:
                row.append(f"{avg:,.0f}")

        if vals:
            if key in ("total_tokens", "latency_ms", "agent_hops"):
                best_idx = min(range(len(vals)), key=lambda i: vals[i])
                best_val = vals[best_idx]
                best_label = f"{best_val:,.0f}" if key != "latency_ms" else f"{best_val/1000:.1f}s"
            else:
                best_idx = max(range(len(vals)), key=lambda i: vals[i])
                best_val = vals[best_idx]
                best_label = f"{best_val:,.0f}"

            improvement = ""
            if key in ("total_tokens", "latency_ms", "agent_hops") and vals and vals[0] > 0 and best_idx != 0:
                improvement = f" ({vals[best_idx]/vals[0]:.1f}x vs Centralized)"
            elif key in ("total_tokens", "latency_ms", "agent_hops") and best_idx == 0:
                improvement = " (baseline)"

            row.append(f"[green]{best_label}{improvement}[/green]")
        else:
            row.append("—")

        table.add_row(*row)

    console.print(table)
    console.print()


def print_architecture_analysis(all_metrics, mcp_count):
    separator("TOKEN COST ANALYSIS — WHY EACH ARCHITECTURE COSTS WHAT IT DOES")

    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count))

    for arch in arch_keys:
        matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch]
        if not matches:
            continue
        m = matches[0]
        explanation = m.explanation or "No explanation available."

        avg_tokens = sum(m2.total_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_schema = sum(m2.tool_schema_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_hops = sum(m2.agent_hops for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)

        from rich.panel import Panel
        panel = Panel(
            f"[bold]{arch}[/bold]\n"
            f"{explanation}\n\n"
            f"[bold]Token Profile:[/bold]\n"
            f"  Avg Total Tokens:   {avg_tokens:>8,.0f}\n"
            f"  Avg Schema Tokens:  {avg_schema:>8,.0f}  ({avg_schema/max(avg_tokens,1)*100:.0f}% of total)\n"
            f"  Avg Agent Hops:     {avg_hops:>8,.1f}\n\n"
            f"[bold]Why this matters:[/bold]\n"
            f"  Schema tokens are a fixed cost per architecture — they don't depend on task complexity.\n"
            f"  Agent hops multiply the per-turn cost (overhead + schema + completion).\n"
            f"  Architectures with fewer hops pay the schema overhead FEWER times.",
            border_style="blue",
        )
        console.print(panel)
        console.print()


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

    print_token_cost_methodology()

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

    for mcp_count in mcp_counts:
        separator(f"DETAILED COMPARISON — MCP Count: {mcp_count}")
        from workflows import WORKFLOWS as WF_LIST
        for wf in WF_LIST:
            print_per_workflow_table(tracker.metrics, mcp_count, wf["name"])
        print_summary_table(tracker.metrics, mcp_count)
        print_architecture_analysis(tracker.metrics, mcp_count)

    os.makedirs("results", exist_ok=True)
    results = tracker.get_results()
    with open("results/scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[dim]Results saved to results/scalability_results.json[/dim]")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{ts}.html", "w") as f:
        f.write(console.export_html())
    console.print(f"[dim]HTML report saved to reports/report_{ts}.html[/dim]")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
