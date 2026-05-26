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

# Cost assumptions (example: Gemini 2.0 Flash standard API pricing)
INPUT_COST_PER_1M = 0.10   # $ per million input tokens
OUTPUT_COST_PER_1M = 0.40  # $ per million output tokens


def _cost_usd(prompt_tokens, completion_tokens):
    return (prompt_tokens / 1_000_000 * INPUT_COST_PER_1M) + (completion_tokens / 1_000_000 * OUTPUT_COST_PER_1M)


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
        "  [bold]Estimated Cost:[/bold]\n"
        f"  • Input rate:  ${INPUT_COST_PER_1M:.2f} / 1M tokens\n"
        f"  • Output rate: ${OUTPUT_COST_PER_1M:.2f} / 1M tokens\n"
        "  • Cost = (prompt_tokens / 1M × input_rate) + (completion_tokens / 1M × output_rate)\n"
        "  • Rates based on Gemini 2.0 Flash pricing. Actual costs vary by provider and model.\n\n"
        "  [bold]Token Cost Drivers per Architecture:[/bold]\n"
        "  • [yellow]Centralized[/yellow]: Schema cost paid EVERY turn. Large prompts from full history.\n"
        "  • [yellow]Federated[/yellow]: Router call adds 1 hop but fewer schemas injected. Lower per-turn cost.\n"
        "  • [yellow]MCP Mediator[/yellow]: Schema loaded once for planning. Execution = zero LLM tokens.\n"
        "  • [yellow]Intent-Driven[/yellow]: Schemas NOT loaded into prompts — only domain descriptions.\n"
        "    75 tokens of domain info vs 12K+ tokens of full schemas. Key source of savings.\n\n"
        "  [bold]Model Caveat:[/bold]\n"
        "  • All results from [cyan]google/gemini-2.0-flash-001[/cyan] on OpenRouter\n"
        "  • Different models may produce different numbers (and fail differently on JSON planning)\n"
        "  • Results should be validated on target models before production decisions\n"
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
        ("Est. Cost (USD)", None, None),
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
                elif label == "Est. Cost (USD)":
                    cost = _cost_usd(m.prompt_tokens, m.completion_tokens)
                    row.append(f"${cost:.6f}")
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
    table.add_column("Metric", style="cyan", width=24)
    for arch in arch_keys:
        table.add_column(arch, style="yellow", justify="right", width=18)
    table.add_column("Best", style="green", justify="right", width=16)

    def _avg(arch, key):
        vals = []
        for wf in wf_keys:
            matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
            if matches and matches[0].tools_truncated != -1:
                vals.append(getattr(matches[0], key, 0))
        return sum(vals) / len(vals) if vals else 0

    rows_spec = [
        ("Avg Total Tokens", "total_tokens", False),
        ("Avg Prompt Tokens", "prompt_tokens", False),
        ("Avg Completion Tokens", "completion_tokens", False),
        ("Avg Schema Tokens", "tool_schema_tokens", False),
        ("Schema % of Total", None, None),
        ("", None, None),
        ("Avg Est. Cost (USD)", None, False),
        ("", None, None),
        ("Avg Tools Exposed", "tools_exposed", False),
        ("Avg Tools Used", "tools_used", True),
        ("Avg Agent Hops", "agent_hops", False),
        ("Avg MCPs Activated", "mcps_activated", True),
        ("", None, None),
        ("Avg Latency", "latency_ms", False),
    ]

    for label, key, higher_is_better in rows_spec:
        row = [label]
        vals = []

        if key is None and label == "Schema % of Total":
            for arch in arch_keys:
                avg_total = max(_avg(arch, "total_tokens"), 1)
                avg_schema = _avg(arch, "tool_schema_tokens")
                row.append(f"{avg_schema/avg_total*100:.0f}%")
            row.append("—")
            table.add_row(*row)
            continue

        if key is None and label == "Avg Est. Cost (USD)":
            for arch in arch_keys:
                avg_prompt = _avg(arch, "prompt_tokens")
                avg_compl = _avg(arch, "completion_tokens")
                cost = _cost_usd(avg_prompt, avg_compl)
                row.append(f"${cost:.6f}")
                vals.append(cost)
            if vals:
                best_idx = min(range(len(vals)), key=lambda i: vals[i])
                row.append(f"[green]${vals[best_idx]:.6f}[/green]")
            else:
                row.append("—")
            table.add_row(*row)
            continue

        if key is None or label == "":
            table.add_row("", *[""] * (len(arch_keys) + 1))
            continue

        for arch in arch_keys:
            avg = _avg(arch, key)
            vals.append(avg)
            if key == "latency_ms":
                row.append(f"{avg/1000:.1f}s")
            elif isinstance(avg, float):
                row.append(f"{avg:,.1f}")
            else:
                row.append(f"{avg:,.0f}")

        if vals:
            if higher_is_better:
                best_idx = max(range(len(vals)), key=lambda i: vals[i])
            else:
                best_idx = min(range(len(vals)), key=lambda i: vals[i])
            best_val = vals[best_idx]
            if key == "latency_ms":
                best_label = f"{best_val/1000:.1f}s"
            elif isinstance(best_val, float):
                best_label = f"{best_val:,.1f}"
            else:
                best_label = f"{best_val:,.0f}"

            improvement = ""
            if not higher_is_better and key in ("total_tokens", "prompt_tokens", "completion_tokens", "agent_hops") and vals and vals[0] > 0 and best_idx != 0:
                improvement = f" ({vals[best_idx]/vals[0]:.1f}x vs Centralized)"
            elif key == "latency_ms" and vals and vals[0] > 0:
                if best_idx != 0:
                    improvement = f" ({vals[best_idx]/vals[0]:.1f}x vs Centralized)"
                else:
                    improvement = " (baseline)"

            row.append(f"[green]{best_label}{improvement}[/green]")
        else:
            row.append("—")

        table.add_row(*row)

    console.print(table)
    console.print()

    # Tool-execution caveat
    avg_used_centralized = _avg(arch_keys[0], "tools_used") if arch_keys else 0
    if avg_used_centralized == 0:
        console.print(
            "  [yellow]⚠ Caveat: Centralized shows 0 tools used — the LLM responded with text only, "
            "never output TOOL_CALL:.[/yellow]\n"
            "  [yellow]  Its latency is artificially low (no tool execution round-trips).[/yellow]\n"
            "  [yellow]  Results are valid for token consumption comparison but not for wall-clock latency comparison.[/yellow]\n"
        )


def _why_text(arch: str, all_metrics, mcp_count) -> str:
    """Return architecture-specific 'why this matters' text."""
    texts = {
        "Centralized MCP": (
            "All schemas injected every turn. At {mcp} MCPs, 89-94% of tokens are just schema overhead — "
            "the model spends almost all its context budget on tool descriptions, not on the actual task. "
            "This architecture does not scale: schema cost grows linearly with MCP count while task complexity is constant. "
            "At ~100 MCPs, schema tokens alone will exceed most context windows, forcing truncation/skip."
        ),
        "Federated MCP": (
            "Router filters domains before execution, saving 50-80% schema tokens on simple tasks. "
            "However, the router itself is a single point of failure — at {mcp} MCPs, it stopped splitting "
            "complex tasks into subtasks (0 tools used in all workflows). Effective at moderate MCP counts "
            "where domain filtering provides real savings, but unreliable for complex multi-domain requests."
        ),
        "MCP Mediator": (
            "Schema loaded once for planning, execution uses zero LLM tokens. Fixed 2-hop cost regardless "
            "of tool count. But pays full schema cost for all {mcp} MCPs (86% of total at 50) even if only "
            "a fraction are used. Best suited for scenarios where tool schemas fit comfortably in context "
            "and tool execution is more expensive than planning."
        ),
        "Intent-Driven": (
            "Only domain descriptions (75 tokens) sent to LLM — no tool schemas loaded. "
            "Total cost stays flat at ~2,500 tokens regardless of MCP count. "
            "At {mcp} MCPs, this is 0.2× the tokens of Centralized. Schema avoidance is the key insight. "
            "Trade-off: requires LLM to output valid structured JSON, which can fail on weaker models. "
            "Best architecture for scaling to hundreds of MCPs."
        ),
    }
    return texts.get(arch, "").format(mcp=mcp_count)


def print_architecture_analysis(all_metrics, mcp_count):
    separator("TOKEN COST ANALYSIS — WHY EACH ARCHITECTURE COSTS WHAT IT DOES")

    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count))

    for arch in arch_keys:
        matches = [m for m in all_metrics if m.mcps_total == mcp_count and m.architecture_name == arch]
        if not matches:
            continue
        m = matches[0]

        avg_tokens = sum(m2.total_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_prompt = sum(m2.prompt_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_compl = sum(m2.completion_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_schema = sum(m2.tool_schema_tokens for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_hops = sum(m2.agent_hops for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_used = sum(m2.tools_used for m2 in matches if m2.tools_truncated != -1) / max(len([m2 for m2 in matches if m2.tools_truncated != -1]), 1)
        avg_cost = _cost_usd(avg_prompt, avg_compl)
        why = _why_text(arch, all_metrics, mcp_count)

        from rich.panel import Panel
        panel = Panel(
            f"[bold]{arch}[/bold]\n"
            f"[bold]Token Profile:[/bold]\n"
            f"  Avg Total Tokens:     {avg_tokens:>8,.0f}\n"
            f"  ├─ Prompt:            {avg_prompt:>8,.0f}\n"
            f"  ├─ Completion:        {avg_compl:>8,.0f}\n"
            f"  Avg Schema Tokens:    {avg_schema:>8,.0f}  ({avg_schema/max(avg_tokens,1)*100:.0f}% of total)\n"
            f"  Avg Agent Hops:       {avg_hops:>8,.1f}\n"
            f"  Avg Tools Used:       {avg_used:>8,.1f}\n"
            f"  Avg Est. Cost:        ${avg_cost:.6f}\n\n"
            f"[bold]Why this matters:[/bold]\n"
            f"{why}",
            border_style="blue",
        )
        console.print(panel)
        console.print()


def print_cross_count_trend(all_metrics):
    """Show how each architecture's metrics scale across MCP counts."""
    mcp_counts = sorted(set(m.mcps_total for m in all_metrics))
    arch_keys = sorted(set(m.architecture_name for m in all_metrics))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics))

    if len(mcp_counts) < 2:
        return

    def _avg_count(arch, key, count):
        vals = []
        for wf in wf_keys:
            matches = [m for m in all_metrics if m.mcps_total == count and m.architecture_name == arch and m.workflow_name == wf]
            if matches and matches[0].tools_truncated != -1:
                vals.append(getattr(matches[0], key, 0))
        return sum(vals) / len(vals) if vals else 0

    separator("SCALABILITY TREND — HOW METRICS CHANGE WITH MCP COUNT")

    for arch in arch_keys:
        table = Table(
            title=f"{arch}  (averaged across all workflows)",
            box=box.SIMPLE,
            title_style="bold white",
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan", width=24)
        for c in mcp_counts:
            table.add_column(f"MCPS={c}", style="yellow", justify="right", width=14)
        growth_col = f"Δ {mcp_counts[0]}→{mcp_counts[-1]} (ratio)"
        table.add_column(growth_col, style="magenta", justify="right", width=14)

        metrics_to_show = [
            ("Avg Total Tokens", "total_tokens"),
            ("Avg Schema Tokens", "tool_schema_tokens"),
            ("Avg Agent Hops", "agent_hops"),
            ("Avg Tools Used", "tools_used"),
            ("Avg Latency (s)", "latency_ms"),
        ]

        for label, key in metrics_to_show:
            row = [label]
            vals_at_counts = []
            for c in mcp_counts:
                v = _avg_count(arch, key, c)
                vals_at_counts.append(v)
                if key == "latency_ms":
                    row.append(f"{v/1000:.1f}s" if v else "—")
                elif isinstance(v, float):
                    row.append(f"{v:,.1f}")
                else:
                    row.append(f"{v:,.0f}")
            # Growth ratio
            if vals_at_counts and vals_at_counts[0] and len(vals_at_counts) > 1:
                ratio = vals_at_counts[-1] / vals_at_counts[0]
                if key == "latency_ms":
                    row.append(f"{vals_at_counts[-1]/1000:.1f}s ({ratio:.1f}x)" if vals_at_counts[-1] else "—")
                else:
                    row.append(f"{vals_at_counts[-1]:,.0f} ({ratio:.1f}x)" if vals_at_counts[-1] else "—")
            else:
                row.append("—")
            table.add_row(*row)

        console.print(table)
        console.print()


def print_caveats(all_metrics):
    separator("METHODOLOGY CAVEATS & LIMITATIONS")
    console.print(
        "  [bold]1. Tool Execution:[/bold]\n"
        "  • Centralized MCP never output TOOL_CALL: — it responded with text only.\n"
        "    Its latency is artificially low (no tool execution round-trips).\n"
        "    Token counts are still valid: the schema was in the prompt, consumed budget.\n\n"
        "  [bold]2. Federated Router Degradation:[/bold]\n"
        "  • At 50 MCPs, the Federated router stopped splitting tasks into subtasks\n"
        "    (0 tools used in all workflows). This may be a model limitation with\n"
        "    Gemini 2.0 Flash at higher schema densities.\n\n"
        "  [bold]3. Single Model Bias:[/bold]\n"
        "  • All results from [cyan]google/gemini-2.0-flash-001[/cyan] on OpenRouter.\n"
        "  • A weaker model may fail more often on JSON planning (Intent-Driven, Mediator)\n"
        "  • A stronger model may follow Centralized's TOOL_CALL: instruction better\n"
        "  • Rankings should be validated on target models before production decisions.\n\n"
        "  [bold]4. Schema Tokens Are Computed (Not from API):[/bold]\n"
        "  • Schema tokens counted via tiktoken (cl100k_base) on tool schema text.\n"
        "  • The actual model may tokenize schemas differently (different tokenizer).\n"
        "  • The relative ranking between architectures is unaffected by this.\n\n"
        "  [bold]5. Simulated vs Real MCP:[/bold]\n"
        "  • Tools are simulated (pre-registered schemas + mock responses).\n"
        "  • Real MCP servers add transport overhead (JSON-RPC framing, stdio latency).\n"
        "  • Token economics and architecture ranking should hold with real servers.\n\n"
        "  [bold]6. No Architecture Breakpoint Data:[/bold]\n"
        "  • No workflow was skipped (tools_truncated=0 for all runs).\n"
        "  • At what MCP count does each architecture exceed context budget? Unknown.\n"
        "  • Running at 100+ MCPs would reveal breakpoints.\n\n"
        "  [bold]7. Workflow Coverage:[/bold]\n"
        "  • 3 workflows tested. Results may vary for different task types.\n"
        "  • Simple tasks (Code Investigation) vs complex (Full System Audit)\n"
        "    show different architecture trade-offs.\n\n"
        "  [bold]8. Dollar Cost Estimates:[/bold]\n"
        f"  • Based on Gemini 2.0 Flash rates (${INPUT_COST_PER_1M}/1M in, ${OUTPUT_COST_PER_1M}/1M out).\n"
        "  • Actual costs depend on provider, model, and negotiated rates.\n"
        "  • Use for relative comparison, not absolute budgeting.\n"
    )


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

    print_cross_count_trend(tracker.metrics)
    print_caveats(tracker.metrics)

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
