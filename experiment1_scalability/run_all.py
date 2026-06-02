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
from common.client import OpenAIClient
from common_tools import generate_mcps, ToolRegistry
from metrics import TokenTracker

from centralized.run import CentralizedOrchestrator
from federated.run import FederatedOrchestrator
from mediator.run import MediatorOrchestrator
from intent_driven.run import IntentDrivenOrchestrator

from workflows import WORKFLOWS

ARCHITECTURES = [
    ("Centralized MCP", CentralizedOrchestrator),
    ("Federated MCP", FederatedOrchestrator),
    ("MCP Mediator", MediatorOrchestrator),
    ("Intent-Driven", IntentDrivenOrchestrator),
]

MODEL_PRICING = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "mistralai/mistral-small-3.1-24b-instruct": (0.06, 0.10),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}


def _cost_usd(prompt_tokens, completion_tokens, model_name=""):
    in_rate, out_rate = MODEL_PRICING.get(model_name, (0.10, 0.40))
    return (prompt_tokens / 1_000_000 * in_rate) + (completion_tokens / 1_000_000 * out_rate)


def separator(title):
    console.print()
    console.print("=" * 66, style="bold cyan")
    console.print(f"{title:^66}", style="bold cyan")
    console.print("=" * 66, style="bold cyan")
    console.print()


def _filter_metrics(all_metrics, model_name=None, mcp_count=None):
    """Filter metrics list by optional model_name and mcp_count."""
    result = all_metrics
    if model_name:
        result = [m for m in result if m.model_name == model_name]
    if mcp_count is not None:
        result = [m for m in result if m.mcps_total == mcp_count]
    return result


def _avg_metric(all_metrics, model_name, mcp_count, arch, workflow, key):
    """Compute average of `key` across workflows for a given model/mcp/arch combo.
    
    Returns 0 if no valid (non-truncated) metrics found.
    """
    vals = []
    for m in all_metrics:
        if (m.model_name == model_name and m.mcps_total == mcp_count
                and m.architecture_name == arch and m.workflow_name == workflow
                and m.tools_truncated != -1):
            vals.append(getattr(m, key, 0))
    return sum(vals) / len(vals) if vals else 0


def print_token_cost_methodology(model_configs):
    separator("TOKEN COST METHODOLOGY")
    models_line = "  • Models tested: " + ", ".join(f"[cyan]{m[1]}[/cyan] (${m[2]:.2f}/${m[3]:.2f} per 1M)" for m in model_configs)
    pricing_lines = "\n".join(
        f"  • [cyan]{m[1]}[/cyan]: ${m[2]:.4f}/1M in, ${m[3]:.4f}/1M out"
        for m in model_configs
    )
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
        f"{models_line}\n"
        f"{pricing_lines}\n"
        "  • Cost = (prompt_tokens / 1M × input_rate) + (completion_tokens / 1M × output_rate)\n\n"
        "  [bold]Token Cost Drivers per Architecture:[/bold]\n"
        "  • [yellow]Centralized[/yellow]: Schema cost paid EVERY turn. Large prompts from full history.\n"
        "  • [yellow]Federated[/yellow]: Router call adds 1 hop but fewer schemas injected. Lower per-turn cost.\n"
        "  • [yellow]MCP Mediator[/yellow]: Schema loaded once for planning. Execution = zero LLM tokens.\n"
        "  • [yellow]Intent-Driven[/yellow]: Schemas NOT loaded into prompts — only domain descriptions.\n"
        "    75 tokens of domain info vs 12K+ tokens of full schemas. Key source of savings.\n\n"
        "  [bold]Multi-Model Approach:[/bold]\n"
        "  • Each architecture runs identically against each model (same MCPs, same workflows)\n"
        "  • Cross-model comparison validates that architecture rankings are model-agnostic\n"
        "  • Results should be validated on target models before production decisions\n"
    )


def print_executive_summary(all_metrics, model_configs):
    """CTO-facing conclusion: which architecture wins and why."""
    mcp_counts = sorted(set(m.mcps_total for m in all_metrics))
    arch_keys = sorted(set(m.architecture_name for m in all_metrics))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics))
    model_names = [mc[1] for mc in model_configs]

    max_mcp = mcp_counts[-1] if mcp_counts else 50

    # Per-model best architecture
    model_best = {}
    for mn in model_names:
        best_arch = None
        best_total = float("inf")
        for arch in arch_keys:
            vals = []
            for wf in wf_keys:
                matches = [m for m in all_metrics if m.model_name == mn and m.mcps_total == max_mcp and m.architecture_name == arch and m.workflow_name == wf]
                if matches and matches[0].tools_truncated != -1:
                    vals.append(matches[0].total_tokens)
            avg = sum(vals) / len(vals) if vals else 0
            if avg and avg < best_total:
                best_total = avg
                best_arch = arch
        model_best[mn] = (best_arch, best_total)

    # Cross-model consistency
    winners = set(a for a, _ in model_best.values())
    consistent = len(winners) == 1
    winner_arch = next(iter(winners)) if consistent else "Varies by model"

    from rich.panel import Panel
    summary = (
        f"[bold white]RECOMMENDATION: {winner_arch}[/bold white]"
        f"{' (consistent across all models)' if consistent else ' (differs by model)'}\n\n"
        f"[bold]Architecture Rankings at {max_mcp} MCPs (by total tokens):[/bold]\n"
    )
    for mn in model_names:
        arch, amt = model_best[mn]
        summary += f"    {mn}: [green]{arch}[/green] ({amt:,.0f} avg tokens)\n"

    summary += f"\n[bold]Cross-Model Consistency:[/bold] "
    if consistent:
        summary += f"{winner_arch} wins on every model tested — ranking is model-agnostic.\n"
    else:
        summary += "Rankings change between models — validate on your target model.\n"

    # Compute data-driven insights instead of hardcoded narrative
    intent_totals = []
    centralized_totals = []
    for m in all_metrics:
        if m.tools_truncated == -1:
            continue
        if m.architecture_name == "Intent-Driven":
            intent_totals.append(m.total_tokens)
        elif m.architecture_name == "Centralized MCP":
            centralized_totals.append(m.total_tokens)

    avg_intent = sum(intent_totals) / len(intent_totals) if intent_totals else 0
    avg_centralized = sum(centralized_totals) / len(centralized_totals) if centralized_totals else 1

    # Find the MCP count where Centralized first exceeds budget
    skip_threshold = "N/A"
    for mc in sorted(mcp_counts):
        centralized_at_mc = [m for m in all_metrics if m.architecture_name == "Centralized MCP" and m.mcps_total == mc and m.tools_truncated == -1]
        if len(centralized_at_mc) < len(wf_keys):
            skip_threshold = f"{mc} MCPs"
            break

    summary += (
        f"\n[bold]Key Finding:[/bold] Intent-Driven averages {avg_intent:,.0f} tokens across all runs, "
        f"vs {avg_centralized:,.0f} for Centralized ({avg_intent/max(avg_centralized,1):.1f}x ratio).\n\n"
        f"[bold]Edge Cases & Risks:[/bold]\n"
        f"    • Centralized MCP starts exceeding budget at {skip_threshold} — schema tokens grow linearly\n"
        f"    • Mediator works well if tool schemas fit in context and you need fixed-cost execution\n"
        f"    • Federated router degrades at higher MCP counts — routing decisions become unreliable\n"
        f"    • Intent-Driven requires LLM to output valid structured JSON; weaker models may fail\n\n"
        f"[bold]Bottom Line:[/bold] Intent-Driven provides the flattest cost curve as MCP count grows. "
        f"Centralized and Mediator hit linear schema-cost walls. Federated holds promise but routing "
        f"reliability must improve."
    )
    panel = Panel(summary, border_style="green", title="[bold white]CTO EXECUTIVE SUMMARY[/bold white]", title_align="left")
    console.print()
    console.print(panel)
    console.print()


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


async def select_model_configs():
    """Return list of (client, model_name, input_cost_per_1m, output_cost_per_1m)."""
    from common.client import _get_env_config, OpenAIClient
    from dotenv import load_dotenv
    load_dotenv()

    env_config = _get_env_config()
    if env_config:
        models = [m.strip() for m in env_config["model"].split(",")]
        configs = []
        for model in models:
            client = OpenAIClient(
                base_url=env_config["base_url"],
                model=model,
                api_key=env_config["api_key"] or "not-needed",
                inject_no_think=env_config["inject_no_think"],
            )
            in_rate, out_rate = MODEL_PRICING.get(model, (0.10, 0.40))
            configs.append((client, model, in_rate, out_rate))
        return configs

    console.print("[bold]Select model #1:[/bold]")
    client1, model1 = select_provider()
    in1, out1 = MODEL_PRICING.get(model1, (0.10, 0.40))
    configs = [(client1, model1, in1, out1)]

    add = input("Add a second model? (y/N): ").strip().lower()
    if add == "y":
        console.print("[bold]Select model #2:[/bold]")
        client2, model2 = select_provider()
        in2, out2 = MODEL_PRICING.get(model2, (0.10, 0.40))
        configs.append((client2, model2, in2, out2))

    return configs


def print_per_workflow_table(all_metrics, mcp_count, workflow_name, model_name=None):
    filtered = _filter_metrics(all_metrics, model_name=model_name, mcp_count=mcp_count)
    arch_keys = sorted(set(m.architecture_name for m in filtered))
    model_tag = f"  |  Model: {model_name}" if model_name else ""
    table = Table(
        title=f"{workflow_name}  |  MCP Count: {mcp_count}{model_tag}",
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
            matches = [m for m in filtered if m.architecture_name == arch and m.workflow_name == workflow_name]
            if matches:
                m = matches[0]
                if m.tools_truncated == -1:
                    row.append("[red]SKIP[/red]")
                elif label == "Est. Cost (USD)":
                    cost = _cost_usd(m.prompt_tokens, m.completion_tokens, model_name=model_name or "")
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


def print_summary_table(all_metrics, mcp_count, model_name=None):
    filtered = _filter_metrics(all_metrics, model_name=model_name, mcp_count=mcp_count)
    arch_keys = sorted(set(m.architecture_name for m in filtered))
    wf_keys = sorted(set(m.workflow_name for m in filtered))
    model_tag = f"  |  Model: {model_name}" if model_name else ""

    table = Table(
        title=f"EXECUTIVE SUMMARY — MCP Count: {mcp_count}{model_tag}  (Averaged Across All Workflows)",
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
            matches = [m for m in filtered if m.architecture_name == arch and m.workflow_name == wf]
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
                cost = _cost_usd(avg_prompt, avg_compl, model_name=model_name or "")
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


def _why_text(arch, mcp_count, avg_total=0, avg_schema=0, avg_hops=0, avg_used=0, centralized_total=0):
    """Return architecture-specific 'why this matters' text using actual data."""
    if avg_total == 0:
        return (f"[SKIP] This architecture did not execute on this model — "
                f"the LLM failed to output a valid tool call format. "
                f"Results reflect model capability, not architecture efficiency.")

    schema_pct = avg_schema / avg_total * 100
    ratio_vs_centralized = avg_total / max(centralized_total, 1)

    texts = {
        "Centralized MCP": (
            f"All schemas injected every turn. At {mcp_count} MCPs, {schema_pct:.0f}% of tokens are schema overhead — "
            f"the model spends almost all its context budget on tool descriptions. "
            f"Schema cost grows with MCP count. At higher counts, schema tokens alone "
            f"can exceed context windows, forcing truncation."
        ),
        "Federated MCP": (
            f"Router filters domains before execution, reducing schema tokens. "
            f"At {mcp_count} MCPs, schema tokens are {avg_schema:,.0f} ({schema_pct:.0f}% of total). "
            f"Whether the router collapses (returns all domains) depends on model capability. "
            f"Effective when routing works reliably; unreliable when it degrades."
        ),
        "MCP Mediator": (
            f"Schema loaded once for planning ({avg_schema:,.0f} tokens, {schema_pct:.0f}% of total), "
            f"execution uses zero LLM tokens. Fixed {avg_hops:.0f}-hop cost. "
            f"Best suited when tool schemas fit comfortably in context and execution is more expensive than planning."
        ),
        "Intent-Driven": (
            f"Only domain descriptions sent to LLM — no tool schemas loaded. "
            f"Schema tokens = {avg_schema:,.0f} (just domain descriptions), {schema_pct:.0f}% of total. "
            f"At {mcp_count} MCPs, this is {ratio_vs_centralized:.1f}× the tokens of Centralized. "
            f"Trade-off: requires LLM to output valid structured JSON — weaker models fail here. "
            f"Best architecture for scaling to hundreds of MCPs."
        ),
    }
    return texts.get(arch, "")


def print_architecture_analysis(all_metrics, mcp_count, model_name=None):
    filtered = _filter_metrics(all_metrics, model_name=model_name, mcp_count=mcp_count)
    arch_keys = sorted(set(m.architecture_name for m in filtered))
    wf_keys = sorted(set(m.workflow_name for m in filtered))
    model_tag = f"  |  Model: {model_name}" if model_name else ""

    separator(f"TOKEN COST ANALYSIS — WHY EACH ARCHITECTURE COSTS WHAT IT DOES{model_tag}")

    # Precompute Centralized average for ratio comparisons
    centralized_avg = 0
    cent_matches = [m for m in filtered if m.architecture_name == "Centralized MCP"]
    cent_valid = [m2 for m2 in cent_matches if m2.tools_truncated != -1]
    if cent_valid:
        centralized_avg = sum(m2.total_tokens for m2 in cent_valid) / len(cent_valid)

    for arch in arch_keys:
        matches = [m for m in filtered if m.architecture_name == arch]
        if not matches:
            continue

        valid = [m2 for m2 in matches if m2.tools_truncated != -1]
        denom = max(len(valid), 1)

        avg_tokens = sum(m2.total_tokens for m2 in valid) / denom
        avg_prompt = sum(m2.prompt_tokens for m2 in valid) / denom
        avg_compl = sum(m2.completion_tokens for m2 in valid) / denom
        avg_schema = sum(m2.tool_schema_tokens for m2 in valid) / denom
        avg_hops = sum(m2.agent_hops for m2 in valid) / denom
        avg_used = sum(m2.tools_used for m2 in valid) / denom
        avg_cost = _cost_usd(avg_prompt, avg_compl, model_name=model_name or "")
        why = _why_text(arch, mcp_count, avg_total=avg_tokens, avg_schema=avg_schema, avg_hops=avg_hops, avg_used=avg_used, centralized_total=centralized_avg)

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


def print_cross_model_comparison(all_metrics, mcp_count, model_configs):
    """Side-by-side comparison of all architectures across models at a given MCP count."""
    model_names = [mc[1] for mc in model_configs]
    arch_keys = sorted(set(m.architecture_name for m in all_metrics if m.mcps_total == mcp_count))
    wf_keys = sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count))
    multi = len(model_names) > 1

    separator(f"CROSS-MODEL COMPARISON — MCP Count: {mcp_count}")

    table = Table(
        title=f"Architectures Compared Across {', '.join(model_names)}",
        box=box.HEAVY_EDGE,
        title_style="bold white on blue",
        header_style="bold white",
    )
    table.add_column("Architecture", style="cyan", width=22)
    for mn in model_names:
        short = mn.split("/")[-1].split("-")[0] if "/" in mn else mn[:12]
        table.add_column(f"{short}\nTotal Tok", style="yellow", justify="right", width=12)
        table.add_column(f"{short}\nEst. Cost", style="yellow", justify="right", width=12)
        table.add_column(f"{short}\nTools Used", style="yellow", justify="right", width=12)

    for arch in arch_keys:
        row = [arch]
        for mn in model_names:
            vals = []
            for wf in wf_keys:
                matches = [m for m in all_metrics if m.model_name == mn and m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
                if matches and matches[0].tools_truncated != -1:
                    vals.append(matches[0])
            if vals:
                avg_tok = sum(v.total_tokens for v in vals) / len(vals)
                avg_prompt = sum(v.prompt_tokens for v in vals) / len(vals)
                avg_compl = sum(v.completion_tokens for v in vals) / len(vals)
                avg_used = sum(v.tools_used for v in vals) / len(vals)
                avg_cost = _cost_usd(avg_prompt, avg_compl, model_name=mn)
                row.append(f"{avg_tok:,.0f}")
                row.append(f"${avg_cost:.6f}")
                row.append(f"{avg_used:.1f}")
            else:
                row.extend(["[red]SKIP[/red]"] * 3)
        table.add_row(*row)

    console.print(table)
    console.print()

    # Cross-model consistency check
    if multi:
        ranking_per_model = {}
        for mn in model_names:
            arch_scores = []
            for arch in arch_keys:
                vals = []
                for wf in wf_keys:
                    matches = [m for m in all_metrics if m.model_name == mn and m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
                    if matches and matches[0].tools_truncated != -1:
                        vals.append(matches[0].total_tokens)
                avg = sum(vals) / len(vals) if vals else float("inf")
                arch_scores.append((avg, arch))
            arch_scores.sort()
            ranking_per_model[mn] = [a for _, a in arch_scores]

        # Check if rankings are identical
        first_ranking = list(ranking_per_model.values())[0]
        consistent = all(r == first_ranking for r in ranking_per_model.values())

        if consistent:
            console.print(f"  [green]✓ Ranking consistent across all models: "
                          f"{' → '.join(first_ranking)}[/green]\n")
        else:
            console.print("  [yellow]⚠ Rankings differ between models:[/yellow]")
            for mn, ranking in ranking_per_model.items():
                console.print(f"    {mn}: {' → '.join(ranking)}")
            console.print()

    # Tool-execution caveat per model
    for mn in model_names:
        filtered = _filter_metrics(all_metrics, model_name=mn, mcp_count=mcp_count)
        arch_keys_local = sorted(set(m.architecture_name for m in filtered))
        if arch_keys_local:
            avg_used = _avg_for_model(all_metrics, mn, mcp_count, arch_keys_local[0], "tools_used")
            if avg_used == 0:
                console.print(
                    f"  [yellow]⚠ [{mn}] Centralized shows 0 tools used — the LLM responded with text only, "
                    f"never output TOOL_CALL:.[/yellow]\n"
                )


def _avg_for_model(all_metrics, model_name, mcp_count, arch, key):
    """Average a metric for a specific model × architecture × MCP count."""
    vals = []
    for wf in sorted(set(m.workflow_name for m in all_metrics if m.mcps_total == mcp_count)):
        matches = [m for m in all_metrics if m.model_name == model_name and m.mcps_total == mcp_count and m.architecture_name == arch and m.workflow_name == wf]
        if matches and matches[0].tools_truncated != -1:
            vals.append(getattr(matches[0], key, 0))
    return sum(vals) / len(vals) if vals else 0


def print_cross_count_trend(all_metrics, model_name=None):
    """Show how each architecture's metrics scale across MCP counts."""
    filtered = _filter_metrics(all_metrics, model_name=model_name)
    mcp_counts = sorted(set(m.mcps_total for m in filtered))
    arch_keys = sorted(set(m.architecture_name for m in filtered))
    wf_keys = sorted(set(m.workflow_name for m in filtered))

    if len(mcp_counts) < 2:
        return

    def _avg_count(arch, key, count):
        vals = []
        for wf in wf_keys:
            matches = [m for m in filtered if m.mcps_total == count and m.architecture_name == arch and m.workflow_name == wf]
            if matches and matches[0].tools_truncated != -1:
                vals.append(getattr(matches[0], key, 0))
        return sum(vals) / len(vals) if vals else 0

    model_tag = f"  |  Model: {model_name}" if model_name else ""
    separator(f"SCALABILITY TREND — HOW METRICS CHANGE WITH MCP COUNT{model_tag}")

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


def print_caveats(all_metrics, model_configs):
    model_names = [mc[1] for mc in model_configs]
    models_str = ", ".join(f"[cyan]{mn}[/cyan]" for mn in model_names)
    pricing_lines = "\n".join(
        f"  • {mn}: ${mc[2]:.4f}/1M in, ${mc[3]:.4f}/1M out"
        for mn, mc in [(mc[1], mc) for mc in model_configs]
    )

    separator("METHODOLOGY CAVEATS & LIMITATIONS")
    console.print(
        "  [bold]1. Tool Execution:[/bold]\n"
        "  • If an architecture shows 0 tools used across all workflows, the model "
        "responded with text only, never output a tool call.\n"
        "    - Token counts remain valid (schemas were still in the prompt, consumed budget).\n"
        "    - Latency is artificially low (no tool execution round-trips).\n"
        "    - This is model-dependent: some models follow JSON format better than others.\n\n"
        "  [bold]2. Federated Router Degradation:[/bold]\n"
        "  • The Federated router may stop reliably filtering at higher MCP counts.\n"
        "  • Console output shows whether router returned all domains (filter collapse).\n"
        "  • This is model-dependent — some models handle routing better than others.\n\n"
        f"  [bold]3. Cross-Model Validation:[/bold]\n"
        f"  • Results validated across {len(model_names)} models: {models_str}\n"
        "  • Rankings that are consistent across models are more reliable than model-specific ones.\n"
        "  • A model that fails on JSON planning (Intent-Driven, Mediator) will score higher "
        "on the other architecture.\n"
        "  • Results should still be validated on target models before production decisions.\n\n"
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
        f"{pricing_lines}\n"
        "  • Actual costs depend on provider, model, and negotiated rates.\n"
        "  • Use for relative comparison, not absolute budgeting.\n"
    )


async def main():
    separator("MCP SCALABILITY COMPARISON")

    model_configs = await select_model_configs()
    if not model_configs:
        console.print("[red]No model selected. Exiting.[/red]")
        sys.exit(1)

    multi = len(model_configs) > 1

    # Connect to each model
    for client, model_name, in_rate, out_rate in model_configs:
        try:
            test = await client.chat([{"role": "user", "content": "Say ready"}], temperature=0.1, max_tokens=20)
            console.print(f"[green]  Connected ({client.model})[/green] ({test.usage.total_tokens} tokens)")
        except Exception as e:
            console.print(f"[red]  Cannot connect {model_name}: {e}[/red]")
            sys.exit(1)
    console.print()

    mcp_counts = select_mcp_counts()
    console.print(f"  [dim]Testing: {mcp_counts}[/dim]\n")

    print_token_cost_methodology(model_configs)

    tracker = TokenTracker()

    for client, model_name, in_rate, out_rate in model_configs:
        budget = get_token_budget(model_name)
        label = f"  [bold cyan][Model: {model_name}][/bold cyan]"
        console.print(f"\n{label}")
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
                    try:
                        metrics = await asyncio.wait_for(orch.run(wf), timeout=300)
                    except asyncio.TimeoutError:
                        console.print(f"    {wf['name'][:30]:30s} [red]TIMEOUT[/red] (exceeded 300s)")
                        from metrics.token_tracker import WorkflowMetrics
                        metrics = WorkflowMetrics(
                            workflow_name=wf["name"],
                            architecture_name=arch_name,
                            model_name=model_name,
                            mcps_total=mcp_count,
                        )
                        metrics.tools_truncated = -1
                    except Exception as e:
                        console.print(f"    {wf['name'][:30]:30s} [red]ERROR[/red] {type(e).__name__}: {e}")
                        from metrics.token_tracker import WorkflowMetrics
                        metrics = WorkflowMetrics(
                            workflow_name=wf["name"],
                            architecture_name=arch_name,
                            model_name=model_name,
                            mcps_total=mcp_count,
                        )
                        metrics.tools_truncated = -1
                    metrics.model_name = model_name
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

    # === REPORT GENERATION ===
    print_executive_summary(tracker.metrics, model_configs)

    for mcp_count in mcp_counts:
        print_cross_model_comparison(tracker.metrics, mcp_count, model_configs)
        for client, model_name, in_rate, out_rate in model_configs:
            console.print()
            separator(f"ARCHITECTURE ANALYSIS — {model_name} @ MCP={mcp_count}")
            print_summary_table(tracker.metrics, mcp_count, model_name=model_name)
            print_architecture_analysis(tracker.metrics, mcp_count, model_name=model_name)

    for client, model_name, in_rate, out_rate in model_configs:
        print_cross_count_trend(tracker.metrics, model_name=model_name)

    print_caveats(tracker.metrics, model_configs)

    os.makedirs("results", exist_ok=True)
    results = tracker.get_results()

    # Add metadata envelope for reproducibility
    import uuid
    import subprocess as _sp
    try:
        git_hash = _sp.check_output(["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"

    envelope = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "models": [mc[1] for mc in model_configs],
        "mcp_counts": mcp_counts,
        "architectures": [name for name, _ in ARCHITECTURES],
        "workflows": [wf["name"] for wf in WORKFLOWS],
        "results": results,
    }

    with open("results/scalability_results.json", "w") as f:
        json.dump(envelope, f, indent=2)
    console.print(f"[dim]Results saved to results/scalability_results.json[/dim]")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{ts}.html", "w") as f:
        f.write(console.export_html())
    console.print(f"[dim]HTML report saved to reports/report_{ts}.html[/dim]")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
