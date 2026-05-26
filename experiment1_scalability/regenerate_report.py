#!/usr/bin/env python3
"""Regenerate HTML report from saved JSON results (no API calls)."""
import sys
import json
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import WorkflowMetrics
from run_all import (
    MODEL_PRICING, console,
    print_token_cost_methodology, print_executive_summary,
    print_cross_model_comparison, print_summary_table,
    print_architecture_analysis, print_cross_count_trend, print_caveats,
)

def main():
    results_path = Path("results/scalability_results.json")
    if not results_path.exists():
        print(f"[red]No results found at {results_path}[/red]")
        sys.exit(1)

    with open(results_path) as f:
        raw = json.load(f)

    # Reconstruct WorkflowMetrics
    metrics_list = []
    for d in raw:
        m = WorkflowMetrics(
            workflow_name=d["workflow"],
            architecture_name=d["architecture"],
            model_name=d.get("model", ""),
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            reasoning_tokens=d.get("reasoning_tokens", 0),
            total_tokens=d["total_tokens"],
            tool_schema_tokens=d["tool_schema_tokens"],
            tools_exposed=d["tools_exposed"],
            tools_used=d["tools_used"],
            tools_truncated=d["tools_truncated"],
            agent_hops=d["agent_hops"],
            mcps_total=d["mcps_total"],
            mcps_activated=d["mcps_activated"],
            latency_ms=d["latency_ms"],
            explanation=d.get("explanation", ""),
        )
        metrics_list.append(m)

    # Reconstruct model configs from data
    model_names = sorted(set(m.model_name for m in metrics_list if m.model_name))
    model_configs = []
    for mn in model_names:
        in_rate, out_rate = MODEL_PRICING.get(mn, (0.10, 0.40))
        # Dummy client — not used in report generation
        from common.client import OpenAIClient
        dummy = OpenAIClient(base_url="", model=mn, api_key="")
        model_configs.append((dummy, mn, in_rate, out_rate))

    mcp_counts = sorted(set(m.mcps_total for m in metrics_list))

    print_token_cost_methodology(model_configs)
    print_executive_summary(metrics_list, model_configs)

    for mc in mcp_counts:
        print_cross_model_comparison(metrics_list, mc, model_configs)
        for _, mn, _, _ in model_configs:
            console.print()
            from run_all import separator
            separator(f"ARCHITECTURE ANALYSIS — {mn} @ MCP={mc}")
            print_summary_table(metrics_list, mc, model_name=mn)
            print_architecture_analysis(metrics_list, mc, model_name=mn)

    for _, mn, _, _ in model_configs:
        print_cross_count_trend(metrics_list, model_name=mn)

    print_caveats(metrics_list, model_configs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{ts}.html", "w") as f:
        f.write(console.export_html())
    print(f"[dim]HTML report saved to reports/report_{ts}.html[/dim]")

if __name__ == "__main__":
    main()
