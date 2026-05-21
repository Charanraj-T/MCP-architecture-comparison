# Experiment 1: Token Efficiency

Compares token consumption across **3 architectures** running the same 3 workflows against 4 domain MCPs (10 tools total). MCPs are simulated as domain-tagged tool groups with mock responses — no real servers.

## Architectures

| Architecture | How It Works |
|---|---|
| **Centralized MCP** | Single agent system prompt with all 10 tools. Simplest. |
| **Federated MCP** | Router LLM classifies request into relevant domains → injects only those tools. |
| **Multi-Agent** | Supervisor LLM decomposes task → per-domain worker agents run independently → supervisor compiles results. |

## Workflows

1. **Incident Investigation** (8 steps) — search repo, grep logs, k8s status, deployments, docs, create ticket, send notification, summarize
2. **Root Cause Analysis** (4 steps) — search docs, read file, grep logs, summarize
3. **Performance Analysis** (6 steps) — CPU metrics, deployments, search repo, search docs, create ticket, summarize

## MCPs (Simulated)

4 domains defined in `federated/router.py`:

| Domain | Tools |
|---|---|
| dev | `search_repo`, `read_file`, `grep_logs` |
| infra | `kubernetes_status`, `deployment_logs`, `cpu_metrics` |
| docs | `search_docs`, `summarize_data` |
| pm | `create_ticket`, `send_notification` |

- Tools are dicts with `{name, description, input_schema, mock_response}`
- LLM output parsed for `TOOL_CALL:` → mock response returned
- No real MCP servers are launched

## Metrics Collected

- **Per-LLM-call:** prompt/completion/reasoning tokens, latency
- **Token breakdown estimates:** user prompt, tool schema, orchestration, memory/replay, inter-agent, output tokens
- **Operational:** tools exposed, tools used, agent hops, MCPs activated

## Expected Results

| Metric | Centralized | Federated | Multi-Agent |
|---|---|---|---|
| Tool schema tokens | Highest (10 tools) | Filtered per domain | Per-worker subset |
| Total tokens | High | Lowest | Highest (duplication + inter-agent) |
| Orchestration overhead | Minimal | Router call | Decompose + N workers + compile |
| Agent hops | 1 | 2 (router + exec) | N+2 |

- **Federated** should win on token efficiency (smallest active context)
- **Centralized** should have highest prompt bloat
- **Multi-Agent** should show highest total tokens (repeated system prompts, inter-agent logs)

## Run

```bash
source ../.venv/bin/activate
python run_all.py
```

Output: rich tables to stdout, `results/comparison_results.json`, traces in `traces/`.
