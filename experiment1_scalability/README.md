# Experiment 1: Scalability

Tests how each architecture scales as the number of MCPs grows: **10, 20, 50, 100**. MCPs are simulated via a factory that generates N evenly distributed across 10 domains.

## Architectures

| Architecture | How It Works |
|---|---|
| **Centralized MCP** | All MCP tools in one system prompt. **Skipped** when schema tokens exceed 3,500 budget. |
| **Federated MCP** | Router selects relevant domains → injects only their tools. Naturally filters noise. |
| **Multi-Agent** | Supervisor decomposes → domain workers run independently → compile. More hops but isolates schema per agent. |

## Workflows

1. **Code Investigation** — search codebase, read files, find usages
2. **Incident Investigation** — check pods, grep logs, CPU metrics, deployments, create ticket
3. **Full System Audit** — all domains (code, logs, deployment, monitoring, DB, networking, security, docs, project mgmt, notifications)

## MCPs (Generated)

10 domain types from `common_tools/factory.py`:

`code_search`, `log_analysis`, `deployment`, `monitoring`, `database`, `networking`, `security`, `documentation`, `project_mgmt`, `notifications`

Each domain has 3 tool templates. At count=N, each domain gets N/10 MCP instances, each with 3 tools (suffixed `_01`, `_02`, ...). Example at N=100: 10 domains × 10 instances × 3 tools = **300 tools total**.

| MCP Count | Total Tools | Schema Tokens (approx) |
|---|---|---|
| 10 | 30 | ~1,200 |
| 20 | 60 | ~2,400 |
| 50 | 150 | ~6,000 |
| 100 | 300 | ~12,000 |

## Token Budget & Skip Logic

- Hard cap: **3,500 tool schema tokens**
- Architectures exceeding budget are **SKIPPED** (`tools_truncated = -1`)
- Centralized typically fails at 50+ MCPs (all tools unconditionally injected)
- Federated may survive longer by filtering unneeded domains
- Multi-Agent distributes schema across workers

## Expected Results

- **Centralized:** Fails earliest — schema tokens grow linearly with total MCP count; no filtering
- **Federated:** Scales best — router prunes irrelevant domains, keeping schema tokens manageable
- **Multi-Agent:** Degrades gracefully — worker isolation prevents any single prompt from being too large, but total system tokens grow with each active worker

## Metrics

Same as Experiment 1 plus `tools_truncated` flag and `mcps_total` count. No token breakdown estimates (focused purely on whether architecture fits in budget).

## Run

```bash
source ../.venv/bin/activate
python run_all.py
```

Output: rich tables per MCP count + final comparison table, `results/scalability_results.json`, traces in `traces/`.
