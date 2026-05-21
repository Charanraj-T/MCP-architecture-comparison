# Experiment 3: Real MCP Orchestration

Evolves Exp 1/2 from simulated MCPs (tool dicts with mock responses) to **real MCP servers** running as subprocesses with stdio transport, real schema discovery, real tool execution, and real agent isolation.

## What's Real vs What's Mocked

| Layer | Exp 1/2 | Exp 3 |
|---|---|---|
| MCP servers | Tool dicts in Python | Real FastMCP subprocesses, stdio transport |
| Tool schemas | Hardcoded dicts | Dynamically discovered via `tools/list` |
| Tool execution | Predefined mock JSON | Real file I/O, git commands, SQL queries, URL fetches |
| MCP protocol | None (direct dict lookup) | Real JSON-RPC 2.0 over stdio |
| Agent isolation | Same process, same context | Separate MCP connections per worker |

## MCP Servers

4 real MCP servers, each a standalone Python process using `FastMCP`:

| Server | Tools | Real Behavior |
|---|---|---|
| **Dev MCP** | `read_file`, `search_files`, `grep_files`, `git_log`, `git_diff`, `git_status` | Reads/writes actual filesystem, runs real git commands |
| **Docs MCP** | `fetch_url`, `store_memory`, `recall_memory`, `search_memory`, `list_memory` | Fetches real URLs via httpx, in-process memory store |
| **Data MCP** | `query`, `list_tables`, `describe_table`, `seed_sample_data` | Real SQLite queries against an in-memory DB seeded with sample incidents/deployments/metrics |
| **Reasoning MCP** | `sequential_think`, `decompose_task`, `summarize` | Structured reasoning, calls LM Studio for decomposition/summarization |

Each server is launched as `asyncio.create_subprocess_exec` and communicates via JSON-RPC 2.0 over stdio. The `mcp_client.py` module handles the full MCP lifecycle: `initialize` → `tools/list` → `tools/call` → shutdown.

## Architectures

### 1. Centralized MCP
- Connects to **all 4 MCP servers** immediately
- Loads **all tool schemas** into one system prompt
- Single LLM loop: `TOOL_CALL:` → routes to the correct MCP server → returns real result
- **Tests:** schema injection cost, context saturation, centralized orchestration overhead with real tool latency

### 2. Federated MCP
- **Semantic router** (LLM-based) classifies the request into relevant MCP domains
- Only connects to **selected MCPs** (e.g., "deployment issue" → dev + data, not docs)
- Only selected schemas enter context
- **Tests:** routing accuracy, schema reduction efficiency, federation overhead

### 3. Multi-Agent
- **Supervisor** decomposes task into subtasks
- **Parallel workers** (Dev, Docs, Data, Planning) each connect to their **own dedicated MCP** with fully **isolated context**
- Workers run via `asyncio.gather()` for parallel tool execution
- Supervisor compiles all results into final answer
- **Tests:** inter-agent token duplication, context isolation benefits, parallel execution tradeoffs

## Workflows

1. **Incident Investigation** — DB queries, deployment history, code search, git log, metrics analysis, URL fetch, summarization (all 4 MCPs)
2. **Deployment Analysis** — git history, DB deployments, code reading, git diff, reasoning (dev + data + reasoning)
3. **Full System Audit** — all tables + schemas, full data query, git status + commits, code grep, URL fetch, memory store, plan + summarize (all 4 MCPs)

## Metrics

Same base set from Exp 1/2 plus:
- `router_tokens` — tokens consumed by the semantic routing call
- `real_tool_calls` — count of actual MCP tool invocations
- `mcp_servers_connected` — how many MCP processes were spawned

## Expected Findings

| Metric | Centralized | Federated | Multi-Agent |
|---|---|---|---|
| Schema tokens | All 4 MCPs | Filtered by router | Per-worker subset |
| Total tokens | Highest (all schemas) | Lowest (filtered) | Highest (duplicated per worker) |
| MCP connections | 4 (all) | Selected subset | Per-worker (parallel) |
| Real tool calls | Serial, all available | Serial, filtered | Parallel, isolated |
| Agent hops | 1 | 1 + router | N + 2 (decompose + workers + compile) |
| Latency | Tool exec sequential | Tool exec sequential | Workers parallel |

## Run

```bash
source ../.venv/bin/activate
python run_all.py
```

Requires LM Studio running at `http://localhost:1234/v1` with `qwen/qwen3-8b` loaded.
