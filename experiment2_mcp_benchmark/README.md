# Experiment 2: Real MCP Benchmark

Tests 3 architectures — **Centralized (1MCP), Federated (Bifrost Code Mode), Multi-Agent** — against **real npx/uvx MCP servers**: filesystem, git, fetch, memory, SQLite, and sequential thinking, running as subprocesses with stdio transport.

## MCP Servers (via npx/uvx, no global installs)

| Server | Runner | Tools | Real Behavior |
|---|---|---|---|
| **Filesystem** | `npx @modelcontextprotocol/server-filesystem` | 14 | Reads/writes files, creates directories, searches |
| **Git** | `uvx mcp-server-git` | 12 | Real git log/diff/status/branch on sandbox/repo |
| **Fetch** | `npx mcp-server-fetch-typescript` | 4 | Fetches real URLs, converts to markdown |
| **Memory** | `npx @modelcontextprotocol/server-memory` | 9 | In-memory key-value store |
| **Data** | `uvx mcp-server-sqlite` | 6 | Real SQLite queries against test.db with users/orders |
| **Reasoning** | `npx @modelcontextprotocol/server-sequential-thinking` | 1 | Step-by-step structured reasoning |

All servers auto-download on first run via `npx -y` / `uvx`. Nothing installed globally.

## Architectures

### 1. Centralized (1MCP)
- All 6 servers behind [`@1mcp/agent`](https://github.com/1mcp-app/agent) aggregator proxy — 1 stdio subprocess
- Tools namespaced as `{server}_1mcp_{tool}` — **all 46 tools in system prompt**
- Measures proxy overhead, connection efficiency (1 vs 6), 1MCP routing

### 2. Federated (Bifrost Code Mode)
- All 6 servers behind [`Bifrost`](https://github.com/maximhq/bifrost) AI gateway with Code Mode
- Only 4 meta-tools exposed: `listToolFiles`, `readToolFile`, `getToolDocs`, `executeToolCode`
- LLM discovers servers on-demand and writes Python (Starlark) to orchestrate tools in a sandbox
- Multiple MCP tool calls happen inside a single `executeToolCode` call — 50%+ token reduction
- Measures Code Mode efficiency, dynamic discovery overhead, sandbox execution cost

### 3. Multi-Agent
- Supervisor decomposes task → parallel workers with isolated MCP connections
- Workers each connect only to their needed servers (dev→filesystem+git, docs→fetch+memory, data→sqlite, planning→sequential-thinking)
- Measures inter-agent token duplication, parallel execution, domain isolation

## Workflows

1. **File Operations** — list dirs, read files, git log
2. **Web Research** — fetch URL, save to memory
3. **Data Analysis** — SQL queries, user orders, aggregation
4. **Reasoning Task** — step-by-step problem solving
5. **Multi-Step Task** — file create + DB query + calculation

## Sandbox Test Data

Located in `sandbox/`:

- `sandbox/repo/` — Git repo with README.md, hello.py, src/foo.py
- `sandbox/data/test.db` — SQLite with `users` (2 rows) and `orders` (3 rows)
- `sandbox/test.txt` — Sample text file for filesystem ops
- `sandbox/data/notes.txt` — Another text file

## Metrics Tracked

- **Input/Output/Reasoning/Total Tokens** — LLM token consumption from API responses
- **Tool Schema Tokens** — bytes of tool schemas injected into system prompt
- **Orchestration Prompt Tokens** — architecture-specific instruction overhead
- **Router Tokens** — tokens consumed by semantic routing (0 for Bifrost — Code Mode discovery is in normal turns)
- **Inter-Agent Tokens** — serialized worker results passed between agents (multi-agent only)
- **Tools Exposed / Used / Truncated** — filtering and skip behaviour
- **Real Tool Calls** — count of actual MCP tool invocations
- **Agent Hops** — sequential LLM calls per workflow
- **MCP Connections / Activated** — subprocess spawning efficiency
- **Latency** — end-to-end wall clock in ms

## Run

```bash
# From repo root
source .venv/bin/activate
cd experiment2_mcp_benchmark
python run_all.py
```

```bash
# Or one-liner from repo root
printf "L\n" | .venv/bin/python experiment2_mcp_benchmark/run_all.py
```

Prompts for provider: **L** (LM Studio local, qwen/qwen3-8b) or **G** (Groq, llama-3.3-70b-versatile).
Output: rich tables per architecture + final comparison, `results/comparison_results.json`, `reports/report_*.html`.
