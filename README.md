# MCP Architecture Comparison

A benchmark framework comparing Model Context Protocol (MCP) orchestration architectures for scalability, cost, and tool-use efficiency.

## What This Does

Evaluates how different MCP architecture patterns perform as the number of tool-providing servers grows:

| Architecture | Pattern | Key Trade-off |
|---|---|---|
| **Centralized MCP** | All tools in one system prompt | Simple but schema cost scales linearly |
| **Federated MCP** | Router filters relevant domains | Lower per-turn cost but routing can degrade |
| **Intent-Driven** | Domain descriptions only, no schemas | Flattest cost curve but requires structured JSON output |
| **MCP Mediator** | Plan-then-execute with zero-cost execution | Fixed 2-hop cost but schemas loaded once for planning |
| **Multi-Agent** | Supervisor + domain workers | Isolates schemas but multiplies inter-agent tokens |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Charanraj-T/MCP-architecture-comparison.git
cd MCP-architecture-comparison

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure (choose one)
cp .env.example .env
# Edit .env with your API key, OR use LM Studio locally

# Run tests
pytest

# Run Experiment 1 (simulated scaling)
cd experiment1_scalability
python run_all.py

# Run Experiment 2 (real MCP servers)
cd experiment2_mcp_benchmark
python run_all.py
```

## Experiments

### Experiment 1: Scalability

Simulates MCPs at scale (10, 20, 50, 100) to measure schema budget fit and skip behavior.

- MCPs are generated via a factory across 10 domains (code, logs, deployment, monitoring, database, networking, security, documentation, project management, notifications)
- Each domain has 3 tool templates; at count=N, each domain gets N/10 instances
- Architectures that exceed the 3,500 schema token budget are skipped

### Experiment 2: Real MCP Benchmark

Tests against real npx/uvx MCP servers (filesystem, git, fetch, memory, SQLite, sequential-thinking) running as subprocesses with stdio transport.

## Repository Structure

```
common/                          # Shared LLM client infrastructure
experiment1_scalability/         # Experiment 1: Simulated scaling
  run_all.py                     # Main orchestrator + report generation
  centralized/                   # All tools in one prompt
  federated/                     # Router filters domains
  intent_driven/                 # Domain descriptions, not schemas
  mediator/                      # Plan-then-execute
  multiagent/                    # Supervisor + workers
  common_tools/                  # Factory, parser, registry
  metrics/                       # Token tracking
  workflows/                     # Workflow definitions
experiment2_mcp_benchmark/       # Experiment 2: Real MCP servers
  run_all.py                     # Main orchestrator
  mcp_client.py                  # MCP stdio client
  centralized/                   # @1mcp/agent aggregator
  federated/                     # Bifrost Code Mode
  intent_driven/                 # Domain-based tool discovery
  mediator/                      # Plan-then-execute
  multiagent/                    # Supervisor + workers
tests/                           # Unit tests
```

## LLM Provider Options

| Provider | Setup | Cost |
|---|---|---|
| **LM Studio** (local) | Run with `qwen/qwen3-8b` at `localhost:1234` | Free |
| **Ollama** (local) | `ollama serve` then select [O] | Free |
| **Groq** | Set `GROQ_API_KEY` in `.env` | Free tier |
| **OpenRouter** | Set `OPENROUTER_API_KEY` in `.env` | Free tier available |
| **Azure OpenAI** | Set `AZURE_OPENAI_*` vars in `.env` | Paid |
| **GitHub Models** | Set `GITHUB_TOKEN` in `.env` | Free |

## Output

- **Rich terminal tables** with per-MCP-count comparisons
- **HTML reports** in `reports/` with CTO executive summary
- **JSON results** in `results/scalability_results.json` with full run metadata
- **Event traces** in `traces/` with per-step LLM telemetry

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards, and PR process.

## Security

See [SECURITY.md](SECURITY.md) for credential handling and vulnerability reporting.

## License

MIT — see [LICENSE](LICENSE).
