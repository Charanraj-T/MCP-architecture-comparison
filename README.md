# MCP Architecture Comparison POC

Two experiments comparing MCP orchestration architectures:

- [Experiment 1: Scalability](experiment1_scalability/README.md) — simulated MCPs at scale (10/20/50/100), measures schema budget fit and skip behavior
- [Experiment 2: Real MCP Benchmark](experiment2_mcp_benchmark/README.md) — real npx/uvx MCP servers (filesystem, git, fetch, memory, SQLite, sequential-thinking) with Centralized (@1mcp/agent), Federated (smartmcp FAISS router), and Multi-Agent architectures

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install smartmcp-router  # for federated architecture
```

LM Studio with `qwen/qwen3-8b` must be running at `http://localhost:1234/v1`.
