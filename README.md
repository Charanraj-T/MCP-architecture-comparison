# MCP Architecture Comparison POC

Two experiments comparing MCP orchestration architectures:

- [Experiment 1: Scalability](experiment1_scalability/README.md) — simulated MCPs at scale (10/20/50/100), measures schema budget fit and skip behavior
- [Experiment 2: Real MCP Orchestration](experiment2_real_mcp/README.md) — real MCP servers with stdio transport, real tool execution, real agent isolation

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

LM Studio with `qwen/qwen3-8b` must be running at `http://localhost:1234/v1`.
