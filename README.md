# MCP Architecture Comparison POC

Three experiments comparing MCP orchestration architectures:

- [Experiment 1: Token Efficiency](experiment1_token_efficiency/README.md) — simulated MCPs, measures prompt structure efficiency across 3 architectures
- [Experiment 2: Scalability](experiment2_scalability/README.md) — simulated MCPs at scale (10/20/50/100 MCPs), measures schema budget fit
- [Experiment 3: Real MCP Orchestration](experiment3_real_mcp/README.md) — real MCP servers with stdio transport, real tool execution, real agent isolation

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

LM Studio with `qwen/qwen3-8b` must be running at `http://localhost:1234/v1`.
