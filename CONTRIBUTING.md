# Contributing to MCP Architecture Review

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running with `qwen/qwen3-8b` at `http://localhost:1234/v1`, OR a Groq/OpenRouter API key
- For experiment 2: Node.js (for `npx`) and `uvx` (for `uvx`)

### Setup

```bash
# Clone the repository
git clone https://github.com/Charanraj-T/MCP-architecture-comparison.git
cd MCP-architecture-comparison

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy common/ experiment1_scalability/ experiment2_mcp_benchmark/
```

## Development Workflow

1. **Create a branch** from `main` for your changes
2. **Make your changes** following the coding standards below
3. **Write tests** for new functionality
4. **Run the test suite** to verify nothing is broken
5. **Submit a pull request** with a clear description

## Coding Standards

### Python Style

- Follow PEP 8 with 120-character line length
- Use `ruff` for linting: `ruff check .`
- Use type hints on all public functions
- Prefer `pathlib.Path` over `os.path`

### Naming Conventions

- `snake_case` for functions, variables, and modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Private methods prefixed with `_`

### Architecture Files

Each architecture (centralized, federated, intent_driven, mediator) has a `run.py` (experiment 1) or `orchestrator.py` (experiment 2) that implements the same interface:

```python
class ArchitectureOrchestrator:
    def __init__(self, client, registry, mcps, ...): ...
    async def run(self, workflow) -> WorkflowMetrics: ...
    async def close(self): ...
```

When adding a new architecture:
1. Create a new directory under `experiment1_scalability/` and `experiment2_mcp_benchmark/`
2. Implement the orchestrator interface
3. Register it in `run_all.py`'s `ARCHITECTURES` dict
4. Add it to the comparison tables

### Testing

- Write unit tests for utility functions (parser, factory, client)
- Use `pytest.mark.slow` for tests that hit real APIs
- Use `pytest.mark.integration` for tests requiring running services
- Mock LLM responses in unit tests

### Commits

- Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Keep commits focused — one logical change per commit
- Reference issues where applicable

## Architecture Overview

```
common/                          # Shared infrastructure
  client.py                      # LLM client with rate limiting + retry

experiment1_scalability/         # Experiment 1: Simulated scaling
  run_all.py                     # Main orchestrator + report generation
  centralized/run.py             # All tools in one prompt
  federated/run.py               # Router filters domains
  intent_driven/run.py           # Domain descriptions, not schemas
  mediator/run.py                # Plan-then-execute
  multiagent/run.py              # Supervisor + workers
  common_tools/                  # Shared: factory, parser, registry
  metrics/                       # Token tracking
  workflows/                     # Workflow definitions

experiment2_mcp_benchmark/       # Experiment 2: Real MCP servers
  run_all.py                     # Main orchestrator
  mcp_client.py                  # MCP stdio client
  centralized/orchestrator.py    # @1mcp/agent aggregator
  federated/orchestrator.py      # Bifrost Code Mode
  intent_driven/orchestrator.py  # Domain-based tool discovery
  mediator/orchestrator.py       # Plan-then-execute
  multiagent/orchestrator.py     # Supervisor + workers
  metrics/                       # Token tracking + traces
  sandbox/                       # Test data for MCP servers
```

## Research Methodology

When modifying benchmark logic, ensure:

1. **Fair comparison** — All architectures receive identical inputs (same MCPs, same workflows, same model)
2. **Reproducibility** — Results can be regenerated from saved traces
3. **Completeness** — All architectures are tested, not just the one being developed
4. **Traceability** — JSON outputs include run metadata (model, config, git hash)

## Reporting Issues

When reporting bugs, please include:

- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output (without API keys!)
- Which experiment and architecture was affected

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
