# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-27

### Added

- **Experiment 1: Scalability** — Simulated MCP scaling at 10/20/50/100 counts
  - Centralized MCP, Federated MCP, Intent-Driven, MCP Mediator, Multi-Agent
  - Per-workflow metrics: tokens, latency, tools used, agent hops
  - CTO executive summary with cross-model comparison
  - HTML report generation with rich terminal output

- **Experiment 2: Real MCP Benchmark** — Real npx/uvx MCP servers
  - Filesystem, Git, Fetch, Memory, SQLite, Sequential Thinking
  - Centralized (@1mcp/agent), Federated (Bifrost), Multi-Agent architectures
  - MCP stdio client with JSON-RPC 2.0 transport

- **Shared Infrastructure**
  - `common/client.py` — OpenAI-compatible client with rate limiting and retry
  - `common_tools/` — MCP factory, tool parser, tool registry
  - `metrics/` — Token tracking, event tracing

- **Enterprise Hardening**
  - `pyproject.toml` with pinned dependencies
  - MIT License
  - 43 unit tests (parser, factory, registry, client)
  - CI/CD with GitHub Actions (pytest + ruff + mypy)
  - `BaseOrchestrator` ABC eliminating ~70% code duplication
  - Security: getpass for API keys, URL validation, no env leaks
  - Pre-commit hooks configuration
  - Docker support for reproducible runs

- **Documentation**
  - Architecture comparison table
  - LLM provider setup guide
  - Contributing guidelines
  - Security policy
  - Product roadmap

### Fixed

- Executive summary fabricating "~2,500 tokens" — now derived from actual metrics
- Retry count mismatch ("3 retries" → "5 retries")
- `response.choices[0]` unguarded against empty responses
- `Ministral-3B` context limit (128K → 32K)
- `assert` replaced with `ValueError` in factory.py
- Experiment2 multiagent broken imports (nonexistent supervisor/workers)
- Missing `sys.exit` → graceful error handling in library code

### Security

- API keys use `getpass` for terminal input (no history leakage)
- API keys no longer written to `os.environ` at runtime
- Custom URL provider validates `http://` or `https://` schemes only
- `.env` file permissions restricted to owner-only (chmod 600)
- `.gitignore` extended with `.env.*`, `*.pem`, `*.key`, `secrets/`
