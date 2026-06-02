# MCP Architecture Comparison — Product Roadmap

**Vision:** Transform this research prototype into a publication-ready, enterprise-grade benchmark framework for MCP architecture evaluation — citable in arXiv, usable by enterprises, maintainable by the open-source community.

**Current Maturity:** 4/10 → **Target:** 8/10

---

## Phase 1: Publication Blockers (Must Complete)

> These are hard blockers for arXiv citation and enterprise adoption.
> Estimated: 2-3 focused sessions.

### Security

- [x] **Rotate exposed Groq API key** — Live `gsk_...` key in `.env`. Rotate immediately via Groq console.
- [x] **`chmod 600 .env`** — Restrict file permissions on plaintext secrets.
- [x] **Add `.env.*`, `*.pem`, `*.key` to `.gitignore`** — Prevent future credential commits.
- [x] **Complete `.env.example`** — Document all env vars used in `client.py` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_INJECT_NO_THINK`, Azure vars).

### Reproducibility

- [x] **Add `pyproject.toml`** — Pin all 8 dependencies to exact versions. Add `[project.optional-dependencies] dev = ["pytest>=7.0", "ruff>=0.4", "mypy>=1.0"]`.
- [x] **Add run metadata envelope to JSON output** — Wrap `scalability_results.json` in `{"schema_version": "1.0", "run_id": "<uuid>", "timestamp": "<iso>", "model": "<name>", "mcp_counts": [...], "git_hash": "<sha>", "results": [...]}`.
- [x] **Fix executive summary fabricating data** — Replace hardcoded `"~2,500 tokens"` with actual computed average from metrics.

### Correctness

- [x] **Fix retry count mismatch** — `client.py:229` says "3 retries" but loop runs 5 times.
- [x] **Guard `response.choices[0]`** — Add bounds check to prevent `IndexError` on empty responses.
- [x] **Fix `Ministral-3B` context limit** — Currently 128K, actual is 32K (`client.py:521`).
- [x] **Replace `assert` with `ValueError`** — `factory.py:432` uses `assert` for input validation (stripped with `python -O`).

### Resilience

- [x] **Add try/except + timeout around `orch.run(wf)`** — Single failure currently crashes everything, losing all results. Wrap in `asyncio.wait_for(timeout=300)` + graceful degradation.
- [x] **Add intermediate result saving** — Flush results to disk after each MCP count. Crash no longer loses everything.

### Legal & Community

- [x] **Add `LICENSE`** (MIT) — Required for citation and open-source usage.
- [x] **Add `SECURITY.md`** — Responsible disclosure process for credential/dependency issues.
- [x] **Add `CONTRIBUTING.md`** — Setup, coding standards, PR process.

### Testing

- [x] **Add `tests/` directory** with pytest config
- [x] **Unit tests for `common_tools/parser.py`** — Test `parse_tool_calls()` with valid JSON, malformed JSON, multiple tool calls, edge cases.
- [x] **Unit tests for `common_tools/factory.py`** — Test `generate_mcps()` at counts 10, 20, 50. Verify domain distribution.
- [x] **Unit tests for `common/client.py`** — Mock LLM responses, test retry logic, rate limiting, `get_token_budget()`.
- [ ] **Smoke test for `run_all.py`** — Verify it runs end-to-end with mocked LLM, produces valid JSON + HTML.

---

## Phase 2: Enterprise Hardening (Recommended)

> Improves maintainability, reduces duplication, enables CI/CD.
> Estimated: 3-4 focused sessions.

### Code Quality

- [x] **Extract shared architecture base class** — Create `common_tools/architecture_base.py` with `_count_tokens`, `_parse_tool_calls`, `_estimate_next_request_tokens`, `_get_tools_for_domains`, `_chat_with_timeout`, usage accumulation. Eliminate 10x duplication.
- [ ] **Extract `_avg_metric()` helper** — Replace 6 duplicated aggregation blocks in `run_all.py` with single function.
- [ ] **Centralize MCP server definitions** — Single `MCP_DEFINITIONS` in `mcp_client.py`, import everywhere (eliminates 3x duplication).
- [x] **Remove dead code** — `LMStudioClient` alias, `estimate_tokens_per_model()`, unused `MultiAgentOrchestrator` import, unused `TokenTracker` class, `import json` in `client.py`.

### Security Hardening

- [x] **Use `getpass` for API key input** — Prevent terminal history leakage of Azure/Groq keys.
- [x] **Stop writing keys to `os.environ`** — Store in instance variables, pass to client constructor.
- [x] **Add URL validation for custom provider** — Reject `file://`, non-HTTP schemes.
- [ ] **Replace `sys.exit(1)` with exceptions** in library code — Allow callers to handle errors gracefully.
- [ ] **Add input validation to `select_provider()`** — Reject invalid choices, validate MCP counts (max 200), validate model slugs.

### Observability

- [ ] **Add structured logging** — Replace `print()` with `logging` module throughout. Add `experiment_id` to log context.
- [ ] **Add experiment metadata to traces** — Each event trace JSON should include run_id, model, mcp_count, architecture.
- [ ] **Add failure diagnostics** — Log which workflow failed, why, and at what step.

### Documentation

- [x] **Rewrite root README** — Add: architecture diagram (mermaid), quickstart guide, env var reference, example output, contributing link.
- [ ] **Add methodology section** — Formal benchmark design: controls, variables, threats to validity, statistical approach.
- [ ] **Add architecture decision records (ADRs)** — Document why each architecture pattern was chosen for comparison.

### CI/CD

- [x] **Add `.github/workflows/ci.yml`** — pytest + ruff + mypy on push/PR.
- [ ] **Add `.github/workflows/publish.yml`** — Tag-triggered PyPI publish (future).
- [ ] **Add pre-commit config** — `ruff check`, `mypy`, trailing-whitespace.

### Configuration

- [x] **Fix experiment2 broken imports** — `multiagent/orchestrator.py` imported from nonexistent `multiagent.supervisor` and `multiagent.workers.*`. Self-contained implementation with inline prompts.
- [ ] **Make `MODEL_PRICING` configurable** — Move from hardcoded dict to `config.json` or env vars.
- [ ] **Replace `input()` with CLI args** — `argparse` for non-interactive execution (CI/CD friendly).
- [ ] **Add experiment config files** — YAML/JSON defining MCP counts, models, architectures per run.

---

## Phase 3: Research Publication Quality (Future)

> Makes this a first-class research artifact.
> Estimated: 1-2 focused sessions.

### Statistical Rigor

- [ ] **Add confidence intervals** — Bootstrap or t-interval on token counts across runs.
- [ ] **Add effect size reporting** — Cohen's d or Cliff's delta for architecture comparisons.
- [ ] **Add significance testing** — Paired t-test or Wilcoxon signed-rank for architecture rankings.
- [ ] **Add multiple run support** — Run each config N times, report mean ± std.

### Reproducibility

- [ ] **Add seed/determinism controls** — Pin random seeds, document non-determinism sources.
- [ ] **Add Docker support** — `Dockerfile` + `docker-compose.yml` for fully reproducible environment.
- [ ] **Add `tox.ini`** — Multi-Python testing (3.10, 3.11, 3.12).
- [ ] **Pin MCP server versions** — `npx -y @modelcontextprotocol/server-filesystem@1.0.0` instead of latest.

### Analysis

- [ ] **Add benchmark comparison dashboard** — Static HTML from results JSON with interactive charts.
- [ ] **Add trend analysis** — `print_cross_count_trend()` improvements with visualization.
- [ ] **Add cost projection modeling** — Extrapolate costs to 100/500/1000 MCP counts.

### Packaging

- [ ] **Add `py.typed` marker** — Enable `mypy --strict` for full type checking.
- [ ] **Add type annotations to all public APIs** — Full `mypy` compliance.
- [ ] **Add pre-commit hooks** — `ruff format`, `ruff check --fix`, `mypy`.
- [ ] **Add `CHANGELOG.md`** — Semantic versioning, document all changes.
- [ ] **Add `pyproject.toml` `[project.scripts]`** — CLI entry point: `mcp-bench run`, `mcp-bench report`.

### Scaling

- [ ] **Add async parallel execution** — Run multiple architectures concurrently where possible.
- [ ] **Add result caching** — Skip re-running identical configurations.
- [ ] **Add incremental saves** — Write results after each architecture×model×mcp_count combination.

---

## Success Criteria

### For arXiv Publication
- [ ] All Phase 1 items complete
- [ ] LICENSE present (MIT or Apache-2.0)
- [ ] Run metadata in all JSON outputs
- [ ] No fabricated narrative text in reports
- [ ] Test coverage > 50% on core modules
- [ ] methodology.md describes experimental design formally

### For Enterprise Adoption
- [ ] All Phase 1 + Phase 2 items complete
- [ ] CI/CD passing on all PRs
- [ ] `pip install -e .` works
- [ ] Non-interactive mode via CLI args
- [ ] Structured logging with experiment context
- [ ] Docker one-click reproduction

### For Open Source Community
- [ ] LICENSE + CONTRIBUTING + SECURITY.md present
- [ ] README enables self-service onboarding
- [ ] Architecture docs explain design decisions
- [ ] Example workflows demonstrate usage
- [ ] Issue templates for bug reports and feature requests

---

## Tracking

| Phase | Items | Status |
|-------|------:|--------|
| Phase 1 | 16 | ✅ Complete |
| Phase 2 | 20 | 🔄 In Progress |
| Phase 3 | 16 | ⬜ Not Started |
| **Total** | **52** | |

Last updated: May 27, 2026
