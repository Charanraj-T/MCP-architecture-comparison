# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Scope

This project handles:

- **API keys** for LLM providers (Groq, OpenRouter, Azure, GitHub, Cloudflare)
- **LLM interactions** via OpenAI-compatible APIs
- **Local file system access** via MCP servers (experiment 2)
- **Subprocess execution** for MCP server processes

## Security Practices

### Credential Handling

- `.env` files are gitignored and must never be committed
- API keys should be set via environment variables, not hardcoded
- Use `chmod 600 .env` to restrict file permissions
- Never log or print API keys

### Dependencies

- Dependencies are pinned in `pyproject.toml` for reproducibility
- Run `pip-audit` periodically to check for known vulnerabilities
- The `smartmcp-router` package should be verified for trustworthiness

### MCP Servers (Experiment 2)

- MCP servers run as subprocesses with stdio transport
- Servers are auto-downloaded via `npx -y` / `uvx` — verify server packages before running
- Sandbox data is isolated in `experiment2_mcp_benchmark/sandbox/`
- `stderr` from MCP servers is captured for debugging

### Runtime

- LLM interactions use HTTPS where possible
- Timeout limits are enforced on API calls (default 120s)
- Rate limiting is applied to prevent API abuse
- Input validation is performed on user-provided configuration

## Known Limitations

- This is a research benchmark, not a production service
- The `select_provider()` function uses `input()` for interactive configuration — not suitable for untrusted environments
- MCP server subprocesses inherit the parent process environment — ensure no sensitive env vars are exposed
- Local LLM servers (LM Studio, Ollama) run on localhost with no authentication

## Updates

Security fixes will be released as patch versions (e.g., 0.1.1 → 0.1.2).

Check the [CHANGELOG](CHANGELOG.md) for security-related updates.
