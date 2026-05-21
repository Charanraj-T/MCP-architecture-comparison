DOMAIN_MAP = {
    "dev": {
        "name": "Dev MCP",
        "tools": ["search_repo", "read_file", "grep_logs"],
    },
    "infra": {
        "name": "Infra MCP",
        "tools": ["kubernetes_status", "deployment_logs", "cpu_metrics"],
    },
    "docs": {
        "name": "Docs MCP",
        "tools": ["search_docs", "summarize_data"],
    },
    "pm": {
        "name": "PM MCP",
        "tools": ["create_ticket", "send_notification"],
    },
}


def all_tools_by_domain() -> dict[str, list[str]]:
    return {k: v["tools"] for k, v in DOMAIN_MAP.items()}


def domain_for_tool(tool_name: str) -> str | None:
    for domain, info in DOMAIN_MAP.items():
        if tool_name in info["tools"]:
            return domain
    return None
