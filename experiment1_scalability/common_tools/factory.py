import copy

DOMAIN_TYPES = [
    "code_search",
    "log_analysis",
    "deployment",
    "monitoring",
    "database",
    "networking",
    "security",
    "documentation",
    "project_mgmt",
    "notifications",
]

DOMAIN_TOOL_TEMPLATES = {
    "code_search": [
        {
            "name": "search_code",
            "description": "Search the codebase for a function, class, or pattern matching a query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "language": {"type": "string", "description": "Language filter", "enum": ["python", "go", "js", "java", "ruby"]},
                },
                "required": ["query"],
            },
            "mock_response": {"matches": [{"file": "src/handler.go", "line": 42, "match": "func ProcessPayment()"}]},
        },
        {
            "name": "read_file",
            "description": "Read a source file from the repository by path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "Start line"},
                    "line_count": {"type": "integer", "description": "Number of lines"},
                },
                "required": ["path"],
            },
            "mock_response": {"path": "src/handler.go", "content": "package main\n\nfunc main() {\n\tprintln(\"hello\")\n}\n"},
        },
        {
            "name": "find_usage",
            "description": "Find all usages of a symbol across the codebase.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name to find"},
                    "scope": {"type": "string", "description": "Search scope (repo, module, file)"},
                },
                "required": ["symbol", "scope"],
            },
            "mock_response": {"symbol": "ProcessPayment", "usages": [{"file": "src/handler.go", "line": 10}, {"file": "src/router.go", "line": 25}]},
        },
    ],
    "log_analysis": [
        {
            "name": "tail_logs",
            "description": "Tail recent log entries for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "lines": {"type": "integer", "description": "Number of lines"},
                    "pattern": {"type": "string", "description": "Optional pattern filter"},
                },
                "required": ["service"],
            },
            "mock_response": {"entries": [{"time": "10:00:00", "level": "INFO", "msg": "started"}]},
        },
        {
            "name": "grep_errors",
            "description": "Search logs for error patterns for a given service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Error pattern"},
                    "service": {"type": "string", "description": "Service name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["pattern", "service"],
            },
            "mock_response": {"matches": [{"time": "10:00:00", "level": "ERROR", "msg": "connection refused"}]},
        },
        {
            "name": "log_summary",
            "description": "Get a summary of log activity for a service over a time range.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["service", "time_range"],
            },
            "mock_response": {"service": "payment-api", "total_lines": 5000, "errors": 23, "warnings": 45},
        },
    ],
    "deployment": [
        {
            "name": "list_deployments",
            "description": "List recent deployments for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["service"],
            },
            "mock_response": {"deployments": [{"version": "v3.2.1", "status": "success", "time": "2025-05-21T10:00:00Z"}]},
        },
        {
            "name": "rollback_status",
            "description": "Check if a rollback is in progress for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
            "mock_response": {"service": "payment-api", "rollback_in_progress": False, "current_version": "v3.2.1"},
        },
        {
            "name": "release_notes",
            "description": "Get release notes for a specific version of a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "version": {"type": "string", "description": "Version tag"},
                },
                "required": ["service", "version"],
            },
            "mock_response": {"service": "payment-api", "version": "v3.2.1", "changes": ["Fixed payment timeout", "Updated DB driver"]},
        },
    ],
    "monitoring": [
        {
            "name": "cpu_metrics",
            "description": "Get CPU usage metrics for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["service", "time_range"],
            },
            "mock_response": {"service": "payment-api", "average": 57.0, "peak": 92.0},
        },
        {
            "name": "memory_metrics",
            "description": "Get memory usage metrics for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["service", "time_range"],
            },
            "mock_response": {"service": "payment-api", "average_mb": 512, "peak_mb": 1024},
        },
        {
            "name": "latency_check",
            "description": "Check API latency percentiles for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["service", "time_range"],
            },
            "mock_response": {"service": "payment-api", "p50_ms": 120, "p95_ms": 450, "p99_ms": 1200},
        },
    ],
    "database": [
        {
            "name": "db_connections",
            "description": "Check active database connections for a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                },
                "required": ["service"],
            },
            "mock_response": {"service": "payment-api", "active_connections": 42, "max_connections": 100, "pool_usage_pct": 42.0},
        },
        {
            "name": "query_performance",
            "description": "Get slow query performance data for a database.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "database": {"type": "string", "description": "Database name"},
                    "threshold_ms": {"type": "integer", "description": "Slow query threshold in ms"},
                },
                "required": ["database", "threshold_ms"],
            },
            "mock_response": {"database": "payment-db", "slow_queries": [{"query": "SELECT * FROM transactions", "avg_ms": 2500}]},
        },
        {
            "name": "schema_check",
            "description": "Check the database schema version and migration status.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "database": {"type": "string", "description": "Database name"},
                },
                "required": ["database"],
            },
            "mock_response": {"database": "payment-db", "schema_version": 42, "pending_migrations": 0, "status": "up_to_date"},
        },
    ],
    "networking": [
        {
            "name": "check_routes",
            "description": "Check network routing table for a service mesh.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source service"},
                    "destination": {"type": "string", "description": "Destination service"},
                },
                "required": ["source", "destination"],
            },
            "mock_response": {"source": "payment-api", "destination": "payment-db", "latency_ms": 5, "healthy": True},
        },
        {
            "name": "dns_lookup",
            "description": "Perform a DNS lookup for a service endpoint.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string", "description": "Hostname to resolve"},
                },
                "required": ["hostname"],
            },
            "mock_response": {"hostname": "payment-api.internal", "resolved_ip": "10.0.1.42", "ttl": 300},
        },
        {
            "name": "bandwidth_check",
            "description": "Check bandwidth usage between services.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source service"},
                    "destination": {"type": "string", "description": "Destination service"},
                },
                "required": ["source", "destination"],
            },
            "mock_response": {"source": "payment-api", "destination": "payment-db", "mbps": 150, "peak_mbps": 800},
        },
    ],
    "security": [
        {
            "name": "scan_vulns",
            "description": "Run a vulnerability scan on a service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service to scan"},
                },
                "required": ["service"],
            },
            "mock_response": {"service": "payment-api", "vulnerabilities": [{"severity": "low", "count": 3}]},
        },
        {
            "name": "check_access",
            "description": "Check access control policies for a resource.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string", "description": "Resource name"},
                    "principal": {"type": "string", "description": "User or role"},
                },
                "required": ["resource", "principal"],
            },
            "mock_response": {"resource": "payment-db", "principal": "payment-api-sa", "access": "granted"},
        },
        {
            "name": "audit_logs",
            "description": "Retrieve audit log entries for a service or resource.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string", "description": "Resource name"},
                    "time_range": {"type": "string", "description": "Time range"},
                },
                "required": ["resource", "time_range"],
            },
            "mock_response": {"entries": [{"time": "10:00:00", "action": "deploy", "actor": "ci-bot", "status": "approved"}]},
        },
    ],
    "documentation": [
        {
            "name": "search_docs",
            "description": "Search internal documentation for a topic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Optional category"},
                },
                "required": ["query"],
            },
            "mock_response": {"results": [{"title": "Payment API Guide", "snippet": "How to use the payment API..."}]},
        },
        {
            "name": "read_doc",
            "description": "Read a documentation page by its ID or path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID or path"},
                },
                "required": ["doc_id"],
            },
            "mock_response": {"doc_id": "payment-api-guide", "content": "# Payment API Guide\n\n## Overview\nThe payment API handles..."},
        },
        {
            "name": "list_docs",
            "description": "List available documentation for a category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Documentation category"},
                },
                "required": ["category"],
            },
            "mock_response": {"category": "api", "docs": ["payment-api-guide", "user-service-guide", "deployment-manual"]},
        },
    ],
    "project_mgmt": [
        {
            "name": "create_ticket",
            "description": "Create a new ticket in the project management system.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Ticket title"},
                    "description": {"type": "string", "description": "Ticket description"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["title", "description", "priority"],
            },
            "mock_response": {"ticket_id": "INC-0042", "status": "created"},
        },
        {
            "name": "update_ticket",
            "description": "Update an existing ticket with new information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "Ticket ID"},
                    "comment": {"type": "string", "description": "Comment to add"},
                },
                "required": ["ticket_id", "comment"],
            },
            "mock_response": {"ticket_id": "INC-0042", "status": "updated", "comment_added": True},
        },
        {
            "name": "list_tickets",
            "description": "List tickets matching filters.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["open", "in_progress", "resolved"], "description": "Status filter"},
                    "assignee": {"type": "string", "description": "Assignee filter"},
                },
                "required": [],
            },
            "mock_response": {"tickets": [{"id": "INC-0042", "title": "Payment API incident", "status": "open"}]},
        },
    ],
    "notifications": [
        {
            "name": "send_alert",
            "description": "Send an alert to a notification channel.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Target channel"},
                    "message": {"type": "string", "description": "Alert message"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                },
                "required": ["channel", "message", "severity"],
            },
            "mock_response": {"status": "sent", "channel": "#incidents"},
        },
        {
            "name": "post_update",
            "description": "Post a status update to the status page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "status": {"type": "string", "enum": ["operational", "degraded", "down", "maintenance"]},
                    "message": {"type": "string", "description": "Update message"},
                },
                "required": ["service", "status", "message"],
            },
            "mock_response": {"status": "updated", "service": "payment-api", "current_status": "degraded"},
        },
        {
            "name": "schedule_notify",
            "description": "Schedule a notification for a future time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Target channel"},
                    "message": {"type": "string", "description": "Notification message"},
                    "delay_minutes": {"type": "integer", "description": "Delay in minutes"},
                },
                "required": ["channel", "message", "delay_minutes"],
            },
            "mock_response": {"status": "scheduled", "channel": "#incidents", "deliver_at": "2025-05-21T11:00:00Z"},
        },
    ],
}


def generate_mcps(count: int) -> list[dict]:
    mcps_per_domain = count // len(DOMAIN_TYPES)
    assert count % len(DOMAIN_TYPES) == 0, f"Count {count} must be divisible by domain count {len(DOMAIN_TYPES)}"
    mcps = []
    for domain in DOMAIN_TYPES:
        template_tools = DOMAIN_TOOL_TEMPLATES[domain]
        for i in range(mcps_per_domain):
            suffix = f"{i+1:02d}"
            tools = []
            for tpl in template_tools:
                tool = copy.deepcopy(tpl)
                base_name = tool["name"]
                tool["name"] = f"{base_name}_{suffix}"
                tool["description"] = f"{tool['description']} (instance {suffix})"
                tools.append(tool)
            mcps.append({
                "mcp_name": f"{domain}_{suffix}",
                "domain": domain,
                "mcp_index": i,
                "domain_index": DOMAIN_TYPES.index(domain),
                "total_mcps": count,
                "tools": tools,
            })
    return mcps



