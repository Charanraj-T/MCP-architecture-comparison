DOCS_TOOLS = [
    {
        "name": "search_docs",
        "description": "Search internal documentation, API references, and knowledge base articles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for documentation",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to narrow search (api, deployment, architecture, troubleshooting)",
                },
            },
            "required": ["query"],
        },
        "mock_response": {
            "results": [
                {
                    "title": "Payment API — Deployment Guide",
                    "url": "docs/deployment/payment-api",
                    "snippet": "The payment-api service requires database migrations before deployment. Ensure DB_URL env is set.",
                },
                {
                    "title": "Incident Response — Payment Timeouts",
                    "url": "docs/runbooks/payment-timeout",
                    "snippet": "If payment-api pods are in CrashLoopBackOff, check database connectivity and write permissions.",
                },
                {
                    "title": "Architecture — Payment Processing Flow",
                    "url": "docs/architecture/payment-flow",
                    "snippet": "Payment flow: API Gateway → Payment Service → Ledger → Notification.",
                },
            ],
        },
    },
    {
        "name": "summarize_data",
        "description": "Summarize a body of text or structured data into a concise executive summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The text or data to summarize",
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum words in the summary",
                },
            },
            "required": ["data"],
        },
        "mock_response": {
            "summary": "Analysis of payment-api incident: CPU spiked to 92% at 10:20 UTC coinciding with deployment v2.4.1 rollout. Pod entered CrashLoopBackOff due to health check timeout. Root cause appears to be database connection pool exhaustion after schema migration.",
            "original_length": 450,
            "summary_length": 120,
        },
    },
]
