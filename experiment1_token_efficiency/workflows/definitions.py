WORKFLOWS = [
    {
        "name": "Incident Investigation",
        "prompt": (
            "We are investigating a production incident. The payment-api service is degraded. "
            "Perform the following steps:\n"
            "1. Search the repository for any recent changes to payment processing code\n"
            "2. Grep application logs for errors in the payment-api service\n"
            "3. Check Kubernetes status for the production namespace\n"
            "4. Look at recent deployment logs for payment-api\n"
            "5. Search documentation for payment-api runbooks\n"
            "6. Create an incident ticket with high priority\n"
            "7. Send a notification to the #incidents channel\n"
            "8. Summarize your findings\n\n"
            "Proceed step by step and use the available tools."
        ),
        "expected_steps": 8,
        "expected_tools": ["search_repo", "grep_logs", "kubernetes_status", "deployment_logs", "search_docs", "create_ticket", "send_notification", "summarize_data"],
    },
    {
        "name": "Root Cause Analysis",
        "prompt": (
            "We need to find the root cause of repeated payment failures. "
            "Perform the following steps:\n"
            "1. Search documentation for known payment timeout issues\n"
            "2. Read the payment processing source file to understand the timeout logic\n"
            "3. Grep logs for timeout-related errors in the last hour\n"
            "4. Summarize the probable root cause\n\n"
            "Use the available tools to investigate and provide a concise analysis."
        ),
        "expected_steps": 4,
        "expected_tools": ["search_docs", "read_file", "grep_logs", "summarize_data"],
    },
    {
        "name": "Performance Analysis",
        "prompt": (
            "The payment-api CPU has been spiking. We need an executive summary. "
            "Perform the following steps:\n"
            "1. Check CPU metrics for payment-api over the last hour\n"
            "2. Find deployments in the same timeframe\n"
            "3. Search the repo for code changes related to the latest deployment\n"
            "4. Search documentation for performance tuning guides\n"
            "5. Create a ticket for the performance investigation\n"
            "6. Prepare an executive summary of findings\n\n"
            "Use the available tools and provide a comprehensive report."
        ),
        "expected_steps": 6,
        "expected_tools": ["cpu_metrics", "deployment_logs", "search_repo", "search_docs", "create_ticket", "summarize_data"],
    },
]
