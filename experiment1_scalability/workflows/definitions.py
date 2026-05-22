WORKFLOWS = [
    {
        "name": "Code Investigation",
        "prompt": (
            "Search the codebase for the function 'ProcessPayment' and find all its usages. "
            "Read the main implementation file to understand the logic. "
            "Report what you find."
        ),
    },
    {
        "name": "Incident Investigation",
        "prompt": (
            "Investigate a production incident affecting the payment-api service: "
            "check pod status in the production namespace, search logs for errors, "
            "query CPU metrics for the last hour, list recent deployments, "
            "and create a high-priority incident ticket. Summarize all findings."
        ),
    },
    {
        "name": "Full System Audit",
        "prompt": (
            "Perform a comprehensive system health check. Cover all aspects: "
            "search code for recent changes to critical services, "
            "check all logs and deployments, "
            "query CPU, memory, and latency metrics, "
            "check database connections and schema, "
            "verify network routes and DNS, "
            "run a vulnerability scan and check access controls, "
            "search documentation for incident runbooks, "
            "create tickets for any issues found, "
            "and send alerts if problems are detected. "
            "Provide a complete audit report."
        ),
    },
]
