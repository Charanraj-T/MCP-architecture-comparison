WORKFLOWS = [
    {
        "name": "Incident Investigation",
        "prompt": (
            "Investigate an incident affecting the payment-api service.\n"
            "1. Query the database for recent incidents and their status\n"
            "2. Check deployment history for payment-api\n"
            "3. Search for the payment processing source code and read the relevant file\n"
            "4. Check git log for recent changes\n"
            "5. Analyze CPU and memory metrics\n"
            "6. Fetch documentation from a relevant URL\n"
            "7. Summarize everything into a coherent incident report"
        ),
        "expected_mcps": ["dev", "data", "docs", "reasoning"],
    },
    {
        "name": "Deployment Analysis",
        "prompt": (
            "Analyze the recent deployment history for payment-api.\n"
            "1. Query the database for recent deployments\n"
            "2. Check git log for commits related to payment-api\n"
            "3. Read the current source code for payment processing\n"
            "4. Check git diff between the last successful and failed deployment\n"
            "5. Summarize what went wrong and what changed"
        ),
        "expected_mcps": ["dev", "data", "reasoning"],
    },
    {
        "name": "Full System Audit",
        "prompt": (
            "Perform a comprehensive system audit.\n"
            "1. List all database tables and describe their schemas\n"
            "2. Query all incidents, deployments, and metrics from the database\n"
            "3. Check the current git status and recent commits\n"
            "4. Search the codebase for critical patterns\n"
            "5. Fetch documentation on incident response\n"
            "6. Store key findings in memory\n"
            "7. Create a structured reasoning plan for remediation\n"
            "8. Compile a complete audit report"
        ),
        "expected_mcps": ["dev", "data", "docs", "reasoning"],
    },
]
