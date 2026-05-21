DEV_TOOLS = [
    {
        "name": "search_repo",
        "description": "Search the repository for code matching a query. Searches file names, contents, and code patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g. function name, error message, class name)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to filter (e.g. '*.py', '*.ts')",
                },
            },
            "required": ["query"],
        },
        "mock_response": {
            "results": [
                {"file": "src/services/payment.py", "line": 42, "match": "def process_payment():"},
                {"file": "src/services/payment.py", "line": 85, "match": "class PaymentError(Exception):"},
                {"file": "tests/test_payment.py", "line": 15, "match": "def test_process_payment():"},
            ],
            "total_matches": 3,
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file in the repository. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file in the repository",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional starting line number (1-indexed)",
                },
                "line_count": {
                    "type": "integer",
                    "description": "Optional number of lines to read",
                },
            },
            "required": ["file_path"],
        },
        "mock_response": {
            "file": "src/services/payment.py",
            "content": "1: import os\n2: import json\n3: \n4: def process_payment(amount: float):\n5:     \"\"\"Process a payment transaction.\"\"\"\n6:     if amount <= 0:\n7:         raise ValueError(\"Amount must be positive\")\n8:     # FIXME: timeout not handled\n9:     response = api.call(amount)\n10:     return response.json()\n",
            "total_lines": 120,
        },
    },
    {
        "name": "grep_logs",
        "description": "Search application logs for patterns. Useful for finding errors, warnings, or specific events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for in logs (e.g. 'ERROR', 'timeout', 'payment')",
                },
                "service": {
                    "type": "string",
                    "description": "Optional service name to filter logs",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of recent log lines to search (default: 100)",
                },
            },
            "required": ["pattern"],
        },
        "mock_response": {
            "service": "payment-api",
            "matches": [
                {"timestamp": "2025-05-21T10:23:15Z", "level": "ERROR", "message": "Connection timeout after 30s"},
                {"timestamp": "2025-05-21T10:23:14Z", "level": "WARN", "message": "Retry attempt 2/3"},
                {"timestamp": "2025-05-21T10:23:12Z", "level": "ERROR", "message": "Failed to process payment: id=txn_9876"},
                {"timestamp": "2025-05-21T10:23:10Z", "level": "INFO", "message": "Deployment v2.4.1 rolled out"},
            ],
            "total_matches": 4,
        },
    },
]
