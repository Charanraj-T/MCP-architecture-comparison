WORKFLOWS = [
    {
        "name": "File Operations",
        "description": "List directory contents, read a text file, and inspect git history.",
        "expected_mcps": ["filesystem", "git"],
        "prompt": (
            "List the contents of the current directory, then read the file test.txt. "
            "After that, show me the git log in the repo directory."
        ),
    },
    {
        "name": "Web Research",
        "description": "Fetch content from a URL and persist a summary to the memory store.",
        "expected_mcps": ["fetch", "memory"],
        "prompt": (
            "Fetch the content from https://example.com and summarize it. "
            "Then save the summary to memory with the key 'example_summary'."
        ),
    },
    {
        "name": "Reasoning Task",
        "description": "Solve a classic word puzzle using structured step-by-step reasoning.",
        "expected_mcps": ["sequential-thinking"],
        "prompt": (
            "Think through this problem step by step: "
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Walk through your reasoning carefully."
        ),
    },
    {
        "name": "Data Analysis",
        "description": "Query the database for all users and their orders with total spend per user.",
        "expected_mcps": ["SQLite"],
        "prompt": (
            "Show me all users and their orders from the database. "
            "Include the total amount spent per user."
        ),
    },
    {
        "name": "Multi-Step Task",
        "description": "Create a file, query a database, and compute revenue using reasoning.",
        "expected_mcps": ["filesystem", "SQLite", "sequential-thinking"],
        "prompt": (
            "First, create a file called 'analysis.txt' with the content 'Analysis complete'. "
            "Then query the database to get all product names. "
            "Finally, use sequential thinking to analyze the sales data: "
            "Widget costs $29.99 and Gadget costs $49.99. Total sales were 3 units. Calculate the total revenue."
        ),
    },
]
