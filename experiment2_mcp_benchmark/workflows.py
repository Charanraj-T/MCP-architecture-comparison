WORKFLOWS = [
    {
        "name": "File Operations",
        "prompt": (
            "List the contents of the current directory, then read the file test.txt. "
            "After that, show me the git log in the repo directory."
        ),
    },
    {
        "name": "Web Research",
        "prompt": (
            "Fetch the content from https://example.com and summarize it. "
            "Then save the summary to memory with the key 'example_summary'."
        ),
    },
    {
        "name": "Reasoning Task",
        "prompt": (
            "Think through this problem step by step: "
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Walk through your reasoning carefully."
        ),
    },
    {
        "name": "Multi-Step Task",
        "prompt": (
            "First, create a file called 'analysis.txt' with the content 'Analysis complete'. "
            "Then query the database to get all product names. "
            "Finally, use sequential thinking to analyze the sales data: "
            "Widget costs $29.99 and Gadget costs $49.99. Total sales were 3 units. Calculate the total revenue."
        ),
    },
]
