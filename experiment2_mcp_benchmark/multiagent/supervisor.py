DECOMPOSE_PROMPT = (
    "You are a supervisor AI. Break the following task into subtasks "
    "for specialized worker agents. Available agents:\n"
    "- dev_agent: File system ops, git ops, create/read/write files\n"
    "- docs_agent: URL fetching, memory storage/recall\n"
    "- data_agent: SQLite database queries, schema inspection\n"
    "- planning_agent: Structured reasoning, task decomposition, summarization\n\n"
    "Output a JSON array of subtasks:\n"
    '[{"agent": "dev_agent", "task": "Describe the task for this agent"}, ...]\n\n'
    "Only include agents that are actually needed. Each task should be self-contained.\n\n"
    "Task:\n"
)

COMPILE_PROMPT = (
    "You are a supervisor AI. Below are the results from your worker agents. "
    "Compile them into a coherent final answer.\n\n"
    "Worker Results:\n"
)
