SUPERVISOR_DECOMPOSE_PROMPT = (
    "You are a supervisor AI. Break the following task into subtasks "
    "for specialized domain agents. Available agents:\n"
    "- dev_agent: Code search, file reading, log analysis\n"
    "- infra_agent: Kubernetes, deployments, CPU metrics\n"
    "- docs_agent: Documentation search, data summarization\n"
    "- pm_agent: Ticket creation, notifications\n\n"
    "Output a JSON array of subtasks:\n"
    '[{"agent": "dev_agent", "task": "..."}, ...]\n\n'
    "Only include agents that are actually needed.\n\n"
    "Task:\n"
)

SUPERVISOR_COMPILE_PROMPT = (
    "You are a supervisor AI. Below are the results from your worker agents. "
    "Compile them into a coherent final answer.\n\n"
    "Worker Results:\n"
)
