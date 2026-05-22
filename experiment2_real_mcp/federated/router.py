import json
import re
from lmstudio_client import LMStudioClient
from mcp_client import MCP_DEFINITIONS


async def classify_mcps(client: LMStudioClient, user_prompt: str) -> tuple[list[str], int]:
    mcp_descriptions = "\n".join(
        f"- {key}: {defn['description']}"
        for key, defn in MCP_DEFINITIONS.items()
    )
    router_prompt = (
        "You are an MCP routing engine. Classify the following request into relevant MCP domains.\n"
        "Only select MCPs that are absolutely necessary to fulfill the request.\n\n"
        f"Available MCPs:\n{mcp_descriptions}\n\n"
        'Respond with ONLY a JSON object:\n{"mcps": ["dev", "docs", "data", "reasoning"]}\n\n'
        f"Request: {user_prompt}"
    )

    response = await client.chat(
        [{"role": "user", "content": router_prompt}],
        temperature=0.1, max_tokens=256,
    )

    content = response.content or ""
    try:
        match = re.search(r'\{.*?"mcps".*?\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            mcps = data.get("mcps", [])
            valid = [m for m in mcps if m in MCP_DEFINITIONS]
            if valid:
                return valid, response.usage.total_tokens
    except (json.JSONDecodeError, KeyError):
        pass

    return list(MCP_DEFINITIONS.keys()), response.usage.total_tokens
