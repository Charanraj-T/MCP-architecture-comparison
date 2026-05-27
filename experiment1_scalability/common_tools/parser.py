import json
import re


def _extract_json_objects(text: str) -> list[dict]:
    """Extract all valid JSON objects from text using multiple strategies."""
    results = []

    # Strategy 1: TOOL_CALL: prefix
    i = 0
    while True:
        idx = text.find("TOOL_CALL:", i)
        if idx == -1:
            break
        json_start = text.find("{", idx)
        if json_start == -1:
            i = idx + 10
            continue
        depth = 0
        json_end = -1
        for j in range(json_start, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_end = j + 1
                    break
        if depth == 0 and json_end > json_start:
            raw = text[json_start:json_end]
            try:
                results.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        i = json_end if json_end > 0 else idx + 10

    # Strategy 2: ```json ... ``` code blocks
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                results.extend(obj)
            elif isinstance(obj, dict) and "name" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Standalone JSON objects ({"name": ..., "arguments": ...})
    for match in re.finditer(r'\{[^}]*(?:"name"|"tool")[^}]*\}', text, re.DOTALL):
        raw = match.group()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                if "tool" in obj and "name" not in obj:
                    obj["name"] = obj.pop("tool")
                results.append(obj)
        except json.JSONDecodeError:
            pass

    return results


def parse_tool_calls(text) -> list[dict]:
    """Parse tool calls from text. Ensures text is a string first."""
    # Defensive: ensure text is always a string
    if isinstance(text, list):
        text = "".join(str(item) for item in text)
    elif not isinstance(text, str):
        text = str(text)
    
    return _extract_json_objects(text)
