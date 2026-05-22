import json


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
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
                calls.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        i = json_end if json_end > 0 else idx + 10
    return calls
