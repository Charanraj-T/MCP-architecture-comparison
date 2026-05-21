import json
import datetime


class MessageBus:
    def __init__(self):
        self._messages: list[dict] = []

    def send(self, sender: str, receiver: str, msg_type: str, payload: dict):
        self._messages.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "from": sender,
            "to": receiver,
            "type": msg_type,
            "payload": payload,
        })

    def get_agent_results(self) -> list[dict]:
        return [
            m for m in self._messages
            if m["type"] == "result"
        ]

    def format_compile_input(self) -> str:
        results = self.get_agent_results()
        parts = []
        for r in results:
            agent = r["from"]
            answer = r["payload"].get("answer", "")
            parts.append(f"<{agent}>\n{answer}\n</{agent}>")
        return "\n\n".join(parts)

    def get_inter_agent_text(self) -> str:
        return json.dumps(self._messages, indent=2)
