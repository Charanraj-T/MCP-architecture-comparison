import json
import datetime
from pathlib import Path

class TraceCollector:
    def __init__(self):
        self.events: list[dict] = []

    def record(self, category: str, label: str, details: dict = None):
        self.events.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "category": category,
            "label": label,
            "details": details or {},
        })

    def flush(self, filepath: str = None) -> str:
        if filepath is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = f"traces/trace_{ts}.json"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)
        self.events.clear()
        return str(path)
