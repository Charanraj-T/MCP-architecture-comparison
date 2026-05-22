import json
import datetime
from pathlib import Path

class JSONLogger:
    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, workflow: str, architecture: str, event: str, details: dict = None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "workflow": workflow,
            "architecture": architecture,
            "event": event,
            "details": details or {},
        }
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.trace_dir / f"events_{architecture}_{workflow}_{ts}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2, default=str)
