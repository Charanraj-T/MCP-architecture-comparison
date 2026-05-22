import json
import datetime
from pathlib import Path


class JSONLogger:
    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, workflow: str, architecture: str, event: str, details: dict = None):
        now = datetime.datetime.now()
        entry = {
            "timestamp": now.isoformat(),
            "workflow": workflow,
            "architecture": architecture,
            "event": event,
            "details": details or {},
        }
        ts = now.strftime("%Y%m%d_%H%M%S_%f")
        sanitized_arch = architecture.replace(" ", "_")
        sanitized_wf = workflow.replace(" ", "_")
        path = self.trace_dir / f"events_{sanitized_arch}_{sanitized_wf}_{ts}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2, default=str)
