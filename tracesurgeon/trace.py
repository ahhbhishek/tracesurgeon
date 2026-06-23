import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    event_type: str          # "node_start" | "node_end" | "tool_start" | "tool_end" | "error"
    step_id: str
    parent_step_id: str | None
    node_name: str
    inputs: Any
    outputs: Any
    timestamp: str
    duration_ms: float | None = None
    error: str | None = None
    success: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class TraceSession:
    def __init__(self, session_id: str | None = None, traces_dir: str = "traces"):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(exist_ok=True)
        self.events: list[TraceEvent] = []
        self._output_path = self.traces_dir / f"run_{self.session_id}.jsonl"

    def add_event(self, event: TraceEvent):
        self.events.append(event)
        # write immediately so nothing is lost if the agent crashes
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def output_path(self) -> Path:
        return self._output_path

    @staticmethod
    def load(path: str) -> list[dict]:
        events = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()
