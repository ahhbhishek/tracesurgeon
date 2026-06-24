import time
import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .trace import TraceEvent, TraceSession


# Per-field character cap for captured strings. Large enough for real LLM
# outputs and tool results, small enough that a giant RAG document or 200KB
# log dump doesn't bloat the trace file or slow analysis.
_MAX_FIELD_CHARS = 8000
_HEAD_CHARS = 5000  # keep the head (the actual content)
_TAIL_CHARS = 2500  # AND the tail (where stack traces / final errors live)


def _truncate(text: str) -> str:
    """
    Cap an over-long string while PRESERVING both ends.

    A naive head-only cut would drop errors that surface at the end of a long
    output (e.g. a traceback after pages of logs). Keeping head + tail means the
    blame scorer still sees error signals wherever they occur.
    """
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    omitted = len(text) - _HEAD_CHARS - _TAIL_CHARS
    return (
        text[:_HEAD_CHARS]
        + f"\n…[TraceSurgeon: {omitted} chars omitted]…\n"
        + text[-_TAIL_CHARS:]
    )


def _safe_serialize(obj: Any, _depth: int = 0) -> Any:
    """Convert anything to something JSON-safe, with large strings truncated."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate(obj)
    if isinstance(obj, (int, float, bool)):
        return obj
    if _depth > 6:  # guard against pathologically deep / cyclic structures
        return _truncate(str(obj))
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i, _depth + 1) for i in obj]
    # fallback: stringify (also truncated)
    return _truncate(str(obj))


class TraceInterceptor(BaseCallbackHandler):
    """
    Hooks into every LangGraph node and tool call.
    Records inputs, outputs, timing, and parent-child relationships
    into a .jsonl file for later analysis.
    """

    def __init__(self, session: TraceSession):
        super().__init__()
        self.session = session
        # maps run_id → { step_id, start_time, node_name, parent_step_id }
        self._active: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    #  Chain (LangGraph node) events                                       #
    # ------------------------------------------------------------------ #

    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        node_name = self._extract_name(serialized, tags, metadata)

        self._active[str(run_id)] = {
            "step_id": step_id,
            "start_time": time.monotonic(),
            "node_name": node_name,
            "parent_step_id": parent_step_id,
        }

        self.session.add_event(TraceEvent(
            event_type="node_start",
            step_id=step_id,
            parent_step_id=parent_step_id,
            node_name=node_name,
            inputs=_safe_serialize(inputs),
            outputs=None,
            timestamp=TraceSession.now(),
        ))

    def on_chain_end(
        self,
        outputs: dict,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        meta = self._active.pop(str(run_id), None)
        if not meta:
            return

        duration = (time.monotonic() - meta["start_time"]) * 1000

        self.session.add_event(TraceEvent(
            event_type="node_end",
            step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"],
            node_name=meta["node_name"],
            inputs=None,
            outputs=_safe_serialize(outputs),
            timestamp=TraceSession.now(),
            duration_ms=round(duration, 2),
            success=True,
        ))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        meta = self._active.pop(str(run_id), None)
        if not meta:
            return

        duration = (time.monotonic() - meta["start_time"]) * 1000

        self.session.add_event(TraceEvent(
            event_type="error",
            step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"],
            node_name=meta["node_name"],
            inputs=None,
            outputs=None,
            timestamp=TraceSession.now(),
            duration_ms=round(duration, 2),
            error=str(error),
            success=False,
        ))

    # ------------------------------------------------------------------ #
    #  Tool events                                                         #
    # ------------------------------------------------------------------ #

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs,
    ):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        tool_name = (serialized or {}).get("name", "unknown_tool")

        self._active[str(run_id)] = {
            "step_id": step_id,
            "start_time": time.monotonic(),
            "node_name": f"tool:{tool_name}",
            "parent_step_id": parent_step_id,
        }

        self.session.add_event(TraceEvent(
            event_type="tool_start",
            step_id=step_id,
            parent_step_id=parent_step_id,
            node_name=f"tool:{tool_name}",
            inputs=_safe_serialize(input_str),
            outputs=None,
            timestamp=TraceSession.now(),
        ))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        meta = self._active.pop(str(run_id), None)
        if not meta:
            return

        duration = (time.monotonic() - meta["start_time"]) * 1000

        self.session.add_event(TraceEvent(
            event_type="tool_end",
            step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"],
            node_name=meta["node_name"],
            inputs=None,
            outputs=_safe_serialize(str(output)),
            timestamp=TraceSession.now(),
            duration_ms=round(duration, 2),
            success=True,
        ))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        meta = self._active.pop(str(run_id), None)
        if not meta:
            return

        duration = (time.monotonic() - meta["start_time"]) * 1000

        self.session.add_event(TraceEvent(
            event_type="error",
            step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"],
            node_name=meta["node_name"],
            inputs=None,
            outputs=None,
            timestamp=TraceSession.now(),
            duration_ms=round(duration, 2),
            error=str(error),
            success=False,
        ))

    # ------------------------------------------------------------------ #
    #  LLM events (capture model calls)                                   #
    # ------------------------------------------------------------------ #

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        model_name = (serialized or {}).get("kwargs", {}).get("model_name", "llm")

        self._active[str(run_id)] = {
            "step_id": step_id,
            "start_time": time.monotonic(),
            "node_name": f"llm:{model_name}",
            "parent_step_id": parent_step_id,
        }

        self.session.add_event(TraceEvent(
            event_type="node_start",
            step_id=step_id,
            parent_step_id=parent_step_id,
            node_name=f"llm:{model_name}",
            inputs=_safe_serialize(prompts),
            outputs=None,
            timestamp=TraceSession.now(),
        ))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        meta = self._active.pop(str(run_id), None)
        if not meta:
            return

        duration = (time.monotonic() - meta["start_time"]) * 1000
        output_text = None
        if response.generations:
            output_text = response.generations[0][0].text if response.generations[0] else None

        self.session.add_event(TraceEvent(
            event_type="node_end",
            step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"],
            node_name=meta["node_name"],
            inputs=None,
            outputs=_safe_serialize(output_text),
            timestamp=TraceSession.now(),
            duration_ms=round(duration, 2),
            success=True,
        ))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_parent(self, parent_run_id: UUID | None) -> str | None:
        if parent_run_id is None:
            return None
        return self._active.get(str(parent_run_id), {}).get("step_id")

    def _extract_name(
        self,
        serialized: dict | None,
        tags: list[str] | None,
        metadata: dict | None = None,
    ) -> str:
        # LangGraph stamps the real node name here — by far the most useful label
        if metadata and metadata.get("langgraph_node"):
            return metadata["langgraph_node"]
        serialized = serialized or {}
        if serialized.get("name"):
            return serialized["name"]
        if serialized.get("id"):
            return serialized["id"][-1]
        if tags:
            return tags[0]
        return "unknown_node"
