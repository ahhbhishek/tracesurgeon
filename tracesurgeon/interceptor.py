import threading
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
    try:
        return _truncate(str(obj))
    except Exception:
        return "<unserializable>"


def _guard(method):
    """
    Decorator: a tracing callback must NEVER crash the host agent.

    LangChain already isolates handler exceptions to some degree, but we add our
    own belt-and-suspenders: any error inside a handler is swallowed (optionally
    surfaced once via the session) so the user's agent keeps running even if we
    hit an un-serializable object or an internal bug.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - intentional catch-all
            self._note_internal_error(method.__name__, e)
            return None
    wrapper.__name__ = method.__name__
    return wrapper


class TraceInterceptor(BaseCallbackHandler):
    """
    Hooks into every LangGraph node, tool, and LLM call and records inputs,
    outputs, timing, and parent-child relationships into a .jsonl file.

    Production-safe:
      - shared state (`_active`) guarded by a lock for concurrent/parallel nodes
      - every callback wrapped so it can never crash the host agent
      - async callbacks supported (delegate to the same sync logic)
    """

    # we don't want LangChain to retry/raise on handler errors
    raise_error: bool = False

    def __init__(self, session: TraceSession):
        super().__init__()
        self.session = session
        # maps run_id → { step_id, start_time, node_name, parent_step_id }
        self._active: dict[str, dict] = {}
        self._active_lock = threading.Lock()
        self._internal_errors: list[str] = []

    def _note_internal_error(self, where: str, err: Exception) -> None:
        msg = f"{where}: {type(err).__name__}: {err}"
        # keep a small bounded record; never raise
        if len(self._internal_errors) < 50:
            self._internal_errors.append(msg)

    # ------------------------------------------------------------------ #
    #  shared-state helpers (locked)                                      #
    # ------------------------------------------------------------------ #

    def _register(self, run_id: UUID, step_id: str, node_name: str,
                  parent_step_id: str | None) -> None:
        with self._active_lock:
            self._active[str(run_id)] = {
                "step_id": step_id,
                "start_time": time.monotonic(),
                "node_name": node_name,
                "parent_step_id": parent_step_id,
            }

    def _pop(self, run_id: UUID) -> dict | None:
        with self._active_lock:
            return self._active.pop(str(run_id), None)

    def _resolve_parent(self, parent_run_id: UUID | None) -> str | None:
        if parent_run_id is None:
            return None
        with self._active_lock:
            return self._active.get(str(parent_run_id), {}).get("step_id")

    # ------------------------------------------------------------------ #
    #  Chain (LangGraph node) events                                      #
    # ------------------------------------------------------------------ #

    @_guard
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        node_name = self._extract_name(serialized, tags, metadata)
        self._register(run_id, step_id, node_name, parent_step_id)
        self.session.add_event(TraceEvent(
            event_type="node_start", step_id=step_id, parent_step_id=parent_step_id,
            node_name=node_name, inputs=_safe_serialize(inputs), outputs=None,
            timestamp=TraceSession.now(),
        ))

    @_guard
    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        self.session.add_event(TraceEvent(
            event_type="node_end", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=_safe_serialize(outputs),
            timestamp=TraceSession.now(), duration_ms=round(duration, 2), success=True,
        ))

    @_guard
    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        self.session.add_event(TraceEvent(
            event_type="error", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=None, timestamp=TraceSession.now(),
            duration_ms=round(duration, 2), error=str(error), success=False,
        ))

    # ------------------------------------------------------------------ #
    #  Tool events                                                        #
    # ------------------------------------------------------------------ #

    @_guard
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, **kwargs):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        tool_name = (serialized or {}).get("name", "unknown_tool")
        self._register(run_id, step_id, f"tool:{tool_name}", parent_step_id)
        self.session.add_event(TraceEvent(
            event_type="tool_start", step_id=step_id, parent_step_id=parent_step_id,
            node_name=f"tool:{tool_name}", inputs=_safe_serialize(input_str),
            outputs=None, timestamp=TraceSession.now(),
        ))

    @_guard
    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        self.session.add_event(TraceEvent(
            event_type="tool_end", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=_safe_serialize(str(output)),
            timestamp=TraceSession.now(), duration_ms=round(duration, 2), success=True,
        ))

    @_guard
    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        self.session.add_event(TraceEvent(
            event_type="error", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=None, timestamp=TraceSession.now(),
            duration_ms=round(duration, 2), error=str(error), success=False,
        ))

    # ------------------------------------------------------------------ #
    #  LLM events                                                         #
    # ------------------------------------------------------------------ #

    @_guard
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        step_id = str(uuid.uuid4())[:8]
        parent_step_id = self._resolve_parent(parent_run_id)
        model_name = (serialized or {}).get("kwargs", {}).get("model_name", "llm")
        self._register(run_id, step_id, f"llm:{model_name}", parent_step_id)
        self.session.add_event(TraceEvent(
            event_type="node_start", step_id=step_id, parent_step_id=parent_step_id,
            node_name=f"llm:{model_name}", inputs=_safe_serialize(prompts),
            outputs=None, timestamp=TraceSession.now(),
        ))

    @_guard
    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        output_text = None
        try:
            if response.generations and response.generations[0]:
                output_text = response.generations[0][0].text
        except Exception:
            output_text = None
        self.session.add_event(TraceEvent(
            event_type="node_end", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=_safe_serialize(output_text),
            timestamp=TraceSession.now(), duration_ms=round(duration, 2), success=True,
        ))

    @_guard
    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        meta = self._pop(run_id)
        if not meta:
            return
        duration = (time.monotonic() - meta["start_time"]) * 1000
        self.session.add_event(TraceEvent(
            event_type="error", step_id=meta["step_id"],
            parent_step_id=meta["parent_step_id"], node_name=meta["node_name"],
            inputs=None, outputs=None, timestamp=TraceSession.now(),
            duration_ms=round(duration, 2), error=str(error), success=False,
        ))

    # ------------------------------------------------------------------ #
    #  Async agents (ainvoke / astream)                                   #
    #                                                                     #
    #  This is a sync BaseCallbackHandler. LangChain's AsyncCallbackManager
    #  automatically runs sync handlers in a thread-pool executor for async
    #  runs, so these same methods fire correctly under ainvoke/astream.
    #  Our shared state is lock-guarded and the file write is atomic, so
    #  running in worker threads is safe. No separate async methods needed.
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _extract_name(self, serialized, tags, metadata=None) -> str:
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
