"""
Edge-case stress suite — tries to BREAK TraceSurgeon.

Each test builds a synthetic trace (or a real LangGraph run) for a tricky
real-world situation and asserts the blame analysis does the right thing.
This is the "make sure it works fine" gate before shipping Phase 1-3.

Run with:
    cd Projects/tracesurgeon
    python tests/test_edge_cases.py
"""

import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon import build_dataflow_dag, run_blame_analysis
from tracesurgeon.trace import TraceSession, TraceEvent
from tracesurgeon.interceptor import _truncate, _safe_serialize, _MAX_FIELD_CHARS


PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))


# ------------------------------------------------------------------ #
#  Synthetic trace builder (full control over edge cases)            #
# ------------------------------------------------------------------ #

def write_trace(name: str, steps: list[dict]) -> str:
    """
    steps: list of {name, outputs, success?, nested?(bool), ts}
    Builds a minimal but valid .jsonl trace with a root + pipeline + nested tools.
    Returns the path.
    """
    sess = TraceSession(session_id=f"edge_{name}", traces_dir="traces")
    root_id = "root0000"
    base_ts = "2026-06-24T00:00:00.{:06d}+00:00"

    # root start
    sess.add_event(TraceEvent("node_start", root_id, None, "unknown_node",
                              {"q": "x"}, None, base_ts.format(0)))
    seq = 1
    pipeline_ids = []
    for i, s in enumerate(steps):
        sid = s.get("id") or uuid.uuid4().hex[:8]
        nested = s.get("nested", False)
        parent = pipeline_ids[-1] if (nested and pipeline_ids) else root_id
        name_ = s["name"]
        ev_start = "tool_start" if nested else "node_start"
        ev_end = "tool_end" if nested else "node_end"
        sess.add_event(TraceEvent(ev_start, sid, parent, name_,
                                  {"in": i}, None, base_ts.format(seq))); seq += 1
        if s.get("success", True):
            sess.add_event(TraceEvent(ev_end, sid, parent, name_,
                                      None, s.get("outputs"), base_ts.format(seq),
                                      duration_ms=1.0, success=True)); seq += 1
        else:
            sess.add_event(TraceEvent("error", sid, parent, name_,
                                      None, None, base_ts.format(seq),
                                      duration_ms=1.0, error=s.get("outputs", "boom"),
                                      success=False)); seq += 1
        if not nested:
            pipeline_ids.append(sid)
    sess.add_event(TraceEvent("node_end", root_id, None, "unknown_node",
                              None, {"done": True}, base_ts.format(seq)))
    return str(sess.output_path())


def root_cause_name(path: str):
    res = run_blame_analysis(build_dataflow_dag(path))
    if not res["has_failure"]:
        return None
    return res["root_cause"]["node_name"]


# ------------------------------------------------------------------ #
#  TESTS                                                             #
# ------------------------------------------------------------------ #

def t_truncation_preserves_tail_error():
    big = "x" * 50000 + " ERROR: boom at the very end"
    out = _truncate(big)
    check("truncation: caps size", len(out) < _MAX_FIELD_CHARS + 200,
          f"len={len(out)}")
    check("truncation: preserves tail error", "ERROR: boom at the very end" in out)


def t_truncation_serialize_nested():
    obj = {"logs": ["a" * 20000, {"deep": "z" * 20000}]}
    s = _safe_serialize(obj)
    flat = json.dumps(s)
    check("serialize: large nested truncated", len(flat) < 60000, f"len={len(flat)}")


def t_clean_run_no_false_positive():
    path = write_trace("clean", [
        {"name": "plan", "outputs": {"msg": "all good"}},
        {"name": "search", "outputs": {"data": "Paris is the capital. No errors."}},
        {"name": "report", "outputs": {"final": "Done successfully with 0 errors."}},
    ])
    res = run_blame_analysis(build_dataflow_dag(path))
    check("clean run: no failure", not res["has_failure"])


def t_single_poisoned_tool():
    path = write_trace("poison1", [
        {"name": "plan", "outputs": {"msg": "ok"}},
        {"name": "search", "outputs": {"x": "fine"}},
        {"name": "tool:fetch", "outputs": "ERROR: 503 service unavailable", "nested": True},
        {"name": "analyze", "outputs": {"a": "incomplete due to bad data"}},
        {"name": "report", "outputs": {"r": "could not finish"}},
    ])
    rc = root_cause_name(path)
    check("single poison: blames the tool", rc == "tool:fetch", f"got {rc}")


def t_node_introduced_error():
    # no tool involved; a node itself produces bad data
    path = write_trace("nodeintro", [
        {"name": "plan", "outputs": {"msg": "ok"}},
        {"name": "compute", "outputs": {"v": "invalid result: division by zero"}},
        {"name": "report", "outputs": {"r": "propagated invalid value"}},
    ])
    rc = root_cause_name(path)
    check("node-introduced error: blames origin node", rc == "compute", f"got {rc}")


def t_real_exception_anchored():
    path = write_trace("exc", [
        {"name": "plan", "outputs": {"msg": "ok"}},
        {"name": "step2", "outputs": {"x": "fine"}},
        {"name": "crasher", "outputs": "RuntimeError: kaboom", "success": False},
    ])
    res = run_blame_analysis(build_dataflow_dag(path))
    rc = root_cause_name(path)
    check("real exception: detected as failure", res["has_failure"])
    check("real exception: blames crashing node", rc == "crasher", f"got {rc}")


def t_benign_error_word_not_flagged():
    path = write_trace("benign", [
        {"name": "plan", "outputs": {"msg": "implementing robust error handling"}},
        {"name": "review", "outputs": {"note": "the error case is covered; no errors found"}},
        {"name": "done", "outputs": {"r": "completed with no errors"}},
    ])
    res = run_blame_analysis(build_dataflow_dag(path))
    check("benign error-words: no false positive", not res["has_failure"])


def t_clean_tool_not_inflated():
    # two tools: one clean, one poisoned. Clean one must NOT be root cause.
    path = write_trace("twotools", [
        {"name": "plan", "outputs": {"msg": "ok"}},
        {"name": "search", "outputs": {"x": "fine"}},
        {"name": "tool:good", "outputs": "valid data: 42", "nested": True},
        {"name": "fetch", "outputs": {"x": "fine"}},
        {"name": "tool:bad", "outputs": "ERROR: timeout", "nested": True},
        {"name": "report", "outputs": {"r": "failed"}},
    ])
    res = run_blame_analysis(build_dataflow_dag(path))
    rc = res["root_cause"]["node_name"] if res["has_failure"] else None
    # find the clean tool's score
    good = next((s for s in res["suspects"] if s["node_name"] == "tool:good"), None)
    check("clean tool not blamed", rc == "tool:bad", f"got {rc}")
    check("clean tool low score", good is not None and good["score"] < 0.3,
          f"good score={good['score'] if good else 'n/a'}")


def t_empty_trace():
    sess = TraceSession(session_id="edge_empty", traces_dir="traces")
    path = str(sess.output_path())
    Path(path).write_text("", encoding="utf-8")
    try:
        res = run_blame_analysis(build_dataflow_dag(path))
        check("empty trace: no crash, no failure", not res["has_failure"])
    except Exception as e:
        check("empty trace: no crash, no failure", False, str(e))


def t_single_node_trace():
    path = write_trace("single", [
        {"name": "only", "outputs": {"r": "ERROR: lonely failure"}},
    ])
    try:
        rc = root_cause_name(path)
        check("single-node failure: blames it", rc == "only", f"got {rc}")
    except Exception as e:
        check("single-node failure: blames it", False, str(e))


def t_unicode_and_weird_output():
    path = write_trace("unicode", [
        {"name": "plan", "outputs": {"msg": "café ☕ 日本語"}},
        {"name": "tool:x", "outputs": "ERROR: 失败 — 503 ☠", "nested": True},
        {"name": "report", "outputs": {"r": "could not complete"}},
    ])
    try:
        rc = root_cause_name(path)
        check("unicode output: blames tool", rc == "tool:x", f"got {rc}")
    except Exception as e:
        check("unicode output: blames tool", False, str(e))


def t_error_buried_in_huge_output():
    # the error is far past the head cut — must still be found via tail preservation
    huge = "log line\n" * 2000 + "fatal: ERROR 500 at the end"
    path = write_trace("buried", [
        {"name": "plan", "outputs": {"msg": "ok"}},
        {"name": "tool:noisy", "outputs": huge, "nested": True},
        {"name": "report", "outputs": {"r": "could not parse"}},
    ])
    rc = root_cause_name(path)
    check("buried error in huge output: still found", rc == "tool:noisy", f"got {rc}")


if __name__ == "__main__":
    print("Edge-case stress suite:\n")
    for fn in [
        t_truncation_preserves_tail_error,
        t_truncation_serialize_nested,
        t_clean_run_no_false_positive,
        t_single_poisoned_tool,
        t_node_introduced_error,
        t_real_exception_anchored,
        t_benign_error_word_not_flagged,
        t_clean_tool_not_inflated,
        t_empty_trace,
        t_single_node_trace,
        t_unicode_and_weird_output,
        t_error_buried_in_huge_output,
    ]:
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"threw {type(e).__name__}: {e}")

    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:", ", ".join(FAIL))
    sys.exit(0 if not FAIL else 1)
