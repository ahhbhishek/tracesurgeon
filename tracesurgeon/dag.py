"""
DAG Builder — Phase 2 (hardened).

LangGraph emits TWO event layers per run:
  - an outer task layer with generic names (graph:step:N) where TOOLS attach
  - an inner node layer with the REAL names (plan, search, analyze) but flat

To get one clean graph we:
  1. prefer the real-named nodes as the pipeline (fall back to graph:step:N for
     old traces that predate node-name capture)
  2. reconstruct data flow from TIMESTAMPS, not LangGraph's parent links:
     a tool that ran inside a node's [start, end] window belongs to that node.

Result (linear agent):  search_tool ─► search ─► fetch ─► analyze ─► report
                          fetch_tool ─►        ┘
Now blame can flow back from the symptom to the tool that introduced the error.
"""

import json
import re
from pathlib import Path

import networkx as nx


# names that are LangGraph plumbing, never real user nodes
_WRAPPER_RE = re.compile(
    r"^(unknown_node|LangGraph|RunnableSequence|__start__|__end__|_write|ChannelWrite.*)$"
)
_GENERIC_RE = re.compile(r"^graph:step:\d+$")


def _is_nested(name: str) -> bool:
    return name.startswith("tool:") or name.startswith("llm:")


# ------------------------------------------------------------------ #
#  Shared parsing                                                      #
# ------------------------------------------------------------------ #

def _parse_steps(trace_path: str) -> dict[str, dict]:
    """
    Merge start/end events per step_id into one record (with start & end ts).

    Robust to real-world damage: a trace whose agent crashed mid-write may have a
    truncated final line; a hand-edited or concatenated file may have stray junk.
    We skip unparseable / malformed lines instead of crashing, so a partial trace
    still yields a partial-but-usable analysis.
    """
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    steps: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # truncated/corrupt line — skip, keep what we can
            if not isinstance(e, dict):
                continue
            sid = e.get("step_id")
            etype = e.get("event_type")
            if not sid or not etype:
                continue  # not a TraceSurgeon event; ignore
            if sid not in steps:
                steps[sid] = {
                    "step_id": sid,
                    "parent_step_id": e.get("parent_step_id"),
                    "node_name": e.get("node_name") or "unknown_node",
                    "inputs": None, "outputs": None,
                    "duration_ms": None, "success": True, "error": None,
                    "start_ts": None, "end_ts": None,
                }
            if etype.endswith("_start"):
                if steps[sid]["start_ts"] is None:
                    steps[sid]["start_ts"] = e.get("timestamp")
                if e.get("inputs") is not None:
                    steps[sid]["inputs"] = e["inputs"]
            if etype.endswith("_end"):
                steps[sid]["end_ts"] = e.get("timestamp")
                if e.get("outputs") is not None:
                    steps[sid]["outputs"] = e["outputs"]
            if e.get("duration_ms") is not None:
                steps[sid]["duration_ms"] = e["duration_ms"]
            if etype == "error":
                steps[sid]["success"] = False
                steps[sid]["error"] = e.get("error")
                steps[sid]["end_ts"] = e.get("timestamp")
    return steps


# ------------------------------------------------------------------ #
#  Call tree (raw, for debugging only)                                #
# ------------------------------------------------------------------ #

def build_dag(trace_path: str) -> nx.DiGraph:
    """Raw call tree from parent_step_id links. Kept for low-level inspection."""
    steps = _parse_steps(trace_path)
    dag = nx.DiGraph()
    for sid, a in steps.items():
        dag.add_node(sid, **a)
    for sid, a in steps.items():
        p = a.get("parent_step_id")
        if p and p in steps:
            dag.add_edge(p, sid)
    return dag


# ------------------------------------------------------------------ #
#  Data-flow graph (for blame)                                        #
# ------------------------------------------------------------------ #

def _pick_host(tool: dict, pipeline: list[dict]) -> dict | None:
    """Find the pipeline node whose time window contains this tool's start."""
    ts = tool.get("start_ts")
    if not pipeline:
        return None
    if ts is None:
        return pipeline[0]
    # nodes whose [start, end] window contains the tool's start
    contained = [
        p for p in pipeline
        if (p.get("start_ts") or "") <= ts and (p.get("end_ts") or "~") >= ts
    ]
    pool = contained or [p for p in pipeline if (p.get("start_ts") or "") <= ts]
    if not pool:
        return pipeline[0]
    return max(pool, key=lambda p: p.get("start_ts") or "")


def _merge_dual_layer(pipeline: list[dict]) -> list[dict]:
    """
    LangGraph emits an outer-task and an inner-node event for the SAME logical
    step (both real-named after metadata capture). They overlap in time and share
    a name. Collapse each such pair into one node so the pipeline reads cleanly.
    Genuine repeated nodes in a loop are separated by other nodes, so they never
    merge.
    """
    pipeline = sorted(pipeline, key=lambda a: a.get("start_ts") or "")
    merged: list[dict] = []
    for n in pipeline:
        prev = merged[-1] if merged else None
        overlaps = (
            prev is not None
            and prev["node_name"] == n["node_name"]
            and (n.get("start_ts") or "") <= (prev.get("end_ts") or "~")
        )
        if overlaps:
            prev["end_ts"] = max(prev.get("end_ts") or "", n.get("end_ts") or "")
            if prev.get("outputs") is None:
                prev["outputs"] = n.get("outputs")
            if not n.get("success", True):
                prev["success"] = False
                prev["error"] = prev.get("error") or n.get("error")
            # keep the larger measured duration
            pd, nd = prev.get("duration_ms"), n.get("duration_ms")
            prev["duration_ms"] = max([x for x in (pd, nd) if x is not None], default=None)
        else:
            merged.append(dict(n))
    return merged


def build_dataflow_dag(trace_path: str) -> nx.DiGraph:
    """Build the timestamp-reconstructed data-flow DAG used for blame analysis."""
    steps = _parse_steps(trace_path)

    nested = [a for a in steps.values() if _is_nested(a["node_name"])]

    # prefer real-named nodes; fall back to graph:step:N for legacy traces
    real = [
        a for a in steps.values()
        if not _is_nested(a["node_name"])
        and not _WRAPPER_RE.match(a["node_name"])
        and not _GENERIC_RE.match(a["node_name"])
    ]
    if real:
        pipeline = real
    else:
        pipeline = [
            a for a in steps.values()
            if not _is_nested(a["node_name"]) and _GENERIC_RE.match(a["node_name"])
        ]

    pipeline = _merge_dual_layer(pipeline)
    pipeline.sort(key=lambda a: a.get("start_ts") or "")

    dag = nx.DiGraph()
    for a in pipeline + nested:
        dag.add_node(a["step_id"], **a)

    # sequential edges between pipeline steps
    for prev, nxt in zip(pipeline, pipeline[1:]):
        dag.add_edge(prev["step_id"], nxt["step_id"], edge_type="sequential")

    # attach each tool/llm to its host node by time window
    for t in nested:
        host = _pick_host(t, pipeline)
        if host and host["step_id"] != t["step_id"]:
            dag.add_edge(t["step_id"], host["step_id"], edge_type="nested")

    return dag


# ------------------------------------------------------------------ #
#  Display                                                            #
# ------------------------------------------------------------------ #

def print_pipeline(flow: nx.DiGraph, highlight: list[str] | None = None) -> None:
    """Print the data-flow pipeline top-to-bottom with tools nested under nodes."""
    hl = set(highlight or [])
    pipe = [n for n, d in flow.nodes(data=True) if not _is_nested(d["node_name"])]
    pipe.sort(key=lambda n: flow.nodes[n].get("start_ts") or "")

    for n in pipe:
        d = flow.nodes[n]
        dur = f" [{d['duration_ms']:.1f}ms]" if d.get("duration_ms") else ""
        status = "✓" if d.get("success", True) else "✗ FAIL"
        marker = "   ◄ ROOT CAUSE" if n in hl else ""
        print(f"  ▼ {d['node_name']} ({n[:6]}) {status}{dur}{marker}")
        # tools that feed this node
        for pred in flow.predecessors(n):
            pd = flow.nodes[pred]
            if _is_nested(pd["node_name"]):
                pmark = "   ◄ ROOT CAUSE" if pred in hl else ""
                pstat = "✓" if pd.get("success", True) else "✗ FAIL"
                print(f"      └─ {pd['node_name']} ({pred[:6]}) {pstat}{pmark}")


def print_tree(dag: nx.DiGraph, highlight: list[str] | None = None) -> None:
    """ASCII call-tree printer (raw build_dag graph)."""
    hl = set(highlight or [])
    roots = [n for n in dag.nodes if dag.in_degree(n) == 0]

    def _render(node_id, prefix, is_last):
        conn = "└── " if is_last else "├── "
        d = dag.nodes[node_id]
        dur = f" [{d['duration_ms']:.1f}ms]" if d.get("duration_ms") else ""
        status = "✓" if d.get("success", True) else "✗ FAIL"
        marker = "  ◄ ROOT CAUSE" if node_id in hl else ""
        print(f"{prefix}{conn}{d.get('node_name', node_id)} ({node_id[:6]}) {status}{dur}{marker}")
        kids = list(dag.successors(node_id))
        cp = prefix + ("    " if is_last else "│   ")
        for i, k in enumerate(kids):
            _render(k, cp, i == len(kids) - 1)

    for root in roots:
        d = dag.nodes[root]
        marker = "  ◄ ROOT CAUSE" if root in hl else ""
        print(f"{d.get('node_name', root)} ({root[:6]}){marker}")
        kids = list(dag.successors(root))
        for i, k in enumerate(kids):
            _render(k, "", i == len(kids) - 1)


def find_failure_path(dag: nx.DiGraph) -> list[str]:
    failed = [n for n, d in dag.nodes(data=True) if not d.get("success", True)]
    if not failed:
        return []
    roots = [n for n in dag.nodes if dag.in_degree(n) == 0]
    if not roots:
        return [failed[0]]
    try:
        return nx.shortest_path(dag, source=roots[0], target=failed[0])
    except nx.NetworkXNoPath:
        return [failed[0]]


def summarize_dag(dag: nx.DiGraph) -> dict:
    nodes = list(dag.nodes(data=True))
    failed = [n for n, d in nodes if not d.get("success", True)]
    tools = [n for n, d in nodes if _is_nested(d.get("node_name", ""))]
    durs = [d["duration_ms"] for _, d in nodes if d.get("duration_ms")]
    return {
        "total_steps": len(nodes),
        "failed_steps": len(failed),
        "tool_calls": len(tools),
        "total_duration_ms": round(sum(durs), 2) if durs else 0,
        "has_failure": len(failed) > 0,
    }
