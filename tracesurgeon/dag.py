"""
DAG Builder — Phase 2.

TWO graph views are built from the same trace:

  build_dag()            -> CALL TREE   (who-called-whom; great for display)
  build_dataflow_dag()   -> DATA FLOW   (whose-output-fed-whose-input; for blame)

The call tree is what LangGraph hands us natively: step1/step2/step3 are all
siblings under the root graph node. That's useless for blame because the tool's
output can't "flow" to a sibling. The data-flow DAG fixes this by:
  - chaining the pipeline steps in execution order (step1 -> step2 -> step3)
  - pointing nested tool/llm calls INTO their step (tool -> step1)
  - dropping the root aggregate node (it just accumulates everything)

Result for a linear agent:  tool -> step1 -> step2 -> step3
Now ancestors(step3) = {tool, step1, step2} and blame can flow back to the tool.
"""

import json
from pathlib import Path

import networkx as nx


# ------------------------------------------------------------------ #
#  Shared parsing                                                      #
# ------------------------------------------------------------------ #

def _parse_steps(trace_path: str) -> dict[str, dict]:
    """Merge start/end events per step_id into one record each."""
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    steps: dict[str, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            sid = event["step_id"]

            if sid not in steps:
                steps[sid] = {
                    "step_id": sid,
                    "parent_step_id": event.get("parent_step_id"),
                    "node_name": event["node_name"],
                    "inputs": None,
                    "outputs": None,
                    "duration_ms": None,
                    "success": True,
                    "error": None,
                    "start_ts": None,
                }

            etype = event["event_type"]
            if etype.endswith("_start"):
                # keep the earliest start timestamp for ordering siblings
                if steps[sid]["start_ts"] is None:
                    steps[sid]["start_ts"] = event.get("timestamp")
                if event.get("inputs") is not None:
                    steps[sid]["inputs"] = event["inputs"]
            if etype.endswith("_end") and event.get("outputs") is not None:
                steps[sid]["outputs"] = event["outputs"]
            if event.get("duration_ms") is not None:
                steps[sid]["duration_ms"] = event["duration_ms"]
            if etype == "error":
                steps[sid]["success"] = False
                steps[sid]["error"] = event.get("error")

    return steps


# ------------------------------------------------------------------ #
#  Call tree (for display)                                             #
# ------------------------------------------------------------------ #

def build_dag(trace_path: str) -> nx.DiGraph:
    """Call tree: parent_step_id -> step_id edges. Good for printing."""
    steps = _parse_steps(trace_path)
    dag = nx.DiGraph()
    for sid, attrs in steps.items():
        dag.add_node(sid, **attrs)
    for sid, attrs in steps.items():
        parent = attrs.get("parent_step_id")
        if parent and parent in steps:
            dag.add_edge(parent, sid)
    return dag


# ------------------------------------------------------------------ #
#  Data-flow graph (for blame)                                        #
# ------------------------------------------------------------------ #

def build_dataflow_dag(trace_path: str) -> nx.DiGraph:
    """
    Build a data-flow DAG where edges follow how data actually moves.
    See module docstring for the transformation rules.
    """
    steps = _parse_steps(trace_path)

    # root aggregate node(s): no parent
    root_ids = {sid for sid, a in steps.items() if not a.get("parent_step_id")}

    dag = nx.DiGraph()
    for sid, attrs in steps.items():
        if sid in root_ids:
            continue  # drop the aggregate root from the data-flow view
        dag.add_node(sid, **attrs)

    # pipeline nodes = direct children of a root, ordered by start time
    pipeline = sorted(
        [a for sid, a in steps.items() if a.get("parent_step_id") in root_ids],
        key=lambda a: a.get("start_ts") or "",
    )
    # chain them sequentially: step1 -> step2 -> step3
    for prev, nxt in zip(pipeline, pipeline[1:]):
        dag.add_edge(prev["step_id"], nxt["step_id"], edge_type="sequential")

    # nested nodes (tool/llm under a step) feed INTO their step
    for sid, a in steps.items():
        parent = a.get("parent_step_id")
        if parent and parent not in root_ids and parent in dag:
            dag.add_edge(sid, parent, edge_type="nested")

    return dag


# ------------------------------------------------------------------ #
#  Traversal helpers                                                  #
# ------------------------------------------------------------------ #

def find_failure_path(dag: nx.DiGraph) -> list[str]:
    """Chain of step_ids from root to the first failed node (exceptions only)."""
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


def print_tree(dag: nx.DiGraph, highlight: list[str] | None = None) -> None:
    """Print the CALL-TREE dag as an ASCII tree. Marks highlighted nodes."""
    hl = set(highlight or [])
    roots = [n for n in dag.nodes if dag.in_degree(n) == 0]

    def _render(node_id: str, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        data = dag.nodes[node_id]
        name = data.get("node_name", node_id)
        dur = f" [{data['duration_ms']:.1f}ms]" if data.get("duration_ms") else ""
        status = "✓" if data.get("success", True) else "✗ FAIL"
        marker = "  ◄ ROOT CAUSE" if node_id in hl else ""
        print(f"{prefix}{connector}{name} ({node_id[:6]}) {status}{dur}{marker}")
        kids = list(dag.successors(node_id))
        cp = prefix + ("    " if is_last else "│   ")
        for i, k in enumerate(kids):
            _render(k, cp, i == len(kids) - 1)

    for root in roots:
        data = dag.nodes[root]
        marker = "  ◄ ROOT CAUSE" if root in hl else ""
        print(f"{data.get('node_name', root)} ({root[:6]}){marker}")
        kids = list(dag.successors(root))
        for i, k in enumerate(kids):
            _render(k, "", i == len(kids) - 1)


def summarize_dag(dag: nx.DiGraph) -> dict:
    nodes = list(dag.nodes(data=True))
    failed = [n for n, d in nodes if not d.get("success", True)]
    tools = [n for n, d in nodes if "tool:" in d.get("node_name", "")]
    durs = [d["duration_ms"] for _, d in nodes if d.get("duration_ms")]
    return {
        "total_steps": len(nodes),
        "failed_steps": len(failed),
        "tool_calls": len(tools),
        "total_duration_ms": round(sum(durs), 2) if durs else 0,
        "has_failure": len(failed) > 0,
    }
