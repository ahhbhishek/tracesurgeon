"""
Blame Scorer — Phase 3.

Operates on the DATA-FLOW dag (from build_dataflow_dag), not the call tree.

Two jobs:
  1. Detect failures — including SILENT ones (error text in output, no exception).
  2. Rank suspects so the ORIGIN of the error wins, not the loudest downstream
     symptom. This is the anti-"blame-hoarding" core of TraceSurgeon.

Scoring heuristic (no ML yet):
  +0.50  INTRODUCER — node has the error but none of its data-flow inputs did
  +0.25  is a tool call (external data sources are the usual origin)
  +0.20  output contains an error signal
  +0.15  proximity / direct feed into the symptom
  ──────
  capped at 1.00

The introducer bonus is what flows blame backward past the symptom.
"""

import re

import networkx as nx


# error signals. \b word boundaries stop "invalid" matching "invalid_tool_calls".
_ERROR_PATTERNS = [
    r"\berror\b", r"\bexception\b", r"\bfailed\b", r"\btraceback\b",
    r"\binvalid\b", r"\bmalformed\b", r"\bcould not\b", r"\bunable to\b",
    r"\btimed out\b", r"\brate limit\b", r"\bunauthorized\b",
    r"\b404\b", r"\b500\b",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)


def _output_str(data: dict) -> str:
    out = data.get("outputs")
    return "" if out is None else str(out)


def _has_error(data: dict) -> bool:
    return bool(_ERROR_RE.search(_output_str(data)))


def detect_symptom(dag: nx.DiGraph) -> str | None:
    """
    The SYMPTOM is where the failure becomes visible.
      - explicit exception (success=False) wins
      - else the most DOWNSTREAM node whose output carries an error signal
    Returns step_id or None.
    """
    for node_id, data in dag.nodes(data=True):
        if not data.get("success", True):
            return node_id

    try:
        order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        order = list(dag.nodes)

    # walk downstream; last error node in topo order = the visible symptom
    symptom = None
    for node_id in order:
        if _has_error(dag.nodes[node_id]):
            symptom = node_id
    return symptom


def _is_introducer(dag: nx.DiGraph, node_id: str) -> bool:
    """True if this node has an error but no data-flow predecessor does."""
    if not _has_error(dag.nodes[node_id]):
        return False
    for pred in dag.predecessors(node_id):
        if _has_error(dag.nodes[pred]):
            return False
    return True


def score_suspects(dag: nx.DiGraph, symptom_id: str) -> list[dict]:
    """Rank the symptom's ancestors (plus the symptom) by likelihood of being root cause."""
    candidates = set(nx.ancestors(dag, symptom_id))
    candidates.add(symptom_id)  # symptom can also be the introducer
    if not candidates:
        return []

    # distances to the symptom for the proximity bonus
    distances: dict[str, int] = {}
    for c in candidates:
        if c == symptom_id:
            distances[c] = 0
            continue
        try:
            distances[c] = nx.shortest_path_length(dag, c, symptom_id)
        except nx.NetworkXNoPath:
            distances[c] = 99
    max_dist = max(distances.values()) or 1

    results = []
    for node_id in candidates:
        data = dag.nodes[node_id]
        score = 0.0
        reasons = []

        if _is_introducer(dag, node_id):
            score += 0.50
            reasons.append("INTRODUCED the error (inputs were clean)")

        if "tool:" in data.get("node_name", ""):
            score += 0.25
            reasons.append("tool call (external data source)")

        if _has_error(data):
            score += 0.20
            reasons.append("output contains error signal")

        dist = distances.get(node_id, max_dist)
        if node_id == symptom_id or dag.has_edge(node_id, symptom_id):
            score += 0.15
            reasons.append("feeds directly into the symptom")
        else:
            proximity = 1.0 - (dist - 1) / max(max_dist, 1)
            score += 0.15 * max(proximity, 0)

        results.append({
            "step_id": node_id,
            "node_name": data.get("node_name", node_id),
            "score": round(min(score, 1.0), 3),
            "is_symptom": node_id == symptom_id,
            "reasons": reasons,
            "outputs_preview": _output_str(data)[:120] or "(no output)",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def run_blame_analysis(dataflow_dag: nx.DiGraph) -> dict:
    """Top-level: detect the symptom, rank suspects, identify the root cause."""
    symptom_id = detect_symptom(dataflow_dag)
    if not symptom_id:
        return {"has_failure": False, "symptom": None, "suspects": [], "root_cause": None}

    suspects = score_suspects(dataflow_dag, symptom_id)
    sdata = dataflow_dag.nodes[symptom_id]

    return {
        "has_failure": True,
        "symptom": {
            "step_id": symptom_id,
            "node_name": sdata.get("node_name", symptom_id),
            "outputs_preview": _output_str(sdata)[:120],
        },
        "suspects": suspects,
        "root_cause": suspects[0] if suspects else None,
    }
