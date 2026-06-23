"""
Hardening test — a BRANCHING multi-tool agent.

The Phase 1 agent was linear. Real agents branch, call several tools, and
sometimes throw real exceptions. This file builds a tougher agent and runs
4 scenarios to stress the blame logic:

    Topology:
        plan ──► search_tool ──► analyze ──► report
                      │                ▲
                      └──► fetch_tool ─┘     (two tools feed analyze)

    Scenarios:
        1. clean           — nothing wrong, must report NO failure
        2. poison_search   — search_tool returns junk; blame must land on search_tool
        3. poison_fetch    — fetch_tool returns junk; blame must land on fetch_tool
        4. crash_analyze   — analyze raises a real exception; blame must land there

Run with:
    cd Projects/tracesurgeon
    python tests/test_branching_agent.py
"""

import sys
import operator
from pathlib import Path
from typing import TypedDict, Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from tracesurgeon import TraceInterceptor, TraceSession, build_dataflow_dag, print_pipeline, run_blame_analysis


# ------------------------------------------------------------------ #
#  State                                                              #
# ------------------------------------------------------------------ #

class State(TypedDict):
    query: str
    search_result: str
    fetch_result: str
    analysis: str
    report: str
    messages: Annotated[list, operator.add]


# scenario flags — set per run
SCENARIO = "clean"


# ------------------------------------------------------------------ #
#  Tools                                                              #
# ------------------------------------------------------------------ #

@tool
def search_tool(query: str) -> str:
    """Search the web."""
    if SCENARIO == "poison_search":
        return "ERROR: search API returned 500 internal server error"
    return f"Search hits for '{query}': France is in Europe. Capital region: Île-de-France."


@tool
def fetch_tool(query: str) -> str:
    """Fetch structured facts."""
    if SCENARIO == "poison_fetch":
        return "ERROR: malformed response, could not parse facts"
    return f"Facts for '{query}': Paris, population 2.1M, founded 3rd century BC."


# ------------------------------------------------------------------ #
#  Nodes                                                             #
# ------------------------------------------------------------------ #

def plan_node(state: State) -> dict:
    return {"messages": [AIMessage(content=f"Plan: research '{state['query']}'")]}


def search_node(state: State) -> dict:
    res = search_tool.invoke({"query": state["query"]})
    return {"search_result": res, "messages": [AIMessage(content="searched")]}


def fetch_node(state: State) -> dict:
    res = fetch_tool.invoke({"query": state["query"]})
    return {"fetch_result": res, "messages": [AIMessage(content="fetched")]}


def analyze_node(state: State) -> dict:
    search = state.get("search_result", "")
    fetch = state.get("fetch_result", "")

    if SCENARIO == "crash_analyze":
        raise ValueError("analyze_node crashed: division by zero in scoring")

    combined = f"{search} | {fetch}"
    if "ERROR" in combined:
        analysis = f"Analysis incomplete — bad upstream data: {combined[:90]}"
    else:
        analysis = f"Analysis: combined {len(combined)} chars of clean data."
    return {"analysis": analysis, "messages": [AIMessage(content=analysis)]}


def report_node(state: State) -> dict:
    report = f"=== REPORT ===\n{state.get('analysis', 'N/A')}"
    return {"report": report, "messages": [AIMessage(content=report)]}


# ------------------------------------------------------------------ #
#  Graph                                                             #
# ------------------------------------------------------------------ #

def build_agent():
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("search", search_node)
    g.add_node("fetch", fetch_node)
    g.add_node("analyze", analyze_node)
    g.add_node("report", report_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "fetch")     # sequential so ordering is deterministic
    g.add_edge("fetch", "analyze")
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile()


# ------------------------------------------------------------------ #
#  Runner                                                            #
# ------------------------------------------------------------------ #

def run_scenario(scenario: str, expected_root: str | None):
    global SCENARIO
    SCENARIO = scenario

    print(f"\n{'='*62}")
    print(f"  SCENARIO: {scenario}   (expect root cause: {expected_root or 'none'})")
    print(f"{'='*62}")

    session = TraceSession(session_id=f"branch_{scenario}", traces_dir="traces")
    interceptor = TraceInterceptor(session)
    agent = build_agent()

    crashed = False
    try:
        agent.invoke(
            {"query": "Tell me about Paris", "messages": []},
            config={"callbacks": [interceptor]},
        )
    except Exception as e:
        crashed = True
        print(f"  (agent raised: {type(e).__name__}: {e})")

    path = str(session.output_path())
    flow = build_dataflow_dag(path)
    result = run_blame_analysis(flow)
    last_flow = flow  # noqa: keep for debug

    if not result["has_failure"]:
        verdict = "PASS" if expected_root is None else "FAIL"
        print(f"  Result: no failure detected.   [{verdict}]")
        return verdict == "PASS"

    rc = result["root_cause"]
    got = rc["node_name"]
    # node names may be "search_tool", "tool:search_tool", or "analyze"
    ok = expected_root is not None and expected_root in got
    verdict = "PASS" if ok else "FAIL"

    print(f"  Symptom    : {result['symptom']['node_name']}")
    print(f"  Root cause : {got}  (score {rc['score']})")
    print(f"  Reasons    : {', '.join(rc['reasons'])}")
    print(f"  Verdict    : [{verdict}]")

    print(f"  --- data-flow pipeline ---")
    print_pipeline(flow, highlight=[rc["step_id"]])

    if not ok:
        print(f"  --- full ranking ---")
        for i, s in enumerate(result["suspects"]):
            print(f"    {i+1}. {s['score']:.3f}  {s['node_name']}  ({', '.join(s['reasons'])})")

    return ok


if __name__ == "__main__":
    results = []
    results.append(run_scenario("clean", None))
    results.append(run_scenario("poison_search", "search_tool"))
    results.append(run_scenario("poison_fetch", "fetch_tool"))
    results.append(run_scenario("crash_analyze", "analyze"))

    passed = sum(results)
    print(f"\n{'='*62}")
    print(f"  RESULTS: {passed}/{len(results)} scenarios passed")
    print(f"{'='*62}")
    sys.exit(0 if passed == len(results) else 1)
