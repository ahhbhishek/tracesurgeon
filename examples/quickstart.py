"""
TraceSurgeon quickstart — copy this, swap in YOUR agent, run it.

This is the whole integration: 3 lines around your existing agent.invoke().
No API key needed to run this file — it uses a tiny self-contained LangGraph
agent with a deliberately poisoned tool so you can see a real root-cause report.

    cd Projects/tracesurgeon
    python examples/quickstart.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# 1) import TraceSurgeon ------------------------------------------------------
from tracesurgeon import instrument, diagnose


# --- your agent (replace this whole block with your real LangGraph agent) ----
class State(TypedDict):
    query: str
    facts: str
    answer: str
    messages: Annotated[list, operator.add]


@tool
def lookup_facts(query: str) -> str:
    """Look up facts for the query (this one is broken on purpose)."""
    return "ERROR: knowledge base returned HTTP 503, service unavailable"


def research(state: State) -> dict:
    return {"facts": lookup_facts.invoke({"query": state["query"]}),
            "messages": [AIMessage(content="researched")]}


def answer(state: State) -> dict:
    facts = state["facts"]
    ans = ("I could not answer — the lookup failed." if "ERROR" in facts
           else f"Based on {facts}, here is the answer.")
    return {"answer": ans, "messages": [AIMessage(content=ans)]}


def build_agent():
    g = StateGraph(State)
    g.add_node("research", research)
    g.add_node("answer", answer)
    g.set_entry_point("research")
    g.add_edge("research", "answer")
    g.add_edge("answer", END)
    return g.compile()


# --- the actual integration --------------------------------------------------
if __name__ == "__main__":
    agent = build_agent()

    # 2) instrument + run your agent as normal
    inst = instrument(session_id="quickstart")
    agent.invoke({"query": "capital of France", "messages": []}, config=inst.config)

    # 3) diagnose the run
    diag = diagnose(inst.path)
    diag.print()

    # ...or use it programmatically / in CI:
    if diag.has_failure:
        rc = diag.root_cause
        print(f"[CI] root cause = {rc['node_name']} "
              f"({rc['remediation']['category']})")

    # the trace is saved to inst.path; you can also analyze it from the CLI:
    #   tracesurgeon debug traces/run_quickstart.jsonl
    #   tracesurgeon debug --json traces/run_quickstart.jsonl
    print(f"\nTrace saved to: {inst.path}")
