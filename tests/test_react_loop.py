"""
Real-world topology test — a ReAct LOOP.

create_react_agent and most production agents are CYCLIC:

    agent ──► tools ──► agent ──► tools ──► agent ──► END
      ▲realises it needs data    ▲realises it needs more    ▲answers

The same node name ("agent", "tools") appears many times. This stresses:
  - repeated node names across loop iterations
  - timestamp-based ordering (not parent links) holding the causal chain
  - blame landing on the RIGHT iteration's tool call, not just any "tools" node

Scenario: the SECOND tool call returns poisoned data; the final answer is wrong.
TraceSurgeon must blame the second tool call, not the final agent reasoning.

Run with:
    cd Projects/tracesurgeon
    python tests/test_react_loop.py
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

from tracesurgeon import instrument, diagnose


class State(TypedDict):
    query: str
    scratchpad: Annotated[list, operator.add]
    answer: str


# loop control + which iteration to poison
_MAX_TURNS = 3
_POISON_TURN = 2


@tool
def lookup(fact: str) -> str:
    """Look up a fact."""
    # poison only fires on the configured turn via the shared counter
    if _state["turn"] == _POISON_TURN:
        return "ERROR: knowledge base timed out, no data returned"
    return f"Fact[{fact}]: verified value {_state['turn']}00"


# tiny shared state so the fake nodes can count turns deterministically
_state = {"turn": 0}


def agent_node(state: State) -> dict:
    turn = _state["turn"]
    if turn >= _MAX_TURNS:
        # final answer — built from whatever the scratchpad holds
        bad = any("ERROR" in s for s in state.get("scratchpad", []))
        ans = "Answer: incomplete (bad data)" if bad else "Answer: 200 confirmed"
        return {"answer": ans, "scratchpad": [AIMessage(content="final")]}
    return {"scratchpad": [AIMessage(content=f"thinking turn {turn}")]}


def tools_node(state: State) -> dict:
    _state["turn"] += 1
    res = lookup.invoke({"fact": state["query"]})
    return {"scratchpad": [AIMessage(content=res)]}


def should_continue(state: State) -> str:
    return "end" if _state["turn"] >= _MAX_TURNS else "tools"


def build_agent():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()


def run():
    _state["turn"] = 0
    print(f"{'='*60}")
    print(f"  ReAct loop — poisoning tool call on turn {_POISON_TURN}")
    print(f"{'='*60}")

    inst = instrument(session_id="react_loop")
    agent = build_agent()
    out = agent.invoke({"query": "value of X", "scratchpad": []}, config=inst.config)

    print(f"\n  Final answer: {out.get('answer')!r}")

    diag = diagnose(inst.path)
    diag.print()

    # validation: root cause should be a tool node carrying the error
    rc = diag.root_cause
    ok = (
        diag.has_failure
        and rc is not None
        and "tool" in rc["node_name"].lower()
        and "INTRODUCED the error (inputs were clean)" in rc["reasons"]
    )
    print(f"  VALIDATION: {'PASS' if ok else 'FAIL'} — "
          f"blamed {rc['node_name'] if rc else 'nothing'} in a loop with "
          f"{_MAX_TURNS} tool calls")
    return ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
