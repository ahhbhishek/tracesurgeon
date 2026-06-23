"""
Test agent for TraceSurgeon Phase 1.

This is a FAKE agent — no real API calls, no API key needed.
It simulates a 3-step research agent:
  1. fetch_data tool  (we'll poison this in one scenario)
  2. summarize node
  3. format_output node

Run with:
    cd Projects/tracesurgeon
    python tests/test_agent.py
"""

import sys
import json
from pathlib import Path

# make sure local package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated
import operator

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from tracesurgeon import TraceInterceptor, TraceSession


# ------------------------------------------------------------------ #
#  Agent State                                                         #
# ------------------------------------------------------------------ #

class ResearchState(TypedDict):
    query: str
    raw_data: str
    summary: str
    final_output: str
    messages: Annotated[list, operator.add]


# ------------------------------------------------------------------ #
#  Fake Tools (no API key needed)                                      #
# ------------------------------------------------------------------ #

POISON_MODE = False  # flip to True to simulate a poisoned tool output


@tool
def fetch_data(query: str) -> str:
    """Fetches data for a given query."""
    if POISON_MODE:
        # this is the "poisoned" output we'll track later
        return "ERROR: upstream service returned malformed JSON: {{{invalid}}}"
    return f"Data for '{query}': The capital of France is Paris. Population: 2.1 million."


@tool
def web_search(query: str) -> str:
    """Runs a web search."""
    return f"Search results for '{query}': [result 1] [result 2] [result 3]"


# ------------------------------------------------------------------ #
#  Agent Nodes                                                         #
# ------------------------------------------------------------------ #

def fetch_node(state: ResearchState) -> dict:
    result = fetch_data.invoke({"query": state["query"]})
    return {
        "raw_data": result,
        "messages": [AIMessage(content=f"Fetched: {result}")]
    }


def summarize_node(state: ResearchState) -> dict:
    raw = state["raw_data"]
    # fake "LLM" summarization — just trims and adds a label
    if "ERROR" in raw:
        summary = f"Summary failed: could not parse data. Raw: {raw[:80]}"
    else:
        summary = f"Summary: {raw[:100]}"
    return {
        "summary": summary,
        "messages": [AIMessage(content=summary)]
    }


def format_output_node(state: ResearchState) -> dict:
    summary = state["summary"]
    output = f"=== FINAL REPORT ===\nQuery: {state['query']}\n{summary}"
    return {
        "final_output": output,
        "messages": [AIMessage(content=output)]
    }


# ------------------------------------------------------------------ #
#  Build the Graph                                                     #
# ------------------------------------------------------------------ #

def build_agent():
    graph = StateGraph(ResearchState)

    graph.add_node("fetch", fetch_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("format_output", format_output_node)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_edge("summarize", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #

def run(poison: bool = False):
    global POISON_MODE
    POISON_MODE = poison

    label = "POISONED" if poison else "HEALTHY"
    print(f"\n{'='*50}")
    print(f"  Running agent in {label} mode")
    print(f"{'='*50}\n")

    session = TraceSession(
        session_id=f"test_{label.lower()}",
        traces_dir="traces"
    )
    interceptor = TraceInterceptor(session)

    agent = build_agent()

    result = agent.invoke(
        {"query": "What is the capital of France?", "messages": []},
        config={"callbacks": [interceptor]}
    )

    print("Agent output:")
    print(result["final_output"])
    print(f"\nTrace saved to: {session.output_path()}")

    # pretty-print the trace
    print(f"\nTrace events captured ({len(session.events)} total):")
    for i, event in enumerate(session.events):
        indent = "  " if event.parent_step_id else ""
        status = "✓" if event.success else "✗ ERROR"
        duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else "..."
        print(f"  {i+1:02d}. {indent}[{event.event_type}] {event.node_name} "
              f"(id={event.step_id}, parent={event.parent_step_id}) "
              f"{status} {duration}")

    return session


if __name__ == "__main__":
    # Run healthy first, then poisoned
    run(poison=False)
    run(poison=True)

    print("\n✓ Phase 1 complete — check the traces/ folder for .jsonl files")
