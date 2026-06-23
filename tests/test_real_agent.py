"""
REAL-WORLD test — a production-style ReAct agent.

We build the agent with langgraph.prebuilt.create_react_agent — the exact
prebuilt production code uses. The graph, the ToolNode, the shared message
state, the multi-turn tool-calling loop, and the tool execution are all 100%
real. The only scripted part is token generation: a deterministic model that
emits genuine AIMessage tool_calls.

(Why scripted tokens? This environment's ANTHROPIC_API_KEY routes to Claude
Code, not the raw API, so direct model calls are gated. TraceSurgeon analyses
the execution TRACE, not the model's reasoning, so a scripted brain over a real
graph is a faithful — and reproducible — test.)

Scenario: "Combined population of the capitals of France and Japan?"
  turn 1: get_capital(France), get_capital(Japan)      -> Paris, Tokyo
  turn 2: get_population(Paris), get_population(Tokyo)  -> Paris OUTAGE (poison)
  turn 3: the model surfaces the failure in its final answer (downstream symptom)

TraceSurgeon must blame get_population (the Paris call) — the upstream origin —
not the model's final answer, which is where the error is most visible.

Run with:
    cd Projects/tracesurgeon
    python tests/test_real_agent.py
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()
warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from tracesurgeon import instrument, diagnose


POISON = True


# ------------------------------------------------------------------ #
#  Real tools                                                        #
# ------------------------------------------------------------------ #

@tool
def get_capital(country: str) -> str:
    """Return the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo", "germany": "Berlin"}
    return capitals.get(country.strip().lower(), f"Unknown country: {country}")


@tool
def get_population(city: str) -> str:
    """Return the population of a city, in millions."""
    if POISON and city.strip().lower() == "paris":
        return "ERROR: census API returned HTTP 503, data unavailable for this city"
    pops = {"paris": "2.1", "tokyo": "14.0", "berlin": "3.7"}
    val = pops.get(city.strip().lower())
    return f"{val} million" if val else f"No population data for {city}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '2.1 + 14.0'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"ERROR: invalid characters in expression: {expression!r}"
    try:
        return str(eval(expression))  # noqa: S307 - charset-sandboxed above
    except Exception as e:
        return f"ERROR: could not evaluate {expression!r}: {e}"


# ------------------------------------------------------------------ #
#  Deterministic tool-calling model (drives the REAL graph)          #
# ------------------------------------------------------------------ #

class ScriptedToolCaller(BaseChatModel):
    """A real BaseChatModel that emits scripted tool_calls so create_react_agent
    executes a genuine multi-turn loop without a live API."""

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-caller"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolCaller":
        return self  # tool schemas not needed for a scripted brain

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        ai_turns = sum(isinstance(m, AIMessage) for m in messages)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        if ai_turns == 0:
            msg = AIMessage(content="I'll look up both capitals.", tool_calls=[
                {"name": "get_capital", "args": {"country": "France"}, "id": "cap_fr"},
                {"name": "get_capital", "args": {"country": "Japan"}, "id": "cap_jp"},
            ])
        elif ai_turns == 1:
            msg = AIMessage(content="Now I'll look up their populations.", tool_calls=[
                {"name": "get_population", "args": {"city": "Paris"}, "id": "pop_paris"},
                {"name": "get_population", "args": {"city": "Tokyo"}, "id": "pop_tokyo"},
            ])
        elif ai_turns == 2:
            recent = " | ".join(str(m.content) for m in tool_msgs[-2:])
            if "ERROR" in recent:
                # the model parrots the failure into its answer — the loud symptom
                msg = AIMessage(content=(
                    "I could not compute the combined population: a population "
                    f"lookup failed. Raw tool output: {recent}"
                ))
            else:
                msg = AIMessage(content="Adding them up.", tool_calls=[
                    {"name": "calculator", "args": {"expression": "2.1 + 14.0"}, "id": "calc"},
                ])
        else:
            total = str(tool_msgs[-1].content) if tool_msgs else "unknown"
            msg = AIMessage(content=f"The combined population is {total} million.")

        return ChatResult(generations=[ChatGeneration(message=msg)])


# ------------------------------------------------------------------ #
#  Run                                                               #
# ------------------------------------------------------------------ #

def run(poison: bool):
    global POISON
    POISON = poison

    model = ScriptedToolCaller()
    agent = create_react_agent(model, tools=[get_capital, get_population, calculator])

    question = ("What is the combined population, in millions, of the capital "
                "cities of France and Japan?")

    print("=" * 62)
    print(f"  REAL create_react_agent graph — poison={'ON' if poison else 'OFF'}")
    print("=" * 62)

    inst = instrument(session_id=f"real_agent_{'poison' if poison else 'clean'}")
    result = agent.invoke({"messages": [("user", question)]}, config=inst.config)

    print(f"  Final answer:\n    {result['messages'][-1].content}\n")

    diag = diagnose(inst.path)
    diag.print()

    rc = diag.root_cause
    if poison:
        ok = rc is not None and "get_population" in rc["node_name"]
        print(f"  EXPECT root cause tool:get_population — GOT "
              f"{rc['node_name'] if rc else 'none'} — [{'PASS' if ok else 'FAIL'}]")
    else:
        ok = not diag.has_failure
        print(f"  EXPECT no failure — GOT "
              f"{'clean' if not diag.has_failure else 'false positive'} — "
              f"[{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    a = run(poison=True)
    b = run(poison=False)
    print(f"\n{'='*62}\n  REAL-AGENT RESULTS: {int(a) + int(b)}/2 passed\n{'='*62}")
    sys.exit(0 if (a and b) else 1)
