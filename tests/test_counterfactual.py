"""
Counterfactual verification tests — Phase 6.

Real LangGraph agents (no API key). We poison a tool so it returns an error,
diagnose the failure, then PATCH that tool's output via the counterfactual engine
and check whether the failure flips — the difference between correlation and proof.

Run with:
    cd Projects/tracesurgeon
    python tests/test_counterfactual.py
"""

import asyncio
import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from tracesurgeon import instrument, diagnose, counterfactual

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ------------------------------------------------------------------ #
#  A small real agent with two tools, one poisoned                    #
# ------------------------------------------------------------------ #

class State(TypedDict):
    query: str
    capital: str
    population: str
    answer: str
    messages: Annotated[list, operator.add]


@tool
def get_capital(country: str) -> str:
    """Capital of a country."""
    return "Paris"


@tool
def get_population(city: str) -> str:
    """Population of a city (poisoned: always errors)."""
    return "ERROR: census API returned HTTP 503, data unavailable"


def n_capital(s: State) -> dict:
    return {"capital": get_capital.invoke({"country": s["query"]}),
            "messages": [AIMessage(content="cap")]}


def n_population(s: State) -> dict:
    return {"population": get_population.invoke({"city": s.get("capital", "?")}),
            "messages": [AIMessage(content="pop")]}


def n_answer(s: State) -> dict:
    pop = s.get("population", "")
    ans = ("I could not answer; the population lookup failed."
           if "ERROR" in pop else f"The population is {pop}.")
    return {"answer": ans, "messages": [AIMessage(content=ans)]}


def build_agent():
    g = StateGraph(State)
    g.add_node("capital", n_capital)
    g.add_node("population", n_population)
    g.add_node("answer", n_answer)
    g.set_entry_point("capital")
    g.add_edge("capital", "population")
    g.add_edge("population", "answer")
    g.add_edge("answer", END)
    return g.compile()


INPUTS = {"query": "France", "messages": []}


def _baseline():
    agent = build_agent()
    inst = instrument(session_id="cf_baseline")
    agent.invoke(INPUTS, config=inst.config)
    return agent, diagnose(inst.path)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #

def test_confirmed():
    agent, diag = _baseline()
    check("baseline blames the poisoned tool",
          diag.has_failure and "get_population" in diag.root_cause["node_name"],
          diag.root_cause["node_name"] if diag.has_failure else "none")

    proof = diag.verify(
        lambda cfg: agent.invoke(INPUTS, config=cfg),
        replacement="2.1 million",
    )
    check("CONFIRMED: patching the tool flips the run to clean",
          proof.verdict == "CONFIRMED" and proof.flipped,
          f"verdict={proof.verdict}")


def test_not_confirmed():
    agent, diag = _baseline()
    # patch an INNOCENT tool — the real cause remains
    proof = counterfactual(
        lambda cfg: agent.invoke(INPUTS, config=cfg),
        patch={"get_capital": "Lyon"},
        baseline=diag,
    )
    check("NOT_CONFIRMED: patching an innocent tool doesn't fix it",
          proof.verdict == "NOT_CONFIRMED" and not proof.flipped,
          f"verdict={proof.verdict}")


def test_direct_patch_value_and_callable():
    agent, diag = _baseline()
    # callable patch value
    proof = counterfactual(
        lambda cfg: agent.invoke(INPUTS, config=cfg),
        patch={"tool:get_population": lambda: "3.0 million"},  # 'tool:' prefix tolerated
        baseline=diag,
    )
    check("callable patch + 'tool:' prefix tolerated",
          proof.verdict == "CONFIRMED")


def test_async():
    agent = build_agent()
    inst = instrument(session_id="cf_async_base")

    async def base():
        await agent.ainvoke(INPUTS, config=inst.config)
    asyncio.run(base())
    diag = diagnose(inst.path)

    proof = diag.verify(
        lambda cfg: asyncio.run(agent.ainvoke(INPUTS, config=cfg)),
        replacement="2.1 million",
    )
    check("async agent: counterfactual flips to clean",
          proof.verdict == "CONFIRMED" and proof.flipped,
          f"verdict={proof.verdict}")


def test_to_dict_serializable():
    agent, diag = _baseline()
    proof = diag.verify(lambda cfg: agent.invoke(INPUTS, config=cfg),
                        replacement="2.1 million")
    try:
        s = json.dumps(proof.to_dict())
        check("Counterfactual.to_dict is JSON-serializable", len(s) > 0)
    except Exception as e:
        check("Counterfactual.to_dict is JSON-serializable", False, str(e))


def test_prebuilt_react_agent():
    """The prebuilt create_react_agent / ToolNode expects tool.invoke() to return
    a ToolMessage — verifies our func-swap preserves the correct return type."""
    from langchain_core.messages import ToolMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.prebuilt import create_react_agent

    @tool
    def population(city: str) -> str:
        """population lookup (poisoned)"""
        return "ERROR: census API HTTP 503 unavailable"

    class M(BaseChatModel):
        @property
        def _llm_type(self): return "m"
        def bind_tools(self, t, **k): return self
        def _generate(self, messages, stop=None, run_manager=None, **k):
            ai = sum(isinstance(m, AIMessage) for m in messages)
            tms = [m for m in messages if isinstance(m, ToolMessage)]
            if ai == 0:
                msg = AIMessage(content="", tool_calls=[
                    {"name": "population", "args": {"city": "Paris"}, "id": "p1"}])
            else:
                last = str(tms[-1].content) if tms else ""
                msg = AIMessage(content=("Could not answer: " + last)
                                if "ERROR" in last else f"Population is {last}.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    agent = create_react_agent(M(), tools=[population])
    inp = {"messages": [("user", "population of Paris?")]}

    inst = instrument(session_id="cf_prebuilt")
    agent.invoke(inp, config=inst.config)
    diag = diagnose(inst.path)

    proof = diag.verify(lambda cfg: agent.invoke(inp, config=cfg),
                        replacement="2.1 million")
    check("prebuilt create_react_agent: counterfactual CONFIRMED",
          proof.verdict == "CONFIRMED" and proof.flipped, f"verdict={proof.verdict}")


def test_silent_wrong_data_with_check():
    """A tool returns plausible-but-wrong data with NO error markers. Detection
    sees nothing; counterfactual with check= proves causation by output."""
    @tool
    def usd_to_eur(amount_usd: str) -> str:
        """convert USD to EUR (silently wrong rate 10.0 instead of 0.92)"""
        return str(float(amount_usd) * 10.0)

    class S(TypedDict):
        amount: str
        result: str
        messages: Annotated[list, operator.add]

    def convert(s):
        return {"result": usd_to_eur.invoke({"amount_usd": s["amount"]}),
                "messages": [AIMessage(content="converted")]}

    def report(s):
        return {"messages": [AIMessage(content=f"EUR total: {s['result']}")]}

    g = StateGraph(S)
    g.add_node("convert", convert)
    g.add_node("report", report)
    g.set_entry_point("convert")
    g.add_edge("convert", "report")
    g.add_edge("report", END)
    agent = g.compile()
    inp = {"amount": "100", "messages": []}

    inst = instrument(session_id="cf_silent")
    agent.invoke(inp, config=inst.config)
    diag = diagnose(inst.path)
    # honest: no error markers -> detection finds nothing
    check("silent failure: detection correctly finds no error", not diag.has_failure)

    def correct(res):
        import re
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", res["messages"][-1].content)]
        return any(80 <= n <= 100 for n in nums)  # near 92

    proof = counterfactual(
        lambda cfg: agent.invoke(inp, config=cfg),
        patch={"usd_to_eur": str(100 * 0.92)},
        check=correct,
    )
    check("silent failure: check= proves causation (CONFIRMED)",
          proof.verdict == "CONFIRMED" and proof.checked, f"verdict={proof.verdict}")

    # patching to a STILL-wrong value must NOT confirm
    proof2 = counterfactual(
        lambda cfg: agent.invoke(inp, config=cfg),
        patch={"usd_to_eur": "5000"},
        check=correct,
    )
    check("silent failure: wrong patch -> NOT_CONFIRMED",
          proof2.verdict == "NOT_CONFIRMED")


def test_inconclusive_on_clean_baseline():
    # baseline with no failure -> nothing to verify
    @tool
    def good(country: str) -> str:
        """clean tool"""
        return "Paris"

    class S2(TypedDict):
        query: str
        out: str
        messages: Annotated[list, operator.add]

    def n(s):
        return {"out": good.invoke({"country": s["query"]}), "messages": [AIMessage(content="x")]}

    g = StateGraph(S2)
    g.add_node("only", n)
    g.set_entry_point("only")
    g.add_edge("only", END)
    agent = g.compile()

    inst = instrument(session_id="cf_clean")
    agent.invoke({"query": "France", "messages": []}, config=inst.config)
    diag = diagnose(inst.path)

    proof = counterfactual(
        lambda cfg: agent.invoke({"query": "France", "messages": []}, config=cfg),
        patch={"good": "Lyon"},
        baseline=diag,
    )
    check("INCONCLUSIVE when baseline is clean", proof.verdict == "INCONCLUSIVE")


if __name__ == "__main__":
    print("Counterfactual verification tests:\n")
    for fn in [test_confirmed, test_not_confirmed, test_direct_patch_value_and_callable,
               test_async, test_to_dict_serializable, test_prebuilt_react_agent,
               test_silent_wrong_data_with_check, test_inconclusive_on_clean_baseline]:
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"threw {type(e).__name__}: {e}")

    # show one full report for eyeballing
    try:
        agent, diag = _baseline()
        diag.verify(lambda cfg: agent.invoke(INPUTS, config=cfg),
                    replacement="2.1 million").print()
    except Exception:
        pass

    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:", ", ".join(FAIL))
    sys.exit(0 if not FAIL else 1)
