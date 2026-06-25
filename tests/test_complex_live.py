"""
COMPLEX live agent scenarios — real LLM (OpenRouter), real create_react_agent.

These go beyond the toy "add two populations" demo and stress what production
agents actually hit:

  1. DEEP PROPAGATION  — a 5-tool financial pipeline where an early tool errors;
     blame must flow back past 3-4 downstream nodes to the true origin.

  2. SILENT WRONG-DATA — a tool returns plausible-but-wrong data with NO error
     markers, silently corrupting the final number. Signal detection sees nothing
     (honest limitation), but counterfactual verification with a `check=` predicate
     PROVES which tool caused the wrong answer. This is the unique capability.

  3. MULTI-FAULT       — two independent tools are poisoned; both must surface.

Setup (in your terminal, not chat):
    export OPENROUTER_API_KEY=sk-or-...
    export OPENROUTER_MODEL=openai/gpt-oss-20b:free   # optional; this is the default

Run:
    python tests/test_complex_live.py
"""

import os
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()
warnings.filterwarnings("ignore")

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from tracesurgeon import instrument, diagnose


def _model():
    if not os.getenv("OPENROUTER_API_KEY"):
        return None, None
    from langchain_openai import ChatOpenAI
    m = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    return ChatOpenAI(model=m, temperature=0,
                      base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"]), m


# ============================================================== #
#  Scenario 1 — DEEP PROPAGATION (5-tool financial pipeline)     #
# ============================================================== #
# get_revenue -> get_cost -> profit -> margin -> grade
# get_cost is poisoned (HTTP 503). The error must be blamed on get_cost, not on
# the profit/margin/grade tools that merely propagated it.

@tool
def get_revenue(quarter: str) -> str:
    """Return revenue for a quarter, in millions USD."""
    return "1200"


@tool
def get_cost(quarter: str) -> str:
    """Return total cost for a quarter, in millions USD."""
    return "ERROR: finance DB connection refused (HTTP 503)"   # poisoned origin


@tool
def profit(revenue: str, cost: str) -> str:
    """Compute profit = revenue - cost (both in millions)."""
    try:
        return str(float(revenue) - float(cost))
    except Exception as e:
        return f"ERROR: cannot compute profit from {revenue!r},{cost!r}: {e}"


@tool
def margin(profit_value: str, revenue: str) -> str:
    """Compute margin percent = profit / revenue * 100."""
    try:
        return f"{float(profit_value)/float(revenue)*100:.1f}%"
    except Exception as e:
        return f"ERROR: cannot compute margin: {e}"


@tool
def grade(margin_percent: str) -> str:
    """Grade the margin: A (>20%), B (>10%), else C."""
    try:
        v = float(margin_percent.strip().rstrip('%'))
        return "A" if v > 20 else "B" if v > 10 else "C"
    except Exception as e:
        return f"ERROR: cannot grade {margin_percent!r}: {e}"


def scenario_deep_propagation():
    model, label = _model()
    agent = create_react_agent(model, tools=[get_revenue, get_cost, profit, margin, grade])
    q = ("For quarter Q3, get revenue and cost, then compute profit, then margin, "
         "then grade the margin. Report the final grade.")
    inputs = {"messages": [("user", q)]}

    print("\n" + "#" * 64)
    print(f"# SCENARIO 1 — DEEP PROPAGATION (5-tool chain)   [{label}]")
    print("#" * 64)
    inst = instrument(session_id="cx_deep")
    try:
        agent.invoke(inputs, config=inst.config)
    except Exception as e:
        print(f"  (agent raised {type(e).__name__})")
    diag = diagnose(inst.path)
    if diag.has_failure:
        rc = diag.root_cause["node_name"]
        print(f"  -> root cause: {rc}  ({diag.root_cause['remediation']['category']})")
        ok = "get_cost" in rc
        print(f"  -> EXPECT tool:get_cost (the origin, not profit/margin/grade): "
              f"{'PASS' if ok else 'FAIL — got ' + rc}")
        return ok
    print("  -> no failure detected (unexpected)")
    return False


# ============================================================== #
#  Scenario 2 — SILENT WRONG-DATA (no error markers)             #
# ============================================================== #
# convert_currency returns a plausible-but-WRONG rate (no error string).
# The final total is wrong but looks fine; detection sees nothing.
# We PROVE the cause with a check= predicate.

@tool
def get_price_usd(item: str) -> str:
    """Return the USD price of an item."""
    return "100"


@tool
def usd_to_eur(amount_usd: str) -> str:
    """Convert USD to EUR. (silently wrong: uses 10.0 instead of ~0.92)"""
    return str(float(amount_usd) * 10.0)   # WRONG rate, but no error text


def scenario_silent_wrong_data():
    model, label = _model()
    agent = create_react_agent(model, tools=[get_price_usd, usd_to_eur])
    q = ("Get the USD price of 'widget', convert it to EUR, and report the EUR "
         "amount as a plain number.")
    inputs = {"messages": [("user", q)]}

    print("\n" + "#" * 64)
    print(f"# SCENARIO 2 — SILENT WRONG-DATA (no error markers)   [{label}]")
    print("#" * 64)
    inst = instrument(session_id="cx_silent")
    result = agent.invoke(inputs, config=inst.config)
    final = result["messages"][-1].content
    print(f"  agent's final answer: {final!r}")

    diag = diagnose(inst.path)
    print(f"  -> detection says: "
          f"{'FAILURE' if diag.has_failure else 'no failure (correct — no error markers)'}")

    # correct answer is ~92 EUR (100 * 0.92); the wrong tool gives 1000.
    def looks_correct(res):
        text = res["messages"][-1].content
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", text)]
        return any(80 <= n <= 100 for n in nums)   # near 92

    print("  -> proving causation via counterfactual check= (correct rate 0.92):")
    proof = diag.verify(
        lambda cfg: agent.invoke(inputs, config=cfg),
        replacement=str(100 * 0.92),     # the corrected conversion output
        tool="usd_to_eur",
        check=looks_correct,
    )
    proof.print()
    ok = proof.verdict == "CONFIRMED"
    print(f"  -> EXPECT CONFIRMED (patching usd_to_eur fixes the answer): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================== #
#  Scenario 3 — MULTI-FAULT (two independent poisoned tools)     #
# ============================================================== #

@tool
def weather(city: str) -> str:
    """Weather for a city (poisoned)."""
    return "ERROR: weather API timed out"


@tool
def traffic(city: str) -> str:
    """Traffic for a city (also poisoned)."""
    return "ERROR: traffic service returned HTTP 500"


def scenario_multi_fault():
    model, label = _model()
    agent = create_react_agent(model, tools=[weather, traffic])
    q = ("Check both the weather and the traffic for Tokyo, then summarize whether "
         "it's a good day to travel.")
    inputs = {"messages": [("user", q)]}

    print("\n" + "#" * 64)
    print(f"# SCENARIO 3 — MULTI-FAULT (two poisoned tools)   [{label}]")
    print("#" * 64)
    inst = instrument(session_id="cx_multi")
    try:
        agent.invoke(inputs, config=inst.config)
    except Exception as e:
        print(f"  (agent raised {type(e).__name__})")
    diag = diagnose(inst.path)
    rep = diag.to_dict()
    # collect distinct errored tool nodes among suspects
    bad_tools = {s["node_name"] for s in rep["suspects"]
                 if s["node_name"].startswith("tool:") and s["score"] >= 0.8}
    print(f"  -> high-confidence faulty tools: {sorted(bad_tools)}")
    ok = any("weather" in t for t in bad_tools) and any("traffic" in t for t in bad_tools)
    print(f"  -> EXPECT both tool:weather and tool:traffic flagged: "
          f"{'PASS' if ok else 'PARTIAL/FAIL'}")
    return ok


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY in your terminal (not in chat) and re-run.")
        sys.exit(2)

    results = {}
    for name, fn in [("deep_propagation", scenario_deep_propagation),
                     ("silent_wrong_data", scenario_silent_wrong_data),
                     ("multi_fault", scenario_multi_fault)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  scenario {name} threw: {type(e).__name__}: {e}")
            results[name] = False

    print("\n" + "=" * 64)
    passed = sum(1 for v in results.values() if v)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'CHECK'}  {k}")
    print(f"  {passed}/{len(results)} complex scenarios passed")
    print("=" * 64)
    sys.exit(0 if passed == len(results) else 1)
