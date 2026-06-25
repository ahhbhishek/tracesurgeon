"""
LIVE end-to-end test — a REAL LLM (Google Gemini) driving a real LangGraph agent.

This is the validation no scripted test can substitute: a genuine model decides
which tools to call, reads their outputs, and writes the final answer. One tool
is poisoned (returns an outage error). We then:

    1. run the real agent           -> real trace
    2. diagnose()                   -> which tool is the root cause?
    3. verify()                     -> patch that tool, re-run, prove causality

The API key is read ONLY from the environment — it is never printed or stored.

Setup (in YOUR terminal, never in chat):
    export GOOGLE_API_KEY=<your-key>        # git-bash / linux / mac
    set GOOGLE_API_KEY=<your-key>           # windows cmd
    $env:GOOGLE_API_KEY="<your-key>"        # powershell

Run:
    python tests/test_real_agent_live.py

Cost: a few Gemini calls on a cheap flash model — fractions of a cent.
"""

import os
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


# ------------------------------------------------------------------ #
#  Real tools — get_population(Paris) is poisoned                     #
# ------------------------------------------------------------------ #

@tool
def get_capital(country: str) -> str:
    """Return the capital city of a country."""
    return {"france": "Paris", "japan": "Tokyo", "germany": "Berlin"}.get(
        country.strip().lower(), f"Unknown: {country}")


@tool
def get_population(city: str) -> str:
    """Return the population of a city in millions."""
    if city.strip().lower() == "paris":
        # the poisoned upstream — a realistic transient outage
        return "ERROR: census API returned HTTP 503, data unavailable for this city"
    return {"tokyo": "14.0 million", "berlin": "3.7 million"}.get(
        city.strip().lower(), f"No data for {city}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '2.1 + 14.0'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"ERROR: invalid characters in {expression!r}"
    try:
        return str(eval(expression))  # noqa: S307 - charset-sandboxed
    except Exception as e:
        return f"ERROR: could not evaluate {expression!r}: {e}"


QUESTION = ("What is the combined population, in millions, of the capital cities "
            "of France and Japan? Look up each capital, then each population, "
            "then add them with the calculator. Give the final number.")


def _make_model():
    """Pick whichever provider has a key in the environment. Cheap/free models."""
    # OpenRouter — OpenAI-compatible endpoint, free tool-capable models
    if os.getenv("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        ), f"OpenRouter {model}"
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0), "OpenAI gpt-4o-mini"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0), "Gemini 1.5 flash"
    return None, None


def main() -> int:
    model, label = _make_model()
    if model is None:
        print("No API key found. Set OPENROUTER_API_KEY / OPENAI_API_KEY / "
              "GOOGLE_API_KEY in your terminal (not in chat) and re-run.")
        return 2
    agent = create_react_agent(model, tools=[get_capital, get_population, calculator])
    inputs = {"messages": [("user", QUESTION)]}

    print("=" * 62)
    print(f"  LIVE: real {label} + real create_react_agent")
    print("=" * 62)
    print(f"  Q: {QUESTION}\n")

    # 1) run the real agent (crash-proof: diagnose even if the agent raises)
    inst = instrument(session_id="live_real")
    try:
        result = agent.invoke(inputs, config=inst.config)
        print("  Final answer from the agent:")
        print("   ", result["messages"][-1].content.replace("\n", "\n    "), "\n")
    except Exception as e:
        print(f"  (agent raised {type(e).__name__}: {str(e)[:120]} — diagnosing the "
              f"captured trace anyway)\n")

    # 2) diagnose
    diag = diagnose(inst.path)
    diag.print()

    # 3) verify (only if a failure was found and it's a tool)
    if diag.has_failure and diag.root_cause["node_name"].startswith("tool:"):
        proof = diag.verify(
            lambda config: agent.invoke(inputs, config=config),
            replacement="2.1 million",
        )
        proof.print()
        verdict = proof.verdict
    else:
        verdict = "N/A"

    print("=" * 62)
    if not diag.has_failure:
        print("  RESULT: the live run came back clean — the model may have")
        print("  recovered from the poisoned tool. Inspect the trace:")
        print(f"    tracesurgeon debug {inst.path}")
    else:
        rc = diag.root_cause["node_name"]
        print(f"  RESULT: root cause = {rc}   |   verification = {verdict}")
        print(f"  Trace: {inst.path}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
