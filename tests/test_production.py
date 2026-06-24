"""
Production-hardening tests:
  - async agents (ainvoke) capture correctly
  - concurrent/parallel tool calls don't corrupt the trace
  - a buggy/un-serializable payload never crashes the host agent
  - the callback never raises into the agent

Run with:
    cd Projects/tracesurgeon
    python tests/test_production.py
"""

import asyncio
import json
import operator
import sys
import threading
from pathlib import Path
from typing import TypedDict, Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from tracesurgeon import instrument, diagnose
from tracesurgeon.trace import TraceSession

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ------------------------------------------------------------------ #
#  1. Async agent capture                                            #
# ------------------------------------------------------------------ #

class S(TypedDict):
    q: str
    out: str
    messages: Annotated[list, operator.add]


@tool
def afetch(q: str) -> str:
    """async-used tool"""
    return "ERROR: async upstream failed 503"


def n1(state: S) -> dict:
    return {"out": afetch.invoke({"q": state["q"]}), "messages": [AIMessage(content="x")]}


def n2(state: S) -> dict:
    return {"out": f"final: {state['out']}", "messages": [AIMessage(content="y")]}


def build():
    g = StateGraph(S)
    g.add_node("step1", n1)
    g.add_node("step2", n2)
    g.set_entry_point("step1")
    g.add_edge("step1", "step2")
    g.add_edge("step2", END)
    return g.compile()


def test_async():
    agent = build()
    inst = instrument(session_id="prod_async")

    async def run():
        return await agent.ainvoke({"q": "hi", "messages": []}, config=inst.config)

    asyncio.run(run())
    # trace file must be valid and contain the tool
    events = TraceSession.load(inst.path)
    names = {e["node_name"] for e in events}
    check("async: trace captured", len(events) > 0)
    check("async: tool recorded", any("afetch" in n for n in names), str(names))
    diag = diagnose(inst.path)
    check("async: blames the tool",
          diag.has_failure and "afetch" in diag.root_cause["node_name"],
          diag.root_cause["node_name"] if diag.has_failure else "no failure")


# ------------------------------------------------------------------ #
#  2. Concurrency: hammer add_event from many threads                #
# ------------------------------------------------------------------ #

def test_concurrent_writes():
    sess = TraceSession(session_id="prod_concurrent")
    from tracesurgeon.trace import TraceEvent

    def worker(i):
        for j in range(50):
            sess.add_event(TraceEvent(
                "node_end", f"s{i}_{j}", None, f"node{i}",
                None, {"v": i * 1000 + j}, TraceSession.now(),
                duration_ms=1.0, success=True,
            ))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # every line must be valid JSON (no interleaving/corruption)
    raw = Path(sess.output_path()).read_text(encoding="utf-8").splitlines()
    valid = 0
    for line in raw:
        if not line.strip():
            continue
        try:
            json.loads(line)
            valid += 1
        except json.JSONDecodeError:
            pass
    expected = 8 * 50
    check("concurrent: no corrupted lines", valid == expected, f"{valid}/{expected} valid")
    check("concurrent: all events present", len(sess.events) == expected,
          f"{len(sess.events)}/{expected}")


# ------------------------------------------------------------------ #
#  3. Crash-proof: un-serializable payload must not crash the agent  #
# ------------------------------------------------------------------ #

class Exploding:
    """An object that raises whenever you touch it."""
    def __str__(self): raise RuntimeError("boom on str")
    def __repr__(self): raise RuntimeError("boom on repr")


@tool
def nasty(q: str) -> str:
    """returns a normal string but we'll smuggle a landmine into state"""
    return "ok"


def n_bad(state: S) -> dict:
    # put an exploding object into the message stream the tracer will serialize
    return {"out": "ok", "messages": [AIMessage(content="z", additional_kwargs={"x": Exploding()})]}


def test_crash_proof():
    g = StateGraph(S)
    g.add_node("step1", n_bad)
    g.set_entry_point("step1")
    g.add_edge("step1", END)
    agent = g.compile()

    inst = instrument(session_id="prod_crashproof")
    crashed = False
    try:
        result = agent.invoke({"q": "x", "messages": []}, config=inst.config)
        ok_result = result.get("out") == "ok"
    except Exception as e:
        crashed = True
        ok_result = False
    check("crash-proof: agent completed despite landmine payload", not crashed and ok_result)
    # the interceptor may have logged internal errors, but must not have raised
    check("crash-proof: trace file still written", Path(inst.path).exists())


# ------------------------------------------------------------------ #
#  4. _safe_serialize directly handles the landmine                  #
# ------------------------------------------------------------------ #

def test_serialize_landmine():
    from tracesurgeon.interceptor import _safe_serialize
    try:
        out = _safe_serialize({"bad": Exploding(), "good": "fine"})
        ok = out["good"] == "fine" and out["bad"] == "<unserializable>"
        check("serialize: landmine becomes <unserializable>", ok, str(out))
    except Exception as e:
        check("serialize: landmine becomes <unserializable>", False, str(e))


def test_parallel_fanout():
    """Two branches run concurrently from one node; trace must stay valid and
    blame must land on the poisoned branch."""
    import time

    class PS(TypedDict):
        q: str
        a: str
        b: str
        messages: Annotated[list, operator.add]

    @tool
    def good_branch(q: str) -> str:
        """clean branch"""
        time.sleep(0.02)
        return "clean data"

    @tool
    def bad_branch(q: str) -> str:
        """poisoned branch"""
        time.sleep(0.02)
        return "ERROR: branch timed out"

    def fan(state): return {"messages": [AIMessage(content="fan")]}
    def ba(state): return {"a": good_branch.invoke({"q": state["q"]}), "messages": [AIMessage(content="a")]}
    def bb(state): return {"b": bad_branch.invoke({"q": state["q"]}), "messages": [AIMessage(content="b")]}
    def join(state): return {"messages": [AIMessage(content="join")]}

    g = StateGraph(PS)
    for n, f in [("fan", fan), ("branchA", ba), ("branchB", bb), ("join", join)]:
        g.add_node(n, f)
    g.set_entry_point("fan")
    g.add_edge("fan", "branchA")
    g.add_edge("fan", "branchB")
    g.add_edge("branchA", "join")
    g.add_edge("branchB", "join")
    g.add_edge("join", END)
    agent = g.compile()

    inst = instrument(session_id="prod_parallel")
    agent.invoke({"q": "x", "messages": []}, config=inst.config)

    lines = [l for l in Path(inst.path).read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = 0
    for l in lines:
        try:
            json.loads(l); valid += 1
        except json.JSONDecodeError:
            pass
    check("parallel: trace integrity", valid == len(lines), f"{valid}/{len(lines)}")
    diag = diagnose(inst.path)
    check("parallel: blames poisoned branch",
          diag.has_failure and "bad_branch" in diag.root_cause["node_name"],
          diag.root_cause["node_name"] if diag.has_failure else "none")


if __name__ == "__main__":
    print("Production-hardening tests:\n")
    for fn in [test_async, test_concurrent_writes, test_parallel_fanout,
               test_crash_proof, test_serialize_landmine]:
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"threw {type(e).__name__}: {e}")
    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:", ", ".join(FAIL))
    sys.exit(0 if not FAIL else 1)
