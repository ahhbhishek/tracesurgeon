"""
Counterfactual verification — Phase 6.

Diagnosis tells you which node *correlates* with a failure. Counterfactual
verification turns that into *proof*: substitute the suspect tool's output with a
corrected value, re-run the real agent, and see whether the failure disappears.

    FAIL  ──(patch only this one output)──>  PASS   ⇒  causal proof

Mechanism: `BaseTool.run` / `arun` are the single chokepoints every LangGraph
tool call goes through (invoke→run, ainvoke→arun, ToolNode uses invoke/ainvoke).
We temporarily wrap them so a tool whose name is in the patch returns the patched
value directly; every other tool runs for real. The patched value flows through
the real downstream graph, which executes and is traced normally — then we
diagnose the new run and compare.

    from tracesurgeon import diagnose, counterfactual

    diag = diagnose(inst.path)                       # "tool:get_population looks guilty"
    proof = counterfactual(
        lambda cfg: agent.invoke(inputs, config=cfg),
        patch={"get_population": "Paris: 2.1 million"},
        baseline=diag,
    )
    proof.print()                                    # CONFIRMED — failure flipped
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from .api import instrument, diagnose, Diagnosis
from . import report as _report

_MARKUP_RE = re.compile(r"(?<!\\)\[/?[a-z0-9_ #]+\]")


def _strip_tool_prefix(name: str) -> str:
    return name[5:] if name.startswith("tool:") else name


def _coerce(value: Any) -> Any:
    """Patch value may be a static value or a zero-arg callable."""
    return value() if callable(value) else value


# ------------------------------------------------------------------ #
#  Tool-output substitution                                          #
# ------------------------------------------------------------------ #

def _substitute(self, value: str, kwargs: dict):
    """
    Return the patched output in the shape the caller expects, WITHOUT executing
    the real tool and WITHOUT mutating any shared state (so this is fully
    thread-safe / async-safe, no locks).

    - When invoked by a ToolNode the input is a ToolCall and `tool_call_id` is in
      kwargs; the caller expects a ToolMessage, so we build one (this is what
      prebuilt create_react_agent / ToolNode require).
    - Otherwise (e.g. a node calling `tool.invoke({...})` directly) the raw value
      is what's expected.
    """
    tool_call_id = kwargs.get("tool_call_id")
    if tool_call_id is not None:
        from langchain_core.messages import ToolMessage
        return ToolMessage(content=value, tool_call_id=tool_call_id,
                           name=getattr(self, "name", None))
    return value


@contextmanager
def _patch_tool_outputs(patch: dict):
    """
    Temporarily make any tool whose `.name` is in `patch` return the patched
    value instead of executing, then restore the originals on exit.

    Wraps the global BaseTool.run / arun chokepoints (every tool call goes through
    these: invoke→run, ainvoke→arun, ToolNode uses invoke/ainvoke). The patched
    value flows into the downstream graph, which runs for real and is traced
    normally. Stateless short-circuit → no locks, safe under parallel/async tool
    calls. (The patched tool's own span is absent from the new trace; the
    outcome — what verification measures — is unaffected.)
    """
    from langchain_core.tools import BaseTool

    names = {_strip_tool_prefix(k): v for k, v in patch.items()}
    orig_run = BaseTool.run
    orig_arun = BaseTool.arun

    def run(self, *args, **kwargs):
        if getattr(self, "name", None) in names:
            return _substitute(self, str(_coerce(names[self.name])), kwargs)
        return orig_run(self, *args, **kwargs)

    async def arun(self, *args, **kwargs):
        if getattr(self, "name", None) in names:
            return _substitute(self, str(_coerce(names[self.name])), kwargs)
        return await orig_arun(self, *args, **kwargs)

    BaseTool.run = run
    BaseTool.arun = arun
    try:
        yield
    finally:
        BaseTool.run = orig_run
        BaseTool.arun = orig_arun


# ------------------------------------------------------------------ #
#  Result                                                            #
# ------------------------------------------------------------------ #

@dataclass
class Counterfactual:
    patch: dict
    patched: dict                      # report dict of the patched run
    patched_trace_path: str
    baseline: dict | None = None       # report dict of the original run
    verdict: str = "INCONCLUSIVE"      # CONFIRMED | PARTIAL | NOT_CONFIRMED | INCONCLUSIVE
    flipped: bool = False

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "flipped": self.flipped,
            "patch": {k: str(v) for k, v in self.patch.items()},
            "patched_trace_path": self.patched_trace_path,
            "baseline": self.baseline,
            "patched": self.patched,
        }

    def print(self) -> None:
        def p(line: str = ""):
            print(_MARKUP_RE.sub("", str(line)).replace("\\[", "["))

        print("\n" + "=" * 62)
        print("  TraceSurgeon — Counterfactual Verification")
        print("=" * 62)

        patched_kv = ", ".join(f"{k} = {str(v)[:60]!r}" for k, v in self.patch.items())
        p(f"  patched output:  {patched_kv}")

        before = "FAIL " if (self.baseline and self.baseline.get("has_failure")) else "clean"
        after = "FAIL " if self.patched.get("has_failure") else "clean"
        b_rc = (self.baseline or {}).get("root_cause")
        a_rc = self.patched.get("root_cause")
        p(f"  before:  {before}   "
          + (f"root cause = {b_rc['node_name']}" if b_rc else ""))
        p(f"  after:   {after}   "
          + (f"new root cause = {a_rc['node_name']}" if a_rc else "(no failure)"))
        p()

        banner = {
            "CONFIRMED":
                "  ✅ CONFIRMED — fixing this one output flips the run to clean.\n"
                "     This is causal proof, not correlation.",
            "PARTIAL":
                "  🟡 PARTIALLY CONFIRMED — the original cause is resolved, but a\n"
                "     different, independent failure surfaced downstream.",
            "NOT_CONFIRMED":
                "  ❌ NOT CONFIRMED — the run still fails the same way. This output\n"
                "     was likely NOT the true cause (or not the only one).",
            "INCONCLUSIVE":
                "  ⚪ INCONCLUSIVE — the baseline run had no failure to verify.",
        }.get(self.verdict, f"  verdict: {self.verdict}")
        p(banner)
        p()


# ------------------------------------------------------------------ #
#  Engine                                                            #
# ------------------------------------------------------------------ #

def _as_report(baseline) -> dict | None:
    """Normalise baseline (Diagnosis | report dict | trace path) to a report dict.
    A bad trace path raises rather than silently degrading to INCONCLUSIVE."""
    if baseline is None:
        return None
    if isinstance(baseline, Diagnosis):
        return baseline.to_dict()
    if isinstance(baseline, dict):
        return baseline
    if isinstance(baseline, str):
        return diagnose(baseline).to_dict()  # propagate a real FileNotFoundError
    raise TypeError(f"baseline must be a Diagnosis, report dict, trace path, or "
                    f"None — got {type(baseline).__name__}")


def counterfactual(
    run_fn: Callable[[dict], Any],
    patch: dict,
    *,
    baseline=None,
    traces_dir: str = "traces",
    session_id: str | None = None,
) -> Counterfactual:
    """
    Re-run an agent with one or more tool outputs replaced, and report whether the
    failure flipped.

    run_fn(config): a thunk that runs your agent with the given config injected,
                    e.g. `lambda cfg: agent.invoke(inputs, config=cfg)`.
    patch:          {tool_name: replacement_value_or_callable}.  "tool:" prefix
                    on the name is accepted and stripped.
    baseline:       a Diagnosis, a report dict, or a trace path of the original
                    (failing) run, used for the before/after comparison.
    """
    base_rep = _as_report(baseline)

    inst = instrument(session_id=session_id or "counterfactual", traces_dir=traces_dir)
    with _patch_tool_outputs(patch):
        run_fn(inst.config)
    patched = diagnose(inst.path).to_dict()

    cf = Counterfactual(
        patch=patch,
        patched=patched,
        patched_trace_path=inst.path,
        baseline=base_rep,
    )

    base_failed = bool(base_rep and base_rep.get("has_failure"))
    if not base_failed:
        cf.verdict, cf.flipped = "INCONCLUSIVE", False
        return cf

    if not patched.get("has_failure"):
        cf.verdict, cf.flipped = "CONFIRMED", True
        return cf

    # patched run still fails — same cause (patch didn't help) or a new one?
    base_cause = (base_rep.get("root_cause") or {}).get("node_name")
    new_cause = (patched.get("root_cause") or {}).get("node_name")
    if new_cause == base_cause:
        # identical attributed cause (including both None) → the patch changed nothing
        cf.verdict, cf.flipped = "NOT_CONFIRMED", False
    else:
        # the original cause is gone; a different failure surfaced downstream
        cf.verdict, cf.flipped = "PARTIAL", True
    return cf
