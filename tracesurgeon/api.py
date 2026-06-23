"""
Public API — the one-liner surface real users touch.

    from tracesurgeon import instrument, diagnose

    agent = build_my_langgraph_agent()
    session = instrument()                       # make a trace session + handler
    agent.invoke(inputs, config=session.config)  # run as normal

    report = diagnose(session.path)              # analyze the trace
    report.print()                               # human-readable root-cause report
"""

from dataclasses import dataclass

from .interceptor import TraceInterceptor
from .trace import TraceSession
from .dag import build_dataflow_dag, print_pipeline
from .scorer import run_blame_analysis


@dataclass
class Instrumentation:
    session: TraceSession
    handler: TraceInterceptor

    @property
    def path(self) -> str:
        return str(self.session.output_path())

    @property
    def config(self) -> dict:
        """Drop straight into agent.invoke(..., config=session.config)."""
        return {"callbacks": [self.handler]}


def instrument(session_id: str | None = None, traces_dir: str = "traces") -> Instrumentation:
    """Create a trace session + callback handler ready to attach to any agent."""
    session = TraceSession(session_id=session_id, traces_dir=traces_dir)
    return Instrumentation(session=session, handler=TraceInterceptor(session))


@dataclass
class Diagnosis:
    result: dict
    flow: object  # nx.DiGraph

    @property
    def has_failure(self) -> bool:
        return self.result["has_failure"]

    @property
    def root_cause(self) -> dict | None:
        return self.result.get("root_cause")

    def print(self) -> None:
        r = self.result
        print("\n" + "=" * 60)
        print("  TraceSurgeon — Root Cause Report")
        print("=" * 60)
        if not r["has_failure"]:
            print("\n  ✓ No failure detected — the run looks clean.\n")
            return

        rc = r["root_cause"]
        sym = r["symptom"]
        print(f"\n  Symptom (where it surfaced):  {sym['node_name']}")
        print(f"  >>> ROOT CAUSE:  {rc['node_name']}  (confidence {rc['score']:.0%})")
        print(f"      why: {', '.join(rc['reasons'])}")
        print(f"      output: {rc['outputs_preview']!r}")

        print(f"\n  Data-flow (root cause marked):")
        print_pipeline(self.flow, highlight=[rc["step_id"]])

        if len(r["suspects"]) > 1:
            print(f"\n  Other suspects:")
            for s in r["suspects"][1:4]:
                print(f"    {s['score']:.2f}  {s['node_name']}")
        print()


def diagnose(trace_path: str) -> Diagnosis:
    """Build the data-flow graph from a trace and run the full blame analysis."""
    flow = build_dataflow_dag(trace_path)
    result = run_blame_analysis(flow)
    return Diagnosis(result=result, flow=flow)
