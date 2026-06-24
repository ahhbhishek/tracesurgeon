"""
Public API — the one-liner surface real users touch.

    from tracesurgeon import instrument, diagnose

    agent = build_my_langgraph_agent()
    session = instrument()                       # make a trace session + handler
    agent.invoke(inputs, config=session.config)  # run as normal

    report = diagnose(session.path)              # analyze the trace
    report.print()                               # human-readable root-cause report
"""

import re
from dataclasses import dataclass

from .interceptor import TraceInterceptor
from .trace import TraceSession
from .dag import build_dataflow_dag
from .scorer import run_blame_analysis
from . import report as _report

# strip our own markup tags but NOT escaped user brackets (\[)
_MARKUP_RE = re.compile(r"(?<!\\)\[/?[a-z0-9_ #]+\]")


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
        return bool(self.result.get("has_failure"))

    @property
    def root_cause(self) -> dict | None:
        """The enriched root-cause dict (node_name, score, reasons, input_preview,
        output_full, timestamp, remediation). None if the run is clean."""
        return self.to_dict().get("root_cause")

    def to_dict(self) -> dict:
        """Full, JSON-serializable report (inputs, full output, timestamps,
        remediation, pipeline). Ideal for programmatic use / CI."""
        return _report.to_report_dict(self.result, self.flow)

    def print(self) -> None:
        """Human-readable root-cause report (plain text; markup stripped)."""
        def p(line: str = ""):
            print(_MARKUP_RE.sub("", str(line)).replace("\\[", "["))

        print("\n" + "=" * 62)
        print("  TraceSurgeon — Root Cause Report")
        print("=" * 62)
        _report.render(self.to_dict(), self.flow, p)


def diagnose(trace_path: str) -> Diagnosis:
    """Build the data-flow graph from a trace and run the full blame analysis."""
    flow = build_dataflow_dag(trace_path)
    result = run_blame_analysis(flow)
    return Diagnosis(result=result, flow=flow)
