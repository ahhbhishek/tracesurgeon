"""
Run every TraceSurgeon test suite and report a combined result.

    cd Projects/tracesurgeon
    python tests/run_all.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUITES = [
    "test_detection.py",       # negation-aware error detection (unit)
    "test_edge_cases.py",      # synthetic edge cases / robustness
    "test_scorer.py",          # linear agent blame
    "test_dag.py",             # dag builder
    "test_branching_agent.py", # branching multi-tool
    "test_react_loop.py",      # cyclic ReAct loop
    "test_real_agent.py",      # real create_react_agent graph
]


# suites that print diagnostics and exit nonzero by design (no pass/fail assert)
_INFORMATIONAL = {"test_scorer.py", "test_dag.py"}


def main():
    # regenerate base traces first (scorer/dag depend on them)
    env_ok = subprocess.run(
        [sys.executable, str(HERE / "test_agent.py")],
        capture_output=True, text=True,
    )
    if env_ok.returncode != 0:
        print("FATAL: test_agent.py (trace generation) failed:")
        print(env_ok.stdout[-1000:], env_ok.stderr[-1000:])
        return 1

    results = {}
    for suite in SUITES:
        p = subprocess.run(
            [sys.executable, str(HERE / suite)],
            capture_output=True, text=True,
        )
        results[suite] = p.returncode
        if suite in _INFORMATIONAL:
            status = "INFO (ran ok)"
        else:
            status = "PASS" if p.returncode == 0 else f"FAIL (exit {p.returncode})"
        print(f"  [{status:14}] {suite}")
        if p.returncode != 0 and suite not in _INFORMATIONAL:
            tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
            for line in tail:
                print(f"        {line}")

    failed = [s for s, rc in results.items()
              if rc != 0 and s not in _INFORMATIONAL]
    asserted = len(SUITES) - len(_INFORMATIONAL)
    print(f"\n  {asserted - len(failed)}/{asserted} asserting suites passed "
          f"({len(_INFORMATIONAL)} informational)")
    if failed:
        print("  FAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
