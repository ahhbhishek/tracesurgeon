"""
Test the Blame Scorer — Phase 3.

Run with:
    cd Projects/tracesurgeon
    python tests/test_scorer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon.dag import build_dag, build_dataflow_dag, print_tree
from tracesurgeon.scorer import run_blame_analysis


def test_blame(label: str, path: str):
    print(f"\n{'='*60}")
    print(f"  Blame Analysis — {label}")
    print(f"{'='*60}")

    flow = build_dataflow_dag(path)      # for blame
    tree = build_dag(path)               # for display
    result = run_blame_analysis(flow)

    if not result["has_failure"]:
        print("\n✓ No failure detected — run looks clean.")
        return

    sym = result["symptom"]
    rc = result["root_cause"]

    print(f"\n✗ Symptom (where it's visible):  {sym['node_name']} ({sym['step_id'][:6]})")
    print(f"    output: {sym['outputs_preview']!r}")

    print(f"\nBlame Ranking:")
    print(f"  {'#':<3} {'Score':<7} {'Node':<22} Reasons")
    print(f"  {'-'*2} {'-'*6} {'-'*21} {'-'*40}")
    for i, s in enumerate(result["suspects"]):
        tag = " (symptom)" if s["is_symptom"] else ""
        print(f"  {i+1:<3} {s['score']:<7.3f} {s['node_name'][:21]:<22} "
              f"{', '.join(s['reasons'])}{tag}")

    print(f"\n  >>> ROOT CAUSE: {rc['node_name']} (score {rc['score']})")
    print(f"      {rc['outputs_preview']!r}")

    print(f"\nCall tree (root cause marked):")
    print_tree(tree, highlight=[rc["step_id"]])


if __name__ == "__main__":
    healthy = "traces/run_test_healthy.jsonl"
    poisoned = "traces/run_test_poisoned.jsonl"

    if not Path(healthy).exists() or not Path(poisoned).exists():
        print("ERROR: trace files not found. Run python tests/test_agent.py first.")
        sys.exit(1)

    test_blame("HEALTHY run", healthy)
    test_blame("POISONED run", poisoned)

    print("\n\n✓ Phase 3 complete — blame flows back to the origin, not the symptom.")
