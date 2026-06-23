"""
Test the DAG builder — Phase 2.

Reads the .jsonl files produced by test_agent.py and builds graphs from them.

Run with:
    cd Projects/tracesurgeon
    python tests/test_dag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon.dag import build_dag, find_failure_path, print_tree, summarize_dag


def test_trace(label: str, path: str):
    print(f"\n{'='*55}")
    print(f"  DAG Analysis — {label}")
    print(f"{'='*55}")

    dag = build_dag(path)
    stats = summarize_dag(dag)

    print(f"\nStats:")
    print(f"  Total steps  : {stats['total_steps']}")
    print(f"  Tool calls   : {stats['tool_calls']}")
    print(f"  Failed steps : {stats['failed_steps']}")
    print(f"  Total time   : {stats['total_duration_ms']}ms")
    print(f"  Has failure  : {stats['has_failure']}")

    blame_path = find_failure_path(dag)

    print(f"\nExecution Tree:")
    print_tree(dag, highlight_path=blame_path)

    if blame_path:
        print(f"\nBlame chain ({len(blame_path)} steps to root cause):")
        for i, node_id in enumerate(blame_path):
            data = dag.nodes[node_id]
            name = data.get("node_name", node_id)
            outputs_preview = str(data.get("outputs", ""))[:80]
            prefix = "  ROOT →" if i == 0 else ("  CAUSE→" if i == len(blame_path) - 1 else "        ")
            print(f"  {prefix} [{i+1}] {name}  outputs: {outputs_preview!r}")
    else:
        print("\n✓ No failures detected in this trace.")


if __name__ == "__main__":
    healthy = "traces/run_test_healthy.jsonl"
    poisoned = "traces/run_test_poisoned.jsonl"

    if not Path(healthy).exists() or not Path(poisoned).exists():
        print("ERROR: trace files not found.")
        print("Run  python tests/test_agent.py  first.")
        sys.exit(1)

    test_trace("HEALTHY run", healthy)
    test_trace("POISONED run", poisoned)

    print("\n\n✓ Phase 2 complete — DAG builder works on both traces.")
