# TraceSurgeon

Causal, dependency-aware debugging for LangGraph agents.

Standard attribution tools suffer from **blame hoarding** — they blame the final,
loudest reasoning step for a failure, when the real cause was a quiet poisoned
tool output several steps upstream. TraceSurgeon flows the blame *backward*
through the agent's data-flow graph to the node that actually **introduced** the
error.

## Status

| Phase | Feature | State |
|-------|---------|-------|
| 1 | Trace capture (LangGraph callback hook) | ✅ |
| 2 | DAG builder (call-tree + data-flow views) | ✅ |
| 3 | Blame scorer (anti-blame-hoarding) | ✅ |
| 4 | CLI + UI | planned |
| 5 | Semantic Edge Inferrer | planned |
| 6 | Counterfactual verification | planned |

## How it works

```
LangGraph agent
   │  TraceInterceptor (callback handler)
   ▼
traces/run_*.jsonl          # every node's inputs, outputs, timing, errors
   │  build_dataflow_dag()
   ▼
data-flow DAG               # tool → step1 → step2 → step3
   │  run_blame_analysis()
   ▼
ranked root-cause suspects  # the INTRODUCER wins, not the symptom
```

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_agent.py     # produces healthy + poisoned traces
python tests/test_scorer.py    # runs the full blame analysis
```

## Layout

```
tracesurgeon/
  interceptor.py   # LangGraph callback → trace events
  trace.py         # TraceEvent + TraceSession (jsonl writer)
  dag.py           # call-tree and data-flow graph builders
  scorer.py        # failure detection + blame ranking
tests/
  test_agent.py    # fake agents (no API key needed)
  test_dag.py
  test_scorer.py
```
