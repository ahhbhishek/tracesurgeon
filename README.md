# TraceSurgeon

**Causal, dependency-aware root-cause debugging for LangGraph agents.**

When an agent fails, standard tools blame the final, loudest reasoning step —
"the model gave a bad answer". But the *real* cause is usually a quiet poisoned
tool output several steps upstream. This is **blame hoarding**. TraceSurgeon
flows blame *backward* through the agent's data-flow graph to the node that
actually **introduced** the error — and tells you what to do about it.

```
ROOT CAUSE
  node:     tool:get_population   (confidence 100%)
  when:     2026-06-24T16:25:57Z
  why:      INTRODUCED the error (inputs were clean), errored tool call
  input:    {'city': 'Paris'}
  output:   "ERROR: census API returned HTTP 503, data unavailable"
  fix →     Upstream service error (5xx / overloaded). Retry with backoff.  (upstream_5xx)
  symptom:  surfaced at agent      ← where a naive tool would have blamed
```

## Install

```bash
pip install -e .                    # gives you the `tracesurgeon` command
# optional: pip install -e ".[anthropic]"   # to drive a real Claude model
```

## Use it on your agent in 60 seconds

Three lines around your existing `agent.invoke(...)`:

```python
from tracesurgeon import instrument, diagnose

inst = instrument()                              # 1. make a tracer
agent.invoke(inputs, config=inst.config)         # 2. run your agent as normal
diagnose(inst.path).print()                      # 3. get the root-cause report
```

`config=inst.config` just adds a callback handler — it changes nothing about how
your agent runs. Works with any LangGraph agent (custom `StateGraph`,
`create_react_agent`, sync or async). A runnable copy is in
[`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py
```

### Counterfactual verification — causal *proof*

Diagnosis tells you which node *correlates* with the failure. **Verification
proves it**: it re-runs your agent with the suspect tool's output replaced by a
corrected value. If the failure disappears when (and only when) you fix that one
output, that's causal proof — not a guess.

```python
diag = diagnose(inst.path)                      # "tool:get_population looks guilty"

proof = diag.verify(
    lambda config: agent.invoke(inputs, config=config),   # re-runnable thunk
    replacement="Paris: 2.1 million",                     # the corrected output
)
proof.print()
```

```
before:  FAIL    root cause = tool:get_population
after:   clean   (no failure)
✅ CONFIRMED — fixing this one output flips the run to clean. Causal proof, not correlation.
```

Verdicts: **CONFIRMED** (clean after patch) · **PARTIAL** (original cause gone, a
new one surfaced) · **NOT_CONFIRMED** (still fails the same way → wrong
hypothesis) · **INCONCLUSIVE** (baseline had no failure). This catches the one
thing signal-detection can't: a tool returning *plausible-but-wrong* data with no
error markers — patch it, see if the outcome changes.

### Programmatic / CI use

`diagnose()` returns a `Diagnosis` you can inspect — no printing required:

```python
diag = diagnose(inst.path)
if diag.has_failure:
    rc = diag.root_cause
    print(rc["node_name"], rc["score"], rc["remediation"]["category"])

report = diag.to_dict()    # full JSON-serializable report (inputs, output, pipeline…)
```

## CLI

```bash
tracesurgeon debug [trace.jsonl]        # root-cause report (default: newest trace)
tracesurgeon debug --json [trace.jsonl] # machine-readable JSON (pipe to jq, use in CI)
tracesurgeon list [--dir traces]        # all traces + clean/failure status
tracesurgeon list --json                # same, as JSON
tracesurgeon show [trace.jsonl]         # raw execution tree (no scoring)
```

- If you omit the trace path, the **newest** trace in `traces/` is used.
- `debug` **exits non-zero** when a failure is detected, so it drops into CI:
  `tracesurgeon debug && echo "clean"`.
- Traces default to a `traces/` directory (auto-created). Change it with
  `instrument(traces_dir="...")` or the `--dir` flag.

## What you get

| Field | Meaning |
|-------|---------|
| **root cause** | the node that *introduced* the error (not where it surfaced) |
| **confidence** | how strongly the evidence points here (0–100%) |
| **input / output** | the actual data in and out of the failing node |
| **fix →** | a remediation hint, categorised (rate_limit, timeout, auth, upstream_5xx, bad_data, network, not_found, context_length, resource, process) |
| **symptom** | where the failure became visible — what a naive tool would blame |
| **blame ranking** | every suspect, scored, with the reasons |
| **data-flow** | the execution pipeline with the root cause marked |

## How it works

```
LangGraph agent
   │  TraceInterceptor (callback handler — thread-safe, crash-proof)
   ▼
traces/run_*.jsonl          # every node's inputs, outputs, timing, errors
   │  build_dataflow_dag()  # reconstructs data flow from timestamps
   ▼
data-flow DAG               # tool → step1 → step2 → step3
   │  run_blame_analysis()  # the INTRODUCER (clean inputs, errored output) wins
   ▼
report                      # ranked root cause + remediation
```

The core idea: a node is the **introducer** if it has an error but *none of its
upstream ancestors do*. That single rule is what flows blame past the loud
downstream symptom to the quiet origin. Then `verify()` re-runs the agent with
that node's output corrected to **prove** the link causally.

## Production-hardened

- **Thread-safe** — concurrent/parallel/async (`ainvoke`) runs never corrupt the
  trace (verified: 400 events / 8 threads, 0 corruption).
- **Crash-proof** — the interceptor can never crash your agent; an
  un-serializable payload degrades to `<unserializable>` and the run finishes.
- **Resilient analysis** — corrupted or truncated traces (agent crashed
  mid-write) are parsed best-effort, not fatally.
- **Comprehensive error detection** — tuned against real OpenAI/Anthropic, HTTP,
  network, stdlib, database, and infra errors with negation-awareness so
  "no errors" / "timeout handling" don't false-trigger.

## Validated topologies

Linear, branching/multi-tool, cyclic ReAct loops, and a production-style
`create_react_agent` graph (multi-turn, parallel tool calls, real `ToolNode` +
message state). In every case blame lands on the upstream origin.

## Run the tests

```bash
python tests/run_all.py        # full suite
```

## Limitations

- A tool that returns **plausible-but-wrong** data with *no* error markers won't
  be flagged by automatic detection — but you can still **prove or rule it out**
  with `diag.verify(...)` (counterfactual re-execution).
- Counterfactual verification currently patches **tool** outputs; root causes that
  are `llm:`/plain graph nodes need an explicit `tool=` or aren't patchable yet.
- Verification is a Python-API feature (it must run your agent), not a CLI command.
- Data-flow between parallel branches is reconstructed from timestamps, so the
  *visual* order of truly-concurrent nodes is approximate (blame is still correct
  via the ancestor cone).

## Layout

```
tracesurgeon/
  interceptor.py   # LangGraph callback → trace events (thread-safe, crash-proof)
  trace.py         # TraceEvent + TraceSession (atomic jsonl writer)
  dag.py           # call-tree + timestamp data-flow graph builders
  scorer.py        # failure detection + blame ranking
  report.py        # remediation + JSON report + unified renderer
  api.py           # instrument() / diagnose() — the public surface
  cli.py           # the `tracesurgeon` command
examples/
  quickstart.py    # copy-paste integration on a tiny real agent
tests/
  run_all.py       # runs every suite
```
