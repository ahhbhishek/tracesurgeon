# Design — how TraceSurgeon works

This is the engineering behind the one-liner. The pipeline is five stages:

```
capture → reconstruct data-flow → attribute (introducer) → remediate → prove
```

---

## 1. Capture — `interceptor.py`

`TraceInterceptor` is a `langchain_core` `BaseCallbackHandler`. Attaching it via
`config={"callbacks": [handler]}` makes LangChain fire `on_chain_*`,
`on_tool_*`, and `on_llm_*` for every node, tool, and model call. Each becomes a
`TraceEvent` (`trace.py`) written immediately to a `.jsonl`, so a crash mid-run
still leaves an analysable trace.

Three properties make it safe on real agents:

- **Thread-safe.** Real agents run nodes concurrently (parallel branches, async,
  thread pools). `TraceSession.add_event` serializes the list-append + file-write
  under a `threading.Lock`, and the interceptor's `_active` run-id map is
  lock-guarded. Verified: 400 events from 8 threads → 0 corrupted lines.
- **Crash-proof.** Every handler is wrapped by `@_guard` — any internal exception
  is swallowed and recorded, never raised into the host agent. `_safe_serialize`
  survives objects that throw on `str()`/`repr()` (→ `<unserializable>`).
- **Bounded.** `_truncate` caps each field at 8 KB but keeps **both ends**, so an
  error in the tail of a huge tool output (a traceback after pages of logs)
  still survives for detection.

**Async.** This is a sync handler; LangChain's `AsyncCallbackManager` runs sync
handlers in a thread-pool for `ainvoke`/`astream`, so the same code path covers
both with no async-specific methods.

## 2. Reconstruct data-flow — `dag.py`

The blame algorithm needs a **data-flow** graph (whose output fed whose input),
but LangGraph's callbacks give a **call tree** (who invoked whom) — and worse, it
emits *two layers* of events per logical node (an outer task wrapper with a
generic name like `graph:step:3`, and the inner node with the real name). Tools
attach to the outer layer; real names live on the inner layer.

`build_dataflow_dag()` turns this into a clean data-flow graph:

- **Pick the real-named nodes** as the pipeline (falling back to `graph:step:N`
  for legacy traces); drop wrapper noise via `_WRAPPER_RE`.
- **Merge the dual layer** (`_merge_dual_layer`): collapse consecutive same-named
  nodes whose time windows overlap. Genuine repeats in a loop are separated by
  other nodes, so they never wrongly merge.
- **Order by timestamp**, chain the pipeline sequentially, and **attach each
  tool/LLM to the node whose `[start, end]` window contains it** (`_pick_host`).
  This reconstructs data flow from *time*, not from LangGraph's parent links —
  which makes it robust to version changes, loops, and parallel branches.

The result for a linear agent: `tool → step1 → step2 → step3`.

## 3. Attribute — the introducer rule — `scorer.py`

**Failure detection** (`_has_error`): a node "has an error" if it raised an
exception (`success == False`) **or** its output contains an un-negated error
signal. The signal regex (`_ERROR_PATTERNS`) is tuned against real provider,
HTTP, network, stdlib, database, and infra errors; a negation pass
(`_NEGATION_RE`) discards benign mentions like *"no errors"*, *"error handling"*,
*"completed with 0 errors"*. On a 75-case corpus: 100% recall, 100% precision.

**Symptom** (`detect_symptom`): the *most downstream* node that has an error
(last in topological order) — where the failure surfaces.

**Introducer** (`_is_introducer`): a node that has an error but **none of its
upstream ancestors do**. Using the full ancestor cone (not just direct
predecessors) is what makes blame robust — an error that originated several hops
up disqualifies every downstream node, so only the true origin survives.

**Scoring** (`score_suspects`), highest first:

| Signal | Weight |
|---|---|
| INTRODUCED the error (clean inputs, errored output) | +0.50 |
| errored **tool** call (external data sources are the usual origin) | +0.25 |
| output carries an error signal | +0.20 |
| proximity / feeds directly into the symptom | +0.15 |

The introducer bonus is the lever that beats "blame hoarding": the loud
downstream symptom scores on proximity/error-signal, but only the quiet origin
gets the +0.50, so it wins.

## 4. Remediate — `report.py`

`classify_error` maps the failure text to one of ~10 categories
(`rate_limit`, `timeout`, `auth`, `upstream_5xx`, `bad_data`, `network`,
`not_found`, `context_length`, `resource`, `process`) with a concrete fix hint.
`report.py` is also the single source of truth for rendering — the CLI and
`Diagnosis.print()` share one renderer, and `to_report_dict()` produces the
JSON-serializable report (inputs, full output, timestamps, remediation, pipeline)
used by `--json` and programmatic callers. User-supplied text is escaped before
the markup pass so tool outputs containing `[...]` can't break rendering.

## 5. Prove — counterfactual verification — `counterfactual.py`

Diagnosis is correlation; verification is proof. To substitute a tool's output
mid-run we exploit a single fact about LangChain: **every tool call goes through
`BaseTool.run` / `BaseTool.arun`** (`invoke→run`, `ainvoke→arun`, and LangGraph's
`ToolNode` uses `invoke`/`ainvoke`). `_patch_tool_outputs` wraps those two
chokepoints in a context manager: for a tool whose `.name` matches the patch, it
returns the corrected value; all other tools run for real.

The return value is **shaped to the caller's expectation** (`_substitute`): when
invoked by a `ToolNode` the input is a `ToolCall` carrying a `tool_call_id`, so we
return a `ToolMessage` (what prebuilt agents require); otherwise the raw value.
This is a stateless short-circuit — no per-instance mutation, no locks — so it is
safe under parallel and async tool execution.

**Verdict logic.** Re-run the agent inside the patch context, then `diagnose` the
new trace:

- baseline had no failure → `INCONCLUSIVE`
- patched run is clean → `CONFIRMED`
- patched still fails, but at a *different* node → `PARTIAL` (original cause
  resolved, new failure surfaced)
- patched fails the same way → `NOT_CONFIRMED`

**Silent failures.** When a tool returns plausible-but-wrong data with no error
markers, detection sees nothing — so the verdict can't come from error detection.
Pass `check=predicate`: `counterfactual` captures the agent's *result* and the
verdict becomes `CONFIRMED` iff `check(result)` is true after the patch. This
proves causation by **output correctness** when there is nothing to detect.

## Design trade-offs / limitations

- **Tool outputs only (v1).** Counterfactual patches tools; `llm:`/plain
  graph-node causes need an explicit `tool=`. Tools are the canonical
  poisoned-output case and the highest-value target.
- **Timestamp data-flow** is an approximation for *truly* concurrent branches —
  the visual order may differ from wall-clock interleaving, but blame is still
  correct because it uses the ancestor cone, not adjacency.
- **Signal-based detection** can't see silent wrong-data on its own (hence
  `check=`). This is a fundamental limit of any detection-only approach, stated
  honestly rather than papered over.
- **Verification runs the agent**, so it's a Python-API feature, not a
  trace-only CLI command.
