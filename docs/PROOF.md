# Proof / receipts

Real, captured evidence that TraceSurgeon does what it claims. Two parts:

1. **Offline test suites** — deterministic, run on every CI build, no API key.
2. **Live LLM validation** — a real model (`openai/gpt-oss-20b` via OpenRouter)
   driving a real `create_react_agent`. Transcripts are sanitized of all
   credentials.

---

## 1. Offline test suites

```text
$ python tests/run_all.py

  [PASS]  test_detection.py        negation-aware error detection (unit)
  [PASS]  test_error_corpus.py     real-world error recall / precision
  [PASS]  test_edge_cases.py       robustness / synthetic edge cases
  [PASS]  test_polish.py           corrupted traces, remediation, json, render
  [PASS]  test_counterfactual.py   causal proof + silent-failure check=
  [PASS]  test_production.py       async, concurrency, crash-proof
  [INFO]  test_scorer.py           (diagnostic print, no assertions)
  [INFO]  test_dag.py              (diagnostic print, no assertions)
  [PASS]  test_branching_agent.py  branching multi-tool
  [PASS]  test_react_loop.py       cyclic ReAct loop
  [PASS]  test_real_agent.py       real create_react_agent (scripted model)

  9/9 asserting suites passed (2 informational)
```

### Error detection — 75-case corpus

```text
$ python tests/test_error_corpus.py
  recall:    55/55 (100%)     real OpenAI/Anthropic/HTTP/network/stdlib/db/infra errors
  precision: 20/20 (100%)     benign text ("no errors", "timeout handling", bare numbers)
  ALL PASS — recall 55/55, precision 20/20
```

### Production hardening

```text
$ python tests/test_production.py        (10/10)
  [PASS] async: trace captured / tool recorded / blames the tool   (ainvoke)
  [PASS] concurrent: 400 events from 8 threads -> 0 corrupted lines
  [PASS] parallel fan-out: trace integrity + blames the poisoned branch
  [PASS] crash-proof: agent completes despite an un-serializable landmine payload
  [PASS] serialize: landmine becomes <unserializable> (no crash)
```

---

## 2. Live LLM validation

Model: `openai/gpt-oss-20b:free` via OpenRouter, temperature 0, real tool calling.
Harness: [`tests/test_complex_live.py`](../tests/test_complex_live.py).

### Headline: poisoned tool, real agent, causal proof

The model called the tools, hit a poisoned `get_population("Paris")` returning a
503, **retried it 9 times**, then produced a plausible final answer that *masked*
the failure. TraceSurgeon surfaced the hidden cause and proved it.

```text
──────────────────── TraceSurgeon — Root Cause Report ────────────────────
  trace: run_live_real.jsonl   steps: 12   tools: 7   total: 53.45ms

  ROOT CAUSE
    node:     tool:get_population  (confidence 100%)
    when:     2026-06-25T02:36:30Z
    why:      INTRODUCED the error (inputs were clean), errored tool call,
              output contains error signal
    input:    {'city': 'Paris'}
    output:   "ERROR: census API returned HTTP 503, data unavailable for this city"
    fix →     Upstream service error (5xx / overloaded). Retry with backoff. (upstream_5xx)
    symptom:  surfaced at agent

  Blame ranking
    1. 1.000  tool:get_population ×9   INTRODUCED the error, errored tool call, error signal
    2. 0.350  tools (symptom) ×11      output contains error signal, feeds the symptom
    3. 0.150  agent ×11
    4. 0.143  llm:openai/gpt-oss-20b:free ×11
    5. 0.027  tool:get_capital ×2

══════════════════ TraceSurgeon — Counterfactual Verification ══════════════
  patched output:  tool:get_population = '2.1 million'
  before:  FAIL    root cause = tool:get_population
  after:   clean   (no failure)

  ✅ CONFIRMED — fixing this one output flips the run to clean.
     This is causal proof, not correlation.
```

Note the model produced a *correct-looking* final answer ("16.1 million") by
falling back on parametric knowledge — the tool failure was invisible in the
output. TraceSurgeon flagged it anyway, and verification proved it.

### Scenario 1 — deep propagation (5-tool pipeline)

`get_revenue → get_cost → profit → margin → grade`. `get_cost` errors; the error
propagates through `profit`, `margin`, and `grade`.

```text
# SCENARIO 1 — DEEP PROPAGATION (5-tool chain)   [openai/gpt-oss-20b:free]
  -> root cause: tool:get_cost  (upstream_5xx)
  -> EXPECT tool:get_cost (the origin, not profit/margin/grade): PASS
```

Blame flowed back **three nodes** past the propagators to the true origin.

### Scenario 2 — silent wrong-data (no error markers)

`usd_to_eur` silently uses rate `10.0` instead of `0.92`. The final answer is
wrong but contains no error string.

```text
# SCENARIO 2 — SILENT WRONG-DATA (no error markers)   [openai/gpt-oss-20b:free]
  agent's final answer: '1000.0'
  -> detection says: no failure (correct — there are no error markers)
  -> proving causation via counterfactual check= (correct rate 0.92):

  ══════════════ TraceSurgeon — Counterfactual Verification ══════════════
  patched output:  usd_to_eur = '92.0'
  mode:    output check (no error markers needed)
  result:  passed after patch

  ✅ CONFIRMED — patching this one output makes the result correct.
     Causal proof of a SILENT failure (no error markers to detect).
```

Detection honestly found nothing; the `check=` predicate proved causation by
output correctness — the capability signal detection fundamentally cannot provide.

### Scenario 3 — multi-fault (two independent poisoned tools)

```text
# SCENARIO 3 — MULTI-FAULT (two poisoned tools)   [openai/gpt-oss-20b:free]
  -> high-confidence faulty tools: ['tool:traffic', 'tool:weather']
  -> EXPECT both tool:weather and tool:traffic flagged: PASS
```

### Real provider errors (caught + categorised correctly)

Along the way, several real API failures occurred. Each was captured by the
crash-proof interceptor and correctly categorised — credentials redacted:

| Real API event | Root cause node | Remediation category |
|---|---|---|
| Gemini `429 RESOURCE_EXHAUSTED` (quota) | `llm:` | `rate_limit` |
| OpenAI `401` (invalid key) | `llm:` | `auth` |
| OpenRouter `429` (upstream rate-limit) | `llm:` | `rate_limit` |
| OpenRouter `404` (retired model id) | `llm:` | `not_found` |
| Poisoned tool `HTTP 503` | `tool:get_population` | `upstream_5xx` |

Every real error, correctly diagnosed and categorised — even when the agent
crashed mid-run, the trace was still captured and analysed.
