# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic
versioning.

## [0.1.0] — 2026-06-25

First public release. Causal, dependency-aware root-cause debugging for LangGraph
agents, validated live on a real LLM.

### Core
- **Trace capture** — thread-safe, crash-proof `TraceInterceptor` callback that
  records every node/tool/LLM event to `.jsonl`; works for sync and async agents.
- **Data-flow reconstruction** — `build_dataflow_dag()` rebuilds the data-flow
  graph from timestamps, robust to LangGraph's dual-layer events, loops, and
  parallel branches.
- **Blame attribution** — the *introducer* rule (an errored node whose ancestors
  are all clean) flows blame past the loud downstream symptom to the quiet origin.
- **Remediation** — failures categorised (rate_limit, timeout, auth, upstream_5xx,
  bad_data, network, not_found, context_length, resource, process) with fix hints.

### Counterfactual verification
- `Diagnosis.verify()` / `counterfactual()` — patch a suspect tool's output,
  re-run the real agent, and report whether the failure flips (CONFIRMED /
  PARTIAL / NOT_CONFIRMED / INCONCLUSIVE).
- `check=` predicate — prove **silent** failures (plausible-but-wrong data with no
  error markers) by output correctness.

### Interfaces
- One-liner Python API: `instrument()` → run → `diagnose()` → `.print()` / `.to_dict()`.
- `tracesurgeon` CLI: `debug` / `list` / `show`, with `--json` for CI.

### Quality
- 100% recall + precision on a 75-case real-world error corpus.
- 9 offline test suites in CI across Python 3.10–3.12.
- Validated live on `openai/gpt-oss-20b` (via OpenRouter): deep propagation,
  silent wrong-data, multi-fault, and real provider errors — see
  [docs/PROOF.md](docs/PROOF.md).

[0.1.0]: https://github.com/ahhbhishek/tracesurgeon/releases/tag/v0.1.0
