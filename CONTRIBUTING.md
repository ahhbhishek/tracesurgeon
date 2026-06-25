# Contributing

Thanks for your interest in TraceSurgeon.

## Setup

```bash
git clone https://github.com/ahhbhishek/tracesurgeon.git
cd tracesurgeon
pip install -e ".[dev]"
```

Optional provider extras for the live tests:

```bash
pip install -e ".[openai]"   # also covers OpenRouter (OpenAI-compatible)
pip install -e ".[gemini]"
pip install -e ".[anthropic]"
```

## Running tests

```bash
python tests/run_all.py        # full offline suite — no API key needed
```

This is exactly what CI runs. Live tests (`tests/test_real_agent_live.py`,
`tests/test_complex_live.py`) are excluded from the suite; they need an API key
read **only** from the environment (`OPENROUTER_API_KEY` / `OPENAI_API_KEY` /
`GOOGLE_API_KEY`) — never hard-code a key.

## Conventions

- **Match the surrounding code** — naming, comment density, and idiom.
- **Add a test for every behavior change.** Most modules have a focused suite
  (`test_<area>.py`); extend the closest one and register new suites in
  `tests/run_all.py`.
- Keep the public surface small — new capabilities should flow through
  `instrument()` / `diagnose()` / `Diagnosis` where possible.
- No secrets in code, tests, fixtures, or trace files (`traces/*.jsonl` is
  git-ignored).

## Architecture

See [docs/DESIGN.md](docs/DESIGN.md) for the pipeline and the introducer rule.
