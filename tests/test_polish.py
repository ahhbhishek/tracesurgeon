"""
Polish / usability tests:
  - corrupted & partial traces don't crash, still yield a usable analysis
  - remediation classification maps real errors to the right category
  - Diagnosis.to_dict() is JSON-serializable and carries inputs/full output/ts
  - no-root-cause and empty cases render without crashing

Run with:
    cd Projects/tracesurgeon
    python tests/test_polish.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon import diagnose, classify_error, to_report_dict
from tracesurgeon.dag import build_dataflow_dag, _parse_steps
from tracesurgeon.scorer import run_blame_analysis

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


TRACES = Path("traces")
TRACES.mkdir(exist_ok=True)


def _write(name, text):
    p = TRACES / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_corrupted_trace_survives():
    # malformed middle line + truncated final line
    path = _write("polish_corrupt.jsonl",
        '{"event_type":"node_start","step_id":"a","node_name":"plan",'
        '"timestamp":"2026-01-01T00:00:00+00:00","inputs":{"q":"hi"}}\n'
        'NOT JSON AT ALL {{{\n'
        '{"event_type":"tool_start","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:bad","timestamp":"2026-01-01T00:00:01+00:00"}\n'
        '{"event_type":"tool_end","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:bad","timestamp":"2026-01-01T00:00:02+00:00",'
        '"outputs":"ERROR: 503 service unavailable"}\n'
        '{"event_type":"node_en')  # truncated mid-line
    try:
        diag = diagnose(path)
        rc = diag.root_cause
        check("corrupted trace: no crash + blames the tool",
              diag.has_failure and rc and "tool:bad" in rc["node_name"],
              rc["node_name"] if rc else "none")
    except Exception as e:
        check("corrupted trace: no crash + blames the tool", False, f"threw {e}")


def test_missing_fields_skipped():
    path = _write("polish_missing.jsonl",
        '{"foo":"bar"}\n'  # no step_id/event_type
        '{"event_type":"node_start"}\n'  # no step_id
        '{"event_type":"node_start","step_id":"x","node_name":"only",'
        '"timestamp":"2026-01-01T00:00:00+00:00"}\n'
        '{"event_type":"error","step_id":"x","node_name":"only",'
        '"timestamp":"2026-01-01T00:00:01+00:00","error":"KeyError: boom"}')
    try:
        steps = _parse_steps(path)
        diag = diagnose(path)
        check("missing-field lines skipped, valid step kept",
              "x" in steps and diag.has_failure)
    except Exception as e:
        check("missing-field lines skipped, valid step kept", False, str(e))


def test_remediation_categories():
    cases = [
        ("openai.RateLimitError: 429 Too Many Requests", "rate_limit"),
        ("HTTP 503 Service Unavailable", "upstream_5xx"),
        ("AuthenticationError: invalid api key 401", "auth"),
        ("httpx.ReadTimeout: timed out", "timeout"),
        ("json.decoder.JSONDecodeError: Expecting value", "bad_data"),
        ("ConnectionResetError: connection refused", "network"),
        ("KeyError: 'results' not found", "not_found"),
        ("This model's maximum context length is 8192 tokens", "context_length"),
    ]
    ok = 0
    for text, expected in cases:
        got = classify_error(text)
        if got and got["category"] == expected:
            ok += 1
        else:
            print(f"      remediation miss: {text!r} -> {got['category'] if got else None} "
                  f"(expected {expected})")
    check("remediation categories", ok == len(cases), f"{ok}/{len(cases)}")


def test_to_dict_json_serializable():
    # build a quick failing trace
    path = _write("polish_json.jsonl",
        '{"event_type":"node_start","step_id":"a","node_name":"plan",'
        '"timestamp":"2026-01-01T00:00:00+00:00","inputs":{"q":"hi"}}\n'
        '{"event_type":"tool_start","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:x","timestamp":"2026-01-01T00:00:01+00:00","inputs":{"v":1}}\n'
        '{"event_type":"tool_end","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:x","timestamp":"2026-01-01T00:00:02+00:00",'
        '"outputs":"ERROR: timeout"}')
    diag = diagnose(path)
    rep = diag.to_dict()
    try:
        s = json.dumps(rep)  # must serialize cleanly
        rc = rep["root_cause"]
        ok = (
            "input_preview" in rc and "output_full" in rc and
            "timestamp" in rc and "remediation" in rc and
            isinstance(rep["pipeline"], list)
        )
        check("to_dict is JSON-serializable + enriched", ok and len(s) > 0)
    except Exception as e:
        check("to_dict is JSON-serializable + enriched", False, str(e))


def test_clean_run_renders():
    path = _write("polish_clean.jsonl",
        '{"event_type":"node_start","step_id":"a","node_name":"plan",'
        '"timestamp":"2026-01-01T00:00:00+00:00"}\n'
        '{"event_type":"node_end","step_id":"a","node_name":"plan",'
        '"timestamp":"2026-01-01T00:00:01+00:00","outputs":"all good, no errors"}')
    diag = diagnose(path)
    rep = diag.to_dict()
    check("clean run: no failure + serializable", not diag.has_failure and json.dumps(rep))
    try:
        diag.print()  # must not crash
        check("clean run: print() does not crash", True)
    except Exception as e:
        check("clean run: print() does not crash", False, str(e))


def test_bracket_output_does_not_crash_render():
    # tool output full of '[...]' must not break the rich-markup renderer
    path = _write("polish_brackets.jsonl",
        '{"event_type":"node_start","step_id":"a","node_name":"plan",'
        '"timestamp":"2026-01-01T00:00:00+00:00"}\n'
        '{"event_type":"tool_start","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:x","timestamp":"2026-01-01T00:00:01+00:00"}\n'
        '{"event_type":"tool_end","step_id":"b","parent_step_id":"a",'
        '"node_name":"tool:x","timestamp":"2026-01-01T00:00:02+00:00",'
        '"outputs":"ERROR [boombox] at [/path] with [weird markup] and [bold]"}')
    diag = diagnose(path)
    # plain render via Diagnosis.print (markup stripped + unescaped)
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            diag.print()
        out = buf.getvalue()
        ok = "[boombox]" in out and "[/path]" in out and "[weird markup]" in out
        check("bracket output renders literally (no crash)", ok, out[-200:])
    except Exception as e:
        check("bracket output renders literally (no crash)", False, str(e))

    # rich render path must also not raise
    try:
        from tracesurgeon import cli as _cli
        if _cli._RICH:
            import io as _io
            import contextlib as _ctx
            with _ctx.redirect_stdout(_io.StringIO()):
                rc = _cli.main(["debug", path])
            check("bracket output: rich path no crash", rc in (0, 1))
        else:
            check("bracket output: rich path no crash", True)
    except Exception as e:
        check("bracket output: rich path no crash", False, str(e))


if __name__ == "__main__":
    print("Polish / usability tests:\n")
    for fn in [test_corrupted_trace_survives, test_missing_fields_skipped,
               test_remediation_categories, test_to_dict_json_serializable,
               test_clean_run_renders, test_bracket_output_does_not_crash_render]:
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"threw {type(e).__name__}: {e}")
    # cleanup
    for f in TRACES.glob("polish_*.jsonl"):
        f.unlink()
    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:", ", ".join(FAIL))
    sys.exit(0 if not FAIL else 1)
