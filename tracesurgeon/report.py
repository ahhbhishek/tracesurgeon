"""
Report layer — turns a raw blame `result` + data-flow `flow` into:

  * a rich, JSON-serializable report dict (to_report_dict)  -> used by --json + API
  * a single human renderer (render)                         -> used by CLI + Diagnosis.print

Centralising this fixes the old split where the CLI printed rich/colored output
but Diagnosis.print() printed plain text with different content. Now there is one
source of truth, and it carries the extra context real debugging needs: the
failing INPUT, a fuller error payload, the wall-clock timestamp, and a
categorised remediation hint ("HTTP 503 = transient, retry").
"""

import re

from .dag import _is_nested


def _esc(value) -> str:
    """
    Escape user-derived text so it can be safely embedded in a markup line.

    Tool outputs / inputs routinely contain '[' (log prefixes, paths, JSON
    arrays). Rich would try to parse those as style tags and crash. We escape
    '[' as '\\[' — rich renders that as a literal '[', and the plain-text
    fallback strips the backslash back out (see callers).
    """
    return str(value).replace("[", "\\[")


# ------------------------------------------------------------------ #
#  Remediation — classify an error and suggest what to do about it    #
# ------------------------------------------------------------------ #

# (regex, category, hint). First match wins, so order from specific -> general.
_REMEDIATION_RULES = [
    (r"rate[ -]?limit|too many requests|\b429\b|quota|throttl",
     "rate_limit",
     "Transient rate limit. Back off and retry with exponential backoff; "
     "check your plan's rate/quota limits."),
    (r"context (window|length)|max(imum)? tokens?|too long|context_length",
     "context_length",
     "The prompt/input exceeded the model's context window. Trim or chunk the "
     "input, or use a larger-context model."),
    (r"overloaded|\b529\b|\b503\b|service unavailable|\b502\b|bad gateway|"
     r"\b500\b|internal server error|\b504\b|gateway timeout",
     "upstream_5xx",
     "Upstream service error (5xx / overloaded). This is not your input — "
     "retry later, ideally with backoff."),
    (r"unauthori|forbidden|\b401\b|\b403\b|invalid (api ?key|token|credential)|"
     r"authenticat\w* (error|failed)|expired (token|key|session)|permission denied",
     "auth",
     "Authentication/permission problem. Check the API key, token, scopes and "
     "credentials for this tool/model."),
    (r"timed ?out|timeout|read ?timeout|connect ?timeout",
     "timeout",
     "Operation timed out. Increase the timeout, add retries, or check whether "
     "the upstream is slow/unreachable."),
    (r"connection (refused|reset|error|aborted|closed)|unreachable|dns|ssl|"
     r"max retries|broken pipe|network",
     "network",
     "Network/connectivity failure. Check connectivity, DNS, SSL and proxy "
     "settings, then retry."),
    (r"not found|\b404\b|missing|keyerror|no (data|results?|rows?|records?)",
     "not_found",
     "A requested resource/key/record was missing. Verify the id, path, query "
     "or upstream data actually exists."),
    (r"json|parse|decode|malformed|invalid|unexpected (token|character|eof)|"
     r"validation|schema|could not parse",
     "bad_data",
     "The data did not match the expected shape. Validate/repair the payload, "
     "or fix the step that produced it."),
    (r"out of (memory|range|bounds)|memory ?error|recursion|segfault|"
     r"segmentation fault|oom|killed",
     "resource",
     "Resource exhaustion (memory/recursion/limits). Reduce input size or fix "
     "an unbounded loop/recursion."),
    (r"non[- ]?zero|exit (status|code)|subprocess|command failed",
     "process",
     "A subprocess/command failed (non-zero exit). Inspect its stderr and the "
     "arguments it was given."),
]
_COMPILED = [(re.compile(rx, re.IGNORECASE), cat, hint) for rx, cat, hint in _REMEDIATION_RULES]


def classify_error(text: str) -> dict | None:
    """Return {category, hint} for the first matching remediation rule, else a
    generic fallback. None only if there is no text at all."""
    if not text:
        return None
    for rx, cat, hint in _COMPILED:
        if rx.search(text):
            return {"category": cat, "hint": hint}
    return {
        "category": "unknown",
        "hint": "Inspect this node's input and output below to see what went "
                "wrong; the failure did not match a known category.",
    }


# ------------------------------------------------------------------ #
#  Enrichment — pull full context for a node out of the flow graph    #
# ------------------------------------------------------------------ #

def _node_field(flow, step_id: str, field: str, cap: int):
    try:
        val = flow.nodes[step_id].get(field)
    except Exception:
        return None
    if val is None:
        return None
    s = str(val)
    return s if len(s) <= cap else s[:cap] + f"…(+{len(s) - cap} chars)"


def _enrich(flow, suspect: dict) -> dict:
    """Add input/full-output/timestamp/remediation to a suspect or symptom dict."""
    sid = suspect.get("step_id")
    out_full = _node_field(flow, sid, "outputs", 1200) or suspect.get("outputs_preview", "")
    err_text = _node_field(flow, sid, "error", 1200)
    enriched = dict(suspect)
    enriched["input_preview"] = _node_field(flow, sid, "inputs", 400)
    enriched["output_full"] = out_full
    enriched["error"] = err_text
    enriched["timestamp"] = _node_field(flow, sid, "start_ts", 64)
    enriched["duration_ms"] = (flow.nodes[sid].get("duration_ms")
                               if sid in flow.nodes else None)
    enriched["remediation"] = classify_error(f"{err_text or ''} {out_full or ''}")
    return enriched


# ------------------------------------------------------------------ #
#  JSON-serializable report                                           #
# ------------------------------------------------------------------ #

def to_report_dict(result: dict, flow) -> dict:
    """Canonical, JSON-safe report used by --json and Diagnosis.to_dict()."""
    if not result.get("has_failure"):
        return {"has_failure": False, "root_cause": None, "symptom": None,
                "suspects": [], "pipeline": _pipeline_list(flow)}

    rc = _enrich(flow, result["root_cause"]) if result.get("root_cause") else None
    sym = result.get("symptom")
    return {
        "has_failure": True,
        "root_cause": rc,
        "symptom": sym,
        "suspects": [_enrich(flow, s) for s in result.get("suspects", [])],
        "pipeline": _pipeline_list(flow),
    }


def _pipeline_list(flow) -> list[dict]:
    pipe = [n for n, d in flow.nodes(data=True) if not _is_nested(d.get("node_name", ""))]
    pipe.sort(key=lambda n: flow.nodes[n].get("start_ts") or "")
    out = []
    for n in pipe:
        d = flow.nodes[n]
        tools = [flow.nodes[p].get("node_name") for p in flow.predecessors(n)
                 if _is_nested(flow.nodes[p].get("node_name", ""))]
        out.append({
            "step_id": n,
            "node_name": d.get("node_name"),
            "success": d.get("success", True),
            "duration_ms": d.get("duration_ms"),
            "timestamp": d.get("start_ts"),
            "tools": tools,
        })
    return out


# ------------------------------------------------------------------ #
#  Human render — markup lines, emitted via a `p` callback            #
#  (CLI passes a rich-aware p; plain callers strip the markup)        #
# ------------------------------------------------------------------ #

def render(report: dict, flow, p, *, highlight: set | None = None) -> None:
    """Render a human report by calling p(markup_line) repeatedly."""
    if not report["has_failure"]:
        p("  [bold green]✓ No failure detected — the run looks clean.[/bold green]")
        p()
        return

    rc = report["root_cause"]
    sym = report["symptom"]
    hl = highlight if highlight is not None else ({rc["step_id"]} if rc else set())

    # --- root cause block ---
    p("  [bold red]ROOT CAUSE[/bold red]")
    if rc:
        p(f"    node:        [bold]{_esc(rc['node_name'])}[/bold]  "
          f"(confidence [bold red]{rc['score']:.0%}[/bold red])")
        if rc.get("timestamp"):
            p(f"    when:        {_esc(rc['timestamp'])}")
        p(f"    why:         {_esc(', '.join(rc['reasons']))}")
        if rc.get("input_preview"):
            p(f"    input:       {_esc(repr(rc['input_preview']))}")
        payload = rc.get("error") or rc.get("output_full") or ""
        p(f"    output:      {_esc(repr(payload))}")
        rem = rc.get("remediation")
        if rem:
            p(f"    [yellow]fix → {_esc(rem['hint'])}[/yellow]  [dim]({rem['category']})[/dim]")
    p(f"    symptom:     surfaced at [cyan]{_esc(sym['node_name']) if sym else '?'}[/cyan]")
    p()

    # --- ranking ---
    p("  [bold]Blame ranking[/bold]")
    for i, s in enumerate(_dedupe(report["suspects"])[:8]):
        tag = " [dim](symptom)[/dim]" if s.get("is_symptom") else ""
        if s.get("count", 1) > 1:
            tag += f" [dim]×{s['count']}[/dim]"
        sev = "bold red" if s["score"] >= 0.8 else ("yellow" if s["score"] >= 0.5 else "dim")
        p(f"    {i+1}. [{sev}]{s['score']:.3f}[/{sev}]  "
          f"[cyan]{_esc(s['node_name'])}[/cyan]{tag}  "
          f"[dim]{_esc(', '.join(s['reasons']) or '—')}[/dim]")
    p()

    # --- data-flow pipeline ---
    p("  [bold]Data-flow[/bold] (root cause marked):")
    for item in report["pipeline"]:
        n = item["step_id"]
        dur = f" \\[{item['duration_ms']:.1f}ms]" if item.get("duration_ms") else ""
        status = "[green]✓[/green]" if item["success"] else "[bold red]✗ FAIL[/bold red]"
        mark = "  [bold red]◄ ROOT CAUSE[/bold red]" if n in hl else ""
        p(f"    ▼ [cyan]{_esc(item['node_name'])}[/cyan] ({n[:6]}) {status}{dur}{mark}")
        for tname in item["tools"]:
            # mark tool if its step is the highlight
            tmark = ""
            for pred in flow.predecessors(n):
                if pred in hl and flow.nodes[pred].get("node_name") == tname:
                    tmark = "  [bold red]◄ ROOT CAUSE[/bold red]"
            p(f"        └─ [magenta]{_esc(tname)}[/magenta]{tmark}")
    p()


def _dedupe(suspects: list[dict]) -> list[dict]:
    """Collapse repeated node names (loops) for display, keep highest score + count."""
    best: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for s in suspects:
        name = s["node_name"]
        counts[name] = counts.get(name, 0) + 1
        if name not in best or s["score"] > best[name]["score"]:
            best[name] = s
    out = []
    for name, s in best.items():
        item = dict(s)
        item["count"] = counts[name]
        out.append(item)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
