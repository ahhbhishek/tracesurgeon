"""
TraceSurgeon CLI.

    tracesurgeon debug <trace.jsonl> [--json]   root-cause analysis of one trace
    tracesurgeon list [--dir traces] [--json]   list traces + clean/failure status
    tracesurgeon show <trace.jsonl>             raw execution tree (no scoring)

If <trace.jsonl> is omitted for `debug`/`show`, the most recent trace in the
traces dir is used. Run via `tracesurgeon ...` (installed) or `python -m tracesurgeon`.
Rich gives colored output; falls back to plain text automatically.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from ._console import enable_utf8
from .dag import build_dataflow_dag, build_dag, print_tree, summarize_dag
from .scorer import run_blame_analysis
from . import report as _report

try:
    from rich.console import Console
    _console = Console()
    _RICH = True
except Exception:
    _console = None
    _RICH = False

# strip our own markup tags ([bold], [/cyan]…) but NOT escaped user brackets (\[)
_MARKUP_RE = re.compile(r"(?<!\\)\[/?[a-z0-9_ #]+\]")


def _p(msg: str = "") -> None:
    if _RICH:
        _console.print(msg)
    else:
        print(_MARKUP_RE.sub("", str(msg)).replace("\\[", "["))


def _err(msg: str) -> None:
    print(_MARKUP_RE.sub("", msg), file=sys.stderr)


def _resolve_trace(arg: str | None, traces_dir: str = "traces") -> Path | None:
    """Resolve an explicit path, or fall back to the newest trace in traces_dir."""
    if arg:
        p = Path(arg)
        return p if p.exists() else None
    d = Path(traces_dir)
    if not d.exists():
        return None
    traces = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None


def _analyze(path: Path) -> dict:
    """Build flow + report dict for a trace; never raises (returns error marker)."""
    flow = build_dataflow_dag(str(path))
    result = run_blame_analysis(flow)
    rep = _report.to_report_dict(result, flow)
    rep["_flow"] = flow
    rep["_stats"] = summarize_dag(flow)
    return rep


# ------------------------------------------------------------------ #
#  debug                                                             #
# ------------------------------------------------------------------ #

def cmd_debug(args) -> int:
    path = _resolve_trace(getattr(args, "trace", None), args.dir)
    if path is None:
        target = args.trace or f"newest trace in {args.dir}/"
        _err(f"Trace not found: {target}")
        return 2

    try:
        rep = _analyze(path)
    except Exception as e:  # noqa: BLE001 - never crash the CLI on a bad trace
        _err(f"Could not analyze {path.name}: {type(e).__name__}: {e}")
        return 2

    if getattr(args, "json", False):
        out = {k: v for k, v in rep.items() if not k.startswith("_")}
        print(json.dumps({"trace": path.name, **out}, indent=2, default=str))
        return 1 if rep["has_failure"] else 0

    flow, stats = rep["_flow"], rep["_stats"]
    if _RICH:
        _console.rule("[bold cyan]TraceSurgeon — Root Cause Report")
    else:
        _p("=== TraceSurgeon — Root Cause Report ===")
    _p(f"  trace: [cyan]{path.name}[/cyan]   steps: {stats['total_steps']}   "
       f"tools: {stats['tool_calls']}   total: {stats['total_duration_ms']}ms")
    _p()
    _report.render(rep, flow, _p)
    return 1 if rep["has_failure"] else 0


# ------------------------------------------------------------------ #
#  list                                                              #
# ------------------------------------------------------------------ #

def cmd_list(args) -> int:
    d = Path(args.dir)
    if not d.exists():
        _err(f"Directory not found: {d}")
        return 2
    traces = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        _p(f"[dim]No traces in {d}/[/dim]")
        return 0

    rows = []
    for t in traces:
        try:
            rep = _analyze(t)
            rows.append({
                "file": t.name,
                "steps": rep["_stats"]["total_steps"],
                "status": ("failure" if rep["has_failure"] else "clean"),
                "root_cause": (rep["root_cause"]["node_name"]
                               if rep["has_failure"] and rep["root_cause"] else None),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"file": t.name, "steps": None, "status": "error", "error": str(e)})

    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, default=str))
        return 0

    try:
        from rich.table import Table
        table = Table(title=f"Traces in {d}/", header_style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Steps", justify="right")
        table.add_column("Root cause / status")
        for r in rows:
            if r["status"] == "error":
                status = f"[red]error: {r.get('error','')}[/red]"
            elif r["status"] == "failure":
                status = f"[bold red]{r['root_cause']}[/bold red]"
            else:
                status = "[green]clean[/green]"
            table.add_row(r["file"], str(r["steps"] if r["steps"] is not None else "?"), status)
        _console.print(table)
    except Exception:
        for r in rows:
            _p(f"{r['file']}  —  {r.get('root_cause') or r['status']}")
    return 0


# ------------------------------------------------------------------ #
#  show                                                              #
# ------------------------------------------------------------------ #

def cmd_show(args) -> int:
    path = _resolve_trace(getattr(args, "trace", None), args.dir)
    if path is None:
        _err(f"Trace not found: {args.trace or f'newest trace in {args.dir}/'}")
        return 2
    _p(f"[bold]Raw execution tree[/bold] — {path.name}\n")
    try:
        print_tree(build_dag(str(path)))
    except Exception as e:  # noqa: BLE001
        _err(f"Could not read {path.name}: {type(e).__name__}: {e}")
        return 2
    return 0


# ------------------------------------------------------------------ #
#  entry point                                                       #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracesurgeon",
        description="Causal, dependency-aware root-cause debugging for LangGraph agents.",
    )
    sub = parser.add_subparsers(dest="command")

    p_debug = sub.add_parser("debug", help="root-cause analysis of a trace")
    p_debug.add_argument("trace", nargs="?", help="path to a .jsonl trace (default: newest)")
    p_debug.add_argument("--dir", default="traces", help="traces dir for default lookup")
    p_debug.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    p_debug.set_defaults(func=cmd_debug)

    p_list = sub.add_parser("list", help="list trace files and their status")
    p_list.add_argument("--dir", default="traces", help="traces directory (default: traces)")
    p_list.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="print the raw execution tree (no scoring)")
    p_show.add_argument("trace", nargs="?", help="path to a .jsonl trace (default: newest)")
    p_show.add_argument("--dir", default="traces", help="traces dir for default lookup")
    p_show.set_defaults(func=cmd_show)
    return parser


def main(argv=None) -> int:
    enable_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
