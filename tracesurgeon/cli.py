"""
TraceSurgeon CLI.

    tracesurgeon debug <trace.jsonl>     full root-cause analysis
    tracesurgeon list [--dir traces]     list trace files + clean/failure status
    tracesurgeon show <trace.jsonl>      raw execution tree (no scoring)

Run via the installed entry point (`tracesurgeon ...`) or `python -m tracesurgeon`.
Uses argparse (stdlib) + rich (colored output, already a dependency). Falls back
to plain text if rich is unavailable.
"""

import argparse
import re
import sys
from pathlib import Path

from ._console import enable_utf8
from .dag import build_dataflow_dag, build_dag, print_tree, summarize_dag, _is_nested
from .scorer import run_blame_analysis

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _console = Console()
    _RICH = True
except Exception:
    _console = None
    _RICH = False

_MARKUP_RE = re.compile(r"\[/?[a-z0-9_ #]+\]")


def _p(msg: str = "") -> None:
    if _RICH:
        _console.print(msg)
    else:
        print(_MARKUP_RE.sub("", str(msg)))


def _score_style(score: float) -> str:
    if score >= 0.8:
        return "bold red"
    if score >= 0.5:
        return "yellow"
    return "dim"


# ------------------------------------------------------------------ #
#  debug                                                             #
# ------------------------------------------------------------------ #

def _dedupe_suspects(suspects: list[dict]) -> list[dict]:
    """
    Collapse repeated node names for DISPLAY, keeping the highest-scoring
    instance of each. In loops the same logical node runs many times; the user
    wants 'which node', not every iteration. Adds a 'count' when collapsed.
    """
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


def cmd_debug(args) -> int:
    path = Path(args.trace)
    if not path.exists():
        _p(f"[bold red]Trace not found:[/bold red] {path}")
        return 2

    flow = build_dataflow_dag(str(path))
    result = run_blame_analysis(flow)
    stats = summarize_dag(flow)

    if _RICH:
        _console.rule("[bold cyan]TraceSurgeon — Root Cause Report")
    else:
        _p("=== TraceSurgeon — Root Cause Report ===")

    _p(f"  trace: [cyan]{path.name}[/cyan]   steps: {stats['total_steps']}   "
       f"tools: {stats['tool_calls']}   total: {stats['total_duration_ms']}ms")
    _p()

    if not result["has_failure"]:
        _p("  [bold green]✓ No failure detected — the run looks clean.[/bold green]")
        _p()
        return 0

    rc = result["root_cause"]
    sym = result["symptom"]
    body = (
        f"[bold]{rc['node_name']}[/bold]   confidence [bold red]{rc['score']:.0%}[/bold red]\n"
        f"[dim]why:[/dim] {', '.join(rc['reasons'])}\n"
        f"[dim]output:[/dim] {rc['outputs_preview']!r}\n"
        f"[dim]surfaced as symptom at:[/dim] {sym['node_name']}"
    )
    if _RICH:
        _console.print(Panel(body, title="[bold red]ROOT CAUSE", border_style="red"))
    else:
        _p("ROOT CAUSE:")
        _p(body)
    _p()

    if _RICH:
        table = Table(title="Blame Ranking", header_style="bold")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Node", style="cyan", no_wrap=True)
        table.add_column("Signals")
        for i, s in enumerate(_dedupe_suspects(result["suspects"])[:8]):
            tag = " (symptom)" if s["is_symptom"] else ""
            if s.get("count", 1) > 1:
                tag += f" [dim]×{s['count']}[/dim]"
            table.add_row(
                str(i + 1),
                Text(f"{s['score']:.3f}", style=_score_style(s["score"])),
                s["node_name"] + tag,
                ", ".join(s["reasons"]) or "—",
            )
        _console.print(table)
    else:
        _p("Blame Ranking:")
        for i, s in enumerate(_dedupe_suspects(result["suspects"])[:8]):
            _p(f"  {i+1}. {s['score']:.3f}  {s['node_name']}  ({', '.join(s['reasons'])})")
    _p()

    _p("[bold]Data-flow[/bold] (root cause marked):")
    _print_pipeline(flow, {rc["step_id"]})
    _p()
    return 1  # nonzero => a failure was detected (useful for CI)


def _print_pipeline(flow, highlight: set) -> None:
    pipe = [n for n, d in flow.nodes(data=True) if not _is_nested(d["node_name"])]
    pipe.sort(key=lambda n: flow.nodes[n].get("start_ts") or "")
    for n in pipe:
        d = flow.nodes[n]
        dur = f" [{d['duration_ms']:.1f}ms]" if d.get("duration_ms") else ""
        status = "[green]✓[/green]" if d.get("success", True) else "[bold red]✗ FAIL[/bold red]"
        mark = "  [bold red]◄ ROOT CAUSE[/bold red]" if n in highlight else ""
        _p(f"  ▼ [cyan]{d['node_name']}[/cyan] ({n[:6]}) {status}{dur}{mark}")
        for pred in flow.predecessors(n):
            pd = flow.nodes[pred]
            if _is_nested(pd["node_name"]):
                pstat = "[green]✓[/green]" if pd.get("success", True) else "[bold red]✗ FAIL[/bold red]"
                pmark = "  [bold red]◄ ROOT CAUSE[/bold red]" if pred in highlight else ""
                _p(f"      └─ [magenta]{pd['node_name']}[/magenta] ({pred[:6]}) {pstat}{pmark}")


# ------------------------------------------------------------------ #
#  list                                                              #
# ------------------------------------------------------------------ #

def cmd_list(args) -> int:
    d = Path(args.dir)
    if not d.exists():
        _p(f"[bold red]Directory not found:[/bold red] {d}")
        return 2
    traces = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        _p(f"[dim]No traces in {d}/[/dim]")
        return 0

    if _RICH:
        table = Table(title=f"Traces in {d}/", header_style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Steps", justify="right")
        table.add_column("Root cause / status")
        for t in traces:
            try:
                flow = build_dataflow_dag(str(t))
                stats = summarize_dag(flow)
                res = run_blame_analysis(flow)
                if res["has_failure"]:
                    status = f"[bold red]{res['root_cause']['node_name']}[/bold red]"
                else:
                    status = "[green]clean[/green]"
                table.add_row(t.name, str(stats["total_steps"]), status)
            except Exception as e:
                table.add_row(t.name, "?", f"[red]error: {e}[/red]")
        _console.print(table)
    else:
        for t in traces:
            _p(t.name)
    return 0


# ------------------------------------------------------------------ #
#  show                                                              #
# ------------------------------------------------------------------ #

def cmd_show(args) -> int:
    path = Path(args.trace)
    if not path.exists():
        _p(f"[bold red]Trace not found:[/bold red] {path}")
        return 2
    _p(f"[bold]Raw execution tree[/bold] — {path.name}\n")
    print_tree(build_dag(str(path)))
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

    p_debug = sub.add_parser("debug", help="full root-cause analysis of a trace")
    p_debug.add_argument("trace", help="path to a .jsonl trace file")
    p_debug.set_defaults(func=cmd_debug)

    p_list = sub.add_parser("list", help="list trace files and their status")
    p_list.add_argument("--dir", default="traces", help="traces directory (default: traces)")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="print the raw execution tree (no scoring)")
    p_show.add_argument("trace", help="path to a .jsonl trace file")
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
