"""
Blame Scorer — Phase 3.

Operates on the DATA-FLOW dag (from build_dataflow_dag), not the call tree.

Two jobs:
  1. Detect failures — including SILENT ones (error text in output, no exception).
  2. Rank suspects so the ORIGIN of the error wins, not the loudest downstream
     symptom. This is the anti-"blame-hoarding" core of TraceSurgeon.

Scoring heuristic (no ML yet):
  +0.50  INTRODUCER — node has the error but none of its data-flow inputs did
  +0.25  is a tool call (external data sources are the usual origin)
  +0.20  output contains an error signal
  +0.15  proximity / direct feed into the symptom
  ──────
  capped at 1.00

The introducer bonus is what flows blame backward past the symptom.
"""

import re

import networkx as nx


# Error signals, tuned against REAL provider/tool/framework errors. Organized by
# category for maintainability. \b word boundaries stop "invalid" matching
# "invalid_tool_calls". Designed for high recall on real failures while the
# negation pass (below) suppresses benign mentions like "no errors".
_ERROR_PATTERNS = [
    # --- generic keywords ---
    r"\berrors?\b", r"\bexceptions?\b", r"\bfailed\b", r"\bfailures?\b",
    r"\bfailing\b", r"\bfatal\b", r"\bcrash(ed|ing)?\b", r"\btraceback\b",
    r"\baborted?\b", r"\bpanic(ked)?\b", r"\bstack ?trace\b",

    # --- validity / parsing ---
    r"\binvalid\b", r"\bmalformed\b", r"\bcorrupt(ed)?\b", r"\bunparseable\b",
    r"\bnot ?(a )?valid\b", r"\bparse (error|fail)", r"\bdecode (error|fail)",
    r"\bunexpected (token|character|eof|end)", r"\bschema (error|mismatch)",
    r"\bvalidation (error|failed)", r"\bunable to parse\b",

    # --- capability / could-not ---
    r"\bcould ?n[o']t\b", r"\bcannot\b", r"\bunable to\b", r"\bdid ?n[o']t\b",
    r"\bfailed to\b", r"\bunsupported\b", r"\bnot (implemented|supported|allowed)\b",

    # --- timeouts / limits / capacity ---
    r"\btimed ?out\b", r"timeout\b", r"\btimeout(error)?\b",
    r"\brate[ -]?limit", r"\bquota\b", r"\bthrottl(e|ed|ing)\b",
    r"\bexceeded\b", r"\boverloaded\b", r"\bcapacity\b", r"\btoo many\b",
    r"\bcontext (window|length)\b", r"\bmax(imum)? tokens?\b",
    r"\bout of (memory|range|bounds)\b", r"\boom\b", r"\bmemory ?error\b",

    # --- auth / permission ---
    r"\bunauthori[sz]ed\b", r"\bforbidden\b", r"\bpermission denied\b",
    r"\baccess denied\b", r"\bauthenticat\w* (error|failed)\b",
    r"\binvalid (api ?key|token|credential)", r"\bexpired (token|key|session)\b",
    r"\bunauthenticated\b", r"\bnot authori[sz]ed\b",

    # --- network / connectivity ---
    r"\bconnection (refused|reset|error|aborted|closed)\b", r"\bconnreset\b",
    r"\bunreachable\b", r"\bdns\b.{0,15}(fail|error)", r"\bssl\b.{0,15}(error|fail)",
    r"\bmax retries\b", r"\bretries exceeded\b", r"\bbroken pipe\b",
    r"\bnetwork (error|failure|unreachable)\b", r"\bsocket\b.{0,15}(error|closed)",

    # --- HTTP statuses & phrases ---
    r"\bnot found\b", r"\bbad request\b", r"\bbad gateway\b",
    r"\bservice unavailable\b", r"\bgateway timeout\b",
    r"\binternal server error\b", r"\btoo many requests\b",
    r"\bserver error\b", r"\bupstream (error|timeout|fail)",
    r"\b(?:HTTP|status|code|err\w*)\b\W{0,10}[45]\d\d\b",
    r"\b[45]\d\d\b\W{0,4}(?:error|unavailable|gateway|forbidden|unauthori|"
    r"not\s+found|too\s+many|bad\s+request|internal|server|timeout)",

    # --- data / lookup / state ---
    r"\bnot found\b", r"\bmissing\b.{0,20}(key|field|value|argument|param)",
    r"\bno (data|results?|response|rows?|records?) (found|returned|available)\b",
    r"\bnull\b.{0,10}(pointer|reference)", r"\bundefined\b.{0,15}(is not|error)",
    r"\bkeyerror\b", r"\bindexerror\b", r"\battributeerror\b",
    r"\btypeerror\b", r"\bvalueerror\b",

    # --- process / execution / infra ---
    r"\bnon[- ]?zero\b", r"\bexit (status|code)\s*[1-9]", r"\breturned \d+\b.{0,10}error",
    r"\bsegfault\b", r"\bsegmentation fault\b", r"\bcore dumped\b", r"\bkilled\b",
    r"\bdeadlock\b", r"\broll(?:ed)?[ -]?back\b", r"\brejected\b", r"\bdenied\b",
    r"\bdisconnected\b",

    # --- CamelCase exception class names (KeyError, JSONDecodeError, …) ---
    r"\b[A-Z][A-Za-z0-9]*Errors?\b", r"\b[A-Z][A-Za-z0-9]*Exceptions?\b",
    r"\b[A-Z][A-Za-z0-9]*(Timeout|Refused|Failure)\b",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)

# Negation / benign contexts that should NOT count as a failure. Real agents
# routinely say "no errors", "without error", "error handling", "error: none",
# "0 errors", "successfully" near the word error. These would cause false
# positives on healthy runs, so we strip matches that sit in such a context.
_NEGATION_RE = re.compile(
    r"(no|none|without|zero|0|free of|handled?|handling|catch|caught|ignore[ds]?|"
    r"avoid(ed|ing)?|prevent(ed|ing)?)\s+"
    r"(\w+\s+){0,2}(error|exception|failure|fault)s?"
    r"|(error|exception|failure)s?\s*[:=]\s*(none|null|0|\[\]|\{\}|false)"
    r"|(error|exception|failure|timeout|timeouts|retry|retries)\s+"
    r"(handling|handler|handlers|logic|config\w*|case|cases|message|"
    r"messages|boundary|boundaries|rate|recovery|path|state|policy|"
    r"policies|wrapper|strategy|strategies|behaviou?r)"
    r"|(success(fully)?|passed|completed)\s+(\w+\s+){0,3}(no|without|zero)\s+"
    r"(\w+\s+){0,1}(error|failure)"
    r"|no issues",
    re.IGNORECASE,
)


def _output_str(data: dict) -> str:
    out = data.get("outputs")
    return "" if out is None else str(out)


def _text_has_error_signal(text: str) -> bool:
    """
    True if text contains an error signal that is NOT in a negated/benign context.

    Strategy: find each error-keyword hit and inspect a small window around it.
    If that window matches a negation pattern, the hit is discounted. Only an
    un-negated hit counts as a real error signal.
    """
    if not text:
        return False
    for m in _ERROR_RE.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 15)
        window = text[start:end]
        if not _NEGATION_RE.search(window):
            return True
    return False


def _has_error(data: dict) -> bool:
    """A node 'has an error' if it threw an exception OR its output carries an
    un-negated error signal."""
    if not data.get("success", True):
        return True
    return _text_has_error_signal(_output_str(data))


def detect_symptom(dag: nx.DiGraph) -> str | None:
    """
    The SYMPTOM is where the failure becomes visible = the most DOWNSTREAM node
    that has an error (exception OR error-signal output). Walking in topological
    order and taking the LAST hit handles exceptions and silent failures the same
    way, and is stable for loops/branches.
    """
    try:
        order = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        order = list(dag.nodes)

    symptom = None
    for node_id in order:
        if _has_error(dag.nodes[node_id]):
            symptom = node_id
    return symptom


def _is_introducer(dag: nx.DiGraph, node_id: str) -> bool:
    """
    True if this node has an error but NONE of its upstream ancestors do.

    Using the full ancestor cone (not just direct predecessors) is what makes
    blame robust: even if the data-flow graph linearises a merge node, an error
    that originated several hops upstream still disqualifies every downstream
    node from being the 'introducer'. Only the true origin survives.
    """
    if not _has_error(dag.nodes[node_id]):
        return False
    for anc in nx.ancestors(dag, node_id):
        if _has_error(dag.nodes[anc]):
            return False
    return True


def score_suspects(dag: nx.DiGraph, symptom_id: str) -> list[dict]:
    """Rank the symptom's ancestors (plus the symptom) by likelihood of being root cause."""
    candidates = set(nx.ancestors(dag, symptom_id))
    candidates.add(symptom_id)  # symptom can also be the introducer
    if not candidates:
        return []

    # distances to the symptom for the proximity bonus
    distances: dict[str, int] = {}
    for c in candidates:
        if c == symptom_id:
            distances[c] = 0
            continue
        try:
            distances[c] = nx.shortest_path_length(dag, c, symptom_id)
        except nx.NetworkXNoPath:
            distances[c] = 99
    max_dist = max(distances.values()) or 1

    results = []
    for node_id in candidates:
        data = dag.nodes[node_id]
        score = 0.0
        reasons = []

        if _is_introducer(dag, node_id):
            score += 0.50
            reasons.append("INTRODUCED the error (inputs were clean)")

        node_has_error = _has_error(data)

        # tool bonus only when the tool ITSELF carries the error — a clean tool
        # that merely ran near the failure should not be inflated.
        if "tool:" in data.get("node_name", "") and node_has_error:
            score += 0.25
            reasons.append("errored tool call (external data source)")

        if node_has_error:
            score += 0.20
            reasons.append("output contains error signal")

        dist = distances.get(node_id, max_dist)
        if node_id == symptom_id or dag.has_edge(node_id, symptom_id):
            score += 0.15
            reasons.append("feeds directly into the symptom")
        else:
            proximity = 1.0 - (dist - 1) / max(max_dist, 1)
            score += 0.15 * max(proximity, 0)

        results.append({
            "step_id": node_id,
            "node_name": data.get("node_name", node_id),
            "score": round(min(score, 1.0), 3),
            "is_symptom": node_id == symptom_id,
            "reasons": reasons,
            "outputs_preview": _output_str(data)[:120] or "(no output)",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def run_blame_analysis(dataflow_dag: nx.DiGraph) -> dict:
    """Top-level: detect the symptom, rank suspects, identify the root cause."""
    symptom_id = detect_symptom(dataflow_dag)
    if not symptom_id:
        return {"has_failure": False, "symptom": None, "suspects": [], "root_cause": None}

    suspects = score_suspects(dataflow_dag, symptom_id)
    sdata = dataflow_dag.nodes[symptom_id]

    return {
        "has_failure": True,
        "symptom": {
            "step_id": symptom_id,
            "node_name": sdata.get("node_name", symptom_id),
            "outputs_preview": _output_str(sdata)[:120],
        },
        "suspects": suspects,
        "root_cause": suspects[0] if suspects else None,
    }
