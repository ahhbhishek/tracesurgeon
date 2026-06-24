"""
Comprehensive real-world error-detection corpus.

Two goals, both must score high:
  RECALL    — catch real errors from actual providers/tools/frameworks
  PRECISION — do NOT flag benign text that merely mentions error-ish words

This is the gate that says "almost every real-world error pattern is catered".

Run with:
    cd Projects/tracesurgeon
    python tests/test_error_corpus.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon.scorer import _text_has_error_signal as E


# ---- REAL ERRORS (must be detected = True) ---------------------------------
ERRORS = [
    # OpenAI / Anthropic / LLM providers
    "openai.RateLimitError: Error code: 429 - Rate limit reached for gpt-4",
    "anthropic.APIStatusError: Error code: 529 - Overloaded",
    "openai.AuthenticationError: Error code: 401 - Incorrect API key provided",
    "anthropic.BadRequestError: Error code: 400 - prompt is too long",
    "openai.APITimeoutError: Request timed out.",
    "openai.APIConnectionError: Connection error.",
    "litellm.ContextWindowExceededError: max tokens exceeded for this model",
    "anthropic.InternalServerError: Error code: 500",
    "This model's maximum context length is 8192 tokens",
    "InvalidRequestError: The model `gpt-5` does not exist",
    "openai.PermissionDeniedError: 403",
    # HTTP / network
    "requests.exceptions.ConnectionError: Max retries exceeded with url",
    "httpx.ReadTimeout",
    "httpx.ConnectTimeout: timed out",
    "HTTP 502 Bad Gateway",
    "HTTP 429 Too Many Requests",
    "HTTP 503 Service Unavailable",
    "HTTP 504 Gateway Timeout",
    "urllib3.exceptions.MaxRetryError",
    "ssl.SSLError: certificate verify failed",
    "socket.gaierror: Name or service not known",
    "ConnectionResetError: [Errno 104] Connection reset by peer",
    "BrokenPipeError: [Errno 32] Broken pipe",
    # stdlib exceptions
    "json.decoder.JSONDecodeError: Expecting value: line 1 column 1",
    "KeyError: 'results'",
    "IndexError: list index out of range",
    "TypeError: 'NoneType' object is not subscriptable",
    "ValueError: invalid literal for int() with base 10",
    "AttributeError: 'dict' object has no attribute 'foo'",
    "FileNotFoundError: [Errno 2] No such file or directory",
    "asyncio.TimeoutError",
    "RecursionError: maximum recursion depth exceeded",
    "MemoryError",
    "ZeroDivisionError: division by zero",
    "Traceback (most recent call last):",
    # database
    "sqlalchemy.exc.OperationalError: connection refused",
    "psycopg2.OperationalError: could not connect to server",
    "pymongo.errors.ServerSelectionTimeoutError",
    "deadlock detected",
    "transaction rolled back",
    # process / infra
    "subprocess returned non-zero exit status 1",
    "Command failed with exit code 127",
    "Process killed (OOM)",
    "Segmentation fault (core dumped)",
    "docker: Error response from daemon",
    # tool / agent specific
    "Tool execution failed: permission denied",
    "The model produced invalid JSON for tool arguments",
    "ValidationError: 1 validation error for ToolInput",
    "ToolException: search failed",
    "Could not parse LLM output",
    "No results found for the query",
    "Authentication failed: invalid api key",
    "ERROR: census API returned HTTP 503, data unavailable for this city",
    "Request rejected: quota exceeded",
    "Operation aborted after 3 retries",
]


# ---- BENIGN (must NOT be flagged = False) ----------------------------------
BENIGN = [
    "Validation complete: no errors found.",
    "The build completed successfully with 0 errors.",
    "This function includes robust error handling for edge cases.",
    "error: none",
    "All 42 tests passed without failure.",
    "We caught the exception and recovered gracefully.",
    "Paris is the capital of France. Population: 2.1 million.",
    "The combined population is 16.1 million.",
    "Successfully fetched 100 records from the database.",
    "no exceptions were thrown during the run",
    "The error rate decreased to 0% after the fix.",  # tricky: 'error' but benign
    "Implemented retry logic and timeout handling for resilience.",
    "Summary: the data is valid and complete.",
    "Order #404 shipped; total was $500 for 502 units.",  # bare numbers, no http ctx
    "The recipe needs 500 grams of flour and 429 ml of water.",
    "Status: ok. All systems operational.",
    "Found 3 results matching your query.",
    "The exception_handler module is documented here.",
    "Tokyo has a population of 14 million people.",
    "Completed with no issues.",
]


def main():
    rec_pass = sum(1 for t in ERRORS if E(t))
    prec_pass = sum(1 for t in BENIGN if not E(t))

    print("RECALL — real errors that MUST be detected:")
    for t in ERRORS:
        if not E(t):
            print(f"  [MISS] {t}")
    print(f"  recall: {rec_pass}/{len(ERRORS)} "
          f"({100*rec_pass/len(ERRORS):.0f}%)\n")

    print("PRECISION — benign text that must NOT be flagged:")
    for t in BENIGN:
        if E(t):
            print(f"  [FALSE+] {t}")
    print(f"  precision: {prec_pass}/{len(BENIGN)} "
          f"({100*prec_pass/len(BENIGN):.0f}%)\n")

    ok = rec_pass == len(ERRORS) and prec_pass == len(BENIGN)
    print(f"  {'ALL PASS' if ok else 'NEEDS WORK'} — "
          f"recall {rec_pass}/{len(ERRORS)}, precision {prec_pass}/{len(BENIGN)}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
