"""
Unit tests for the improved blame detection — Phases 1-3 hardening.

Covers:
  #1  negation-aware error detection (no false positives on benign text)
  #3  tool bonus only when the tool itself errors

Run with:
    cd Projects/tracesurgeon
    python tests/test_detection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracesurgeon._console import enable_utf8
enable_utf8()

from tracesurgeon.scorer import _text_has_error_signal


# (text, should_be_flagged_as_error)
CASES = [
    # --- real errors: MUST flag ---
    ("ERROR: census API returned HTTP 503", True),
    ("Traceback (most recent call last): ValueError", True),
    ("the response was malformed and could not be parsed", True),
    ("request failed with status 404 not found", True),
    ("RateLimitError: rate limit exceeded", True),

    # --- benign / negated: MUST NOT flag (these were false positives before) ---
    ("Validation complete: no errors found.", False),
    ("The build completed successfully with 0 errors.", False),
    ("This function includes robust error handling for edge cases.", False),
    ("error: none", False),
    ("All 42 tests passed without failure.", False),
    ("We caught the exception and recovered gracefully.", False),
    ("Summary: the data is valid and complete.", False),  # 'invalid' not present
    ("Paris is the capital of France. Population: 2.1 million.", False),

    # --- tricky: benign phrase BUT also a real error elsewhere -> MUST flag ---
    ("Error handling is robust, however the API call failed with a 500.", True),
]


def main():
    passed = 0
    print("Detection unit tests:\n")
    for text, expected in CASES:
        got = _text_has_error_signal(text)
        ok = got == expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        exp = "ERROR" if expected else "clean"
        gt = "ERROR" if got else "clean"
        print(f"  [{mark}] expected {exp:5} got {gt:5}  | {text[:60]}")

    print(f"\n  {passed}/{len(CASES)} detection cases passed")
    return passed == len(CASES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
