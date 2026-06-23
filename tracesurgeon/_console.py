"""Console helpers — make output safe on any terminal (esp. Windows cp1252)."""

import sys


def enable_utf8() -> None:
    """
    Force stdout/stderr to UTF-8 so box-drawing and check glyphs don't crash
    on Windows consoles or when output is piped/redirected to a file.
    Safe no-op on streams that don't support reconfigure.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
