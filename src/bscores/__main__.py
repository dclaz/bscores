"""Allow ``python -m bscores`` alongside the installed ``bscores`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
