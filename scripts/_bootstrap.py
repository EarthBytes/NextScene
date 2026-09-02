"""Side-effect import: add scripts/ and backend/ to sys.path for CLI entry points."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_BACKEND = _SCRIPTS.parent / "backend"
for path in (_SCRIPTS, _BACKEND):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
