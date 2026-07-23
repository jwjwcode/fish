#!/usr/bin/env python3
"""Launch the fish-feeding activity tuning GUI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fish_activity.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
