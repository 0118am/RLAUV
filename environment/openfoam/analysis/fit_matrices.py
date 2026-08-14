#!/usr/bin/env python3
"""Compatibility wrapper for the hydrodynamic-analysis CLI."""

from __future__ import annotations

from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from environment.openfoam.analysis.__main__ import main
else:
    from .__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
