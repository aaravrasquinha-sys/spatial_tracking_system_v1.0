"""Repo-root conftest: makes `pytest` work from a fresh checkout even before
`pip install -e .`, and pins CWD-independent behaviour for the whole suite."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
