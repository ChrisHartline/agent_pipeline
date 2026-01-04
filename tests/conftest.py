import sys
from pathlib import Path

# The project is installed in editable mode during CI and local dev via `pip install -e .`.
# Tests should import packages normally (no sys.path hacks needed).
ROOT = Path(__file__).resolve().parents[1]

