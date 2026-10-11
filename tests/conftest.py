from __future__ import annotations

import sys
from pathlib import Path

# Keep tests runnable via `pytest -q` from the repository root and CI runners.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
