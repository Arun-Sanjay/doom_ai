"""Import this first in training code that uses the agent modules: it puts submission/ on sys.path.

    from training import paths  # noqa: F401
    from doomnet import spec
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
for _p in (str(SUBMISSION), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
