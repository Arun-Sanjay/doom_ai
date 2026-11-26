import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "submission"), str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds (skipped in CI with -m 'not slow')")
