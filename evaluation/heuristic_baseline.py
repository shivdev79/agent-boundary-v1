"""Offline heuristic baseline for quick judgeable comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import run_policy
from evaluation.policies import heuristic_policy


def main() -> None:
    import json

    print(json.dumps(run_policy("heuristic", heuristic_policy), indent=2))


if __name__ == "__main__":
    main()
