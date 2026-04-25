"""Compare multiple offline policies and write plot-ready artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import ROOT, run_policy, write_json
from evaluation.policies import expert_policy, has_trained_policy, heuristic_policy, random_policy, trained_policy, weak_policy


def _flatten_for_csv(results):
    rows = []
    for result in results:
        for episode in result["episodes"]:
            rows.append(
                {
                    "policy": result["policy"],
                    "seed": episode["seed"],
                    "task_id": episode["task_id"],
                    "workflow_mode": episode["workflow_mode"],
                    "cumulative_score": episode["cumulative_score"],
                    "cumulative_reward": episode["cumulative_reward"],
                    "num_steps": len(episode["steps"]),
                }
            )
    return rows


def main() -> None:
    out_dir = ROOT / "artifacts" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_policy("random", random_policy),
        run_policy("weak", weak_policy),
        run_policy("heuristic", heuristic_policy),
        run_policy("expert", expert_policy),
    ]
    if has_trained_policy():
        results.append(run_policy("trained", trained_policy))

    aggregate = {
        "results": results,
        "policy_summary": [
            {
                "policy": result["policy"],
                "average_score": result["average_score"],
                "average_reward": result["average_reward"],
            }
            for result in results
        ],
    }
    write_json(out_dir / "policy_comparison.json", aggregate)

    rows = _flatten_for_csv(results)
    with (out_dir / "policy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "seed",
                "task_id",
                "workflow_mode",
                "cumulative_score",
                "cumulative_reward",
                "num_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
