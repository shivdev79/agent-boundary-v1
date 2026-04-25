"""Generate readable comparison plots from evaluation artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    artifact = ROOT / "artifacts" / "evaluation" / "policy_comparison.json"
    if not artifact.exists():
        raise FileNotFoundError("Run evaluation/compare_policies.py first.")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    summary = payload["policy_summary"]
    training_artifact = ROOT / "artifacts" / "training" / "training_history.json"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install the train extras.") from exc

    out_dir = ROOT / "artifacts" / "evaluation"
    policies = [row["policy"] for row in summary]
    scores = [row["average_score"] for row in summary]
    rewards = [row["average_reward"] for row in summary]

    palette = {
        "random": "#999999",
        "weak": "#b85450",
        "heuristic": "#f6b26b",
        "expert": "#6aa84f",
        "trained": "#3c78d8",
    }
    colors = [palette.get(policy, "#777777") for policy in policies]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(policies, scores, color=colors)
    axes[0].set_title("Average Score by Policy")
    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Normalized cumulative score")
    axes[0].set_ylim(0, max(scores) + 0.5)

    axes[1].bar(policies, rewards, color=colors)
    axes[1].set_title("Average Reward by Policy")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Episode reward")
    axes[1].set_ylim(min(0, min(rewards) - 0.5), max(rewards) + 0.5)

    if training_artifact.exists():
        history = json.loads(training_artifact.read_text(encoding="utf-8"))
        xs = [row["episode"] for row in history["evaluations"]]
        ys = [row["average_reward"] for row in history["evaluations"]]
        axes[2].plot(xs, ys, color=palette["trained"], linewidth=2.5, label="trained policy")
        baselines = history.get("baseline_reward_lines", {})
        for name, value in baselines.items():
            axes[2].axhline(value, linestyle="--", linewidth=1.2, color=palette.get(name, "#777777"), label=name)
        axes[2].set_title("Training Curve")
        axes[2].set_xlabel("Training episode")
        axes[2].set_ylabel("Evaluation reward")
        axes[2].legend(frameon=False, fontsize=8)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Run training/train_grpo.py\nfor training curves.", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "policy_comparison.png", dpi=160)
    print(f"Wrote {out_dir / 'policy_comparison.png'}")


if __name__ == "__main__":
    main()
