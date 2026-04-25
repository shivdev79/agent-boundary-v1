"""Online policy-gradient training against the live AgentBoundary-v1 environment.

Despite the historical filename, this script now runs end-to-end training
directly against the environment instead of optimizing over a static trace file.
It learns a lightweight decision policy, writes reproducible artifacts, and
generates a readable training curve for the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.common import run_policy, write_json
from evaluation.policies import expert_policy, heuristic_policy, random_policy, weak_policy
from policy_learning import ACTIONS, LinearDecisionPolicy, extract_features

try:
    from agentv1.server.agentv1_environment import AgentBoundaryEnvironment
except ImportError:  # pragma: no cover
    from server.agentv1_environment import AgentBoundaryEnvironment


def evaluate_policy(policy: LinearDecisionPolicy, seeds: list[int]) -> dict:
    rng = np.random.default_rng(0)
    return run_policy("trained", lambda obs: policy.choose_action(obs, rng=rng, greedy=True), seeds=seeds)


def _write_training_plot(history: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [row["episode"] for row in history["evaluations"]]
    rewards = [row["average_reward"] for row in history["evaluations"]]
    scores = [row["average_score"] for row in history["evaluations"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(episodes, rewards, color="#3c78d8", linewidth=2.5, label="trained policy")
    for name, value in history["baseline_reward_lines"].items():
        axes[0].axhline(value, linestyle="--", linewidth=1.2, label=name)
    axes[0].set_title("Training Reward Curve")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Average evaluation reward")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(episodes, scores, color="#6aa84f", linewidth=2.5, label="trained policy")
    for name, value in history["baseline_score_lines"].items():
        axes[1].axhline(value, linestyle="--", linewidth=1.2, label=name)
    axes[1].set_title("Training Score Curve")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Average normalized score")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)


def main() -> None:
    output_dir = ROOT / "artifacts" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = AgentBoundaryEnvironment()
    policy = LinearDecisionPolicy()
    rng = np.random.default_rng(7)

    train_episodes = 600
    eval_interval = 25
    learning_rate = 0.045
    entropy_bonus = 0.01
    baseline = 0.0
    eval_seeds = list(range(5))

    baseline_runs = {
        "random": run_policy("random", random_policy, seeds=eval_seeds),
        "weak": run_policy("weak", weak_policy, seeds=eval_seeds),
        "heuristic": run_policy("heuristic", heuristic_policy, seeds=eval_seeds),
        "expert": run_policy("expert", expert_policy, seeds=eval_seeds),
    }

    history = {
        "config": {
            "train_episodes": train_episodes,
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "entropy_bonus": entropy_bonus,
        },
        "baseline_reward_lines": {name: run["average_reward"] for name, run in baseline_runs.items()},
        "baseline_score_lines": {name: run["average_score"] for name, run in baseline_runs.items()},
        "evaluations": [],
    }

    for episode_idx in range(1, train_episodes + 1):
        obs = env.reset(seed=int(rng.integers(0, 10_000)))
        trajectory = []

        while True:
            features = extract_features(obs)
            probs = policy.action_probs(obs)
            action = policy.choose_action(obs, rng=rng, greedy=False)
            next_obs = env.step(action)
            trajectory.append(
                {
                    "features": features,
                    "probs": probs,
                    "action_index": [a.value for a in ACTIONS].index(action.decision.value),
                    "reward": next_obs.reward or 0.0,
                }
            )
            obs = next_obs
            if obs.done:
                break

        returns = []
        running_return = 0.0
        for step in reversed(trajectory):
            running_return += step["reward"]
            returns.append(running_return)
        returns.reverse()

        for step, return_value in zip(trajectory, returns):
            baseline = 0.9 * baseline + 0.1 * return_value
            advantage = return_value - baseline
            one_hot = np.zeros(len(ACTIONS), dtype=np.float64)
            one_hot[step["action_index"]] = 1.0
            grad_logits = (one_hot - step["probs"])[:, None] * step["features"][None, :]
            entropy_grad = (step["probs"] - (1.0 / len(ACTIONS)))[:, None] * step["features"][None, :]
            policy.weights += learning_rate * (advantage * grad_logits - entropy_bonus * entropy_grad)

        if episode_idx % eval_interval == 0 or episode_idx == 1:
            evaluation = evaluate_policy(policy, seeds=eval_seeds)
            history["evaluations"].append(
                {
                    "episode": episode_idx,
                    "average_reward": evaluation["average_reward"],
                    "average_score": evaluation["average_score"],
                }
            )

    policy.save(output_dir / "policy_weights.json")
    write_json(output_dir / "training_history.json", history)

    final_evaluation = evaluate_policy(policy, seeds=eval_seeds)
    write_json(
        output_dir / "training_summary.json",
        {
            "final_trained_policy": {
                "average_reward": final_evaluation["average_reward"],
                "average_score": final_evaluation["average_score"],
            },
            "baselines": {
                name: {
                    "average_reward": result["average_reward"],
                    "average_score": result["average_score"],
                }
                for name, result in baseline_runs.items()
            },
        },
    )
    _write_training_plot(history, output_dir / "training_curve.png")
    print(json.dumps({"training_summary": final_evaluation, "artifacts_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
