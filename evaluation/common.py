"""Shared evaluation helpers for AgentBoundary-v1."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agentv1 import AgentBoundaryAction
    from agentv1.server.agentv1_environment import AgentBoundaryEnvironment
except ImportError:  # pragma: no cover
    from models import AgentBoundaryAction
    from server.agentv1_environment import AgentBoundaryEnvironment


PolicyFn = Callable[[object], AgentBoundaryAction]


@dataclass
class StepSummary:
    request_id: str
    decision: str
    reward: float
    cumulative_score: float
    rubric_breakdown: Dict[str, float]


@dataclass
class EpisodeSummary:
    seed: int
    task_id: str
    workflow_mode: str
    cumulative_score: float
    cumulative_reward: float
    decision_history: List[str]
    steps: List[StepSummary]


def run_policy(policy_name: str, choose_action: PolicyFn, seeds: Sequence[int] | None = None) -> Dict:
    env = AgentBoundaryEnvironment()
    seeds = list(seeds) if seeds is not None else list(range(5))
    episodes: List[EpisodeSummary] = []

    for seed in seeds:
        obs = env.reset(seed=seed)
        step_summaries: List[StepSummary] = []
        while True:
            action = choose_action(obs)
            obs = env.step(action)
            step_summaries.append(
                StepSummary(
                    request_id=obs.current_request_id,
                    decision=action.decision.value,
                    reward=obs.reward or 0.0,
                    cumulative_score=float(obs.metadata.get("cumulative_score", 0.0)),
                    rubric_breakdown=obs.rubric_breakdown,
                )
            )
            if obs.done:
                episodes.append(
                    EpisodeSummary(
                        seed=seed,
                        task_id=obs.task_id,
                        workflow_mode=obs.workflow_mode.value,
                        cumulative_score=float(obs.metadata.get("cumulative_score", 0.0)),
                        cumulative_reward=float(obs.metadata.get("cumulative_reward", 0.0)),
                        decision_history=obs.decision_history,
                        steps=step_summaries,
                    )
                )
                break

    return {
        "policy": policy_name,
        "episodes": [asdict(ep) for ep in episodes],
        "average_score": round(sum(ep.cumulative_score for ep in episodes) / len(episodes), 3),
        "average_reward": round(sum(ep.cumulative_reward for ep in episodes) / len(episodes), 3),
    }


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
