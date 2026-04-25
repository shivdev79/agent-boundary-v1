"""Client wrapper for the AgentBoundary-v1 environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        AgentBoundaryAction,
        AgentBoundaryObservation,
        AgentBoundaryState,
        TaskDifficulty,
        WorkflowMode,
    )
except ImportError:  # pragma: no cover
    from models import (
        AgentBoundaryAction,
        AgentBoundaryObservation,
        AgentBoundaryState,
        TaskDifficulty,
        WorkflowMode,
    )


class AgentBoundaryEnv(EnvClient[AgentBoundaryAction, AgentBoundaryObservation, AgentBoundaryState]):
    """Client for persistent interaction with the AgentBoundary-v1 environment."""

    def _step_payload(self, action: AgentBoundaryAction) -> Dict:
        return {
            "decision": action.decision.value,
            "justification": action.justification,
            "question_focus": action.question_focus.value,
            "escalation_target": action.escalation_target.value,
            "selected_tool": action.selected_tool,
            "audit_note": action.audit_note,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AgentBoundaryObservation]:
        obs_data = payload.get("observation", {})
        observation = AgentBoundaryObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            task_id=obs_data.get("task_id", ""),
            task_title=obs_data.get("task_title", ""),
            difficulty=TaskDifficulty(obs_data.get("difficulty", TaskDifficulty.EASY.value)),
            workflow_mode=WorkflowMode(obs_data.get("workflow_mode", WorkflowMode.SINGLE.value)),
            turn_index=obs_data.get("turn_index", 0),
            max_turns=obs_data.get("max_turns", 1),
            current_request_id=obs_data.get("current_request_id", ""),
            requester_name=obs_data.get("requester_name", ""),
            requester_role=obs_data.get("requester_role", ""),
            communication_channel=obs_data.get("communication_channel", ""),
            requested_action=obs_data.get("requested_action", ""),
            summary=obs_data.get("summary", ""),
            overall_summary=obs_data.get("overall_summary", ""),
            visible_facts=obs_data.get("visible_facts", []),
            risk_signals=obs_data.get("risk_signals", []),
            policy_snippets=obs_data.get("policy_snippets", []),
            stakeholders=obs_data.get("stakeholders", []),
            available_tools=obs_data.get("available_tools", []),
            queue_overview=obs_data.get("queue_overview", []),
            pending_request_count=obs_data.get("pending_request_count", 0),
            resolved_request_count=obs_data.get("resolved_request_count", 0),
            response_to_question=obs_data.get("response_to_question", ""),
            decision_history=obs_data.get("decision_history", []),
            audit_log=obs_data.get("audit_log", []),
            rubric_breakdown=obs_data.get("rubric_breakdown", {}),
            outcome=obs_data.get("outcome", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AgentBoundaryState:
        return AgentBoundaryState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=TaskDifficulty(payload.get("difficulty", TaskDifficulty.EASY.value)),
            workflow_mode=WorkflowMode(payload.get("workflow_mode", WorkflowMode.SINGLE.value)),
            stage_index=payload.get("stage_index", 0),
            current_request_id=payload.get("current_request_id", ""),
            max_turns=payload.get("max_turns", 1),
            task_seed=payload.get("task_seed", 0),
            used_ask=payload.get("used_ask", False),
            decision_history=payload.get("decision_history", []),
            completed_request_ids=payload.get("completed_request_ids", []),
            audit_log=payload.get("audit_log", []),
            cumulative_score=payload.get("cumulative_score", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )


# Backwards-compatible alias.
Agentv1Env = AgentBoundaryEnv
