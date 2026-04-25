"""AgentBoundary-v1 environment implementation."""

from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AgentBoundaryAction,
        AgentBoundaryObservation,
        AgentBoundaryState,
        WorkflowMode,
    )
    from .grader import grade_action
    from .task_bank import TASK_BANK, TASK_BY_ID, TaskCase, TaskStage
except ImportError:  # pragma: no cover
    from models import AgentBoundaryAction, AgentBoundaryObservation, AgentBoundaryState, WorkflowMode
    from server.grader import grade_action
    from server.task_bank import TASK_BANK, TASK_BY_ID, TaskCase, TaskStage


class AgentBoundaryEnvironment(Environment):
    """OpenEnv environment for training calibrated autonomy and safe delegation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._reset_count = 0
        self._task: TaskCase = TASK_BANK[0]
        self._state = self._new_state(task_seed=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> AgentBoundaryObservation:
        """Reset the environment to a deterministic workflow episode."""
        del kwargs

        if task_id is not None:
            if task_id not in TASK_BY_ID:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. Valid ids: {sorted(TASK_BY_ID.keys())}"
                )
            self._task = TASK_BY_ID[task_id]
            task_seed = seed if seed is not None else self._reset_count
        elif seed is None:
            self._task = TASK_BANK[self._reset_count % len(TASK_BANK)]
            task_seed = self._reset_count
        else:
            self._task = TASK_BANK[seed % len(TASK_BANK)]
            task_seed = seed

        self._reset_count += 1
        self._state = self._new_state(task_seed=task_seed, episode_id=episode_id)
        return self._build_observation(
            response_to_question="",
            outcome="Start by reviewing the active request and deciding whether to ACT, ASK, ESCALATE, or REFUSE.",
            done=False,
            reward=0.0,
            rubric_breakdown={},
        )

    def step(
        self,
        action: AgentBoundaryAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> AgentBoundaryObservation:
        """Advance the workflow by one deterministic judgment step."""
        del timeout_s, kwargs
        stage = self._current_stage
        self._state.step_count += 1
        self._state.decision_history.append(f"{stage.request_id}:{action.decision.value}")

        audit_line = self._format_audit_line(stage, action)
        self._state.audit_log.append(audit_line)

        grade = grade_action(stage, action)
        self._state.cumulative_score = round(self._state.cumulative_score + grade.score, 3)
        self._state.cumulative_reward = round(self._state.cumulative_reward + grade.reward, 3)

        at_turn_limit = self._state.step_count >= self._task.max_turns

        if action.decision.value == "ASK" and stage.next_stage_on_ask is not None and not self._state.used_ask:
            self._state.stage_index = stage.next_stage_on_ask
            self._state.current_request_id = self._task.stages[self._state.stage_index].request_id
            self._state.used_ask = True
            return self._build_observation(
                response_to_question=stage.response_to_question,
                outcome=grade.outcome,
                done=at_turn_limit,
                reward=grade.reward,
                rubric_breakdown=grade.rubric_breakdown,
                action=action,
            )

        self._mark_request_completed(stage.request_id)

        next_index = self._next_stage_index_after_resolution(stage)
        if next_index is not None and not at_turn_limit:
            self._state.stage_index = next_index
            self._state.current_request_id = self._task.stages[self._state.stage_index].request_id
            next_stage = self._current_stage
            return self._build_observation(
                response_to_question="",
                outcome=f"{grade.outcome} Next request: {next_stage.request_summary}",
                done=False,
                reward=grade.reward,
                rubric_breakdown=grade.rubric_breakdown,
                action=action,
            )

        final_outcome = (
            f"{grade.outcome} Workflow complete. "
            f"You resolved {len(self._state.completed_request_ids)} request(s) with "
            f"cumulative score {self._state.cumulative_score:.3f} and reward {self._state.cumulative_reward:.3f}."
        )
        return self._build_observation(
            response_to_question="",
            outcome=final_outcome,
            done=True,
            reward=grade.reward,
            rubric_breakdown=grade.rubric_breakdown,
            action=action,
        )

    @property
    def state(self) -> AgentBoundaryState:
        """Return the serializable environment state."""
        return self._state

    @property
    def _current_stage(self) -> TaskStage:
        return self._task.stages[self._state.stage_index]

    def _new_state(
        self,
        task_seed: int,
        episode_id: Optional[str] = None,
    ) -> AgentBoundaryState:
        first_stage = self._task.stages[0]
        return AgentBoundaryState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            workflow_mode=self._task.workflow_mode,
            stage_index=0,
            current_request_id=first_stage.request_id,
            max_turns=self._task.max_turns,
            task_seed=task_seed,
            used_ask=False,
            decision_history=[],
            completed_request_ids=[],
            audit_log=[],
            cumulative_score=0.0,
            cumulative_reward=0.0,
        )

    def _mark_request_completed(self, request_id: str) -> None:
        if request_id not in self._state.completed_request_ids:
            self._state.completed_request_ids.append(request_id)

    def _next_stage_index_after_resolution(self, stage: TaskStage) -> Optional[int]:
        next_index = self._state.stage_index + 1
        while next_index < len(self._task.stages):
            if self._task.stages[next_index].request_id != stage.request_id:
                return next_index
            next_index += 1
        return None

    def _queue_overview(self) -> List[str]:
        seen = set()
        lines: List[str] = []
        for stage in self._task.stages:
            if stage.request_id in seen:
                continue
            seen.add(stage.request_id)
            status = "resolved" if stage.request_id in self._state.completed_request_ids else "pending"
            lines.append(f"{stage.request_id}: {stage.request_summary} [{status}]")
        return lines

    def _pending_request_count(self) -> int:
        request_ids = []
        seen = set()
        for stage in self._task.stages:
            if stage.request_id not in seen:
                seen.add(stage.request_id)
                request_ids.append(stage.request_id)
        return len([rid for rid in request_ids if rid not in self._state.completed_request_ids])

    def _format_audit_line(self, stage: TaskStage, action: AgentBoundaryAction) -> str:
        tool_fragment = f" via {action.selected_tool}" if action.selected_tool else ""
        note_fragment = f" note={action.audit_note.strip()}" if action.audit_note.strip() else ""
        return (
            f"{stage.request_id}: {action.decision.value} "
            f"for {stage.requested_action}{tool_fragment};"
            f" justification={action.justification.strip()}{note_fragment}"
        )

    def _build_observation(
        self,
        response_to_question: str,
        outcome: str,
        done: bool,
        reward: float,
        rubric_breakdown: dict,
        action: Optional[AgentBoundaryAction] = None,
    ) -> AgentBoundaryObservation:
        stage = self._current_stage
        metadata = {
            "task_seed": self._state.task_seed,
            "difficulty": self._task.difficulty.value,
            "workflow_mode": self._task.workflow_mode.value,
            "current_request_id": stage.request_id,
            "cumulative_score": self._state.cumulative_score,
            "cumulative_reward": self._state.cumulative_reward,
            "recommended_turns": self._task.max_turns,
        }
        if action is not None:
            metadata["last_decision"] = action.decision.value
            metadata["question_focus"] = action.question_focus.value
            metadata["escalation_target"] = action.escalation_target.value
            metadata["selected_tool"] = action.selected_tool

        return AgentBoundaryObservation(
            done=done,
            reward=reward,
            metadata=metadata,
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            workflow_mode=self._task.workflow_mode,
            turn_index=self._state.step_count,
            max_turns=self._task.max_turns,
            current_request_id=stage.request_id,
            requester_name=stage.requester_name,
            requester_role=stage.requester_role,
            communication_channel=stage.communication_channel,
            requested_action=stage.requested_action,
            summary=stage.request_summary,
            overall_summary=self._task.overall_summary,
            visible_facts=stage.visible_facts,
            risk_signals=stage.risk_signals,
            policy_snippets=stage.policy_snippets,
            stakeholders=stage.stakeholders,
            available_tools=stage.available_tools,
            queue_overview=self._queue_overview(),
            pending_request_count=self._pending_request_count(),
            resolved_request_count=len(self._state.completed_request_ids),
            response_to_question=response_to_question,
            decision_history=list(self._state.decision_history),
            audit_log=list(self._state.audit_log),
            rubric_breakdown=rubric_breakdown,
            outcome=outcome,
        )


# Backwards-compatible alias for existing imports.
Agentv1Environment = AgentBoundaryEnvironment
