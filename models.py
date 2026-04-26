"""Typed models for the AgentBoundary-v1 OpenEnv environment."""

from enum import Enum
from typing import Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


class JudgmentDecision(str, Enum):
    ACT = "ACT"
    ASK = "ASK"
    ESCALATE = "ESCALATE"
    REFUSE = "REFUSE"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class WorkflowMode(str, Enum):
    SINGLE = "single_request"
    BATCH = "batch_triage"
    LONG_HORIZON = "long_horizon_workflow"
    ADVERSARIAL = "adversarial_workflow"


class QuestionFocus(str, Enum):
    NONE = "NONE"
    APPROVAL_PROOF = "APPROVAL_PROOF"
    OUT_OF_BAND_VERIFICATION = "OUT_OF_BAND_VERIFICATION"
    CHANGE_TICKET = "CHANGE_TICKET"
    SCOPE_JUSTIFICATION = "SCOPE_JUSTIFICATION"
    DATA_CLASSIFICATION = "DATA_CLASSIFICATION"
    SECURE_DESTINATION = "SECURE_DESTINATION"
    BUSINESS_JUSTIFICATION = "BUSINESS_JUSTIFICATION"


class EscalationTarget(str, Enum):
    NONE = "NONE"
    MANAGER = "MANAGER"
    SECURITY = "SECURITY"
    FINANCE_CONTROLLER = "FINANCE_CONTROLLER"
    IT_OPERATIONS = "IT_OPERATIONS"
    LEGAL = "LEGAL"
    PRIVACY = "PRIVACY"


class AgentBoundaryAction(Action):
    """Structured decision taken by the agent for a judgment scenario."""

    decision: JudgmentDecision = Field(..., description="Primary judgment decision.")

    @field_validator("decision", "question_focus", "escalation_target", mode="before")
    @classmethod
    def strip_enum_fields(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v
    justification: str = Field(
        default="",
        description="Brief reason for the decision grounded in the scenario.",
    )
    question_focus: QuestionFocus = Field(
        default=QuestionFocus.NONE,
        description="If asking, what clarification the agent requests.",
    )
    escalation_target: EscalationTarget = Field(
        default=EscalationTarget.NONE,
        description="If escalating, who should receive the escalation.",
    )
    selected_tool: str = Field(
        default="",
        description="Optional tool or evidence source the agent relies on.",
    )
    audit_note: str = Field(
        default="",
        description="Structured note left for the audit trail or handoff log.",
    )


class AgentBoundaryObservation(Observation):
    """Observable task context returned at each step."""

    task_id: str = Field(default="", description="Stable deterministic task identifier.")
    task_title: str = Field(default="", description="Human readable task title.")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Scenario difficulty level.",
    )
    workflow_mode: WorkflowMode = Field(
        default=WorkflowMode.SINGLE,
        description="High-level episode structure.",
    )
    turn_index: int = Field(default=0, description="Current turn within the episode.")
    max_turns: int = Field(default=1, description="Maximum number of turns in the task.")
    current_request_id: str = Field(default="", description="Current request identifier.")
    requester_name: str = Field(default="", description="Requester identity.")
    requester_role: str = Field(default="", description="Requester role.")
    communication_channel: str = Field(default="", description="Channel used by requester.")
    requested_action: str = Field(default="", description="What the requester wants.")
    summary: str = Field(default="", description="Current request summary.")
    overall_summary: str = Field(default="", description="Episode-level workflow summary.")
    visible_facts: List[str] = Field(
        default_factory=list,
        description="Facts visible to the agent at this step.",
    )
    risk_signals: List[str] = Field(
        default_factory=list,
        description="Notable signals that should influence judgment.",
    )
    policy_snippets: List[str] = Field(
        default_factory=list,
        description="Relevant policy or approval snippets surfaced for the agent.",
    )
    stakeholders: List[str] = Field(
        default_factory=list,
        description="Relevant people or teams with incentives in the request.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Permitted tools or evidence sources the agent may consult.",
    )
    queue_overview: List[str] = Field(
        default_factory=list,
        description="Ordered summary of pending requests in batch workflows.",
    )
    pending_request_count: int = Field(default=0, description="Requests still unresolved.")
    resolved_request_count: int = Field(default=0, description="Requests already resolved.")
    response_to_question: str = Field(
        default="",
        description="Answer revealed after an ASK action, if any.",
    )
    decision_history: List[str] = Field(
        default_factory=list,
        description="Prior decisions made in this episode.",
    )
    audit_log: List[str] = Field(
        default_factory=list,
        description="Audit trail generated during the workflow.",
    )
    rubric_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Deterministic component scores used in reward shaping.",
    )
    outcome: str = Field(
        default="",
        description="Environment outcome after the latest action.",
    )


class AgentBoundaryState(State):
    """Serializable environment state."""

    task_id: str = Field(default="", description="Current task identifier.")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Current task difficulty.",
    )
    workflow_mode: WorkflowMode = Field(
        default=WorkflowMode.SINGLE,
        description="Current workflow structure.",
    )
    stage_index: int = Field(default=0, description="Current stage in the task.")
    current_request_id: str = Field(default="", description="Current request being handled.")
    max_turns: int = Field(default=1, description="Maximum turns available.")
    task_seed: int = Field(default=0, description="Seed used to select the task.")
    used_ask: bool = Field(default=False, description="Whether ASK has already been used.")
    decision_history: List[str] = Field(
        default_factory=list,
        description="Prior decisions stored for reproducibility.",
    )
    completed_request_ids: List[str] = Field(
        default_factory=list,
        description="Requests fully resolved in the episode.",
    )
    audit_log: List[str] = Field(
        default_factory=list,
        description="Workflow audit log generated so far.",
    )
    cumulative_score: float = Field(default=0.0, description="Running normalized score.")
    cumulative_reward: float = Field(default=0.0, description="Running dense reward total.")


# Backwards-compatible aliases for the original template names.
Agentv1Action = AgentBoundaryAction
Agentv1Observation = AgentBoundaryObservation
Agentv1State = AgentBoundaryState
