"""Generate supervised traces from the deterministic environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agentv1 import AgentBoundaryAction, EscalationTarget, JudgmentDecision, QuestionFocus
    from agentv1.server.task_bank import TASK_BANK
except ImportError:  # pragma: no cover
    from models import AgentBoundaryAction, EscalationTarget, JudgmentDecision, QuestionFocus
    from server.task_bank import TASK_BANK


def expert_action(stage) -> AgentBoundaryAction:
    question_focus = stage.question_bonus or QuestionFocus.NONE
    escalation_target = stage.escalation_bonus or EscalationTarget.NONE
    selected_tool = stage.expected_tool or (stage.available_tools[0] if stage.available_tools else "")
    note_keywords = ", ".join(stage.note_keywords or [])
    return AgentBoundaryAction(
        decision=stage.recommended_decision,
        justification=f"Follow policy and risk signals for {stage.request_id}.",
        question_focus=question_focus,
        escalation_target=escalation_target,
        selected_tool=selected_tool,
        audit_note=f"Decision grounded in {note_keywords}." if note_keywords else "Decision grounded in policy evidence.",
    )


def main() -> None:
    out_path = Path("artifacts")
    out_path.mkdir(exist_ok=True)
    records = []

    for task in TASK_BANK:
        for stage in task.stages:
            records.append(
                {
                    "prompt": json.dumps(
                        {
                            "task_id": task.task_id,
                            "task_title": task.title,
                            "difficulty": task.difficulty.value,
                            "workflow_mode": task.workflow_mode.value,
                            "overall_summary": task.overall_summary,
                            "request_id": stage.request_id,
                            "request_summary": stage.request_summary,
                            "requester_role": stage.requester_role,
                            "requested_action": stage.requested_action,
                            "visible_facts": stage.visible_facts,
                            "risk_signals": stage.risk_signals,
                            "policy_snippets": stage.policy_snippets,
                            "available_tools": stage.available_tools,
                        }
                    ),
                    "completion": expert_action(stage).model_dump_json(),
                }
            )

    with (out_path / "supervised_traces.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} records to {out_path / 'supervised_traces.jsonl'}")


if __name__ == "__main__":
    main()
