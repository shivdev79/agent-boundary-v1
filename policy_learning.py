"""Shared policy-learning helpers for AgentBoundary-v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

try:
    from .models import (
        AgentBoundaryAction,
        EscalationTarget,
        JudgmentDecision,
        QuestionFocus,
    )
except ImportError:  # pragma: no cover
    from models import AgentBoundaryAction, EscalationTarget, JudgmentDecision, QuestionFocus


ACTIONS: List[JudgmentDecision] = [
    JudgmentDecision.ACT,
    JudgmentDecision.ASK,
    JudgmentDecision.ESCALATE,
    JudgmentDecision.REFUSE,
]

FEATURE_NAMES: List[str] = [
    "bias",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "mode_single",
    "mode_batch",
    "mode_long_horizon",
    "mode_adversarial",
    "channel_email",
    "channel_sms",
    "channel_chat",
    "channel_ticket",
    "has_response",
    "pending_ratio",
    "resolved_ratio",
    "turn_progress",
    "kw_public",
    "kw_approved",
    "kw_payroll",
    "kw_external",
    "kw_bank",
    "kw_callback",
    "kw_approval",
    "kw_missing",
    "kw_urgent",
    "kw_secrecy",
    "kw_gmail",
    "kw_social",
    "kw_legal",
    "kw_privacy",
    "kw_access",
    "kw_dashboard",
    "kw_incident",
    "kw_secure",
    "kw_customer",
    "kw_export",
]


def _obs_text(obs) -> str:
    return " ".join(
        [
            obs.requested_action,
            obs.summary,
            obs.response_to_question,
            *obs.visible_facts,
            *obs.risk_signals,
            *obs.policy_snippets,
        ]
    ).lower()


def _has_any(text: str, *terms: str) -> float:
    return float(any(term in text for term in terms))


def extract_features(obs) -> np.ndarray:
    text = _obs_text(obs)
    channel = obs.communication_channel.lower()
    pending_den = max(1, obs.pending_request_count + obs.resolved_request_count)
    turn_den = max(1, obs.max_turns)

    values = [
        1.0,
        float(obs.difficulty.value == "easy"),
        float(obs.difficulty.value == "medium"),
        float(obs.difficulty.value == "hard"),
        float(obs.workflow_mode.value == "single_request"),
        float(obs.workflow_mode.value == "batch_triage"),
        float(obs.workflow_mode.value == "long_horizon_workflow"),
        float(obs.workflow_mode.value == "adversarial_workflow"),
        float("email" in channel),
        float("sms" in channel),
        float("slack" in channel or "chat" in channel),
        float("ticket" in channel),
        float(bool(obs.response_to_question)),
        obs.pending_request_count / pending_den,
        obs.resolved_request_count / pending_den,
        obs.turn_index / turn_den,
        _has_any(text, "public", "status page", "press kit", "website"),
        _has_any(text, "approved", "approval attached", "documented approval"),
        _has_any(text, "payroll", "salary", "bonus"),
        _has_any(text, "external", "public email", "unknown sender", "personal email"),
        _has_any(text, "bank", "treasury", "payment batch"),
        _has_any(text, "callback", "call back", "verification"),
        _has_any(text, "approval", "approved", "manager approval"),
        _has_any(text, "missing", "not completed", "still missing", "no ticket"),
        _has_any(text, "urgent", "immediately", "20 minutes", "cutoff", "tonight"),
        _has_any(text, "secrecy", "do not loop in", "must not loop in", "secret"),
        _has_any(text, "gmail", "personal gmail", "mailbox"),
        _has_any(text, "social engineering", "impersonation", "unrecognized number"),
        _has_any(text, "legal", "matter number", "legal hold"),
        _has_any(text, "privacy", "employee records", "named contacts"),
        _has_any(text, "access", "grant temporary read access"),
        _has_any(text, "dashboard", "restricted dashboard"),
        _has_any(text, "incident", "incident commander", "security playbook"),
        _has_any(text, "secure destination", "secure transfer", "secure path"),
        _has_any(text, "customer", "enterprise customers", "retention"),
        _has_any(text, "export", "send the export", "data export"),
    ]
    return np.asarray(values, dtype=np.float64)


def select_tool(obs) -> str:
    if not obs.available_tools:
        return ""

    text = _obs_text(obs)
    preferred_groups = [
        ("approval", ["approval_metadata", "approval_matrix", "access_request_form"]),
        ("public", ["approval_metadata", "status_page_preview", "cms_change_preview"]),
        ("callback", ["vendor_callback_log", "change_ticket", "controller_roster"]),
        ("bank", ["vendor_callback_log", "controller_roster", "treasury_queue"]),
        ("gmail", ["identity_lookup", "security_playbook", "incident_roster"]),
        ("social", ["security_playbook", "identity_lookup", "incident_roster"]),
        ("payroll", ["data_classification_registry", "vendor_directory", "legal_hold_lookup"]),
        ("incident", ["incident_ticket", "approval_matrix", "incident_roster"]),
        ("access", ["approval_matrix", "incident_ticket", "access_request_form"]),
        ("legal", ["legal_hold_lookup", "incident_roster"]),
    ]
    for keyword, candidates in preferred_groups:
        if keyword in text:
            for candidate in candidates:
                if candidate in obs.available_tools:
                    return candidate
    return obs.available_tools[0]


def infer_question_focus(obs) -> QuestionFocus:
    text = _obs_text(obs)
    if "gmail" in text or "secure" in text or "personal email" in text:
        return QuestionFocus.SECURE_DESTINATION
    if "approval" in text or "manager approval" in text:
        return QuestionFocus.APPROVAL_PROOF
    if "callback" in text or "bank" in text or "verification" in text:
        return QuestionFocus.OUT_OF_BAND_VERIFICATION
    if "classification" in text or "restricted" in text:
        return QuestionFocus.DATA_CLASSIFICATION
    return QuestionFocus.BUSINESS_JUSTIFICATION


def infer_escalation_target(obs) -> EscalationTarget:
    text = _obs_text(obs)
    if "bank" in text or "callback" in text or "controller" in text or "treasury" in text:
        return EscalationTarget.FINANCE_CONTROLLER
    if "gmail" in text or "social engineering" in text or "unrecognized number" in text or "secrecy" in text:
        return EscalationTarget.SECURITY
    if "legal" in text or "employee records" in text or "matter number" in text:
        return EscalationTarget.LEGAL
    if "privacy" in text:
        return EscalationTarget.PRIVACY
    return EscalationTarget.MANAGER


def build_action(obs, decision: JudgmentDecision) -> AgentBoundaryAction:
    text = _obs_text(obs)
    tool = select_tool(obs)

    if decision == JudgmentDecision.ACT:
        justification = "Approvals and visible evidence support routine execution through the standard workflow."
        audit_note = "Approved public request completed through the standard workflow."
        if "public" not in text and "approved" not in text:
            justification = "Visible evidence supports completing the request without extra routing."
            audit_note = "Completed approved request after checking the surfaced evidence."
        return AgentBoundaryAction(
            decision=decision,
            justification=justification,
            selected_tool=tool,
            audit_note=audit_note,
        )

    if decision == JudgmentDecision.ASK:
        focus = infer_question_focus(obs)
        reason_map = {
            QuestionFocus.SECURE_DESTINATION: (
                "Need a verifiable secure destination before any transfer can proceed.",
                "Asked for secure destination evidence before processing request.",
            ),
            QuestionFocus.APPROVAL_PROOF: (
                "Need the missing approval evidence before proceeding.",
                "Asked for documented approval proof before proceeding.",
            ),
            QuestionFocus.OUT_OF_BAND_VERIFICATION: (
                "Need the missing callback or verification evidence before taking action.",
                "Asked for callback verification before processing request.",
            ),
            QuestionFocus.DATA_CLASSIFICATION: (
                "Need the data classification and handling rules clarified before acting.",
                "Asked for data-classification evidence before acting.",
            ),
            QuestionFocus.BUSINESS_JUSTIFICATION: (
                "Need the business justification clarified before taking action.",
                "Asked for business justification before taking action.",
            ),
        }
        justification, audit_note = reason_map[focus]
        return AgentBoundaryAction(
            decision=decision,
            justification=justification,
            question_focus=focus,
            selected_tool=tool,
            audit_note=audit_note,
        )

    if decision == JudgmentDecision.ESCALATE:
        target = infer_escalation_target(obs)
        if target == EscalationTarget.FINANCE_CONTROLLER:
            justification = "Missing bank-change controls require controller review instead of direct action."
            audit_note = "Escalated bank-change request to controller because callback controls are missing."
        elif target == EscalationTarget.SECURITY:
            justification = "Off-channel urgency and sensitive data indicators cross the security incident threshold."
            audit_note = "Escalated likely social-engineering request to security."
        elif target == EscalationTarget.LEGAL:
            justification = "Sensitive records request needs legal owner review before any data movement."
            audit_note = "Escalated sensitive records request to legal for accountable review."
        elif target == EscalationTarget.PRIVACY:
            justification = "Privacy-sensitive records handling belongs with the privacy owner."
            audit_note = "Escalated privacy-sensitive request to privacy owner."
        else:
            justification = "The accountable approval owner should resolve the missing authorization path."
            audit_note = "Escalated request to accountable manager because approval path is incomplete."
        return AgentBoundaryAction(
            decision=decision,
            justification=justification,
            escalation_target=target,
            selected_tool=tool,
            audit_note=audit_note,
        )

    justification = "The request is clearly disallowed by the surfaced policy and should be denied."
    audit_note = "Denied restricted external request based on policy."
    if "payroll" in text:
        justification = "External disclosure of payroll data is prohibited by policy."
        audit_note = "Denied external payroll request and protected restricted employee data."
    return AgentBoundaryAction(
        decision=decision,
        justification=justification,
        selected_tool=tool,
        audit_note=audit_note,
    )


class LinearDecisionPolicy:
    """A lightweight softmax policy over hand-built observation features."""

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else np.zeros(
            (len(ACTIONS), len(FEATURE_NAMES)),
            dtype=np.float64,
        )

    def action_probs(self, obs) -> np.ndarray:
        features = extract_features(obs)
        logits = self.weights @ features
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def choose_decision(self, obs, rng: np.random.Generator, greedy: bool = False) -> JudgmentDecision:
        probs = self.action_probs(obs)
        if greedy:
            return ACTIONS[int(np.argmax(probs))]
        return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]

    def choose_action(self, obs, rng: np.random.Generator, greedy: bool = False) -> AgentBoundaryAction:
        decision = self.choose_decision(obs, rng=rng, greedy=greedy)
        return build_action(obs, decision)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": FEATURE_NAMES,
            "action_names": [action.value for action in ACTIONS],
            "weights": self.weights.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearDecisionPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(weights=np.asarray(payload["weights"], dtype=np.float64))
