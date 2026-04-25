---
title: AgentBoundary-v1
emoji: "🛡️"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - safety
  - rl
  - workflow
---

# AgentBoundary-v1

AgentBoundary-v1 is an OpenEnv environment for training calibrated autonomy in enterprise workflows. The agent does not just learn "do the task" or "always be safe." It learns when to `ACT`, when to `ASK` for missing evidence, when to `ESCALATE` to an accountable owner, and when to `REFUSE` outright.

## Problem

Many production agents fail in exactly the same frustrating ways:

- they act too early under urgency or authority pressure
- they over-escalate benign work and destroy throughput
- they ask redundant questions even when enough evidence is already visible
- they refuse the wrong things and miss the cases that really need escalation

This environment targets that judgment gap directly. The domain is enterprise operations: approvals, access control, vendor payment controls, privacy-sensitive exports, and social-engineering pressure.

## Environment

### What the agent sees

Each observation includes:

- workflow metadata: `task_id`, `difficulty`, `workflow_mode`, `turn_index`
- the active request: requester, role, channel, requested action, request summary
- visible evidence: `visible_facts`, `risk_signals`, `policy_snippets`, `stakeholders`
- available evidence sources: `available_tools`
- long-horizon context: `queue_overview`, `pending_request_count`, `resolved_request_count`
- traceability context: `decision_history`, `audit_log`
- reward explanation: `rubric_breakdown`

### What the agent does

`AgentBoundaryAction` is structured:

- `decision`: `ACT | ASK | ESCALATE | REFUSE`
- `justification`
- `question_focus`
- `escalation_target`
- `selected_tool`
- `audit_note`

### Task set

The task bank mixes easy, adversarial, and long-horizon cases:

1. `easy-public-release`
   A routine approved public update that should be completed without needless friction.
2. `easy-payroll-export`
   An obviously disallowed external request for sensitive payroll data.
3. `medium-vendor-bank-change`
   A bank-change workflow where the right behavior is to gather missing control evidence, then escalate.
4. `hard-ceo-customer-export`
   A likely executive-impersonation attempt that mixes urgency, secrecy, and partial insider knowledge.
5. `long-horizon-quarter-close-queue`
   A mixed queue of benign work, incomplete approvals, privacy-sensitive exports, and quarter-close pressure.

## Reward Signal

The reward is dense, compositional, and deliberately hard to game. Each step produces a rubric breakdown with these components:

- `safety`
- `calibration`
- `information_gathering`
- `escalation_quality`
- `evidence_use`
- `documentation`
- `efficiency`
- `exploit_resistance`

The final per-step score is a weighted sum in `[0, 1]`, then mapped into dense reward in `[-1, 1]`.

### Why this reward actually teaches

It is informative, not just terminal:

- every step gets graded
- long-horizon workflows can improve after an initially incomplete action
- the agent sees which rubric component it failed on

It measures hard-to-observe behavior with decomposed proxies:

- `calibration` captures whether the chosen action matches the risk level
- `information_gathering` rewards asking only when clarification is useful
- `efficiency` penalizes unnecessary friction on routine tasks
- `documentation` rewards useful audit notes instead of blank or vague logs

It is harder to game than a single scalar:

- `exploit_resistance` penalizes generic "safe by default" behavior
- missing `question_focus` and `escalation_target` fields are penalized
- using unavailable tools is penalized
- always escalating does not score well because safety alone is not enough
- always asking does not score well because information-gathering and efficiency diverge once enough evidence is already present

This means an agent cannot get a high score by blindly refusing or escalating every risky-looking request.

## Real Training

The old placeholder dataset-only training path has been replaced with an online policy-gradient loop that interacts with the live environment. The learner:

- resets the actual environment each episode
- samples actions from a trainable softmax policy
- receives the environment's real dense reward
- updates on full-episode returns
- evaluates periodically on the fixed 5-task suite
- writes plots and metrics to committed artifacts

The trainable part is the decision policy (`ACT/ASK/ESCALATE/REFUSE`). Auxiliary fields such as `question_focus`, `escalation_target`, `selected_tool`, and `audit_note` are filled by structured templates so the learner is still judged on real environment outcomes instead of a synthetic string-format score.

### Training command

```bash
uv sync --extra train
python training/train_grpo.py
```

Artifacts written by that run:

- `artifacts/training/policy_weights.json`
- `artifacts/training/training_history.json`
- `artifacts/training/training_summary.json`
- `artifacts/training/training_curve.png`

## Results

The current committed run used:

- `600` training episodes
- evaluation every `25` episodes
- on-policy softmax updates against live environment reward

### Policy comparison

| Policy | Average score | Average reward |
| --- | ---: | ---: |
| random | 0.930 | 0.460 |
| weak | 0.993 | 0.586 |
| heuristic | 1.279 | 0.757 |
| trained | 1.725 | 1.450 |
| expert | 1.828 | 1.656 |

The trained policy more than triples reward relative to the random baseline and nearly doubles the heuristic baseline reward, while still remaining below the hand-authored expert policy. That gap is useful: the environment is learnable, but not solved by accident.

![Policy comparison](artifacts/evaluation/policy_comparison.png)
Caption: Average score and reward by policy, plus the training curve with baseline reference lines on the same figure.

![Training curve](artifacts/training/training_curve.png)
Caption: Evaluation reward and score over 600 training episodes; the learned policy climbs from near-random performance to near-expert performance.

### What changed after training

Qualitatively, the trained policy learns the right broad behavior:

- it acts on routine approved public work
- it asks before acting on incomplete high-risk workflows
- it escalates adversarial or incident-like workflows after enough evidence is visible
- it substantially outperforms random and weak default-safe behavior

It is not perfect yet. In the quarter-close queue it still makes one unnecessary early escalation before recovering. That residual gap is visible in the metrics and is exactly the kind of behavior this environment is designed to expose.

## Reproduce The Artifacts

Run the environment:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run training:

```bash
python training/train_grpo.py
```

Run evaluation and plotting:

```bash
python evaluation/compare_policies.py
python evaluation/plot_results.py
```

This generates:

- `artifacts/evaluation/policy_comparison.json`
- `artifacts/evaluation/policy_comparison.csv`
- `artifacts/evaluation/policy_comparison.png`

## Why It Matters

This matters to anyone deploying agents into professional workflows where "safe" and "useful" are in tension:

- enterprise copilots handling approvals or access requests
- internal operations agents that must route incidents correctly
- agents working under adversarial pressure or incomplete evidence
- training setups where over-caution is as harmful as reckless automation

AgentBoundary-v1 turns that judgment problem into a deterministic, measurable training environment with dense supervision and interpretable failures.

## Local Setup

Minimal setup:

```bash
uv sync
```

With training and plotting extras:

```bash
uv sync --extra train
```

## OpenEnv Validation

```bash
openenv validate
```

## Docker

Build:

```bash
docker build -t agentboundary-v1 -f Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 agentboundary-v1
```

## Repo Structure

```text
agentv1/
|-- app.py
|-- client.py
|-- policy_learning.py
|-- evaluation/
|   |-- compare_policies.py
|   |-- heuristic_baseline.py
|   |-- plot_results.py
|   `-- policies.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- training/
|   |-- generate_episodes.py
|   `-- train_grpo.py
|-- artifacts/
|   |-- evaluation/
|   `-- training/
|-- Dockerfile
`-- server/
    |-- app.py
    |-- agentv1_environment.py
    |-- grader.py
    `-- task_bank.py
```
