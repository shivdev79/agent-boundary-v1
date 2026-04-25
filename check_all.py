"""Full project health check."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from server.agentv1_environment import AgentBoundaryEnvironment
from server.grader import grade_action, RUBRIC_WEIGHTS
from server.task_bank import TASK_BANK
from models import AgentBoundaryAction, JudgmentDecision, EscalationTarget, QuestionFocus
from evaluation.common import run_policy
from evaluation.policies import (random_policy, expert_policy, weak_policy,
                                  heuristic_policy, trained_policy, has_trained_policy)

def read(p): return pathlib.Path(p).read_text(encoding="utf-8")

passed, failed = [], []
def check(name, result, detail=""):
    (passed if result else failed).append((name, detail))

env = AgentBoundaryEnvironment()

# ── 1. Environment ─────────────────────────────────────────────────────────
for task in TASK_BANK:
    obs = env.reset(task_id=task.task_id)
    check("reset:" + task.task_id, obs.task_id == task.task_id and not obs.done and obs.turn_index == 0)

obs = env.reset(task_id="easy-public-release")
obs = env.step(AgentBoundaryAction(decision=JudgmentDecision.ACT,
    justification="Approved public release grounded in policy evidence.",
    audit_note="approved public release completed standard workflow."))
check("step:done",           obs.done)
check("step:reward-range",   -1.0 <= obs.reward <= 1.0)
check("step:8-rubric-keys",  len(obs.rubric_breakdown) == 8)
check("step:audit-log",      bool(obs.audit_log))
check("step:decision-hist",  bool(obs.decision_history))
check("step:outcome",        bool(obs.outcome))

obs = env.reset(task_id="easy-public-release")
obs = env.step(AgentBoundaryAction(decision=JudgmentDecision.ESCALATE,
    justification="Escalating this request for safety review of approval.",
    escalation_target=EscalationTarget.MANAGER,
    audit_note="escalated to manager for safety review."))
check("timeout:hard-cutoff", obs.done)

obs = env.reset(task_id="medium-vendor-bank-change")
obs = env.step(AgentBoundaryAction(decision=JudgmentDecision.ASK,
    justification="Need callback verification before bank change.",
    question_focus=QuestionFocus.OUT_OF_BAND_VERIFICATION,
    audit_note="bank callback verification missing approval."))
check("ask:response-revealed",  bool(obs.response_to_question))
check("ask:not-done",           not obs.done)
obs = env.step(AgentBoundaryAction(decision=JudgmentDecision.ASK,
    justification="Asking again redundantly without new evidence here.",
    question_focus=QuestionFocus.APPROVAL_PROOF,
    audit_note="redundant second ask attempt here again."))
check("ask:second-blocked",     obs.done)

obs = env.reset(task_id="long-horizon-quarter-close-queue")
steps = 0
while not obs.done and steps < 20:
    obs = env.step(AgentBoundaryAction(decision=JudgmentDecision.ESCALATE,
        justification="Escalating for safety review of the request.",
        escalation_target=EscalationTarget.LEGAL,
        audit_note="escalated to legal for privacy review."))
    steps += 1
check("multistep:terminates", obs.done)
check("multistep:no-runaway", steps <= 10, "steps=%d" % steps)

# ── 2. Reward design ───────────────────────────────────────────────────────
check("reward:8-components",   len(RUBRIC_WEIGHTS) == 8)
check("reward:weights-sum-1",  round(sum(RUBRIC_WEIGHTS.values()), 10) == 1.0,
      "sum=%.4f" % sum(RUBRIC_WEIGHTS.values()))

all_beat = True
for task in TASK_BANK:
    for stage in task.stages:
        qf   = stage.question_bonus   or QuestionFocus.NONE
        et   = stage.escalation_bonus or EscalationTarget.NONE
        tool = stage.expected_tool or (stage.available_tools[0] if stage.available_tools else "")
        kw   = " ".join(stage.note_keywords or [])
        expert_a = AgentBoundaryAction(decision=stage.recommended_decision,
            justification="Decision grounded in risk signals and policy evidence for " + kw + ".",
            question_focus=qf, escalation_target=et, selected_tool=tool,
            audit_note=kw + " decision grounded in policy and risk signals.")
        hack_a = AgentBoundaryAction(decision=JudgmentDecision.ESCALATE,
            justification="Escalating by default to stay safe.",
            escalation_target=EscalationTarget.MANAGER, audit_note="Escalated by default.")
        wrong_a = AgentBoundaryAction(decision=JudgmentDecision.ACT,
            justification="ok", audit_note="done")
        ge = grade_action(stage, expert_a)
        gh = grade_action(stage, hack_a)
        gw = grade_action(stage, wrong_a)
        if ge.reward <= gh.reward or ge.reward <= gw.reward:
            all_beat = False
check("reward:expert-beats-all-10-stages", all_beat)

stage0 = TASK_BANK[2].stages[0]
g_grounded = grade_action(stage0, AgentBoundaryAction(decision=JudgmentDecision.ASK,
    justification="Missing callback verification and bank approval controls.",
    question_focus=QuestionFocus.OUT_OF_BAND_VERIFICATION, selected_tool="vendor_callback_log",
    audit_note="bank callback verification missing controller approval."))
g_hollow = grade_action(stage0, AgentBoundaryAction(decision=JudgmentDecision.ASK,
    justification="I need to ask about this before proceeding with anything.",
    question_focus=QuestionFocus.OUT_OF_BAND_VERIFICATION, selected_tool="vendor_callback_log",
    audit_note="bank callback verification missing controller approval."))
check("reward:evidence-grounding", g_grounded.reward > g_hollow.reward,
      "grounded=%+.3f hollow=%+.3f" % (g_grounded.reward, g_hollow.reward))

# ── 3. Policy comparison ───────────────────────────────────────────────────
seeds = list(range(5))
rand_r   = run_policy("random",    random_policy,    seeds=seeds)["average_reward"]
weak_r   = run_policy("weak",      weak_policy,      seeds=seeds)["average_reward"]
heur_r   = run_policy("heuristic", heuristic_policy, seeds=seeds)["average_reward"]
expert_r = run_policy("expert",    expert_policy,    seeds=seeds)["average_reward"]
check("policy:random-positive",        rand_r > 0,        "rand=%.3f" % rand_r)
check("policy:expert-beats-random",    expert_r > rand_r, "exp=%.3f" % expert_r)
check("policy:heuristic-beats-random", heur_r > rand_r)
check("policy:trained-exists",         has_trained_policy())
if has_trained_policy():
    tr_r = run_policy("trained", trained_policy, seeds=seeds)["average_reward"]
    check("policy:trained-beats-random",    tr_r > rand_r,       "tr=%.3f" % tr_r)
    check("policy:trained-beats-heuristic", tr_r > heur_r,       "tr=%.3f h=%.3f" % (tr_r, heur_r))
    check("policy:trained-3x-random",       tr_r > rand_r * 2.5, "%.1fx" % (tr_r / rand_r))

# ── 4. Artifacts ───────────────────────────────────────────────────────────
for name, p in [
    ("training-curve",    "artifacts/training/training_curve.png"),
    ("policy-comparison", "artifacts/evaluation/policy_comparison.json"),
    ("training-summary",  "artifacts/training/training_summary.json"),
    ("policy-weights",    "artifacts/training/policy_weights.json"),
    ("supervised-traces", "artifacts/supervised_traces.jsonl"),
    ("comparison-csv",    "artifacts/evaluation/policy_comparison.csv"),
]:
    check("artifact:" + name, pathlib.Path(p).exists())

summary = json.loads(read("artifacts/training/training_summary.json"))
check("artifact:trained-reward>1", summary["final_trained_policy"]["average_reward"] > 1.0,
      str(summary["final_trained_policy"]["average_reward"]))

# ── 5. Training scripts ────────────────────────────────────────────────────
grpo = read("training/train_grpo.py")
llm  = read("training/train_llm_grpo.py")
check("training:live-env-loop",     "env.step" in grpo)
check("training:curriculum",        "_curriculum_pool" in grpo)
check("training:rubric-monitoring", "rubric_avg" in grpo)
check("training:decision-dist",     "decision_distribution" in grpo)
check("training:neg-ep-count",      "negative_reward_episodes" in grpo)
check("training:grpo-sample-print", "_reward_call_count" in llm)
check("training:malformed-tracked", "malformed" in llm)
check("training:easy-oversample",   "repeat * 2" in llm)
check("training:correct-save",      "merged_16bit" in llm)
check("training:no-naive-merge",    "merge_and_unload" not in llm)
check("training:post-eval",         "run_post_training_eval" in llm)
check("training:dry-run",           "dry-run" in llm)
check("training:fp16-t4",           "fp16=True" in llm and "bf16=False" in llm)

# ── 6. Deploy ──────────────────────────────────────────────────────────────
check("deploy:dockerfile",   pathlib.Path("Dockerfile").exists())
check("deploy:openenv-yaml", pathlib.Path("openenv.yaml").exists())
check("deploy:hf-space",     "huggingface.co/spaces" in read("README.md"))
check("deploy:writeup",      pathlib.Path("WRITEUP.md").exists())
check("deploy:concurrent",   "SUPPORTS_CONCURRENT_SESSIONS: bool = True" in
      read("server/agentv1_environment.py"))

# ── 7. WRITEUP consistency ─────────────────────────────────────────────────
writeup = read("WRITEUP.md")
check("writeup:hf-link",      "huggingface.co/spaces/Shivanshu31" in writeup)
check("writeup:github-link",  "shivdev79/agent-boundary-v1" in writeup)
check("writeup:results-table","1.32" in writeup)
writeup_says_seven = "Seven" in writeup and "eight" not in writeup.lower()
check("writeup:reward-count-matches-code", not writeup_says_seven,
      "WRITEUP says Seven but grader has 8 components")

# ── Summary ────────────────────────────────────────────────────────────────
total = len(passed) + len(failed)
print("PASSED: %d / %d" % (len(passed), total))
if failed:
    print("\nFAILED (%d):" % len(failed))
    for name, detail in failed:
        print("  X  %-48s %s" % (name, detail))
else:
    print("\nAll %d checks passed." % total)
