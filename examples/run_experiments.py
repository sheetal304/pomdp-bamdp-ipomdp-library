"""Runs all experiments, saves plots (PNG) and a results table (Markdown) in results/.
Run from the project root:  python examples/run_experiments.py"""
import os, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_library.pomdp import make_tiger_pomdp
from decision_library.bamdp import run_learning_experiment, TigerBAMDPAgent
from decision_library.ipomdp import compare_with_without_other

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
md = ["# Experiment results", "",
      "All numbers below were produced by `examples/run_experiments.py` (fixed random seeds).", ""]


# ---------------- Experiment 1: POMDP policies ----------------
def play_episode(model, policy, rng, max_steps=50):
    state = model.states[int(rng.integers(2))]
    belief, total = np.array([0.5, 0.5]), 0.0
    for _ in range(max_steps):
        action = policy(model, belief, rng)
        state, obs, reward = model.sample_step(state, action, rng)
        total += reward
        if action != "listen":
            return total
        belief = model.update_belief(belief, action, obs)
    return total


def random_door(model, belief, rng):
    return str(rng.choice(["open-left", "open-right"]))


def belief_greedy(model, belief, rng):
    return model.best_action_now(belief)


model = make_tiger_pomdp(0.85)
n = 5000
md += ["## Experiment 1 - POMDP: does using the belief help? (Tiger, accuracy 0.85)", "",
       f"Average total reward per episode over {n} episodes (higher is better).", "",
       "| Policy | Mean reward | Std. error |", "|---|---|---|"]
names = {"Open a random door immediately": random_door,
         "Belief-based (listen until confident)": belief_greedy}
for name, pol in names.items():
    rng = np.random.default_rng(0)
    r = np.array([play_episode(model, pol, rng) for _ in range(n)])
    md.append(f"| {name} | {r.mean():.2f} | {r.std(ddof=1) / np.sqrt(n):.2f} |")
md.append("")

fig, ax = plt.subplots(figsize=(6, 4))
for acc in [0.65, 0.75, 0.85, 0.95]:
    m, b, ps = make_tiger_pomdp(acc), np.array([0.5, 0.5]), [0.5]
    for _ in range(10):
        b = m.update_belief(b, "listen", "hear-left")
        ps.append(b[0])
    ax.plot(range(11), ps, marker="o", label=f"accuracy {acc}")
ax.axhline(0.9, color="gray", ls="--", label="open-door threshold (0.9)")
ax.set(xlabel="number of 'hear-left' sounds", ylabel="P(tiger-left)",
       title="POMDP: belief after repeated 'hear-left'")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "pomdp_belief.png"), dpi=130); plt.close(fig)

# ---------------- Experiment 2: BAMDP learning ----------------
fig, ax = plt.subplots(figsize=(6, 4))
for seed in range(3):
    out = run_learning_experiment(0.85, 200, 3, seed=seed)
    ax.plot(out["means"], label=f"seed {seed}")
ax.axhline(0.85, color="black", ls="--", label="true accuracy 0.85")
ax.set(xlabel="episode", ylabel="learned accuracy (posterior mean)",
       title="BAMDP: learning the sensor accuracy")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "bamdp_learning.png"), dpi=130); plt.close(fig)

md += ["## Experiment 2 - BAMDP: learning the sensor accuracy", "",
       "200 episodes x 3 listens each, 20 different seeds per row. Error = |learned - true|.", "",
       "| True accuracy | Mean learned | Mean abs. error | Mean std (uncertainty) | Listens needed (learned) |",
       "|---|---|---|---|---|"]
for true_acc in [0.65, 0.75, 0.85, 0.95]:
    runs = [run_learning_experiment(true_acc, 200, 3, seed=s) for s in range(20)]
    final = np.array([r["means"][-1] for r in runs])
    std = np.mean([r["stds"][-1] for r in runs])
    listens = np.mean([r["agent"].listens_needed() for r in runs])
    md.append(f"| {true_acc} | {final.mean():.3f} | {np.abs(final - true_acc).mean():.3f} "
              f"| {std:.3f} | {listens:.1f} |")
md += ["", f"Reference: an agent with no knowledge (prior Beta(1,1)) needs "
           f"{TigerBAMDPAgent().listens_needed()} listens.", ""]

# ---------------- Experiment 3: IPOMDP ----------------
rounds = [2, 3, 4, 6]
res = [compare_with_without_other(2000, r, seed=0) for r in rounds]
md += ["## Experiment 3 - simplified IPOMDP: does watching the other agent help?", "",
       "2000 random trials per row (random tiger side, random type of j). "
       "Value = average probability agent i assigns to the TRUE tiger side.", "",
       "| Rounds | Watching j | Ignoring j | Difference |", "|---|---|---|---|"]
for r, x in zip(rounds, res):
    md.append(f"| {r} | {x['with_other']:.4f} | {x['without_other']:.4f} "
              f"| {x['with_other'] - x['without_other']:+.4f} |")
md.append("")
fig, ax = plt.subplots(figsize=(6, 4))
w = 0.38
xs = np.arange(len(rounds))
ax.bar(xs - w / 2, [x["without_other"] for x in res], w, label="ignoring j")
ax.bar(xs + w / 2, [x["with_other"] for x in res], w, label="watching j")
ax.set_xticks(xs); ax.set_xticklabels(rounds)
ax.set(xlabel="rounds played", ylabel="P(true tiger side)", ylim=(0.8, 1.0),
       title="IPOMDP: value of watching the other agent")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ipomdp_watching.png"), dpi=130); plt.close(fig)

open(os.path.join(RESULTS, "results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
