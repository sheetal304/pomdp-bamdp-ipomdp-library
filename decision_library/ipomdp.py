"""
ipomdp.py - A SIMPLIFIED Interactive POMDP demo (Tiger problem, 2 agents).

SETUP: Agent i (us) and another agent j are both in the Tiger problem.
Each hears the tiger with its OWN independent noisy ear. Agent i can SEE
j's actions (listen / open a door) but NOT j's hearing or j's belief.

j has a hidden TYPE = how confident it must be before opening a door:
    hasty   -> threshold 0.60 (opens after ~1 listen)
    careful -> threshold 0.99 (listens several times first)

Agent i keeps ONE joint belief over (j's type, tiger state), and updates it
from (a) its own hearing and (b) the actions it watches j take.
Watching j helps for two reasons: it reveals j's type, and a confident j
opening a door is a hint about where the tiger is.

LIMITATIONS (be honest in the interview):
  * Only ONE level of reasoning: i models j, but j does NOT model i.
  * j's "mind" is just one number (its threshold) - only 2 simple types.
  * j's behaviour is a fixed rule, not a full solved POMDP policy.
  * This is an illustration of the idea, NOT full I-POMDP theory.
"""
import numpy as np
from .pomdp import make_tiger_pomdp
from .utils import check_distribution, normalize

ACTIONS = ["listen", "open-left", "open-right"]
STATES = ["tiger-left", "tiger-right"]
DEFAULT_TYPES = {"hasty": 0.60, "careful": 0.99}


def steps_to_threshold(threshold, accuracy):
    """Net number of 'same-side' sounds j needs before its belief reaches `threshold`.

    After d net sounds, j's belief = 1 / (1 + ((1-acc)/acc)^d), so we need
    d >= ln(thr/(1-thr)) / ln(acc/(1-acc)).
    """
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must be in (0.5, 1)")
    if not 0.5 < accuracy < 1.0:
        raise ValueError("accuracy must be in (0.5, 1)")
    need = np.log(threshold / (1 - threshold)) / np.log(accuracy / (1 - accuracy))
    return int(np.ceil(need - 1e-9))


def other_action_probs(threshold, accuracy, tiger_state, n_rounds):
    """P(j's action | j still playing, tiger_state), for each round.

    Returns array of shape (n_rounds, 3) with columns [listen, open-left, open-right].
    j keeps a counter d = (#hear-left - #hear-right). It opens a door once
    |d| reaches dstar; it opens the door it believes is SAFE.
    """
    if tiger_state not in STATES:
        raise ValueError(f"unknown state: {tiger_state!r}")
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    dstar = steps_to_threshold(threshold, accuracy)
    size = 2 * dstar + 1                      # index = d + dstar
    dist = np.zeros(size)
    dist[dstar] = 1.0                         # start with d = 0
    p_hear_left = accuracy if tiger_state == "tiger-left" else 1 - accuracy
    out = np.zeros((n_rounds, 3))

    for r in range(n_rounds):
        alive = dist.sum()
        if alive <= 0:
            continue                          # nobody left; row stays 0
        open_right = dist[2 * dstar]          # d = +dstar -> thinks LEFT -> opens right
        open_left = dist[0]                   # d = -dstar -> thinks RIGHT -> opens left
        listen = alive - open_left - open_right
        out[r] = np.array([listen, open_left, open_right]) / alive
        new = np.zeros(size)
        for d in range(-dstar + 1, dstar):    # only those still listening
            m = dist[d + dstar]
            new[d + 1 + dstar] += m * p_hear_left
            new[d - 1 + dstar] += m * (1 - p_hear_left)
        dist = new
    return out


class IPOMDPAgent:
    """Agent i: reasons about the tiger AND about the type of agent j."""

    def __init__(self, other_types=None, own_accuracy=0.85, other_accuracy=0.85,
                 type_prior=None, n_rounds=20):
        self.other_types = dict(other_types or DEFAULT_TYPES)
        self.names = list(self.other_types)
        self.own_accuracy = own_accuracy
        self.n_rounds = n_rounds
        prior = (np.full(len(self.names), 1 / len(self.names))
                 if type_prior is None else check_distribution(type_prior, "type_prior"))
        if len(prior) != len(self.names):
            raise ValueError("type_prior length must equal number of types")
        # joint belief: rows = j's type, columns = tiger state
        self.joint = np.outer(prior, [0.5, 0.5])
        self._probs = [[other_action_probs(self.other_types[t], other_accuracy, s, n_rounds)
                        for s in STATES] for t in self.names]
        self._pomdp = make_tiger_pomdp(own_accuracy)
        self.round = 0

    # ---- reading the belief ----
    def type_belief(self):
        return dict(zip(self.names, self.joint.sum(axis=1)))

    def state_belief(self):
        return self.joint.sum(axis=0)

    def best_action(self):
        return self._pomdp.best_action_now(self.state_belief())

    # ---- updating the belief ----
    def hear(self, observation):
        """Update from agent i's OWN hearing."""
        if observation not in ("hear-left", "hear-right"):
            raise ValueError(f"unknown observation: {observation!r}")
        a = self.own_accuracy
        like = np.array([a, 1 - a]) if observation == "hear-left" else np.array([1 - a, a])
        self._apply(np.tile(like, (len(self.names), 1)))

    def watch_other(self, action):
        """Update from the action agent j took this round."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        if self.round >= self.n_rounds:
            raise ValueError("watched more rounds than n_rounds")
        k = ACTIONS.index(action)
        like = np.array([[self._probs[t][s][self.round, k] for s in range(2)]
                         for t in range(len(self.names))])
        self.round += 1
        self._apply(like)

    def _apply(self, likelihood):
        updated = self.joint * likelihood
        if updated.sum() <= 0:
            raise ValueError("this observation is impossible under the current belief")
        self.joint = updated / updated.sum()


def simulate_interaction(true_type="careful", true_state="tiger-left", n_rounds=6,
                         accuracy=0.85, seed=0, watch_other=True, other_types=None):
    """Play n_rounds. Agent i listens every round; j acts by its true type."""
    types = dict(other_types or DEFAULT_TYPES)
    if true_type not in types:
        raise ValueError(f"unknown type: {true_type!r}")
    if true_state not in STATES:
        raise ValueError(f"unknown state: {true_state!r}")
    rng = np.random.default_rng(seed)
    agent = IPOMDPAgent(types, own_accuracy=accuracy, other_accuracy=accuracy,
                        n_rounds=max(n_rounds, 1))
    dstar = steps_to_threshold(types[true_type], accuracy)
    left = true_state == "tiger-left"
    d, j_active, history = 0, True, []

    for _ in range(n_rounds):
        j_action = "done"                                   # j already opened a door earlier
        if j_active:
            if d >= dstar:
                j_action, j_active = "open-right", False
            elif d <= -dstar:
                j_action, j_active = "open-left", False
            else:
                j_action = "listen"
            if watch_other:
                agent.watch_other(j_action)
            if j_action == "listen":                       # j hears this round
                d += 1 if (rng.random() < accuracy) == left else -1
        heard_left = (rng.random() < accuracy) == left      # i hears
        agent.hear("hear-left" if heard_left else "hear-right")
        history.append({"j_action": j_action,
                        "p_careful": float(agent.type_belief().get("careful", np.nan)),
                        "p_tiger_left": float(agent.state_belief()[0])})
    return {"history": history, "agent": agent}


def compare_with_without_other(n_trials=500, n_rounds=4, accuracy=0.85, seed=0):
    """Does watching j help i find the tiger? Average probability i gives to the TRUE state."""
    rng = np.random.default_rng(seed)
    with_j, without_j = [], []
    for _ in range(n_trials):
        t = rng.choice(list(DEFAULT_TYPES))
        s = STATES[int(rng.integers(2))]
        trial_seed = int(rng.integers(1_000_000))
        for flag, store in [(True, with_j), (False, without_j)]:
            out = simulate_interaction(t, s, n_rounds, accuracy, trial_seed, watch_other=flag)
            store.append(out["agent"].state_belief()[STATES.index(s)])
    return {"with_other": float(np.mean(with_j)), "without_other": float(np.mean(without_j))}
