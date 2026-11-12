"""
bamdp.py - A SIMPLIFIED Bayes-Adaptive MDP demo (Tiger problem version).

IDEA: In the POMDP module the sensor accuracy was GIVEN (e.g. 85%).
Here the agent does NOT know it. It keeps a belief about the accuracy
itself, and updates that belief from experience.

How the belief is stored: a Beta(alpha, beta) distribution.
    alpha = 1 + (number of times the sensor was RIGHT)
    beta  = 1 + (number of times the sensor was WRONG)
    mean  = alpha / (alpha + beta)   <- the agent's current guess of accuracy
Beta(1, 1) is a "know nothing" prior (every accuracy equally likely).

LIMITATIONS (be honest in the interview):
  * Only ONE unknown parameter (sensor accuracy) is learned.
  * The agent plans using its best GUESS (the mean) of the accuracy.
    A full BAMDP planner would reason about the uncertainty itself.
  * Learning needs feedback: after each episode we reveal where the tiger was.
"""
import numpy as np
from .pomdp import make_tiger_pomdp

MIN_ACCURACY = 0.51  # the Tiger POMDP needs accuracy > 0.5, so we clip the guess


class BetaBelief:
    """Belief about an unknown probability p in (0, 1), as Beta(alpha, beta)."""

    def __init__(self, alpha=1.0, beta=1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total ** 2 * (total + 1))

    @property
    def std(self):
        return float(np.sqrt(self.variance))

    def update(self, correct):
        """Bayes update from ONE experience: was the sensor right (True) or wrong (False)?"""
        if not isinstance(correct, (bool, np.bool_)):
            raise ValueError("correct must be True or False")
        if correct:
            self.alpha += 1
        else:
            self.beta += 1


class TigerBAMDPAgent:
    """Tiger agent that learns the sensor accuracy while it plays."""

    def __init__(self, prior=None):
        self.belief = prior if prior is not None else BetaBelief(1, 1)

    def believed_accuracy(self):
        """Current best guess of sensor accuracy (clipped into (0.5, 1])."""
        return float(min(max(self.belief.mean, MIN_ACCURACY), 1.0))

    def current_pomdp(self):
        """Build a Tiger POMDP using the CURRENT guess of the accuracy."""
        return make_tiger_pomdp(self.believed_accuracy())

    def learn_from_listen(self, heard, true_state):
        """Update the belief after we later learn where the tiger really was."""
        if heard not in ("hear-left", "hear-right"):
            raise ValueError(f"unknown observation: {heard!r}")
        if true_state not in ("tiger-left", "tiger-right"):
            raise ValueError(f"unknown state: {true_state!r}")
        correct = (heard == "hear-left") == (true_state == "tiger-left")
        self.belief.update(bool(correct))

    def listens_needed(self, max_listens=100):
        """How many identical 'hear-left' sounds until this agent is confident
        enough to open a door? Shows how the learned accuracy changes behaviour."""
        model = self.current_pomdp()
        b = np.array([0.5, 0.5])
        for n in range(max_listens):
            if model.best_action_now(b) != "listen":
                return n
            b = model.update_belief(b, "listen", "hear-left")
        return max_listens


def run_learning_experiment(true_accuracy=0.85, n_episodes=100,
                            listens_per_episode=3, prior_alpha=1.0,
                            prior_beta=1.0, seed=0):
    """Simulate an agent learning the sensor accuracy from experience.

    Each episode: tiger placed randomly, agent listens a few times,
    then the true tiger position is revealed and the belief is updated.
    Returns the learned mean and std after every episode.
    """
    if not 0.5 < true_accuracy <= 1.0:
        raise ValueError("true_accuracy must be in (0.5, 1.0]")
    if n_episodes < 1 or listens_per_episode < 1:
        raise ValueError("n_episodes and listens_per_episode must be >= 1")

    rng = np.random.default_rng(seed)
    agent = TigerBAMDPAgent(BetaBelief(prior_alpha, prior_beta))
    means, stds = [], []

    for _ in range(n_episodes):
        tiger_left = bool(rng.random() < 0.5)
        true_state = "tiger-left" if tiger_left else "tiger-right"
        for _ in range(listens_per_episode):
            sensor_right = rng.random() < true_accuracy
            hears_left = tiger_left if sensor_right else not tiger_left
            heard = "hear-left" if hears_left else "hear-right"
            agent.learn_from_listen(heard, true_state)
        means.append(agent.belief.mean)
        stds.append(agent.belief.std)

    return {"means": np.array(means), "stds": np.array(stds), "agent": agent}
