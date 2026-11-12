"""
pomdp.py - A small POMDP (Partially Observable Markov Decision Process).

IDEA: The agent does NOT know the true state. It only sees noisy
observations, so it keeps a BELIEF: a probability for each state.

Shapes used (n_s = states, n_a = actions, n_o = observations):
    T[a, s, s2] = P(next state s2 | state s, action a)
    O[a, s2, o] = P(observation o  | next state s2, action a)
    R[s, a]     = reward for doing action a in state s
"""
import numpy as np
from .utils import check_distribution, check_rows_sum_to_one, normalize


class POMDP:
    def __init__(self, states, actions, observations, T, O, R):
        self.states = list(states)
        self.actions = list(actions)
        self.observations = list(observations)
        n_s, n_a, n_o = len(self.states), len(self.actions), len(self.observations)

        self.T = check_rows_sum_to_one(T, "T")
        self.O = check_rows_sum_to_one(O, "O")
        self.R = np.asarray(R, dtype=float)

        if self.T.shape != (n_a, n_s, n_s):
            raise ValueError(f"T must have shape {(n_a, n_s, n_s)}, got {self.T.shape}")
        if self.O.shape != (n_a, n_s, n_o):
            raise ValueError(f"O must have shape {(n_a, n_s, n_o)}, got {self.O.shape}")
        if self.R.shape != (n_s, n_a):
            raise ValueError(f"R must have shape {(n_s, n_a)}, got {self.R.shape}")

    # ---- small helpers: names -> numbers ----
    def _a(self, action):
        if action not in self.actions:
            raise ValueError(f"unknown action: {action!r}")
        return self.actions.index(action)

    def _o(self, obs):
        if obs not in self.observations:
            raise ValueError(f"unknown observation: {obs!r}")
        return self.observations.index(obs)

    # ---- the core math ----
    def update_belief(self, belief, action, observation):
        """Bayesian belief update.

        b'(s2)  is proportional to  O[a, s2, o] * sum_s T[a, s, s2] * b(s)
          step 1 (predict): push belief through the transition model
          step 2 (correct): multiply by how likely the observation is
          step 3: normalize so it sums to 1
        """
        b = check_distribution(belief, "belief")
        if len(b) != len(self.states):
            raise ValueError("belief length must equal number of states")
        a, o = self._a(action), self._o(observation)

        predicted = b @ self.T[a]               # step 1
        unnormalized = self.O[a][:, o] * predicted  # step 2
        if unnormalized.sum() == 0:
            raise ValueError("this observation is impossible under the current belief")
        return normalize(unnormalized)          # step 3

    def expected_reward(self, belief, action):
        """Average immediate reward of `action` under the current belief."""
        b = check_distribution(belief, "belief")
        return float(b @ self.R[:, self._a(action)])

    def best_action_now(self, belief):
        """Greedy choice: the action with the best expected IMMEDIATE reward.
        (Simple on purpose - it does not plan ahead.)"""
        return max(self.actions, key=lambda act: self.expected_reward(belief, act))

    def sample_step(self, state, action, rng):
        """Simulate one step: returns (next_state, observation, reward)."""
        s, a = self.states.index(state), self._a(action)
        reward = float(self.R[s, a])
        s2 = rng.choice(len(self.states), p=self.T[a, s])
        o = rng.choice(len(self.observations), p=self.O[a, s2])
        return self.states[s2], self.observations[o], reward


def make_tiger_pomdp(listen_accuracy=0.85):
    """The classic Tiger problem.

    A tiger is behind the LEFT or RIGHT door. Agent can listen (noisy)
    or open a door. Opening resets the game (tiger placed randomly again).
    """
    if not 0.5 < listen_accuracy <= 1.0:
        raise ValueError("listen_accuracy must be in (0.5, 1.0]")
    states = ["tiger-left", "tiger-right"]
    actions = ["listen", "open-left", "open-right"]
    observations = ["hear-left", "hear-right"]
    p = listen_accuracy

    T = np.array([
        [[1, 0], [0, 1]],          # listen: state does not change
        [[.5, .5], [.5, .5]],      # open-left: reset
        [[.5, .5], [.5, .5]],      # open-right: reset
    ])
    O = np.array([
        [[p, 1 - p], [1 - p, p]],  # listen: hear the correct side with prob p
        [[.5, .5], [.5, .5]],      # opening: observation is uninformative
        [[.5, .5], [.5, .5]],
    ])
    #        listen  open-left  open-right
    R = np.array([
        [-1,  -100,  10],          # tiger-left
        [-1,    10, -100],         # tiger-right
    ])
    return POMDP(states, actions, observations, T, O, R)
