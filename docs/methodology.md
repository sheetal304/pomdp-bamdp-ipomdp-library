# Methodology

This is a **student learning prototype**. It illustrates the core idea of each framework on one
small example (the Tiger problem). It does **not** reproduce the full research-level
BAMDP / POMDP / IPOMDP theory or their planning algorithms.

## The common idea

All three frameworks are about an agent that must act while something relevant is **hidden**.
The agent keeps a **belief** (a probability distribution) about the hidden thing and updates it
with Bayes' rule when it gets new evidence.

| Framework | What is hidden? | Belief is about |
|---|---|---|
| POMDP | the current state of the world | state |
| BAMDP | the parameters (rules) of the world | parameters |
| IPOMDP | the other agent's mind (its type) | other agent's model |

## The running example: Tiger problem

A tiger is behind the left or right door (reward +10 for the safe door, -100 for the tiger door,
-1 for listening). Listening gives a noisy hint: the correct side is heard with probability `p`
(default 0.85). Opening a door resets the problem (tiger placed again at random).

## 1. POMDP (`decision_library/pomdp.py`)

A POMDP is a tuple (S, A, O, T, Z, R):

* `T[a, s, s'] = P(s' | s, a)` transition model
* `Z[a, s', o] = P(o | s', a)` observation model (called `O` in the code)
* `R[s, a]` reward

The agent keeps a belief `b(s)`. After taking action `a` and seeing observation `o`:

```
b'(s') = Z[a, s', o] * sum_s T[a, s, s'] * b(s)  /  P(o | b, a)
```

(predict with `T`, correct with `Z`, normalise). The decision rule in this library is
**greedy**: choose the action with the highest expected *immediate* reward
`sum_s b(s) R[s, a]`. For the Tiger problem this means: listen until the belief
in one side exceeds 0.9, because `10 p - 100 (1 - p) > -1` exactly when `p > 0.9`.

**Experiment 1** compares this belief-based rule with opening a random door.

## 2. BAMDP (`decision_library/bamdp.py`)

The sensor accuracy `theta` is unknown. The agent has a **Beta(alpha, beta)** belief over it.
Beta is *conjugate* for yes/no data, so the Bayes update is just counting:

```
Beta(alpha, beta)  --sensor was right-->  Beta(alpha + 1, beta)
Beta(alpha, beta)  --sensor was wrong-->  Beta(alpha, beta + 1)
mean = alpha / (alpha + beta)        variance = alpha*beta / ((alpha+beta)^2 (alpha+beta+1))
```

After every episode the true tiger position is revealed (this is the "experience"), so we know
whether each sound was right. The agent then plans with its current best guess of `theta`.

**How this differs from POMDP belief updates:** the POMDP belief is over the *state* and the
model is fixed and known. Here the belief is over the *model parameter itself*, and it
accumulates over many episodes, so the agent's *behaviour changes* as it learns
(55 listens needed with no knowledge vs 2 after learning, see results).

**Simplifications (not full BAMDP):** one unknown parameter; the planner uses the posterior mean
(a certainty-equivalent shortcut) instead of planning over the full Bayes-adaptive state
`(state, belief over parameters)`; no explicit explore/exploit planning.

## 3. IPOMDP (`decision_library/ipomdp.py`)

Agent `i` shares the room with agent `j`. Both hear the tiger with independent noisy ears.
Agent `i` sees `j`'s *actions* but not `j`'s hearing or belief. `j` has a hidden **type**:

* `hasty`: opens a door once its belief reaches 0.60 (after 1 net sound)
* `careful`: opens a door once its belief reaches 0.99 (after 3 net sounds)

`j` is modelled as a counter `d = #hear-left - #hear-right` that opens the door it believes is
safe once `|d|` reaches a type-specific threshold. For each type and tiger side, a small dynamic
programme over `d` gives `P(j's action in round r | j still playing)`.

Agent `i` keeps a **joint belief** `b(type, state)` and applies two kinds of Bayes updates:

```
own hearing:       b(t, s) <- b(t, s) * P(o_i | s)
watching j:        b(t, s) <- b(t, s) * P(a_j in round r | t, s)
```

Watching `j` helps for two reasons: it reveals `j`'s type, and a door opened by a confident `j`
is evidence about the tiger's side. **Experiment 3** measures this.

**Simplifications (not full IPOMDP):** only one level of nesting (`i` models `j`, `j` does not
model `i`); only two hand-written types; `j`'s behaviour is a fixed rule, not a solved policy;
agent `i` does not plan interactively (it only reasons). Real IPOMDPs use *interactive states*
and recursively nested beliefs and are much harder to solve.

## Assumptions

* Discrete, tiny state/action/observation spaces; the Tiger problem only.
* Sensor noise is independent between listens (and between agents).
* In BAMDP the true state is revealed after each episode.
* In IPOMDP agent `j` knows the true sensor accuracy.

## Limitations

* No exact POMDP/BAMDP/IPOMDP solvers (no value iteration over beliefs, no point-based methods).
* Greedy one-step decision rule, not long-horizon planning.
* Results come from simulations with fixed seeds; they illustrate behaviour on this toy problem
  and are not benchmarks against other libraries.

## Possible future work

* Exact or point-based POMDP solvers (value iteration over belief points, PBVI/SARSOP-style).
* Joint belief over (state, parameters) with a real Bayes-adaptive planner (e.g. Monte-Carlo tree search).
* Multi-parameter BAMDP (unknown transition and reward models), Dirichlet priors.
* IPOMDP with level-2 nesting and types that are themselves solved POMDP policies.
* More environments than Tiger (grid worlds, a link to a Q-learning navigation project).
