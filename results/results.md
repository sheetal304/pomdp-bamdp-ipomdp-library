# Experiment results

All numbers below were produced by `examples/run_experiments.py` (fixed random seeds).

## Experiment 1 - POMDP: does using the belief help? (Tiger, accuracy 0.85)

Average total reward per episode over 5000 episodes (higher is better).

| Policy | Mean reward | Std. error |
|---|---|---|
| Open a random door immediately | -43.00 | 0.78 |
| Belief-based (listen until confident) | 4.21 | 0.26 |

## Experiment 2 - BAMDP: learning the sensor accuracy

200 episodes x 3 listens each, 20 different seeds per row. Error = |learned - true|.

| True accuracy | Mean learned | Mean abs. error | Mean std (uncertainty) | Listens needed (learned) |
|---|---|---|---|---|
| 0.65 | 0.650 | 0.017 | 0.019 | 4.2 |
| 0.75 | 0.750 | 0.016 | 0.018 | 2.5 |
| 0.85 | 0.852 | 0.014 | 0.014 | 2.0 |
| 0.95 | 0.952 | 0.009 | 0.009 | 1.0 |

Reference: an agent with no knowledge (prior Beta(1,1)) needs 55 listens.

## Experiment 3 - simplified IPOMDP: does watching the other agent help?

2000 random trials per row (random tiger side, random type of j). Value = average probability agent i assigns to the TRUE tiger side.

| Rounds | Watching j | Ignoring j | Difference |
|---|---|---|---|
| 2 | 0.8722 | 0.8422 | +0.0299 |
| 3 | 0.9187 | 0.9036 | +0.0152 |
| 4 | 0.9634 | 0.9316 | +0.0318 |
| 6 | 0.9884 | 0.9696 | +0.0188 |

