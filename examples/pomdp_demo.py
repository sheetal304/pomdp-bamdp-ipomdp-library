"""Demo: POMDP (Tiger problem). Run from the project root:  python examples/pomdp_demo.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from decision_library.pomdp import make_tiger_pomdp


def main():
    model = make_tiger_pomdp(listen_accuracy=0.85)
    belief = np.array([0.5, 0.5])
    print("POMDP DEMO - Tiger problem (sensor accuracy 85%)")
    print(f"{'step':<22}{'P(tiger-left)':>14}{'P(tiger-right)':>16}   best action")
    print(f"{'start':<22}{belief[0]:>14.4f}{belief[1]:>16.4f}   {model.best_action_now(belief)}")
    for i, obs in enumerate(["hear-left", "hear-left", "hear-left"], start=1):
        belief = model.update_belief(belief, "listen", obs)
        print(f"{'listen -> ' + obs:<22}{belief[0]:>14.4f}{belief[1]:>16.4f}   {model.best_action_now(belief)}")

    print("\nSimulated episode (seed=0, true state = tiger-left):")
    rng = np.random.default_rng(0)
    state, belief, total = "tiger-left", np.array([0.5, 0.5]), 0.0
    for step in range(1, 10):
        action = model.best_action_now(belief)
        state_next, obs, reward = model.sample_step(state, action, rng)
        total += reward
        print(f"  step {step}: action={action:<11} reward={reward:>5.0f}")
        if action != "listen":
            break
        belief = model.update_belief(belief, action, obs)
        print(f"          heard {obs}, belief P(left)={belief[0]:.3f}")
        state = state_next
    print(f"  total reward: {total:.0f}")


if __name__ == "__main__":
    main()
