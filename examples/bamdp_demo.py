"""Demo: BAMDP (learning the sensor accuracy). Run:  python examples/bamdp_demo.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decision_library.bamdp import run_learning_experiment, TigerBAMDPAgent


def main():
    print("BAMDP DEMO - agent learns the sensor accuracy (true value = 0.85)")
    out = run_learning_experiment(true_accuracy=0.85, n_episodes=100,
                                  listens_per_episode=3, seed=0)
    print(f"{'episode':>8}{'learned accuracy':>18}{'uncertainty (std)':>20}")
    for ep in [1, 5, 10, 20, 50, 100]:
        print(f"{ep:>8}{out['means'][ep-1]:>18.3f}{out['stds'][ep-1]:>20.3f}")
    print("\nListens needed before opening a door:")
    print(f"  agent that knows nothing (prior Beta(1,1)): {TigerBAMDPAgent().listens_needed()}")
    print(f"  agent after learning for 100 episodes:      {out['agent'].listens_needed()}")


if __name__ == "__main__":
    main()
