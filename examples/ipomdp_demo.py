"""Demo: simplified IPOMDP (2 agents). Run:  python examples/ipomdp_demo.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decision_library.ipomdp import simulate_interaction, compare_with_without_other


def main():
    print("IPOMDP DEMO - agent i reasons about another agent j")
    for true_type in ["hasty", "careful"]:
        out = simulate_interaction(true_type, "tiger-left", n_rounds=6, seed=1)
        print(f"\nTrue type of j = {true_type}, tiger is LEFT")
        print(f"{'round':>6}{'j did':>13}{'P(j careful)':>15}{'P(tiger-left)':>15}")
        for r, h in enumerate(out["history"], start=1):
            print(f"{r:>6}{h['j_action']:>13}{h['p_careful']:>15.3f}{h['p_tiger_left']:>15.3f}")

    print("\nDoes watching j help? (2000 random trials, avg probability given to the TRUE state)")
    print(f"{'rounds':>7}{'watching j':>13}{'ignoring j':>13}")
    for n in [2, 3, 4, 6]:
        r = compare_with_without_other(n_trials=2000, n_rounds=n, seed=0)
        print(f"{n:>7}{r['with_other']:>13.4f}{r['without_other']:>13.4f}")


if __name__ == "__main__":
    main()
