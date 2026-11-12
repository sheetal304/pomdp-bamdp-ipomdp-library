"""
tests/test_ipomdp.py - Tests for decision_library/ipomdp.py
Expected numbers are worked out by hand.
"""
import unittest
import numpy as np

from decision_library.ipomdp import (steps_to_threshold, other_action_probs,
                                     IPOMDPAgent, simulate_interaction,
                                     compare_with_without_other)


class TestOtherModel(unittest.TestCase):
    def test_steps_to_threshold(self):
        self.assertEqual(steps_to_threshold(0.60, 0.85), 1)   # hasty
        self.assertEqual(steps_to_threshold(0.99, 0.85), 3)   # careful

    def test_hasty_listens_once_then_opens(self):
        p = other_action_probs(0.60, 0.85, "tiger-left", 4)
        np.testing.assert_allclose(p[0], [1, 0, 0])
        # heard LEFT (prob .85) -> thinks left -> opens RIGHT
        np.testing.assert_allclose(p[1], [0, 0.15, 0.85])

    def test_careful_opens_only_after_three_net_sounds(self):
        p = other_action_probs(0.99, 0.85, "tiger-left", 5)
        np.testing.assert_allclose(p[1], [1, 0, 0])
        np.testing.assert_allclose(p[2], [1, 0, 0])
        # round 3: d=+3 has prob .85^3, d=-3 has prob .15^3
        np.testing.assert_allclose(p[3], [1 - 0.614125 - 0.003375, 0.003375, 0.614125])

    def test_each_row_sums_to_one(self):
        p = other_action_probs(0.99, 0.85, "tiger-right", 6)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            steps_to_threshold(0.4, 0.85)
        with self.assertRaises(ValueError):
            steps_to_threshold(0.9, 1.0)
        with self.assertRaises(ValueError):
            other_action_probs(0.9, 0.85, "tiger-up", 3)


class TestAgent(unittest.TestCase):
    def test_start_is_uniform(self):
        a = IPOMDPAgent()
        np.testing.assert_allclose(a.state_belief(), [0.5, 0.5])
        self.assertAlmostEqual(a.type_belief()["hasty"], 0.5)

    def test_own_hearing_update(self):
        a = IPOMDPAgent()
        a.hear("hear-left")
        np.testing.assert_allclose(a.state_belief(), [0.85, 0.15])

    def test_first_round_listen_tells_nothing(self):
        a = IPOMDPAgent()
        a.watch_other("listen")
        self.assertAlmostEqual(a.type_belief()["hasty"], 0.5)

    def test_j_opening_early_means_hasty(self):
        a = IPOMDPAgent()
        a.watch_other("listen")
        a.watch_other("open-right")   # careful can't open at round 1
        self.assertAlmostEqual(a.type_belief()["hasty"], 1.0)
        # opened RIGHT -> j believed tiger LEFT: 0.85 vs 0.15
        np.testing.assert_allclose(a.state_belief(), [0.85, 0.15])

    def test_j_still_listening_means_careful(self):
        a = IPOMDPAgent()
        a.watch_other("listen")
        a.watch_other("listen")       # hasty would have opened by now
        self.assertAlmostEqual(a.type_belief()["careful"], 1.0)

    def test_impossible_action_raises(self):
        with self.assertRaises(ValueError):
            IPOMDPAgent().watch_other("open-left")   # nobody opens in round 0

    def test_invalid_inputs_raise(self):
        a = IPOMDPAgent()
        with self.assertRaises(ValueError):
            a.hear("hear-nothing")
        with self.assertRaises(ValueError):
            a.watch_other("jump")
        with self.assertRaises(ValueError):
            IPOMDPAgent(type_prior=[0.7, 0.7])
        with self.assertRaises(ValueError):
            IPOMDPAgent(type_prior=[1.0])

    def test_too_many_rounds_raises(self):
        a = IPOMDPAgent(n_rounds=1)
        a.watch_other("listen")
        with self.assertRaises(ValueError):
            a.watch_other("listen")

    def test_joint_belief_stays_valid(self):
        a = IPOMDPAgent()
        a.hear("hear-left")
        a.watch_other("listen")
        a.watch_other("listen")
        self.assertAlmostEqual(a.joint.sum(), 1.0)


class TestSimulation(unittest.TestCase):
    def test_hasty_agent_is_identified(self):
        out = simulate_interaction("hasty", "tiger-left", n_rounds=3, seed=1)
        self.assertAlmostEqual(out["history"][-1]["p_careful"], 0.0)

    def test_same_seed_same_result(self):
        a = simulate_interaction("careful", "tiger-right", seed=5)["history"]
        b = simulate_interaction("careful", "tiger-right", seed=5)["history"]
        self.assertEqual(a, b)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            simulate_interaction("grumpy")
        with self.assertRaises(ValueError):
            simulate_interaction(true_state="tiger-up")

    def test_experiment_reproducible(self):
        a = compare_with_without_other(n_trials=30, seed=3)
        b = compare_with_without_other(n_trials=30, seed=3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
