"""
tests/test_bamdp.py - Tests for decision_library/bamdp.py
Expected numbers are worked out by hand.
"""
import unittest
import numpy as np

from decision_library.bamdp import (BetaBelief, TigerBAMDPAgent,
                                    run_learning_experiment)


class TestBetaBelief(unittest.TestCase):
    def test_flat_prior_mean_is_half(self):
        self.assertAlmostEqual(BetaBelief(1, 1).mean, 0.5)

    def test_flat_prior_variance(self):
        # 1*1 / (2^2 * 3) = 1/12
        self.assertAlmostEqual(BetaBelief(1, 1).variance, 1 / 12)

    def test_one_correct_gives_two_thirds(self):
        b = BetaBelief(1, 1)
        b.update(True)             # Beta(2, 1) -> mean 2/3
        self.assertAlmostEqual(b.mean, 2 / 3)

    def test_three_right_one_wrong(self):
        b = BetaBelief(1, 1)
        for c in [True, True, True, False]:
            b.update(c)            # Beta(4, 2) -> mean 4/6
        self.assertAlmostEqual(b.mean, 4 / 6)

    def test_order_of_experience_does_not_matter(self):
        a, b = BetaBelief(1, 1), BetaBelief(1, 1)
        for c in [True, False, True]:
            a.update(c)
        for c in [True, True, False]:
            b.update(c)
        self.assertAlmostEqual(a.mean, b.mean)

    def test_more_data_means_less_uncertainty(self):
        b = BetaBelief(1, 1)
        before = b.std
        for _ in range(50):
            b.update(True)
            b.update(False)
        self.assertLess(b.std, before)

    def test_invalid_prior_raises(self):
        with self.assertRaises(ValueError):
            BetaBelief(0, 1)
        with self.assertRaises(ValueError):
            BetaBelief(1, -2)

    def test_update_needs_true_or_false(self):
        with self.assertRaises(ValueError):
            BetaBelief(1, 1).update("yes")


class TestAgent(unittest.TestCase):
    def test_learning_changes_the_belief(self):
        # Key BAMDP idea: the belief about the RULES changes with experience.
        agent = TigerBAMDPAgent()
        before = agent.belief.mean
        agent.learn_from_listen("hear-left", "tiger-left")   # sensor was right
        self.assertGreater(agent.belief.mean, before)

    def test_wrong_sensor_lowers_belief(self):
        agent = TigerBAMDPAgent()
        agent.learn_from_listen("hear-left", "tiger-right")  # sensor was wrong
        self.assertLess(agent.belief.mean, 0.5)

    def test_believed_accuracy_is_clipped(self):
        agent = TigerBAMDPAgent(BetaBelief(1, 50))  # mean far below 0.5
        self.assertGreater(agent.believed_accuracy(), 0.5)
        agent.current_pomdp()  # must not raise

    def test_bad_inputs_raise(self):
        agent = TigerBAMDPAgent()
        with self.assertRaises(ValueError):
            agent.learn_from_listen("hear-nothing", "tiger-left")
        with self.assertRaises(ValueError):
            agent.learn_from_listen("hear-left", "tiger-up")

    def test_better_sensor_means_fewer_listens(self):
        n = {}
        for acc, prior in [("high", BetaBelief(95, 5)),
                           ("mid", BetaBelief(85, 15)),
                           ("low", BetaBelief(65, 35))]:
            n[acc] = TigerBAMDPAgent(prior).listens_needed()
        self.assertLessEqual(n["high"], n["mid"])
        self.assertLess(n["mid"], n["low"])

    def test_listens_needed_for_known_085(self):
        # Same as POMDP: belief 0.85 -> still listen, 0.97 -> open. So 2 listens.
        self.assertEqual(TigerBAMDPAgent(BetaBelief(85, 15)).listens_needed(), 2)


class TestExperiment(unittest.TestCase):
    def test_learns_close_to_true_accuracy(self):
        out = run_learning_experiment(true_accuracy=0.85, n_episodes=500, seed=0)
        self.assertAlmostEqual(out["means"][-1], 0.85, delta=0.05)

    def test_uncertainty_shrinks_over_time(self):
        out = run_learning_experiment(n_episodes=100, seed=1)
        self.assertLess(out["stds"][-1], out["stds"][0])

    def test_same_seed_same_result(self):
        a = run_learning_experiment(seed=7)["means"]
        b = run_learning_experiment(seed=7)["means"]
        np.testing.assert_array_equal(a, b)

    def test_invalid_experiment_inputs_raise(self):
        with self.assertRaises(ValueError):
            run_learning_experiment(true_accuracy=0.4)
        with self.assertRaises(ValueError):
            run_learning_experiment(n_episodes=0)


if __name__ == "__main__":
    unittest.main()
