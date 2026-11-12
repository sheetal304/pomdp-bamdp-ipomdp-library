"""
tests/test_pomdp.py - Tests for decision_library/pomdp.py

Each test checks ONE small thing. Expected numbers are worked out by hand
(Bayes rule), not copied from the code's output.
"""
import unittest
import numpy as np

from decision_library.pomdp import POMDP, make_tiger_pomdp


class TestBeliefUpdate(unittest.TestCase):
    def setUp(self):
        self.model = make_tiger_pomdp(listen_accuracy=0.85)
        self.uniform = np.array([0.5, 0.5])

    def test_first_listen_gives_085(self):
        # By hand: (0.5*0.85) / (0.5*0.85 + 0.5*0.15) = 0.85
        b = self.model.update_belief(self.uniform, "listen", "hear-left")
        np.testing.assert_allclose(b, [0.85, 0.15])

    def test_second_listen_gives_09698(self):
        # By hand: 0.85*0.85 / (0.85*0.85 + 0.15*0.15) = 0.7225 / 0.745
        b = self.model.update_belief(self.uniform, "listen", "hear-left")
        b = self.model.update_belief(b, "listen", "hear-left")
        np.testing.assert_allclose(b, [0.7225 / 0.745, 0.0225 / 0.745])

    def test_belief_always_sums_to_one(self):
        b = self.uniform
        for obs in ["hear-left", "hear-right", "hear-left", "hear-left"]:
            b = self.model.update_belief(b, "listen", obs)
            self.assertAlmostEqual(b.sum(), 1.0)

    def test_hear_right_is_mirror_of_hear_left(self):
        left = self.model.update_belief(self.uniform, "listen", "hear-left")
        right = self.model.update_belief(self.uniform, "listen", "hear-right")
        np.testing.assert_allclose(left, right[::-1])

    def test_opposite_observations_cancel_out(self):
        # one hear-left + one hear-right from 50-50 -> back to 50-50
        b = self.model.update_belief(self.uniform, "listen", "hear-left")
        b = self.model.update_belief(b, "listen", "hear-right")
        np.testing.assert_allclose(b, [0.5, 0.5])

    def test_opening_a_door_resets_belief(self):
        # Opening resets the game, so we know nothing again.
        confident = np.array([0.97, 0.03])
        b = self.model.update_belief(confident, "open-left", "hear-left")
        np.testing.assert_allclose(b, [0.5, 0.5])

    def test_impossible_observation_raises(self):
        perfect = make_tiger_pomdp(listen_accuracy=1.0)
        with self.assertRaises(ValueError):
            # belief says tiger is surely left, but we "hear right" with 100% sensor
            perfect.update_belief([1.0, 0.0], "listen", "hear-right")


class TestDecisions(unittest.TestCase):
    def setUp(self):
        self.model = make_tiger_pomdp()

    def test_expected_reward_values(self):
        b = np.array([0.85, 0.15])
        # open-right: 0.85*10 + 0.15*(-100) = -6.5
        self.assertAlmostEqual(self.model.expected_reward(b, "open-right"), -6.5)
        self.assertAlmostEqual(self.model.expected_reward(b, "listen"), -1.0)

    def test_listens_when_unsure(self):
        self.assertEqual(self.model.best_action_now([0.5, 0.5]), "listen")
        self.assertEqual(self.model.best_action_now([0.85, 0.15]), "listen")

    def test_opens_safe_door_when_confident(self):
        self.assertEqual(self.model.best_action_now([0.97, 0.03]), "open-right")
        self.assertEqual(self.model.best_action_now([0.03, 0.97]), "open-left")


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.model = make_tiger_pomdp()

    def test_belief_not_summing_to_one_raises(self):
        with self.assertRaises(ValueError):
            self.model.update_belief([0.5, 0.6], "listen", "hear-left")

    def test_negative_belief_raises(self):
        with self.assertRaises(ValueError):
            self.model.update_belief([1.5, -0.5], "listen", "hear-left")

    def test_wrong_belief_length_raises(self):
        with self.assertRaises(ValueError):
            self.model.update_belief([0.2, 0.3, 0.5], "listen", "hear-left")

    def test_unknown_action_raises(self):
        with self.assertRaises(ValueError):
            self.model.update_belief([0.5, 0.5], "jump", "hear-left")

    def test_unknown_observation_raises(self):
        with self.assertRaises(ValueError):
            self.model.update_belief([0.5, 0.5], "listen", "hear-nothing")

    def test_bad_transition_matrix_raises(self):
        T = np.array([[[0.9, 0.2], [0, 1]]])  # first row sums to 1.1
        O = np.array([[[0.5, 0.5], [0.5, 0.5]]])
        R = np.zeros((2, 1))
        with self.assertRaises(ValueError):
            POMDP(["a", "b"], ["act"], ["x", "y"], T, O, R)

    def test_wrong_shape_raises(self):
        T = np.array([[[1, 0], [0, 1]]])
        O = np.array([[[0.5, 0.5], [0.5, 0.5]]])
        R = np.zeros((3, 1))  # should be (2, 1)
        with self.assertRaises(ValueError):
            POMDP(["a", "b"], ["act"], ["x", "y"], T, O, R)

    def test_bad_listen_accuracy_raises(self):
        for bad in [0.3, 0.5, 1.2]:
            with self.assertRaises(ValueError):
                make_tiger_pomdp(listen_accuracy=bad)


class TestSimulation(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        model = make_tiger_pomdp()
        r1 = model.sample_step("tiger-left", "listen", np.random.default_rng(42))
        r2 = model.sample_step("tiger-left", "listen", np.random.default_rng(42))
        self.assertEqual(r1, r2)

    def test_listening_does_not_move_the_tiger(self):
        model = make_tiger_pomdp()
        rng = np.random.default_rng(0)
        for _ in range(50):
            next_state, _, reward = model.sample_step("tiger-left", "listen", rng)
            self.assertEqual(next_state, "tiger-left")
            self.assertEqual(reward, -1)


if __name__ == "__main__":
    unittest.main()
