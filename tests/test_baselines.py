"""The Elo baseline used for comparison in the diagnostics."""

from __future__ import annotations

import pytest

from bscores.baselines import Elo


class TestExpectation:
    def test_equal_ratings_are_a_coin_flip(self):
        assert Elo().expect("a", "b") == pytest.approx(0.5)

    def test_matches_equation_4(self):
        model = Elo(k=32.0)
        model._ratings = {"a": 1700.0, "b": 1500.0}
        expected = 1.0 / (1.0 + 10.0 ** ((1500.0 - 1700.0) / 400.0))
        assert model.expect("a", "b") == pytest.approx(expected)

    def test_four_hundred_points_is_ten_to_one(self):
        model = Elo()
        model._ratings = {"a": 1900.0, "b": 1500.0}
        assert model.expect("a", "b") == pytest.approx(10.0 / 11.0)

    def test_complementary(self):
        model = Elo()
        model._ratings = {"a": 1650.0, "b": 1480.0}
        assert model.expect("a", "b") + model.expect("b", "a") == pytest.approx(1.0)

    def test_home_advantage_shifts_the_favourite(self):
        neutral = Elo()
        boosted = Elo(home_advantage=100.0)
        assert boosted.expect("a", "b") > neutral.expect("a", "b")


class TestUpdates:
    def test_winner_gains_and_loser_loses(self):
        model = Elo(k=32.0)
        model.update("a", "b", 1.0)
        assert model.rating("a") > 1500.0
        assert model.rating("b") < 1500.0

    def test_equal_k_conserves_total_rating(self):
        model = Elo(k=32.0)
        model.update("a", "b", 1.0)
        assert model.rating("a") + model.rating("b") == pytest.approx(3000.0)

    def test_expected_result_barely_moves_a_rating(self):
        model = Elo(k=32.0)
        model._ratings = {"a": 2000.0, "b": 1200.0}
        before = model.rating("a")
        model.update("a", "b", 1.0)
        assert model.rating("a") - before < 1.0

    def test_upsets_move_ratings_a_lot(self):
        model = Elo(k=32.0)
        model._ratings = {"a": 2000.0, "b": 1200.0}
        before = model.rating("b")
        model.update("a", "b", 0.0)
        assert model.rating("b") - before > 30.0

    def test_draws_pull_ratings_together(self):
        model = Elo(k=32.0)
        model._ratings = {"a": 1700.0, "b": 1500.0}
        model.update("a", "b", 0.5)
        assert model.rating("a") < 1700.0
        assert model.rating("b") > 1500.0

    def test_experience_schedule_slows_veterans(self):
        model = Elo()
        rookie_k = model._k("newcomer")
        for _ in range(200):
            model.update("veteran", "other", 1.0)
        assert model._k("veteran") < rookie_k

    def test_fixed_k_overrides_the_schedule(self):
        model = Elo(k=24.0)
        assert model._k("anyone") == 24.0


class TestRun:
    def test_probabilities_are_strictly_pre_match(self):
        model = Elo(k=32.0)
        probabilities = model.run(["a", "a"], ["b", "b"], [1.0, 1.0])
        # Nothing is known before the first match.
        assert probabilities[0] == pytest.approx(0.5)
        # The first result is folded in before the second is forecast.
        assert probabilities[1] > 0.5

    def test_learns_a_transitive_hierarchy(self):
        teams = [f"t{i}" for i in range(6)]
        home, away, outcome = [], [], []
        for _ in range(12):
            for i in range(6):
                for j in range(i + 1, 6):
                    home.append(teams[i])
                    away.append(teams[j])
                    outcome.append(1.0)
        model = Elo()
        model.run(home, away, outcome)
        order = sorted(model.ratings, key=lambda t: -model.ratings[t])
        assert order[0] == "t0"
        assert order[-1] == "t5"

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            Elo().run(["a"], ["b", "c"], [1.0])

    def test_ratings_snapshot_is_a_copy(self):
        model = Elo()
        model.update("a", "b", 1.0)
        snapshot = model.ratings
        snapshot["a"] = 0.0
        assert model.rating("a") != 0.0

    def test_repr(self):
        assert "k_scale" in repr(Elo())
        assert "k=24.0" in repr(Elo(k=24.0))
