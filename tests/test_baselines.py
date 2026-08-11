"""The Elo baseline used for comparison in the diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.baselines import DEFAULT_ELO_GRID, Elo, tune_elo
from bscores.metrics import log_loss


@pytest.fixture(scope="module")
def fixtures():
    from .test_backtest import competition

    return competition(n_teams=10, rounds=16, seed=1, upset_rate=0.25)


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


class TestTuneElo:
    def test_returns_constructor_arguments(self, fixtures):
        home, away, outcome, times = fixtures
        best = tune_elo(
            home, away, outcome, times,
            grid={"home_advantage": [0.0, 30.0], "k_scale": [150.0, 300.0]},
            validation_start=0.5,
        )
        assert set(best) == {"home_advantage", "k_scale"}
        Elo(**best)  # splats cleanly

    def test_picks_the_best_of_the_grid(self, fixtures):
        home, away, outcome, times = fixtures
        grid = {"home_advantage": [0.0, 40.0, 90.0]}
        best = tune_elo(home, away, outcome, times, grid=grid, validation_start=0.5)

        start = int(round(0.5 * len(home)))
        scores = {}
        for value in grid["home_advantage"]:
            p = Elo(home_advantage=value).run(home, away, outcome)
            scores[value] = log_loss(np.asarray(outcome)[start:], p[start:])
        assert best["home_advantage"] == min(scores, key=scores.get)

    def test_accuracy_maximises_instead(self, fixtures):
        home, away, outcome, times = fixtures
        best = tune_elo(
            home, away, outcome, times,
            grid={"home_advantage": [0.0, 60.0]}, validation_start=0.5, metric="accuracy",
        )
        assert best["home_advantage"] in (0.0, 60.0)

    def test_scoring_stops_at_validation_end(self, fixtures):
        home, away, outcome, times = fixtures
        grid = {"home_advantage": [0.0, 30.0, 60.0]}
        cut = float(np.quantile(np.asarray(times, dtype=float), 0.75))
        windowed = tune_elo(
            home, away, outcome, times, grid=grid, validation_start=0.4, validation_end=cut
        )
        # Rewriting outcomes after the window must not move the choice.
        tampered = np.asarray(outcome, dtype=float).copy()
        after = np.asarray(times, dtype=float) >= cut
        tampered[after] = 1.0 - tampered[after]
        assert windowed == tune_elo(
            home, away, tampered, times, grid=grid, validation_start=0.4, validation_end=cut
        )

    def test_row_order_does_not_matter(self, fixtures):
        home, away, outcome, times = fixtures
        grid = {"home_advantage": [0.0, 45.0]}
        base = tune_elo(home, away, outcome, times, grid=grid, validation_start=0.5)
        p = np.random.default_rng(0).permutation(len(home))
        shuffled = tune_elo(
            np.asarray(home)[p], np.asarray(away)[p],
            np.asarray(outcome, dtype=float)[p], np.asarray(times, dtype=float)[p],
            grid=grid, validation_start=0.5,
        )
        assert base == shuffled

    def test_empty_grid_rejected(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="grid must not be empty"):
            tune_elo(home, away, outcome, times, grid={}, validation_start=0.5)

    def test_window_must_be_ordered(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="validation_end must fall after"):
            tune_elo(home, away, outcome, times, validation_start=0.8, validation_end=0.2)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            tune_elo(["a"], ["b", "c"], [1.0], [0.0], validation_start=0.5)

    def test_default_grid_is_searchable(self, fixtures):
        home, away, outcome, times = fixtures
        best = tune_elo(home, away, outcome, times, validation_start=0.5)
        assert set(best) == set(DEFAULT_ELO_GRID)
