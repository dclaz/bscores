"""Monte Carlo season simulation."""

from __future__ import annotations

import numpy as np
import pytest

from bscores import BScoreModel
from bscores.simulation import simulate_season

from .test_models import round_robin


@pytest.fixture(scope="module")
def model():
    teams = [f"t{i}" for i in range(6)]
    home, away, outcome, times = round_robin(teams, rounds=8, seed=0, upset_rate=0.2)
    return BScoreModel(alpha=400.0).fit(home, away, outcome, times)


def fixtures(teams, rounds=1):
    home, away = [], []
    for _ in range(rounds):
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home.append(teams[i])
                away.append(teams[j])
    return home, away


class TestShape:
    def test_arrays_are_the_right_size(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(model, home, away, n_simulations=200, seed=0)
        assert season.points.shape == (200, len(model.players))
        assert season.positions.shape == season.points.shape
        assert sorted(season.competitors) == sorted(model.players)

    def test_positions_are_a_permutation_in_every_season(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(model, home, away, n_simulations=100, seed=1)
        expected = np.arange(1, len(model.players) + 1)
        for row in season.positions:
            np.testing.assert_array_equal(np.sort(row), expected)

    def test_points_total_is_fixed_without_draws(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(
            model, home, away, n_simulations=50, seed=2, draw_probability=0.0
        )
        np.testing.assert_allclose(season.points.sum(axis=1), 4.0 * len(home))

    def test_draws_preserve_the_points_total(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(
            model, home, away, n_simulations=50, seed=3, draw_probability=0.2
        )
        np.testing.assert_allclose(season.points.sum(axis=1), 4.0 * len(home))


class TestProbabilities:
    def test_finishing_probabilities_sum_to_one(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(model, home, away, n_simulations=500, seed=4)
        assert sum(season.finish_probability(1).values()) == pytest.approx(1.0)
        assert sum(season.position_distribution("t0")) == pytest.approx(1.0)

    def test_top_n_probabilities_sum_to_n(self, model):
        home, away = fixtures(model.players)
        season = simulate_season(model, home, away, n_simulations=500, seed=5)
        assert sum(season.top_n_probability(3).values()) == pytest.approx(3.0)

    def test_the_best_team_is_the_favourite(self, model):
        home, away = fixtures(model.players, rounds=3)
        season = simulate_season(model, home, away, n_simulations=2000, seed=6)
        best = model.leaderboard(top=1)[0].name
        assert next(iter(season.finish_probability(1))) == best
        assert next(iter(season.expected_points())) == best

    def test_banked_points_carry_over(self, model):
        home, away = fixtures(model.players)
        laggard = model.leaderboard()[-1].name
        season = simulate_season(
            model, home, away, n_simulations=500, seed=7,
            standings={laggard: 1000.0},
        )
        # A thousand-point head start beats any rating difference.
        assert next(iter(season.expected_points())) == laggard
        assert season.finish_probability(1)[laggard] == 1.0

    def test_more_fixtures_sharpen_the_favourite(self, model):
        best = model.leaderboard(top=1)[0].name
        short = simulate_season(model, *fixtures(model.players, 1), n_simulations=2000, seed=8)
        long = simulate_season(model, *fixtures(model.players, 6), n_simulations=2000, seed=8)
        assert long.finish_probability(1)[best] > short.finish_probability(1)[best]

    def test_ties_are_broken_evenly(self):
        # Two identical competitors must split first place roughly 50/50.
        model = BScoreModel(kernel="uniform")
        model.add_matches(["a", "b"], ["b", "a"], [1.0, 1.0], [0.0, 1.0])
        season = simulate_season(
            model, ["a"], ["b"], n_simulations=4000, seed=9, draw_probability=1.0 - 1e-12
        )
        first = season.finish_probability(1)
        assert first["a"] == pytest.approx(0.5, abs=0.05)


class TestDeterminism:
    def test_same_seed_same_season(self, model):
        home, away = fixtures(model.players)
        a = simulate_season(model, home, away, n_simulations=100, seed=11)
        b = simulate_season(model, home, away, n_simulations=100, seed=11)
        np.testing.assert_array_equal(a.points, b.points)

    def test_different_seeds_differ(self, model):
        home, away = fixtures(model.players)
        a = simulate_season(model, home, away, n_simulations=100, seed=11)
        b = simulate_season(model, home, away, n_simulations=100, seed=12)
        assert not np.array_equal(a.points, b.points)


class TestValidation:
    def test_length_mismatch(self, model):
        with pytest.raises(ValueError, match="home/away length mismatch"):
            simulate_season(model, ["t0"], ["t1", "t2"])

    def test_empty_fixture_list(self, model):
        with pytest.raises(ValueError, match="empty fixture list"):
            simulate_season(model, [], [])

    def test_simulation_count(self, model):
        with pytest.raises(ValueError, match="n_simulations"):
            simulate_season(model, ["t0"], ["t1"], n_simulations=0)

    def test_draw_probability_range(self, model):
        with pytest.raises(ValueError, match="draw probability"):
            simulate_season(model, ["t0"], ["t1"], draw_probability=1.0)

    def test_unknown_place(self, model):
        season = simulate_season(model, *fixtures(model.players), n_simulations=20, seed=0)
        with pytest.raises(ValueError, match="place must lie"):
            season.finish_probability(99)
        with pytest.raises(ValueError, match="n must lie"):
            season.top_n_probability(0)
        with pytest.raises(KeyError, match="unknown competitor"):
            season.position_distribution("nobody")

    def test_repr(self, model):
        season = simulate_season(model, *fixtures(model.players), n_simulations=20, seed=0)
        assert "SeasonSimulation(" in repr(season)
