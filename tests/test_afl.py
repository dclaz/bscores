"""End-to-end behaviour on the AFL archive.

These are the tests that would notice if the method quietly stopped working:
the ratings have to make sense against known AFL history, and the out-of-sample
forecasts have to beat the trivial baselines.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from bscores import BScoreModel, Elo, evaluate, log_loss, rolling_forecast
from bscores.datasets import load_afl
from bscores.metrics import diebold_mariano

# The paper's alpha, in days; kept as the reference configuration even though
# the AFL's own optimum is shorter (see test_alpha_sweep_prefers_short_memory).
PAPER_ALPHA = 365.0


@pytest.fixture(scope="module")
def afl():
    return load_afl(as_frame=False)


@pytest.fixture(scope="module")
def fitted(afl):
    model = BScoreModel(alpha=PAPER_ALPHA)
    model.add_matches(afl.home_team, afl.away_team, afl.outcome, afl.date)
    return model


@pytest.fixture(scope="module")
def backtest(afl):
    return rolling_forecast(
        afl.home_team,
        afl.away_team,
        afl.outcome,
        afl.date,
        alpha=PAPER_ALPHA,
        initial_train=0.5,
        refit_every=300,
    )


class TestRatings:
    def test_every_team_is_rated(self, fitted, afl):
        assert len(fitted) == 18
        assert sorted(fitted.players) == afl.teams

    def test_scores_are_a_unit_vector(self, fitted):
        assert np.linalg.norm(fitted.scores()) == pytest.approx(1.0)
        assert np.all(fitted.scores() > 0.0)

    def test_the_strongest_club_of_the_era_rates_top(self, fitted):
        # Geelong made 11 finals series and two grand finals over the window
        # the archive covers, and no other club is close on sustained record.
        assert fitted.leaderboard(top=1)[0].name == "Geelong"

    def test_short_memory_surfaces_current_form(self, afl):
        # The archive ends in April 2022 with Melbourne the reigning premiers
        # and unbeaten; a one-month memory has to see that, and the paper's
        # 365-day memory should not, because it averages over a decade.
        recent = BScoreModel(alpha=30.0)
        recent.add_matches(afl.home_team, afl.away_team, afl.outcome, afl.date)
        assert "Melbourne" in [rating.name for rating in recent.leaderboard(top=3)]

    def test_expansion_clubs_rate_poorly(self, fitted):
        # Gold Coast joined in 2011 and finished no higher than 14th in the
        # period covered; they should sit in the bottom third.
        board = [rating.name for rating in fitted.leaderboard()]
        assert board.index("Gold Coast") >= 12

    def test_match_counts_add_up(self, fitted, afl):
        total = sum(rating.matches for rating in fitted.ratings())
        assert total == 2 * len(afl)

    def test_ratings_move_over_time(self, fitted, afl):
        history = fitted.score_history(afl.date, inclusive=False)
        richmond = history.of("Richmond")
        # Richmond won three flags between 2017 and 2020 after a decade of
        # mediocrity, so their rating must not be flat.
        assert richmond.std() > 0.01
        assert richmond.max() > 2 * richmond.min()

    def test_a_team_that_stops_playing_decays(self, afl):
        # Truncate the fixture list after 2019 for one club and check the
        # network marks it down even though it never loses again.
        keep = ~((afl.home_team == "Sydney") | (afl.away_team == "Sydney"))
        cutoff = np.datetime64("2019-01-01", "D")
        mask = keep | (afl.date < cutoff)
        model = BScoreModel(alpha=PAPER_ALPHA)
        model.add_matches(
            afl.home_team[mask], afl.away_team[mask], afl.outcome[mask], afl.date[mask]
        )
        at_cutoff = model.rating("Sydney", at=cutoff).score
        at_end = model.rating("Sydney", at=afl.date.max()).score
        assert at_end < at_cutoff


class TestForecasting:
    def test_beats_the_home_ground_base_rate(self, backtest, afl):
        train = afl.outcome[: len(afl) - len(backtest)]
        naive = np.full(len(backtest), train.mean())
        assert backtest.metrics()["log_loss"] < log_loss(backtest.outcome, naive)
        assert backtest.metrics()["brier_score"] < np.mean((backtest.outcome - naive) ** 2)

    def test_beats_a_coin_flip(self, backtest):
        scores = backtest.metrics()
        assert scores["log_loss"] < np.log(2.0)
        assert scores["accuracy"] > 0.55

    def test_statistically_better_than_the_base_rate(self, backtest, afl):
        train = afl.outcome[: len(afl) - len(backtest)]
        naive = np.full(len(backtest), train.mean())
        naive_loss = -(
            backtest.outcome * np.log(naive) + (1 - backtest.outcome) * np.log(1 - naive)
        )
        statistic, p_value = diebold_mariano(backtest.losses("log_loss"), naive_loss)
        assert statistic < 0  # negative means B-scores lose less
        assert p_value < 0.01

    def test_competitive_with_elo(self, afl, backtest):
        elo = Elo()
        probabilities = elo.run(afl.home_team, afl.away_team, afl.outcome)
        start = len(afl) - len(backtest)
        elo_scores = evaluate(afl.outcome[start:], probabilities[start:])
        # Elo carries a fitted home-ground edge implicitly through its own
        # history; B-scores at the paper's alpha should land within a whisker.
        assert backtest.metrics()["log_loss"] < elo_scores["log_loss"] + 0.01

    def test_probabilities_are_calibrated_on_average(self, backtest):
        assert backtest.probability.mean() == pytest.approx(
            backtest.outcome.mean(), abs=0.03
        )

    def test_favourites_win_more_often(self, backtest):
        confident = backtest.probability > 0.65
        assert backtest.outcome[confident].mean() > backtest.outcome.mean()

    def test_home_advantage_lands_in_the_intercept(self, backtest, afl):
        # sigmoid(beta_0) should sit near the home win rate, because the two
        # slopes very nearly cancel for evenly matched sides.
        from bscores.calibration import sigmoid

        intercepts = sigmoid(backtest.coefficients[:, 0])
        assert np.all((intercepts > 0.5) & (intercepts < 0.65))

    def test_slopes_have_opposite_signs(self, backtest):
        assert np.all(backtest.coefficients[:, 1] > 0)
        assert np.all(backtest.coefficients[:, 2] < 0)

    def test_deterministic(self, afl):
        def run():
            return rolling_forecast(
                afl.home_team,
                afl.away_team,
                afl.outcome,
                afl.date,
                alpha=PAPER_ALPHA,
                initial_train=0.5,
            ).probability

        np.testing.assert_array_equal(run(), run())


class TestTuning:
    def test_alpha_sweep_prefers_short_memory(self, afl):
        from bscores import sweep_alpha

        rows = sweep_alpha(
            afl.home_team,
            afl.away_team,
            afl.outcome,
            afl.date,
            [21.0, 365.0, 3650.0],
            initial_train=0.5,
        )
        # A season of AFL is 22 rounds; form turns over much faster than the
        # 52-week tennis window the paper's alpha=365 mirrors.
        assert rows[0]["alpha"] == 21.0
        assert rows[-1]["alpha"] == 3650.0

    def test_longer_memory_is_smoother(self, afl):
        short = BScoreModel(alpha=30.0)
        long = BScoreModel(alpha=1095.0)
        for model in (short, long):
            model.add_matches(afl.home_team, afl.away_team, afl.outcome, afl.date)
        dates = afl.date[::10]
        volatility = {
            name: np.abs(np.diff(model.score_history(dates).of("Geelong"))).mean()
            for name, model in (("short", short), ("long", long))
        }
        assert volatility["short"] > volatility["long"]


class TestPerformance:
    def test_full_backtest_is_quick(self, afl):
        start = time.perf_counter()
        rolling_forecast(
            afl.home_team,
            afl.away_team,
            afl.outcome,
            afl.date,
            alpha=PAPER_ALPHA,
            initial_train=0.5,
            refit_every=300,
        )
        elapsed = time.perf_counter() - start
        # ~1400 causal centrality solves over 2534 matches.  The bound is loose
        # enough for a slow CI box but tight enough to catch an accidental
        # per-epoch rebuild of the whole history.
        assert elapsed < 20.0

    def test_warm_starting_saves_iterations(self, afl):
        from bscores.centrality import bonacich_centrality

        model = BScoreModel(alpha=PAPER_ALPHA)
        model.add_matches(afl.home_team, afl.away_team, afl.outcome, afl.date)
        dates = np.unique(afl.date)[-40:]

        cold_total = warm_total = 0
        warm = None
        for matrix in model.network.iter_matrices(dates, transposed=True):
            cold = bonacich_centrality(matrix, transposed=True, return_info=True)
            hot = bonacich_centrality(matrix, transposed=True, x0=warm, return_info=True)
            warm = hot.vector
            cold_total += cold.iterations
            warm_total += hot.iterations
        assert warm_total < cold_total
