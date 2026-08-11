"""The expanding-window back-test, and the leakage guarantees it rests on."""

from __future__ import annotations

import numpy as np
import pytest

from bscores import BScoreModel
from bscores.backtest import _resolve_start, rolling_forecast, sweep_alpha


def competition(n_teams=8, rounds=8, seed=0, upset_rate=0.25):
    """Fixtures with a known strength order and a realistic upset rate."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(n_teams)]
    home, away, outcome, times = [], [], [], []
    day = 0.0
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                swap = rng.random() < 0.5
                a, b = (j, i) if swap else (i, j)
                stronger_at_home = a < b
                if rng.random() < upset_rate:
                    stronger_at_home = not stronger_at_home
                home.append(teams[a])
                away.append(teams[b])
                outcome.append(1.0 if stronger_at_home else 0.0)
                times.append(day)
                day += 1.0
    return home, away, np.array(outcome), np.array(times)


class TestShape:
    def test_forecasts_the_holdout(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, alpha=200.0, initial_train=0.5)
        assert len(result) == len(home) - len(home) // 2
        assert result.probability.shape == result.outcome.shape
        assert np.all((result.probability >= 0.0) & (result.probability <= 1.0))

    def test_carries_the_scores_that_produced_each_forecast(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, alpha=200.0)
        assert result.home_score.shape == result.probability.shape
        assert np.all(result.home_score >= 0.0)

    def test_records_one_coefficient_row_per_refit(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(
            home, away, outcome, times, alpha=200.0, initial_train=0.5, refit_every=50
        )
        expected = int(np.ceil((len(home) - len(home) // 2) / 50))
        assert result.coefficients.shape == (expected, 3)
        assert result.refit_at.size == expected

    def test_train_size_grows_with_each_refit(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(
            home, away, outcome, times, alpha=200.0, initial_train=0.5, refit_every=50
        )
        assert np.all(np.diff(result.train_size) >= 0)
        assert result.train_size[0] == len(home) // 2

    def test_metrics_and_losses_agree(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, alpha=200.0)
        assert result.metrics()["log_loss"] == pytest.approx(result.losses("log_loss").mean())
        assert result.metrics()["brier_score"] == pytest.approx(
            result.losses("brier_score").mean()
        )

    def test_unknown_loss_rejected(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, alpha=200.0)
        with pytest.raises(ValueError, match="unknown loss"):
            result.losses("mystery")

    def test_repr_summarises(self):
        home, away, outcome, times = competition()
        assert "log_loss=" in repr(rolling_forecast(home, away, outcome, times, alpha=200.0))


class TestNoLeakage:
    def test_rewriting_the_future_leaves_earlier_forecasts_untouched(self):
        home, away, outcome, times = competition(seed=1)
        n = len(home)
        cut = n // 2
        # Rewrite history from well inside the test period.  Forecasts before
        # this point may legitimately use results that came before *them*, but
        # nothing at or after ``tamper_from`` can reach back.
        tamper_from = cut + (n - cut) // 2

        def run(results):
            return rolling_forecast(
                home,
                away,
                results,
                times,
                alpha=200.0,
                initial_train=cut,
                refit_every=1_000_000,
            )

        baseline = run(outcome)
        tampered_outcome = outcome.copy()
        tampered_outcome[tamper_from:] = 1.0 - tampered_outcome[tamper_from:]
        tampered = run(tampered_outcome)

        window = tamper_from - cut
        np.testing.assert_allclose(
            baseline.probability[:window], tampered.probability[:window], atol=1e-12
        )
        # ... and the forecasts that *should* react to the rewrite do.
        assert not np.allclose(
            baseline.probability[window:], tampered.probability[window:], atol=1e-6
        )

    def test_a_refit_only_uses_earlier_matches(self):
        home, away, outcome, times = competition(seed=2)
        n = len(home)
        result = rolling_forecast(
            home, away, outcome, times, alpha=200.0, initial_train=n // 2, refit_every=40
        )
        assert np.all(result.train_size < n)
        assert np.all(result.train_size <= np.arange(n // 2, n))

    def test_input_order_does_not_matter(self):
        home, away, outcome, times = competition(seed=3)
        shuffled = np.random.default_rng(0).permutation(len(home))
        straight = rolling_forecast(home, away, outcome, times, alpha=200.0)
        scrambled = rolling_forecast(
            [home[i] for i in shuffled],
            [away[i] for i in shuffled],
            outcome[shuffled],
            times[shuffled],
            alpha=200.0,
        )
        np.testing.assert_allclose(straight.probability, scrambled.probability, atol=1e-10)


class TestQuality:
    def test_beats_the_base_rate(self):
        home, away, outcome, times = competition(rounds=12, seed=4)
        result = rolling_forecast(home, away, outcome, times, alpha=400.0, initial_train=0.5)
        train_rate = outcome[: len(outcome) // 2].mean()
        from bscores.metrics import log_loss

        naive = log_loss(result.outcome, np.full(len(result), train_rate))
        assert result.metrics()["log_loss"] < naive

    def test_recovers_the_strength_order(self):
        home, away, outcome, times = competition(rounds=12, seed=5, upset_rate=0.15)
        result = rolling_forecast(home, away, outcome, times, alpha=1e6)
        board = [r.name for r in result.model.leaderboard()]
        assert board[0] == "t0"
        assert board[-1] == "t7"

    def test_a_perfectly_predictable_competition_is_forecast_confidently(self):
        home, away, outcome, times = competition(rounds=10, seed=6, upset_rate=0.0)
        result = rolling_forecast(home, away, outcome, times, alpha=1e6)
        assert result.metrics()["accuracy"] > 0.9


class TestStartResolution:
    def test_integer_counts_matches(self):
        times = np.arange(10.0)
        assert _resolve_start(times, 4) == 4

    def test_fraction_of_the_fixture_list(self):
        times = np.arange(10.0)
        assert _resolve_start(times, 0.5) == 5

    def test_timestamp_cuts_by_date(self):
        times = np.arange(10.0)
        assert _resolve_start(times, 6.0) == 6

    def test_default_is_the_halfway_point(self):
        assert _resolve_start(np.arange(11.0), None) == 5

    def test_date_strings(self):
        home, away, outcome, _ = competition(n_teams=6, rounds=6)
        dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(len(home))
        result = rolling_forecast(
            home, away, outcome, dates, alpha=200.0, initial_train="2020-03-01"
        )
        assert len(result) > 0
        assert result.times.min() >= float(
            np.datetime64("2020-03-01", "D").astype("int64")
        )


class TestValidation:
    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            rolling_forecast(["a"], ["b", "c"], [1.0], [0.0])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            rolling_forecast([], [], [], [])

    def test_refit_every_must_be_positive(self):
        home, away, outcome, times = competition(n_teams=4, rounds=2)
        with pytest.raises(ValueError, match="refit_every"):
            rolling_forecast(home, away, outcome, times, refit_every=0)

    def test_no_training_data_rejected(self):
        home, away, outcome, times = competition(n_teams=4, rounds=2)
        with pytest.raises(ValueError, match="at least one match to train"):
            rolling_forecast(home, away, outcome, times, initial_train=0)

    def test_no_test_data_rejected(self):
        home, away, outcome, times = competition(n_teams=4, rounds=2)
        with pytest.raises(ValueError, match="no matches to forecast"):
            rolling_forecast(home, away, outcome, times, initial_train=len(home))


class TestOptions:
    def test_supplied_model_is_used_and_returned(self):
        home, away, outcome, times = competition()
        model = BScoreModel(alpha=90.0, draw_weight=0.0)
        result = rolling_forecast(home, away, outcome, times, model=model)
        assert result.model is model
        assert model.is_calibrated

    def test_keep_model_can_be_disabled(self):
        home, away, outcome, times = competition()
        assert rolling_forecast(home, away, outcome, times, keep_model=False).model is None

    def test_symmetric_calibration_drops_a_coefficient(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, symmetric=True)
        assert result.coefficients.shape[1] == 2

    def test_no_intercept(self):
        home, away, outcome, times = competition()
        result = rolling_forecast(home, away, outcome, times, symmetric=True, fit_intercept=False)
        assert result.coefficients.shape[1] == 1

    def test_weights_are_forwarded(self):
        home, away, outcome, times = competition()
        weights = np.full(len(home), 2.0)
        weighted = rolling_forecast(home, away, outcome, times, alpha=200.0, weights=weights)
        # Scaling every arc scales the matrix, and the unit-norm eigenvector is
        # invariant to that, so the forecasts should not move.
        plain = rolling_forecast(home, away, outcome, times, alpha=200.0)
        np.testing.assert_allclose(weighted.probability, plain.probability, atol=1e-8)


class TestSweepAlpha:
    def test_sorted_best_first(self):
        home, away, outcome, times = competition(n_teams=6, rounds=8, seed=7)
        rows = sweep_alpha(home, away, outcome, times, [50.0, 200.0, 1000.0])
        assert len(rows) == 3
        assert [r["log_loss"] for r in rows] == sorted(r["log_loss"] for r in rows)
        assert set(rows[0]) >= {"alpha", "log_loss", "brier_score", "accuracy"}

    def test_accuracy_sorts_the_other_way(self):
        home, away, outcome, times = competition(n_teams=6, rounds=8, seed=8)
        rows = sweep_alpha(home, away, outcome, times, [50.0, 1000.0], metric="accuracy")
        assert rows[0]["accuracy"] >= rows[-1]["accuracy"]

    def test_unknown_metric_rejected(self):
        home, away, outcome, times = competition(n_teams=4, rounds=4)
        with pytest.raises(ValueError, match="unknown metric"):
            sweep_alpha(home, away, outcome, times, [100.0], metric="vibes")
