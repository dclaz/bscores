"""Loss functions, the Diebold-Mariano test and the betting ROI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bscores.metrics import (
    DEFAULT_TOL,
    accuracy,
    brier_score,
    classification_error,
    diebold_mariano,
    evaluate,
    log_loss,
    roi,
)


class TestLogLoss:
    def test_matches_the_definition(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([0.9, 0.1, 0.8, 0.5])
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        assert log_loss(y, p) == pytest.approx(expected)

    def test_perfect_forecasts_score_zero(self):
        assert log_loss([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-14)

    def test_uninformative_forecasts_score_log_two(self):
        assert log_loss([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2.0))

    def test_fractional_outcomes_are_scored(self):
        # A draw predicted at 0.5 is the best possible call for that match.
        assert log_loss([0.5], [0.5]) == pytest.approx(math.log(2.0))

    def test_tol_caps_the_damage_of_a_confident_miss(self):
        loose = log_loss([1.0], [1e-6], tol=0.01)
        tight = log_loss([1.0], [1e-6])
        assert loose < tight
        assert loose == pytest.approx(-math.log(0.01))

    def test_default_tol_is_the_r_value(self):
        # R's .Machine$double.neg.eps
        assert pytest.approx(2.0**-53) == DEFAULT_TOL

    def test_zero_probability_stays_finite(self):
        assert np.isfinite(log_loss([1.0], [0.0]))

    def test_tol_range_checked(self):
        with pytest.raises(ValueError, match="tol"):
            log_loss([1.0], [0.5], tol=0.6)
        with pytest.raises(ValueError, match="tol"):
            log_loss([1.0], [0.5], tol=-0.1)


class TestBrierScore:
    def test_matches_the_definition(self):
        y = np.array([1.0, 0.0, 0.5])
        p = np.array([0.7, 0.2, 0.4])
        assert brier_score(y, p) == pytest.approx(np.mean((y - p) ** 2))

    def test_perfect_forecasts_score_zero(self):
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_worst_case_is_one(self):
        assert brier_score([1.0, 0.0], [0.0, 1.0]) == 1.0


class TestAccuracy:
    def test_counts_rounded_hits(self):
        assert accuracy([1.0, 0.0, 1.0, 1.0], [0.9, 0.4, 0.6, 0.2]) == pytest.approx(0.75)

    def test_rounds_half_to_even_like_r(self):
        # numpy rounds half to even, so an exact coin flip is scored as a
        # predicted loss rather than a toss-up.
        assert accuracy([0.0], [0.5]) == 1.0
        assert accuracy([1.0], [0.5]) == 0.0

    def test_classification_error_is_the_complement(self):
        y, p = [1.0, 0.0, 1.0], [0.9, 0.4, 0.2]
        assert accuracy(y, p) + classification_error(y, p) == pytest.approx(1.0)


class TestEvaluate:
    def test_reports_every_metric(self):
        scores = evaluate([1.0, 0.0], [0.8, 0.3])
        assert set(scores) == {"n", "log_loss", "brier_score", "accuracy", "classification_error"}
        assert scores["n"] == 2.0

    def test_agrees_with_the_individual_functions(self):
        y, p = [1.0, 0.0, 1.0], [0.7, 0.2, 0.55]
        scores = evaluate(y, p)
        assert scores["log_loss"] == pytest.approx(log_loss(y, p))
        assert scores["brier_score"] == pytest.approx(brier_score(y, p))


class TestValidation:
    @pytest.mark.parametrize("fn", [log_loss, brier_score, accuracy, classification_error])
    def test_length_mismatch_rejected(self, fn):
        with pytest.raises(ValueError, match="same length"):
            fn([1.0, 0.0], [0.5])

    @pytest.mark.parametrize("fn", [log_loss, brier_score, accuracy])
    def test_empty_rejected(self, fn):
        with pytest.raises(ValueError, match="empty"):
            fn([], [])


class TestDieboldMariano:
    def test_negative_when_the_first_model_loses_less(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.5, 0.1, 500)
        b = rng.normal(0.6, 0.1, 500)
        statistic, p_value = diebold_mariano(a, b)
        assert statistic < 0
        assert p_value < 0.01

    def test_sign_flips_with_the_arguments(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0.5, 0.1, 300), rng.normal(0.6, 0.1, 300)
        forward, _ = diebold_mariano(a, b)
        backward, _ = diebold_mariano(b, a)
        assert forward == pytest.approx(-backward)

    def test_identical_losses_give_no_evidence(self):
        losses = np.linspace(0.1, 0.9, 50)
        statistic, p_value = diebold_mariano(losses, losses)
        assert statistic == 0.0 and p_value == 1.0

    def test_indistinguishable_models_are_not_rejected(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(0.5, 0.1, 400), rng.normal(0.5, 0.1, 400)
        _, p_value = diebold_mariano(a, b)
        assert p_value > 0.05

    def test_longer_horizons_widen_the_variance(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(0.5, 0.1, 400), rng.normal(0.55, 0.1, 400)
        one, _ = diebold_mariano(a, b, horizon=1)
        five, _ = diebold_mariano(a, b, horizon=5)
        assert abs(five) != pytest.approx(abs(one))

    def test_small_sample_correction_shrinks_the_statistic(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(0.5, 0.1, 20), rng.normal(0.6, 0.1, 20)
        corrected, _ = diebold_mariano(a, b, small_sample=True)
        raw, _ = diebold_mariano(a, b, small_sample=False)
        assert abs(corrected) < abs(raw)

    def test_validation(self):
        with pytest.raises(ValueError, match="same length"):
            diebold_mariano([1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="at least two"):
            diebold_mariano([1.0], [1.0])
        with pytest.raises(ValueError, match="horizon"):
            diebold_mariano([1.0, 2.0], [1.0, 2.0], horizon=0)


class TestRoi:
    def test_a_winning_bet_returns_the_odds(self):
        result = roi([1.0], [0.9], [2.0], threshold=0.5)
        assert result["n_bets"] == 1.0
        assert result["profit"] == pytest.approx(1.0)
        assert result["roi"] == pytest.approx(1.0)

    def test_a_losing_bet_loses_the_stake(self):
        result = roi([0.0], [0.9], [2.0], threshold=0.5)
        assert result["profit"] == pytest.approx(-1.0)
        assert result["roi"] == pytest.approx(-1.0)

    def test_a_draw_returns_the_stake(self):
        result = roi([0.5], [0.9], [2.0], threshold=0.5)
        assert result["profit"] == pytest.approx(0.0)

    def test_threshold_filters_matches(self):
        outcome = [1.0, 1.0]
        prediction = [0.9, 0.4]
        odds = [2.0, 2.0]
        assert roi(outcome, prediction, odds, threshold=0.5)["n_bets"] == 1.0
        assert roi(outcome, prediction, odds, threshold=0.95)["n_bets"] == 0.0

    def test_min_implied_excludes_heavy_underdogs(self):
        # Implied probability 1/11 = 0.09, below the 0.2 floor.
        result = roi([1.0], [0.9], [11.0], threshold=0.5, min_implied=0.2)
        assert result["n_bets"] == 0.0
        assert math.isnan(result["roi"])

    def test_stake_scales_the_totals(self):
        single = roi([1.0, 0.0], [0.9, 0.9], [2.0, 2.0], stake=1.0)
        double = roi([1.0, 0.0], [0.9, 0.9], [2.0, 2.0], stake=2.0)
        assert double["staked"] == pytest.approx(2 * single["staked"])
        assert double["roi"] == pytest.approx(single["roi"])

    def test_break_even_at_fair_odds(self):
        rng = np.random.default_rng(5)
        n = 20_000
        outcome = (rng.random(n) < 0.6).astype(float)
        result = roi(outcome, np.full(n, 0.9), np.full(n, 1 / 0.6))
        assert result["roi"] == pytest.approx(0.0, abs=0.02)

    def test_missing_odds_are_skipped(self):
        result = roi([1.0, 1.0], [0.9, 0.9], [2.0, np.nan])
        assert result["n_bets"] == 1.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="odds length"):
            roi([1.0], [0.9], [2.0, 3.0])
