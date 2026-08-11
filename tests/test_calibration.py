"""The logit layer (Eq. 3) and the IRLS solver behind it."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.calibration import LogitCalibrator, fit_logistic, sigmoid


class TestSigmoid:
    def test_known_values(self):
        np.testing.assert_allclose(sigmoid([0.0]), [0.5])
        np.testing.assert_allclose(sigmoid([np.log(3.0)]), [0.75])

    def test_stable_at_extremes(self):
        values = sigmoid([-1e4, 1e4, -800.0, 800.0])
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, [0.0, 1.0, 0.0, 1.0], atol=1e-300)

    def test_symmetry(self):
        z = np.linspace(-20, 20, 41)
        np.testing.assert_allclose(sigmoid(z) + sigmoid(-z), 1.0)


class TestFitLogistic:
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        n = 60_000
        x = rng.normal(size=(n, 2))
        design = np.column_stack([np.ones(n), x])
        truth = np.array([-0.4, 1.7, -0.9])
        y = (rng.random(n) < sigmoid(design @ truth)).astype(float)
        fit = fit_logistic(design, y, ridge=0.0)
        assert fit.converged
        np.testing.assert_allclose(fit.beta, truth, atol=0.05)

    def test_fractional_targets_equal_the_equivalent_split(self):
        # One observation at y=0.5 must equal two at y=1 and y=0.
        design = np.array([[1.0, 0.3], [1.0, -0.2]])
        fractional = fit_logistic(design, np.array([0.5, 1.0]), ridge=0.0)
        duplicated = fit_logistic(
            np.vstack([design[:1], design[:1], design[1:]]),
            np.array([1.0, 0.0, 1.0]),
            ridge=0.0,
        )
        np.testing.assert_allclose(fractional.beta, duplicated.beta, atol=1e-6)

    def test_sample_weights_match_duplication(self):
        rng = np.random.default_rng(1)
        design = np.column_stack([np.ones(40), rng.normal(size=40)])
        y = (rng.random(40) < 0.5).astype(float)
        weighted = fit_logistic(design, y, sample_weight=np.full(40, 3.0), ridge=0.0)
        stacked = fit_logistic(np.vstack([design] * 3), np.tile(y, 3), ridge=0.0)
        np.testing.assert_allclose(weighted.beta, stacked.beta, atol=1e-6)

    def test_ridge_keeps_separated_data_finite(self):
        # Perfectly separable: the unpenalised MLE diverges.
        design = np.column_stack([np.ones(6), np.array([-3.0, -2, -1, 1, 2, 3])])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        fit = fit_logistic(design, y, ridge=1e-3)
        assert np.all(np.isfinite(fit.beta))
        assert abs(fit.beta[1]) < 500.0

    def test_intercept_can_be_left_unpenalised(self):
        rng = np.random.default_rng(2)
        design = np.column_stack([np.ones(500), rng.normal(size=500)])
        y = (rng.random(500) < 0.8).astype(float)
        fit = fit_logistic(design, y, ridge=1e4, penalize=np.array([False, True]))
        # The heavy penalty flattens the slope but the intercept still tracks
        # the base rate.
        assert abs(fit.beta[1]) < 1e-2
        assert sigmoid(np.array([fit.beta[0]]))[0] == pytest.approx(y.mean(), abs=0.02)

    def test_warm_start_reaches_the_same_optimum(self):
        rng = np.random.default_rng(3)
        design = np.column_stack([np.ones(300), rng.normal(size=300)])
        y = (rng.random(300) < 0.4).astype(float)
        cold = fit_logistic(design, y)
        warm = fit_logistic(design, y, beta0=cold.beta)
        np.testing.assert_allclose(warm.beta, cold.beta, atol=1e-8)
        assert warm.n_iter <= cold.n_iter

    def test_log_likelihood_is_reported(self):
        design = np.ones((10, 1))
        fit = fit_logistic(design, np.full(10, 0.5), ridge=0.0)
        assert fit.beta[0] == pytest.approx(0.0, abs=1e-6)
        assert fit.log_likelihood == pytest.approx(10 * np.log(0.5))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"y": np.array([0.0])}, "length mismatch"),
            ({"y": np.array([2.0, 0.0])}, r"\[0, 1\]"),
        ],
    )
    def test_input_validation(self, kwargs, match):
        design = np.ones((2, 1))
        with pytest.raises(ValueError, match=match):
            fit_logistic(design, **kwargs)

    def test_rejects_1d_design(self):
        with pytest.raises(ValueError, match="2-d"):
            fit_logistic(np.ones(4), np.zeros(4))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="zero observations"):
            fit_logistic(np.ones((0, 1)), np.zeros(0))

    def test_rejects_non_finite_design(self):
        with pytest.raises(ValueError, match="non-finite"):
            fit_logistic(np.array([[np.nan]]), np.array([1.0]))

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            fit_logistic(np.ones((2, 1)), np.array([0.0, 1.0]), sample_weight=np.array([1.0, -1.0]))


class TestLogitCalibrator:
    def _synthetic(self, n=60_000, seed=0):
        rng = np.random.default_rng(seed)
        home = rng.random(n) * 0.4 + 0.05
        away = rng.random(n) * 0.4 + 0.05
        truth = np.array([0.25, 6.0, -6.0])
        p = sigmoid(truth[0] + truth[1] * home + truth[2] * away)
        y = (rng.random(n) < p).astype(float)
        return home, away, y, truth

    def test_recovers_equation_3_coefficients(self):
        home, away, y, truth = self._synthetic()
        model = LogitCalibrator(ridge=0.0).fit(home, away, y)
        np.testing.assert_allclose(model.beta_, truth, atol=0.35)
        assert model.intercept_ == pytest.approx(truth[0], abs=0.2)
        assert model.coef_.shape == (2,)

    def test_probabilities_are_calibrated_in_aggregate(self):
        home, away, y, _ = self._synthetic()
        model = LogitCalibrator().fit(home, away, y)
        assert model.predict_proba(home, away).mean() == pytest.approx(y.mean(), abs=0.01)

    def test_design_layout(self):
        model = LogitCalibrator()
        design = model.design([0.2, 0.3], [0.1, 0.4])
        np.testing.assert_allclose(design, [[1.0, 0.2, 0.1], [1.0, 0.3, 0.4]])

    def test_symmetric_uses_the_difference(self):
        model = LogitCalibrator(symmetric=True)
        design = model.design([0.2], [0.1])
        np.testing.assert_allclose(design, [[1.0, 0.1]])

    def test_symmetric_fit_is_order_reversing(self):
        home, away, y, _ = self._synthetic(n=3000, seed=4)
        model = LogitCalibrator(symmetric=True, fit_intercept=False).fit(home, away, y)
        forward = model.predict_proba(home, away)
        backward = model.predict_proba(away, home)
        np.testing.assert_allclose(forward + backward, 1.0, atol=1e-12)

    def test_no_intercept(self):
        home, away, y, _ = self._synthetic(n=2000, seed=5)
        model = LogitCalibrator(fit_intercept=False).fit(home, away, y)
        assert model.intercept_ == 0.0
        assert model.beta_.shape == (2,)

    def test_extra_covariates(self):
        rng = np.random.default_rng(6)
        n = 4000
        home, away = rng.random(n) * 0.3, rng.random(n) * 0.3
        finals = (rng.random(n) < 0.2).astype(float)
        p = sigmoid(0.1 + 5 * home - 5 * away + 1.2 * finals)
        y = (rng.random(n) < p).astype(float)
        model = LogitCalibrator(ridge=0.0).fit(home, away, y, extra=finals)
        assert model.beta_.shape == (4,)
        assert model.beta_[3] == pytest.approx(1.2, abs=0.3)

    def test_log_transform_floors_zero_scores_on_the_data_scale(self):
        home = np.array([0.2, 0.3, 0.0, 0.25])
        away = np.array([0.1, 0.0, 0.3, 0.2])
        model = LogitCalibrator(transform="log").fit(home, away, np.array([1.0, 1.0, 0.0, 1.0]))
        # 1% of the mean positive score, not a fixed 1e-12 that would map an
        # unrated competitor to a wild outlier.
        positive = np.concatenate([home, away])
        assert model.epsilon_ == pytest.approx(0.01 * positive[positive > 0].mean())
        assert np.all(np.isfinite(model.predict_proba(home, away)))

    def test_explicit_epsilon_is_respected(self):
        model = LogitCalibrator(transform="log", epsilon=0.5)
        model.fit([0.2], [0.1], [1.0])
        assert model.epsilon_ == 0.5

    def test_sqrt_transform(self):
        model = LogitCalibrator(transform="sqrt")
        np.testing.assert_allclose(model.design([0.25], [0.0])[0], [1.0, 0.5, 0.0])

    def test_unknown_transform_rejected(self):
        with pytest.raises(ValueError, match="unknown transform"):
            LogitCalibrator(transform="wat").design([0.1], [0.2])

    def test_predict_before_fit_raises(self):
        model = LogitCalibrator()
        assert not model.is_fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba([0.1], [0.2])
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.coef_
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.intercept_

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="home/away length mismatch"):
            LogitCalibrator().design([0.1, 0.2], [0.3])

    def test_extra_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="extra length mismatch"):
            LogitCalibrator().design([0.1, 0.2], [0.3, 0.4], extra=np.ones(3))

    def test_repr_before_and_after_fit(self):
        model = LogitCalibrator()
        assert "unfitted" in repr(model)
        model.fit([0.2, 0.1], [0.1, 0.2], [1.0, 0.0])
        assert "beta=" in repr(model)
