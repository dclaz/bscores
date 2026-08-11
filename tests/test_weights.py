"""Arc weighting by margin of victory and match importance."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.weights import MARGIN_SCHEMES, importance_weight, margin_weight


class TestMarginWeight:
    @pytest.mark.parametrize("scheme", MARGIN_SCHEMES)
    def test_a_dead_heat_weighs_one(self, scheme):
        assert margin_weight([0], scheme=scheme) == pytest.approx([1.0])

    @pytest.mark.parametrize("scheme", MARGIN_SCHEMES)
    def test_sign_is_ignored(self, scheme):
        np.testing.assert_allclose(
            margin_weight([-40, 40], scheme=scheme), margin_weight([40, 40], scheme=scheme)
        )

    @pytest.mark.parametrize("scheme", ["linear", "sqrt", "log"])
    def test_monotone_in_the_margin(self, scheme):
        weights = margin_weight(np.arange(0, 100), scheme=scheme, cap=None)
        assert np.all(np.diff(weights) > 0)

    def test_uniform_ignores_the_margin(self):
        np.testing.assert_allclose(margin_weight([0, 50, 200], scheme="uniform"), 1.0)

    def test_linear_reaches_two_at_the_scale(self):
        assert margin_weight([36], scheme="linear", scale=36.0, cap=None) == pytest.approx([2.0])

    def test_schemes_are_ordered_by_how_hard_they_bite(self):
        # A blowout: linear rewards it most, log least, uniform not at all.
        blowout = 120.0
        weights = {
            scheme: float(margin_weight([blowout], scheme=scheme, scale=36.0, cap=None)[0])
            for scheme in MARGIN_SCHEMES
        }
        assert weights["uniform"] < weights["sqrt"] < weights["log"] < weights["linear"]

    def test_cap_bounds_freak_scorelines(self):
        assert margin_weight([1000], scheme="linear", cap=3.0) == pytest.approx([3.0])
        assert margin_weight([1000], scheme="linear", cap=None)[0] > 3.0

    def test_scale_controls_the_slope(self):
        steep = margin_weight([36], scheme="linear", scale=18.0, cap=None)[0]
        shallow = margin_weight([36], scheme="linear", scale=72.0, cap=None)[0]
        assert steep > shallow

    def test_validation(self):
        with pytest.raises(ValueError, match="scale"):
            margin_weight([10], scale=0.0)
        with pytest.raises(ValueError, match="cap"):
            margin_weight([10], cap=0.5)
        with pytest.raises(ValueError, match="unknown scheme"):
            margin_weight([10], scheme="vibes")


class TestImportanceWeight:
    def test_flags_are_boosted(self):
        np.testing.assert_allclose(importance_weight([True, False, True], weight=3.0), [3, 1, 3])

    def test_weight_must_be_positive(self):
        with pytest.raises(ValueError, match="weight"):
            importance_weight([True], weight=0.0)


class TestEffectOnRatings:
    def test_uniform_rescaling_leaves_ratings_alone(self):
        # The principal eigenvector does not care about the scale of the matrix,
        # so only *relative* weights can move a rating.
        from bscores import BScoreModel

        home = ["a", "b", "c", "a"]
        away = ["b", "c", "a", "c"]
        outcome = [1.0, 1.0, 1.0, 0.0]
        times = [0.0, 7.0, 14.0, 21.0]

        plain = BScoreModel(alpha=365.0).add_matches(home, away, outcome, times)
        scaled = BScoreModel(alpha=365.0).add_matches(
            home, away, outcome, times, weights=np.full(4, 7.5)
        )
        np.testing.assert_allclose(plain.scores(), scaled.scores(), atol=1e-10)

    def test_margin_weighting_moves_ratings(self):
        from bscores import BScoreModel

        home = ["a", "b", "c", "a"]
        away = ["b", "c", "a", "c"]
        outcome = [1.0, 1.0, 1.0, 0.0]
        times = [0.0, 7.0, 14.0, 21.0]
        margins = [100, 1, 1, -1]

        plain = BScoreModel(alpha=365.0).add_matches(home, away, outcome, times)
        weighted = BScoreModel(alpha=365.0).add_matches(
            home, away, outcome, times, weights=margin_weight(margins, scheme="linear", cap=None)
        )
        assert not np.allclose(plain.scores(), weighted.scores())
