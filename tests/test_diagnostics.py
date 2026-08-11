"""Rating explanations, calibration checks and network health."""

from __future__ import annotations

import numpy as np
import pytest

from bscores import BScoreModel
from bscores.diagnostics import (
    calibration_curve,
    explain_rating,
    head_to_head,
    network_summary,
    rating_churn,
    reliability_table,
    sharpness,
    upset_rate,
)

from .test_models import round_robin


def cyclic_model(alpha=365.0):
    """A small network with a cycle, so the eigenvector solve applies."""
    model = BScoreModel(alpha=alpha)
    for winner, loser, day in [
        ("a", "b", 0.0),
        ("b", "c", 7.0),
        ("c", "a", 14.0),
        ("a", "c", 21.0),
        ("b", "a", 28.0),
        ("c", "b", 35.0),
    ]:
        model.rate_result(winner, loser, at=day)
    return model


class TestExplainRating:
    def test_contributions_sum_to_the_rating(self):
        model = cyclic_model()
        for name in model.players:
            parts = explain_rating(model, name, top=None)
            assert sum(c.contribution for c in parts) == pytest.approx(
                model.rating(name).score, rel=1e-8
            )

    def test_shares_sum_to_one(self):
        model = cyclic_model()
        parts = explain_rating(model, "a", top=None)
        assert sum(c.share for c in parts) == pytest.approx(1.0)

    def test_only_beaten_opponents_appear(self):
        model = BScoreModel(alpha=365.0)
        model.rate_result("a", "b", at=0.0)
        model.rate_result("b", "c", at=7.0)
        model.rate_result("c", "a", at=14.0)
        assert [c.opponent for c in explain_rating(model, "a", top=None)] == ["b"]

    def test_sorted_by_share(self):
        model = cyclic_model()
        shares = [c.share for c in explain_rating(model, "a", top=None)]
        assert shares == sorted(shares, reverse=True)

    def test_top_truncates(self):
        model = cyclic_model()
        assert len(explain_rating(model, "a", top=1)) == 1

    def test_beating_a_strong_opponent_dominates_the_explanation(self):
        model = BScoreModel(alpha=1e6)
        # "strong" beats three others; "weak" beats nobody; "x" beats both.
        for loser in ("p", "q", "r"):
            model.rate_result("strong", loser, at=0.0)
        model.rate_result("x", "strong", at=1.0)
        model.rate_result("x", "weak", at=2.0)
        model.rate_result("p", "q", at=3.0)  # a cycle, so rho > 0
        model.rate_result("q", "p", at=4.0)
        top = explain_rating(model, "x")[0]
        assert top.opponent == "strong"

    def test_acyclic_network_still_explains(self):
        model = BScoreModel(alpha=1e6)
        model.rate_result("a", "b", at=0.0)
        model.rate_result("b", "c", at=1.0)
        parts = explain_rating(model, "a", top=None)
        assert [c.opponent for c in parts] == ["b"]
        assert parts[0].share == pytest.approx(1.0)

    def test_unknown_competitor_rejected(self):
        with pytest.raises(KeyError, match="unknown competitor"):
            explain_rating(cyclic_model(), "nobody")

    def test_repr(self):
        assert "share=" in repr(explain_rating(cyclic_model(), "a")[0])


class TestCalibrationCurve:
    def test_a_perfect_model_lands_on_the_diagonal(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 40_000)
        y = (rng.random(40_000) < p).astype(float)
        curve = calibration_curve(y, p, bins=10)
        np.testing.assert_allclose(curve["observed"], curve["predicted"], atol=0.02)

    def test_an_overconfident_model_bows_away_from_it(self):
        rng = np.random.default_rng(1)
        truth = rng.uniform(0.3, 0.7, 20_000)
        y = (rng.random(20_000) < truth).astype(float)
        overconfident = np.clip((truth - 0.5) * 3.0 + 0.5, 0.01, 0.99)
        curve = calibration_curve(y, overconfident, bins=8)
        high = curve["predicted"] > 0.7
        assert np.all(curve["observed"][high] < curve["predicted"][high])

    def test_counts_add_up(self):
        rng = np.random.default_rng(2)
        p = rng.random(500)
        y = (rng.random(500) < p).astype(float)
        assert calibration_curve(y, p, bins=7)["count"].sum() == 500

    def test_quantile_bins_are_evenly_filled(self):
        rng = np.random.default_rng(3)
        p = rng.beta(6, 6, 1000)
        y = (rng.random(1000) < p).astype(float)
        counts = calibration_curve(y, p, bins=5, strategy="quantile")["count"]
        assert counts.max() - counts.min() <= 2

    def test_empty_bins_are_dropped(self):
        y = np.array([1.0, 0.0])
        p = np.array([0.51, 0.52])
        assert calibration_curve(y, p, bins=10)["count"].size == 1

    def test_validation(self):
        with pytest.raises(ValueError, match="bins"):
            calibration_curve([1.0], [0.5], bins=0)
        with pytest.raises(ValueError, match="unknown strategy"):
            calibration_curve([1.0], [0.5], strategy="vibes")


class TestReliabilityTable:
    def test_renders_one_row_per_bin(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0.1, 0.9, 400)
        y = (rng.random(400) < p).astype(float)
        table = reliability_table(y, p, bins=4)
        assert len(table.splitlines()) == 5
        assert table.startswith("predicted")


class TestSharpness:
    def test_a_model_that_never_commits_scores_zero(self):
        assert sharpness([0.5, 0.5, 0.5]) == 0.0

    def test_a_model_that_always_commits_scores_one(self):
        assert sharpness([0.0, 1.0]) == pytest.approx(1.0)

    def test_ordering(self):
        assert sharpness([0.4, 0.6]) < sharpness([0.2, 0.8])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="no forecasts"):
            sharpness([])


class TestNetworkSummary:
    def test_reports_the_basics(self):
        model = cyclic_model()
        summary = network_summary(model)
        assert summary["competitors"] == 3
        assert summary["results"] == 6
        assert summary["has_cycle"] is True
        assert summary["spectral_radius"] > 0.0
        assert summary["unrated"] == 0
        assert 0.0 < summary["density"] <= 1.0

    def test_flags_an_acyclic_network(self):
        model = BScoreModel(alpha=1e6)
        model.rate_result("a", "b", at=0.0)
        model.rate_result("b", "c", at=1.0)
        summary = network_summary(model)
        assert summary["has_cycle"] is False
        assert summary["spectral_radius"] == 0.0
        assert summary["solver"] == "acyclic"
        assert summary["unrated"] == 1  # "c" has never won

    def test_density_of_a_full_round_robin(self):
        model = BScoreModel(alpha=1e9)
        model.add_matches(*round_robin([f"t{i}" for i in range(5)], rounds=4, upset_rate=0.5))
        assert network_summary(model)["density"] > 0.7

    def test_evaluated_at_a_time(self):
        model = cyclic_model()
        early = network_summary(model, at=7.0)
        late = network_summary(model, at=35.0)
        assert early["arcs"] < late["arcs"]


class TestHeadToHead:
    def test_counts_both_directions(self):
        result = head_to_head(["a", "b"], ["b", "a"], [1.0, 1.0])
        i, j = result["names"].index("a"), result["names"].index("b")
        assert result["wins"][i, j] == 1.0
        assert result["wins"][j, i] == 1.0
        assert result["played"][i, j] == 2

    def test_draws_split_the_win(self):
        result = head_to_head(["a"], ["b"], [0.5])
        i, j = result["names"].index("a"), result["names"].index("b")
        assert result["wins"][i, j] == 0.5 and result["wins"][j, i] == 0.5

    def test_wins_and_played_are_consistent(self):
        home, away, outcome, _ = round_robin(["a", "b", "c"], rounds=5, upset_rate=0.4)
        result = head_to_head(home, away, outcome)
        np.testing.assert_allclose(
            result["wins"] + result["wins"].T, result["played"].astype(float)
        )

    def test_restricting_the_competitor_list(self):
        result = head_to_head(["a", "c"], ["b", "d"], [1.0, 1.0], competitors=["a", "b"])
        assert result["names"] == ["a", "b"]
        assert result["played"].sum() == 2  # the c-d match is skipped

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            head_to_head(["a"], ["b", "c"], [1.0])


class TestRatingChurn:
    def test_short_memory_churns_more_than_long(self):
        home, away, outcome, times = round_robin(
            [f"t{i}" for i in range(8)], rounds=10, upset_rate=0.4
        )
        churn = {}
        for label, alpha in (("short", 20.0), ("long", 2000.0)):
            model = BScoreModel(alpha=alpha).add_matches(home, away, outcome, times)
            history = model.score_history(times)
            churn[label] = rating_churn(history)["rank_change"].mean()
        assert churn["short"] > churn["long"]

    def test_shapes_are_one_shorter_than_the_history(self):
        home, away, outcome, times = round_robin(["a", "b", "c", "d"], rounds=4, upset_rate=0.4)
        model = BScoreModel(alpha=200.0).add_matches(home, away, outcome, times)
        history = model.score_history(times)
        churn = rating_churn(history, top=2)
        assert churn["times"].size == history.times.size - 1
        assert churn["rank_change"].size == churn["times"].size
        assert churn["entered_top"].size == churn["times"].size

    def test_a_static_history_has_no_churn(self):
        model = BScoreModel(kernel="uniform")
        model.add_matches(["a", "b"], ["b", "c"], [1.0, 1.0], [0.0, 1.0])
        history = model.score_history([10.0, 20.0, 30.0])
        np.testing.assert_allclose(rating_churn(history, top=2)["rank_change"], 0.0)

    def test_validation(self):
        model = BScoreModel(alpha=200.0)
        model.add_matches(["a"], ["b"], [1.0], [0.0])
        with pytest.raises(ValueError, match="at least two epochs"):
            rating_churn(model.score_history([0.0]))
        with pytest.raises(ValueError, match="top must lie"):
            rating_churn(model.score_history([0.0, 1.0]), top=99)


class TestUpsetRate:
    def test_matches_the_forecasts_when_calibrated(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0.5, 0.95, 20_000)
        y = (rng.random(20_000) < p).astype(float)
        result = upset_rate(y, p)
        assert result["upset_rate"] == pytest.approx(result["expected"], abs=0.01)

    def test_detects_overconfidence(self):
        p = np.full(1000, 0.9)
        y = (np.arange(1000) % 2).astype(float)  # actually a coin flip
        result = upset_rate(y, p)
        assert result["upset_rate"] > result["expected"] + 0.3

    def test_no_favoured_matches(self):
        result = upset_rate([1.0], [0.4], threshold=0.5)
        assert result["n_favoured"] == 0.0
        assert np.isnan(result["upset_rate"])
