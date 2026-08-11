"""Hyperparameter search, and the leakage discipline it depends on."""

from __future__ import annotations

import numpy as np
import pytest

from bscores.tuning import (
    CALIBRATION_KEYS,
    NETWORK_KEYS,
    TuningResult,
    _expand,
    _split,
    grid_search,
    refit_best,
)

from .test_backtest import competition


@pytest.fixture(scope="module")
def fixtures():
    return competition(n_teams=8, rounds=14, seed=0, upset_rate=0.25)


class TestGridExpansion:
    def test_cartesian_product(self):
        combos = _expand({"a": [1, 2], "b": ["x", "y", "z"]})
        assert len(combos) == 6
        assert {"a": 1, "b": "x"} in combos

    def test_empty_grid_is_one_default_config(self):
        assert _expand({}) == [{}]

    def test_duplicates_are_dropped(self):
        assert len(_expand({"a": [1, 1, 2]})) == 2

    def test_split_separates_network_from_calibration(self):
        network, calibration = _split({"alpha": 30.0, "transform": "log", "symmetric": True})
        assert dict(network) == {"alpha": 30.0}
        assert calibration == {"transform": "log", "symmetric": True}

    def test_every_key_is_classified(self):
        assert not set(NETWORK_KEYS) & set(CALIBRATION_KEYS)


class TestGridSearch:
    def test_returns_one_row_per_configuration(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [30.0, 200.0], "transform": ["identity", "log"]},
            validation_start=0.4,
        )
        assert len(search) == 4
        assert all("log_loss" in row for row in search.rows)

    def test_ranked_best_first(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [10.0, 60.0, 400.0]}, validation_start=0.4,
        )
        scores = [row["log_loss"] for row in search.rows]
        assert scores == sorted(scores)
        assert search.best_score == scores[0]

    def test_accuracy_ranks_the_other_way(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [10.0, 400.0]}, validation_start=0.4, metric="accuracy",
        )
        scores = [row["accuracy"] for row in search.rows]
        assert scores == sorted(scores, reverse=True)

    def test_calibration_variants_share_one_rating_pass(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={
                "alpha": [30.0, 200.0],
                "transform": ["identity", "log", "sqrt"],
                "symmetric": [False, True],
            },
            validation_start=0.4,
        )
        assert len(search) == 12
        # Two alphas, six calibration variants each.
        assert search.n_rating_passes == 2

    def test_best_params_excludes_the_metrics(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [30.0, 200.0], "transform": ["identity"]}, validation_start=0.4,
        )
        assert set(search.best_params) == {"alpha", "transform"}

    def test_sensitivity_reports_best_per_value(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [10.0, 60.0, 400.0], "transform": ["identity", "log"]},
            validation_start=0.4,
        )
        profile = search.sensitivity("alpha")
        assert len(profile) == 3
        assert profile[0][1] == search.best_score
        assert [s for _, s in profile] == sorted(s for _, s in profile)

    def test_sensitivity_rejects_keys_not_in_the_grid(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times, grid={"alpha": [30.0]}, validation_start=0.4
        )
        with pytest.raises(KeyError, match="not part of the grid"):
            search.sensitivity("kernel")

    def test_progress_callback_fires_once_per_config(self, fixtures):
        home, away, outcome, times = fixtures
        seen = []
        grid_search(
            home, away, outcome, times,
            grid={"alpha": [30.0, 200.0]}, validation_start=0.4,
            progress=lambda done, total, config: seen.append((done, total)),
        )
        assert seen == [(1, 2), (2, 2)]

    def test_weight_options_are_selectable(self, fixtures):
        home, away, outcome, times = fixtures
        options = {"flat": np.ones(len(home)), "heavy": np.full(len(home), 1.0)}
        options["heavy"][::2] = 3.0
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [60.0], "weights": ["flat", "heavy"]},
            weight_options=options, validation_start=0.4,
        )
        assert len(search) == 2
        assert {row["weights"] for row in search.rows} == {"flat", "heavy"}

    def test_best_result_is_the_winning_forecast(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [10.0, 60.0, 400.0]}, validation_start=0.4,
        )
        assert search.best_result is not None
        assert search.best_result.metrics()["log_loss"] == pytest.approx(search.best_score)

    def test_repr_and_len(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times, grid={"alpha": [60.0]}, validation_start=0.4
        )
        assert "TuningResult(" in repr(search)
        assert len(search) == 1


class TestValidationWindow:
    def test_scoring_stops_at_validation_end(self, fixtures):
        home, away, outcome, times = fixtures
        cut = float(np.quantile(times, 0.7))
        windowed = grid_search(
            home, away, outcome, times,
            grid={"alpha": [60.0]}, validation_start=0.4, validation_end=cut,
        )
        full = grid_search(
            home, away, outcome, times, grid={"alpha": [60.0]}, validation_start=0.4,
        )
        assert windowed.best["n"] < full.best["n"]
        assert windowed.best_score != full.best_score

    def test_test_period_outcomes_cannot_influence_the_choice(self, fixtures):
        home, away, outcome, times = fixtures
        cut = float(np.quantile(times, 0.7))
        grid = {"alpha": [10.0, 60.0, 400.0]}

        baseline = grid_search(
            home, away, outcome, times, grid=grid, validation_start=0.4, validation_end=cut
        )
        tampered_outcome = np.asarray(outcome, dtype=float).copy()
        after = np.asarray(times) >= cut
        tampered_outcome[after] = 1.0 - tampered_outcome[after]
        tampered = grid_search(
            home, away, outcome=tampered_outcome, times=times,
            grid=grid, validation_start=0.4, validation_end=cut,
        )
        assert baseline.best_params == tampered.best_params
        assert baseline.best_score == pytest.approx(tampered.best_score)

    def test_validation_end_must_follow_the_start(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="validation_end must fall after"):
            grid_search(
                home, away, outcome, times,
                grid={"alpha": [60.0]}, validation_start=0.5, validation_end=float(times[0]),
            )


class TestErrors:
    def test_unknown_grid_key(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="unknown grid keys"):
            grid_search(home, away, outcome, times, grid={"nonsense": [1]}, validation_start=0.4)

    def test_weights_without_options(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="weight_options"):
            grid_search(home, away, outcome, times, grid={"weights": ["a"]}, validation_start=0.4)

    def test_unknown_weight_name(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(KeyError, match="no entry named"):
            grid_search(
                home, away, outcome, times,
                grid={"weights": ["missing"]},
                weight_options={"flat": np.ones(len(home))}, validation_start=0.4,
            )

    def test_unknown_metric(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="unknown metric"):
            grid_search(
                home, away, outcome, times,
                grid={"alpha": [60.0]}, validation_start=0.4, metric="vibes",
            )

    def test_validation_start_must_leave_both_sides(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="matches on both sides"):
            grid_search(home, away, outcome, times, grid={"alpha": [60.0]}, validation_start=0)


class TestRefitBest:
    def test_reproduces_the_chosen_configuration(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [10.0, 60.0], "transform": ["identity", "log"]},
            validation_start=0.4, validation_end=float(np.quantile(times, 0.7)),
        )
        held_out = refit_best(
            home, away, outcome, times, search.best_params,
            test_start=float(np.quantile(times, 0.7)),
        )
        assert len(held_out) > 0
        assert np.all((held_out.probability >= 0.0) & (held_out.probability <= 1.0))

    def test_accepts_a_weight_selection(self, fixtures):
        home, away, outcome, times = fixtures
        options = {"flat": np.ones(len(home))}
        held_out = refit_best(
            home, away, outcome, times, {"alpha": 60.0, "weights": "flat"},
            weight_options=options, test_start=0.7,
        )
        assert len(held_out) > 0

    def test_unknown_weight_name_rejected(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(KeyError, match="no entry named"):
            refit_best(home, away, outcome, times, {"weights": "missing"}, test_start=0.7)


def test_tuning_result_sorts_on_construction():
    rows = [{"alpha": 1.0, "log_loss": 0.7}, {"alpha": 2.0, "log_loss": 0.5}]
    result = TuningResult(rows=rows, metric="log_loss", validation=(0, None))
    assert result.best["alpha"] == 2.0
    assert result.top(1) == [{"alpha": 2.0, "log_loss": 0.5}]
