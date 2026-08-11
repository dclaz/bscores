"""Bayesian hyperparameter search, and the leakage discipline it depends on."""

from __future__ import annotations

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

from bscores.search import (  # noqa: E402
    SearchResult,
    build_tuned,
    config_to_model,
    optuna_search,
    optuna_search_elo,
    suggest_config,
    suggest_elo,
)

from .test_backtest import competition  # noqa: E402


@pytest.fixture(scope="module")
def fixtures():
    return competition(n_teams=8, rounds=16, seed=0, upset_rate=0.25)


@pytest.fixture(scope="module")
def margins(fixtures):
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 30.0, len(fixtures[0]))


def sample(space, seed=0, **kwargs):
    """Draw one configuration from a space function, without running a study."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    drawn = {}

    def objective(trial):
        drawn.update(space(trial, **kwargs) if kwargs else space(trial))
        return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=1)
    return drawn


class TestSearchSpace:
    def test_core_parameters_are_always_sampled(self):
        config = sample(suggest_config)
        assert {"alpha", "kernel", "transform", "regularization"} <= set(config)
        assert 7.0 <= config["alpha"] <= 2000.0

    def test_window_width_only_appears_for_a_window_kernel(self):
        for seed in range(12):
            config = sample(suggest_config, seed=seed)
            assert ("window_width" in config) == (config["kernel"] == "window")

    def test_max_age_only_appears_for_the_heavy_tailed_kernel(self):
        for seed in range(12):
            config = sample(suggest_config, seed=seed)
            if "max_age" in config:
                assert config["kernel"] == "hyperbolic"

    def test_margin_parameters_are_absent_without_margins(self):
        config = sample(suggest_config)
        assert "margin_scheme" not in config

    def test_margin_parameters_travel_together(self, margins):
        for seed in range(8):
            config = sample(suggest_config, seed=seed, margins=margins)
            present = {"margin_scheme", "margin_scale", "margin_cap"} & set(config)
            assert present in (set(), {"margin_scheme", "margin_scale", "margin_cap"})

    def test_elo_schedule_branches_are_exclusive(self):
        for seed in range(12):
            config = sample(suggest_elo, seed=seed)
            fixed = "k" in config
            scheduled = {"k_scale", "k_shape", "k_power"} <= set(config)
            assert fixed != scheduled
            assert 0.0 <= config["home_advantage"] <= 120.0


class TestConfigToModel:
    def test_kernel_names_map_to_kernels(self):
        from bscores.decay import Exponential, Hyperbolic, Window

        hyperbolic = config_to_model({"alpha": 100.0, "kernel": "hyperbolic"})
        exponential = config_to_model({"alpha": 100.0, "kernel": "exponential"})
        assert isinstance(hyperbolic.kernel, Hyperbolic)
        assert isinstance(exponential.kernel, Exponential)
        windowed = config_to_model({"alpha": 100.0, "kernel": "window", "window_width": 300.0})
        assert isinstance(windowed.kernel, Window)

    def test_scalar_settings_are_carried_through(self):
        model = config_to_model(
            {"alpha": 50.0, "kernel": "exponential", "draw_weight": 0.25,
             "regularization": 0.02, "max_age": 400.0}
        )
        assert model.draw_weight == 0.25
        assert model.regularization == 0.02
        assert model.network.max_age == 400.0

    def test_build_tuned_returns_a_calibrated_model(self, fixtures, margins):
        home, away, outcome, times = fixtures
        model = build_tuned(
            {"alpha": 120.0, "kernel": "exponential", "transform": "sqrt",
             "margin_scheme": "linear", "margin_scale": 24.0, "margin_cap": 3.0},
            home, away, outcome, times, margins=margins,
        )
        assert model.calibrator is not None
        assert 0.0 < model.predict_win([[home[0]], [away[0]]])[0] < 1.0


class TestOptunaSearch:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, fixtures):
        home, away, outcome, times = fixtures
        return optuna_search(
            home, away, outcome, times,
            validation_start=0.5, n_trials=12, n_startup_trials=6, seed=0,
        )

    def test_returns_one_row_per_trial(self, result):
        assert isinstance(result, SearchResult)
        assert len(result) == 12
        assert all("log_loss" in row for row in result.rows)

    def test_ranked_best_first(self, result):
        scores = [row["log_loss"] for row in result.rows]
        assert scores == sorted(scores)
        assert result.best_score == pytest.approx(scores[0])

    def test_best_params_rebuild_a_model(self, result):
        assert "alpha" in result.best_params
        config_to_model(result.best_params)

    def test_seeded_search_is_reproducible(self, fixtures):
        home, away, outcome, times = fixtures
        kwargs = dict(validation_start=0.5, n_trials=8, n_startup_trials=4, seed=7)
        first = optuna_search(home, away, outcome, times, **kwargs)
        second = optuna_search(home, away, outcome, times, **kwargs)
        assert first.best_params == second.best_params
        assert first.best_score == pytest.approx(second.best_score)

    def test_accuracy_ranks_the_other_way(self, fixtures):
        home, away, outcome, times = fixtures
        search = optuna_search(
            home, away, outcome, times, validation_start=0.5,
            n_trials=8, n_startup_trials=4, seed=0, metric="accuracy",
        )
        scores = [row["accuracy"] for row in search.rows]
        assert scores == sorted(scores, reverse=True)

    def test_repr_names_the_winner(self, result):
        assert "SearchResult(" in repr(result)
        assert "alpha" in repr(result)

    def test_rejects_a_zero_budget(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            optuna_search(home, away, outcome, times, validation_start=0.5, n_trials=0)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            optuna_search(["a"], ["b", "c"], [1.0], [0.0], validation_start=0.5, n_trials=2)


class TestValidationWindow:
    def test_outcomes_after_the_window_cannot_influence_the_search(self, fixtures):
        home, away, outcome, times = fixtures
        cut = float(np.quantile(np.asarray(times, dtype=float), 0.75))
        kwargs = dict(
            validation_start=0.4, validation_end=cut,
            n_trials=10, n_startup_trials=5, seed=0,
        )
        baseline = optuna_search(home, away, outcome, times, **kwargs)

        tampered = np.asarray(outcome, dtype=float).copy()
        after = np.asarray(times, dtype=float) >= cut
        tampered[after] = 1.0 - tampered[after]
        rerun = optuna_search(home, away, tampered, times, **kwargs)

        assert baseline.best_params == rerun.best_params
        assert baseline.best_score == pytest.approx(rerun.best_score)

    def test_a_shorter_window_scores_differently(self, fixtures):
        home, away, outcome, times = fixtures
        cut = float(np.quantile(np.asarray(times, dtype=float), 0.7))
        kwargs = dict(n_trials=6, n_startup_trials=3, seed=0)
        windowed = optuna_search(
            home, away, outcome, times, validation_start=0.4, validation_end=cut, **kwargs
        )
        full = optuna_search(home, away, outcome, times, validation_start=0.4, **kwargs)
        assert windowed.best_score != full.best_score


class TestOptunaSearchElo:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, fixtures):
        home, away, outcome, times = fixtures
        return optuna_search_elo(
            home, away, outcome, times,
            validation_start=0.5, n_trials=12, n_startup_trials=6, seed=0,
        )

    def test_best_params_construct_an_elo(self, result):
        from bscores.baselines import Elo

        assert "home_advantage" in result.best_params
        Elo(**result.best_params)

    def test_ranked_best_first(self, result):
        scores = [row["log_loss"] for row in result.rows]
        assert scores == sorted(scores)

    def test_beats_or_matches_the_default_on_its_own_window(self, fixtures):
        from bscores.baselines import Elo
        from bscores.metrics import log_loss

        home, away, outcome, times = fixtures
        search = optuna_search_elo(
            home, away, outcome, times, validation_start=0.5,
            n_trials=25, n_startup_trials=10, seed=0,
        )
        start = int(round(0.5 * len(home)))
        default = log_loss(
            np.asarray(outcome)[start:],
            Elo().run(home, away, outcome)[start:],
        )
        assert search.best_score <= default

    def test_window_must_be_ordered(self, fixtures):
        home, away, outcome, times = fixtures
        with pytest.raises(ValueError, match="validation_end must fall after"):
            optuna_search_elo(
                home, away, outcome, times,
                validation_start=0.8, validation_end=0.2, n_trials=2,
            )
