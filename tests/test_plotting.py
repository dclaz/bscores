"""Plotting helpers.

These check that each figure is built without error and carries the data it is
supposed to; they do not compare pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bscores import BScoreModel, rolling_forecast  # noqa: E402
from bscores.decay import Exponential, Hyperbolic  # noqa: E402
from bscores.plotting import (  # noqa: E402
    plot_backtest,
    plot_calibration,
    plot_decay,
    plot_network,
    plot_ratings,
    plot_tuning,
)
from bscores.tuning import grid_search  # noqa: E402

from .test_models import round_robin  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fixtures():
    teams = [f"t{i}" for i in range(6)]
    return round_robin(teams, rounds=8, seed=0, upset_rate=0.25)


@pytest.fixture(scope="module")
def model(fixtures):
    home, away, outcome, times = fixtures
    return BScoreModel(alpha=200.0).fit(home, away, outcome, times)


class TestPlotRatings:
    def test_draws_one_line_per_competitor(self, model, fixtures):
        history = model.score_history(fixtures[3])
        ax = plot_ratings(history, top=3)
        assert len(ax.get_lines()) == 3
        assert ax.get_ylabel() == "B-score"

    def test_explicit_competitors(self, model, fixtures):
        history = model.score_history(fixtures[3])
        ax = plot_ratings(history, competitors=["t0", "t1"])
        assert {line.get_label() for line in ax.get_lines()} == {"t0", "t1"}

    def test_reuses_a_supplied_axes(self, model, fixtures):
        history = model.score_history(fixtures[3])
        _, ax = plt.subplots()
        assert plot_ratings(history, top=2, ax=ax) is ax


class TestPlotCalibration:
    def test_draws_the_diagonal_and_the_curve(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, 600)
        y = (rng.random(600) < p).astype(float)
        ax = plot_calibration(y, p, bins=5)
        assert len(ax.get_lines()) == 2  # diagonal + curve
        assert ax.get_xlim() == (0.0, 1.0)

    def test_labelled(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.2, 0.8, 200)
        y = (rng.random(200) < p).astype(float)
        ax = plot_calibration(y, p, bins=4, label="B-score")
        assert ax.get_legend() is not None


class TestPlotTuning:
    def test_numeric_parameter_gets_a_line(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times, grid={"alpha": [20.0, 100.0, 500.0]}, validation_start=0.4
        )
        ax = plot_tuning(search, "alpha")
        assert len(ax.get_lines()) >= 1
        assert ax.get_xscale() == "log"

    def test_categorical_parameter_gets_bars(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [100.0], "transform": ["identity", "log", "sqrt"]},
            validation_start=0.4,
        )
        ax = plot_tuning(search, "transform")
        assert len(ax.patches) == 3

    def test_linear_axis_on_request(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times, grid={"alpha": [20.0, 100.0]}, validation_start=0.4
        )
        assert plot_tuning(search, "alpha", logx=False).get_xscale() == "linear"


class TestPlotDecay:
    def test_one_line_per_kernel(self):
        ax = plot_decay([Hyperbolic(365.0), Exponential(365.0)])
        assert len(ax.get_lines()) == 2

    def test_the_hyperbolic_tail_is_visibly_fatter(self):
        ages = np.array([1460.0])
        assert Hyperbolic(365.0)(ages)[0] > Exponential(365.0)(ages)[0]


class TestPlotNetwork:
    def test_draws_a_node_per_competitor(self, model):
        ax = plot_network(model)
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == len(model.players)

    def test_labels_every_competitor(self, model):
        ax = plot_network(model)
        # The arcs are drawn as empty-text annotations; skip those.
        labelled = {a.get_text() for a in ax.texts if a.get_text()}
        assert labelled == set(model.players)

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="no competitors"):
            plot_network(BScoreModel())


class TestPlotBacktest:
    def test_cumulative_advantage_is_monotone_in_length(self, fixtures):
        home, away, outcome, times = fixtures
        result = rolling_forecast(home, away, outcome, times, alpha=200.0)
        ax = plot_backtest(result)
        line = ax.get_lines()[0]
        assert len(line.get_ydata()) == len(result)
        assert ax.get_ylabel() == "cumulative log-loss saved"

    def test_explicit_baseline(self, fixtures):
        home, away, outcome, times = fixtures
        result = rolling_forecast(home, away, outcome, times, alpha=200.0)
        ax = plot_backtest(result, baseline=np.full(len(result), 0.5))
        assert "baseline" in ax.get_title()
