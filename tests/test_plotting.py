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

    def test_round_axis_collapses_the_off_season(self, model, fixtures):
        from bscores.datasets import load_afl

        afl = load_afl(as_frame=False)
        rated = BScoreModel(alpha=200.0)
        rated.add_matches(afl.home_team, afl.away_team, afl.outcome, afl.date)
        dates = np.unique(afl.date)
        history = rated.score_history(dates)

        by_date = plot_ratings(history, top=2)
        by_round = plot_ratings(history, top=2, x="round", schedule=afl)

        dates = by_date.get_lines()[0].get_xdata().astype("datetime64[D]").astype(int)
        gaps_by_date = np.diff(dates)
        gaps_by_round = np.diff(by_round.get_lines()[0].get_xdata())
        # The calendar axis has months-long summer jumps; the round axis does not.
        assert gaps_by_date.max() > 100
        assert gaps_by_round.max() <= 2
        assert by_round.get_xlabel() == "season and round"

    def test_round_axis_without_a_schedule_spaces_epochs_evenly(self, model, fixtures):
        history = model.score_history(fixtures[3])
        ax = plot_ratings(history, top=2, x="round")
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_xdata(), np.arange(history.times.size)
        )

    def test_explicit_x_positions(self, model, fixtures):
        history = model.score_history(fixtures[3])
        positions = np.linspace(0.0, 1.0, history.times.size)
        ax = plot_ratings(history, top=1, x=positions)
        np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), positions)

    def test_wrong_length_x_rejected(self, model, fixtures):
        history = model.score_history(fixtures[3])
        with pytest.raises(ValueError, match="positions for"):
            plot_ratings(history, x=np.zeros(3))

    def test_unknown_x_mode_rejected(self, model, fixtures):
        history = model.score_history(fixtures[3])
        with pytest.raises(ValueError, match="unknown x axis"):
            plot_ratings(history, x="vibes")


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

    def test_categorical_parameter_gets_one_point_per_value(self, fixtures):
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [100.0], "transform": ["identity", "log", "sqrt"]},
            validation_start=0.4,
        )
        ax = plot_tuning(search, "transform")
        assert len(ax.collections[-1].get_offsets()) == 3
        assert [t.get_text() for t in ax.get_yticklabels()] == ["identity", "log", "sqrt"]

    def test_categorical_axis_is_zoomed_so_differences_are_visible(self, fixtures):
        # Scores differ in the third decimal; an axis anchored at zero would
        # render every value as the same point.
        home, away, outcome, times = fixtures
        search = grid_search(
            home, away, outcome, times,
            grid={"alpha": [100.0], "transform": ["identity", "log", "sqrt"]},
            validation_start=0.4,
        )
        ax = plot_tuning(search, "transform")
        scores = [s for _, s in search.sensitivity("transform")]
        low, high = ax.get_xlim()
        assert low > 0.0
        assert (high - low) < 4 * (max(scores) - min(scores))

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
