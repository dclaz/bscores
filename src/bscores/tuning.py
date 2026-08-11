"""Choosing the hyperparameters.

The paper fixes ``alpha = 365`` to mirror the 52-week ATP/WTA ranking window and
flags the choice as "worthy of investigation".  It is: on a competition with a
different rhythm, ``alpha`` moves forecast accuracy further than the gap between
B-scores and any competing rating system.

:func:`grid_search` searches out of sample and, importantly, on a window that is
*not* the one you intend to report on.  Tuning and reporting on the same matches
inflates the result by however hard you searched; the three-way split below is
the cheapest way to not fool yourself:

.. code-block:: text

    |------- warm-up -------|--- validation ---|------ test ------|
     model builds a history   grid_search here   report this only

Searching is cheaper than it looks.  Ratings depend only on the network
settings, so configurations that differ solely in how the logit is fitted share
one causal rating pass; the search groups them automatically.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._time import as_days
from .backtest import BacktestResult, _resolve_start, walk_forward
from .calibration import LogitCalibrator
from .metrics import evaluate
from .models import BScoreModel
from .typing import Names

__all__ = [
    "TuningResult",
    "grid_search",
    "refit_best",
    "DEFAULT_GRID",
    "NETWORK_KEYS",
    "CALIBRATION_KEYS",
]

#: Grid keys that change the ratings, and so force a fresh causal sweep.
NETWORK_KEYS: tuple[str, ...] = (
    "alpha",
    "kernel",
    "draw_weight",
    "regularization",
    "max_age",
    "weights",
)

#: Grid keys that only change how ratings are turned into probabilities.  These
#: are swept for free on top of an already-computed rating pass.
CALIBRATION_KEYS: tuple[str, ...] = ("transform", "symmetric", "ridge", "fit_intercept")

#: A reasonable starting grid: a wide sweep of the memory parameter against the
#: two kernel shapes and the two feature transforms.
DEFAULT_GRID: dict[str, list[Any]] = {
    "alpha": [7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0, 180.0, 365.0, 730.0],
    "kernel": ["hyperbolic", "exponential"],
    "transform": ["identity", "log"],
}

#: Settings chosen by ``python examples/afl_tuning.py`` on the bundled AFL data.
#:
#: Selected on 2019-2022 (783 matches) and then scored once on 2023-2026 (828
#: matches), where they reach a 0.585 log-loss against 0.635 for the paper's
#: ``alpha=365`` hyperbolic default and 0.601 for Elo (Diebold-Mariano -5.24 and
#: -1.98).  An earlier search on the shorter 2009-2022 archive, validating on
#: 2016-2018 instead, picked exactly these values — so they are replicated on
#: disjoint validation windows, not fitted to one.
#:
#: Still sport-specific: treat it as a worked example of what :func:`grid_search`
#: produces, not as a default for your own competition.  The ``"weights"`` entry
#: names a margin-of-victory vector; build it with
#: ``bscores.weights.margin_weight(margin, scheme="linear", scale=24.0, cap=3.0)``.
AFL_TUNED: dict[str, Any] = {
    "alpha": 120.0,
    "kernel": "exponential",
    "transform": "sqrt",
    "regularization": 0.03,
    "weights": "mov",
}

#: Metrics where a larger number is better.
_MAXIMISE = frozenset({"accuracy"})


@dataclass
class TuningResult:
    """Every configuration tried, scored on the validation window."""

    rows: list[dict[str, Any]]
    """One dict per configuration: its parameters plus every metric."""

    metric: str
    """The metric that decided the ranking."""

    validation: tuple[Any, Any]
    """``(start, end)`` of the window the scores were computed on."""

    n_rating_passes: int = 0
    """Causal rating sweeps actually run, versus ``len(rows)`` configurations."""

    best_result: BacktestResult | None = field(default=None, repr=False)
    """The winning configuration's forecasts, for inspection."""

    def __post_init__(self) -> None:
        reverse = self.metric in _MAXIMISE
        self.rows.sort(key=lambda row: row[self.metric], reverse=reverse)

    @property
    def best(self) -> dict[str, Any]:
        """The winning row, parameters and metrics together."""
        return self.rows[0]

    @property
    def best_params(self) -> dict[str, Any]:
        """Just the winning parameters, ready to splat into a model."""
        metrics = set(evaluate([1.0], [0.5])) | {"n"}
        return {k: v for k, v in self.rows[0].items() if k not in metrics}

    @property
    def best_score(self) -> float:
        """The winning value of :attr:`metric`."""
        return float(self.rows[0][self.metric])

    def top(self, n: int = 10) -> list[dict[str, Any]]:
        """The ``n`` best configurations."""
        return self.rows[:n]

    def sensitivity(self, key: str) -> list[tuple[Any, float]]:
        """Best achievable score for each value of one parameter.

        Answers "how much does this knob actually matter?" — a flat profile
        means the parameter is not worth tuning on this data.
        """
        if key not in self.rows[0]:
            raise KeyError(f"{key!r} was not part of the grid")
        best: dict[Any, float] = {}
        for row in self.rows:
            value, score = row[key], float(row[self.metric])
            better = (
                score > best.get(value, -np.inf)
                if self.metric in _MAXIMISE
                else score < best.get(value, np.inf)
            )
            if value not in best or better:
                best[value] = score
        ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=self.metric in _MAXIMISE)
        return [(value, float(score)) for value, score in ordered]

    def to_frame(self):  # pragma: no cover - thin pandas adapter
        """Return the search results as a ``pandas.DataFrame``."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.best_params.items())
        return (
            f"TuningResult({len(self.rows)} configs, {self.n_rating_passes} rating passes, "
            f"best {self.metric}={self.best_score:.4f} at {params})"
        )


def _expand(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, as a list of parameter dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    combos = []
    seen: set[str] = set()
    for values in itertools.product(*(list(grid[k]) for k in keys)):
        config = dict(zip(keys, values))
        signature = repr(sorted(config.items(), key=lambda kv: kv[0]))
        if signature in seen:
            continue
        seen.add(signature)
        combos.append(config)
    return combos


def _split(config: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
    """Separate a configuration into its network key and calibration part."""
    network = tuple(sorted((k, v) for k, v in config.items() if k in NETWORK_KEYS))
    calibration = {k: v for k, v in config.items() if k in CALIBRATION_KEYS}
    return network, calibration


def grid_search(
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    grid: Mapping[str, Sequence[Any]] | None = None,
    weight_options: Mapping[str, Any] | None = None,
    metric: str = "log_loss",
    validation_start: Any,
    validation_end: Any = None,
    refit_every: int = 300,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> TuningResult:
    """Search a hyperparameter grid out of sample.

    Parameters
    ----------
    home, away, outcome, times
        The full fixture list, in any order; it is sorted internally.  Matches
        after ``validation_end`` still enter the *network* (they are history for
        nothing, since scoring stops at ``validation_end``) but never influence
        a validation score, so holding a test period back is safe.
    grid
        Mapping of parameter name to the values to try; the cartesian product is
        searched.  Recognised keys are :data:`NETWORK_KEYS` and
        :data:`CALIBRATION_KEYS`.  Defaults to :data:`DEFAULT_GRID`.
    weight_options
        Named per-match weight vectors, selected by putting their names in
        ``grid["weights"]``.  Build them with :mod:`bscores.weights`.
    metric
        Which of ``log_loss``, ``brier_score``, ``accuracy`` or
        ``classification_error`` decides the ranking.
    validation_start
        Where scoring begins — everything before it is the warm-up window the
        first model is trained on.  An ``int`` match count, a ``float`` fraction
        or a date.
    validation_end
        Where scoring stops; ``None`` runs to the end of the fixture list.  Set
        this to the start of your test period.
    refit_every
        Matches forecast between logit refits.
    progress
        Called as ``progress(done, total, config)`` after each configuration.

    Returns
    -------
    TuningResult

    Examples
    --------
    >>> from bscores.datasets import load_afl
    >>> from bscores.tuning import grid_search
    >>> afl = load_afl(as_frame=False)
    >>> search = grid_search(
    ...     afl.home_team, afl.away_team, afl.outcome, afl.date,
    ...     grid={"alpha": [21.0, 365.0]},
    ...     validation_start="2016-01-01", validation_end="2019-01-01",
    ... )
    >>> search.best_params["alpha"]
    21.0
    """
    settings = dict(DEFAULT_GRID if grid is None else grid)
    unknown = set(settings) - set(NETWORK_KEYS) - set(CALIBRATION_KEYS)
    if unknown:
        raise ValueError(
            f"unknown grid keys {sorted(unknown)}; "
            f"choose from {sorted(NETWORK_KEYS + CALIBRATION_KEYS)}"
        )
    if "weights" in settings and weight_options is None:
        raise ValueError("grid['weights'] needs a weight_options mapping to resolve names")

    home_names = np.asarray(list(home), dtype=object)
    away_names = np.asarray(list(away), dtype=object)
    results = np.asarray(outcome, dtype=np.float64).ravel()
    stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    order = np.argsort(stamps, kind="stable")
    home_names, away_names = home_names[order], away_names[order]
    results, stamps = results[order], stamps[order]

    start = _resolve_start(stamps, validation_start)
    if not 0 < start < stamps.size:
        raise ValueError(
            f"validation_start selects {start} of {stamps.size} matches; it must leave "
            "matches on both sides"
        )
    end_day = None if validation_end is None else float(as_days(validation_end))
    if end_day is not None and end_day <= stamps[start]:
        raise ValueError("validation_end must fall after validation_start")

    configs = _expand(settings)
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for config in configs:
        network, calibration = _split(config)
        grouped.setdefault(network, []).append(calibration)

    rows: list[dict[str, Any]] = []
    best_result: BacktestResult | None = None
    best_score = np.inf if metric not in _MAXIMISE else -np.inf
    done = 0

    for network, calibrations in grouped.items():
        network_params = dict(network)
        share = None
        if "weights" in network_params:
            name = network_params["weights"]
            assert weight_options is not None
            if name not in weight_options:
                raise KeyError(f"weight_options has no entry named {name!r}")
            share = np.asarray(weight_options[name], dtype=np.float64).ravel()[order]

        model = BScoreModel(
            alpha=network_params.get("alpha", 365.0),
            kernel=network_params.get("kernel"),
            draw_weight=network_params.get("draw_weight", 0.5),
            regularization=network_params.get("regularization", 0.0),
            max_age=network_params.get("max_age"),
        )
        model.add_matches(home_names, away_names, results, stamps, weights=share)
        home_scores, away_scores = model.match_scores(
            home_names, away_names, stamps, inclusive=False
        )

        for calibration in calibrations:
            calibrator = LogitCalibrator(
                fit_intercept=calibration.get("fit_intercept", True),
                symmetric=calibration.get("symmetric", False),
                transform=calibration.get("transform", "identity"),
                ridge=calibration.get("ridge", 1e-6),
            )
            probability, train_size, coefficients, refit_at = walk_forward(
                home_scores, away_scores, results, start,
                refit_every=refit_every, calibrator=calibrator,
            )
            result = BacktestResult(
                times=stamps[start:],
                home=home_names[start:],
                away=away_names[start:],
                outcome=results[start:],
                probability=probability,
                home_score=home_scores[start:],
                away_score=away_scores[start:],
                train_size=train_size,
                coefficients=coefficients,
                refit_at=refit_at,
                model=None,
            )
            scored = result if end_day is None else result.between(None, end_day)
            metrics = scored.metrics()
            if metric not in metrics:
                raise ValueError(f"unknown metric {metric!r}; choose from {sorted(metrics)}")

            row = {**network_params, **calibration, **metrics}
            rows.append(row)

            score = metrics[metric]
            better = score > best_score if metric in _MAXIMISE else score < best_score
            if better:
                best_score, best_result = score, scored

            done += 1
            if progress is not None:
                progress(done, len(configs), row)

    return TuningResult(
        rows=rows,
        metric=metric,
        validation=(validation_start, validation_end),
        n_rating_passes=len(grouped),
        best_result=best_result,
    )


def refit_best(
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    params: Mapping[str, Any],
    *,
    weight_options: Mapping[str, Any] | None = None,
    test_start: Any,
    refit_every: int = 300,
) -> BacktestResult:
    """Re-run one configuration and score it on a held-out test window.

    The companion to :func:`grid_search`: search on validation, report with
    this.  Takes the same parameter names, so ``refit_best(..., search
    .best_params, test_start=...)`` is the intended call.
    """
    from .backtest import rolling_forecast

    settings = dict(params)
    share = None
    if "weights" in settings:
        name = settings.pop("weights")
        if weight_options is None or name not in weight_options:
            raise KeyError(f"weight_options has no entry named {name!r}")
        share = weight_options[name]

    model = BScoreModel(
        alpha=settings.pop("alpha", 365.0),
        kernel=settings.pop("kernel", None),
        draw_weight=settings.pop("draw_weight", 0.5),
        regularization=settings.pop("regularization", 0.0),
        max_age=settings.pop("max_age", None),
    )
    return rolling_forecast(
        home,
        away,
        outcome,
        times,
        model=model,
        initial_train=test_start,
        refit_every=refit_every,
        weights=share,
        symmetric=settings.pop("symmetric", False),
        transform=settings.pop("transform", "identity"),
        ridge=settings.pop("ridge", 1e-6),
        fit_intercept=settings.pop("fit_intercept", True),
    )
