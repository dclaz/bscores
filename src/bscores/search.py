"""Bayesian hyperparameter search, via Optuna.

:func:`~bscores.tuning.grid_search` is exhaustive and reproducible, which makes
it the right tool for a handful of knobs you want to compare like for like.  It
has two limits this module lifts.

**Cost.** A grid's size is the product of its axes.  Nine parameters at four
values each is 260 000 configurations; a Tree-structured Parzen Estimator finds
a good corner of that space in a few hundred trials, because it spends its
budget where earlier trials looked promising rather than spreading it evenly.

**Shape.** A grid is a box, and the real space is not.  ``window_width`` only
means something when the kernel is ``"window"``; the margin scheme's ``scale``
and ``cap`` only exist if margin weighting is switched on at all.  A grid has to
either enumerate meaningless combinations or leave the parameters out.  Optuna's
define-by-run API samples them only along the branches where they apply, so a
trial that picks the exponential kernel never wastes a draw on a window width.

Optional: ``pip install "bscores[tune]"``.

    from bscores.search import optuna_search

    study = optuna_search(
        home, away, outcome, times,
        validation_start="2019-01-01", validation_end="2023-01-01",
        n_trials=400, n_startup_trials=80,
    )
    study.best_params

The same discipline as :mod:`bscores.tuning` applies, and matters more here
because a smarter search overfits a validation window faster than a dumb one:
tune on the validation window, then report once on a test window neither the
search nor you have looked at.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._time import as_days
from .typing import Names

__all__ = [
    "optuna_search",
    "optuna_search_elo",
    "suggest_elo",
    "SearchResult",
    "suggest_config",
    "config_to_model",
    "build_tuned",
]

#: Metrics where a larger number is better.
_MAXIMISE = frozenset({"accuracy"})


@dataclass
class SearchResult:
    """Outcome of an Optuna study, in the shape :class:`TuningResult` uses."""

    rows: list[dict[str, Any]]
    """One dict per completed trial: its parameters plus every metric."""

    metric: str
    """The metric that decided the ranking."""

    validation: tuple[Any, Any]
    """``(start, end)`` of the window the scores were computed on."""

    best_params: dict[str, Any] = field(default_factory=dict)
    """The winning trial's parameters."""

    best_score: float = float("nan")
    """The winning value of :attr:`metric`."""

    study: Any = field(default=None, repr=False)
    """The underlying :class:`optuna.Study`, for its own plotting and analysis."""

    def top(self, n: int = 10) -> list[dict[str, Any]]:
        """The ``n`` best trials."""
        return self.rows[:n]

    def importances(self, evaluator: Any = None) -> dict[str, float]:
        """Parameter importances, largest first.

        Answers the same question as
        :meth:`~bscores.tuning.TuningResult.sensitivity` but accounts for
        interactions, and copes with parameters that only exist on some
        branches of the search space.

        Defaults to Optuna's PED-ANOVA evaluator, which is pure Python.  The
        fANOVA evaluator is often quoted but pulls in scikit-learn; pass it
        explicitly if you have it and want it.
        """
        import optuna

        if evaluator is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                evaluator = optuna.importance.PedAnovaImportanceEvaluator()
        return dict(optuna.importance.get_param_importances(self.study, evaluator=evaluator))

    def to_frame(self):  # pragma: no cover - thin pandas adapter
        """Return the trials as a ``pandas.DataFrame``."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.best_params.items())
        return (
            f"SearchResult({len(self.rows)} trials, "
            f"best {self.metric}={self.best_score:.4f} at {params})"
        )


def suggest_config(trial: Any, *, margins: Any = None, finals: Any = None) -> dict[str, Any]:
    """Sample one configuration from the conditional search space.

    Exposed so you can extend or restrict the space without reimplementing the
    objective — pass your own version as ``space`` to :func:`optuna_search`.

    Parameters
    ----------
    trial
        The Optuna trial to sample from.
    margins
        Per-match margins.  When given, margin-of-victory weighting becomes part
        of the space, along with its scheme, scale and cap.
    finals
        Per-match boolean flags for important matches.  When given, an
        importance multiplier joins the space.

    Returns
    -------
    dict
        Keys are a mix of :class:`~bscores.BScoreModel` arguments, calibration
        settings, and the weighting choices, ready for :func:`config_to_model`.
    """
    config: dict[str, Any] = {
        # Memory.  Log scale because the interesting range spans two orders of
        # magnitude and the low end is where the resolution is needed.
        "alpha": trial.suggest_float("alpha", 7.0, 2000.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["hyperbolic", "exponential", "window"]),
        # Calibration.
        "transform": trial.suggest_categorical("transform", ["identity", "sqrt", "log"]),
        "symmetric": trial.suggest_categorical("symmetric", [False, True]),
        "regularization": trial.suggest_float("regularization", 1e-4, 1.0, log=True),
        "draw_weight": trial.suggest_float("draw_weight", 0.0, 1.0),
    }

    # Conditional: a window kernel needs a width, the others do not.
    if config["kernel"] == "window":
        config["window_width"] = trial.suggest_float("window_width", 60.0, 2000.0, log=True)
    # Conditional: truncating the tail only makes sense for a heavy-tailed one.
    elif config["kernel"] == "hyperbolic" and trial.suggest_categorical(
        "truncate", [False, True]
    ):
        config["max_age"] = trial.suggest_float("max_age", 180.0, 3650.0, log=True)

    if margins is not None and trial.suggest_categorical("use_margin", [False, True]):
        config["margin_scheme"] = trial.suggest_categorical(
            "margin_scheme", ["linear", "sqrt", "log"]
        )
        config["margin_scale"] = trial.suggest_float("margin_scale", 6.0, 72.0, log=True)
        config["margin_cap"] = trial.suggest_float("margin_cap", 1.5, 6.0)

    if finals is not None and trial.suggest_categorical("use_importance", [False, True]):
        config["importance"] = trial.suggest_float("importance", 1.0, 4.0)

    return config


def config_to_model(config: dict[str, Any]) -> Any:
    """Build a :class:`~bscores.BScoreModel` from a sampled configuration."""
    from .decay import Exponential, Hyperbolic, Window
    from .models import BScoreModel

    alpha = float(config["alpha"])
    name = config.get("kernel", "hyperbolic")
    if name == "window":
        kernel: Any = Window(float(config["window_width"]), Exponential(alpha))
    elif name == "exponential":
        kernel = Exponential(alpha)
    else:
        kernel = Hyperbolic(alpha)

    return BScoreModel(
        kernel=kernel,
        draw_weight=float(config.get("draw_weight", 0.5)),
        regularization=float(config.get("regularization", 0.0)),
        max_age=config.get("max_age"),
    )


def _arc_weights(config: dict[str, Any], margins: Any, finals: Any) -> np.ndarray | None:
    """Combine the weighting choices a trial made into one arc-weight vector."""
    from .weights import importance_weight, margin_weight

    weights: np.ndarray | None = None
    if "margin_scheme" in config:
        weights = margin_weight(
            margins,
            scheme=config["margin_scheme"],
            scale=float(config["margin_scale"]),
            cap=float(config["margin_cap"]),
        )
    if "importance" in config:
        boost = importance_weight(finals, weight=float(config["importance"]))
        weights = boost if weights is None else weights * boost
    return weights


def optuna_search(
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    validation_start: Any,
    validation_end: Any = None,
    margins: Any = None,
    finals: Any = None,
    n_trials: int = 300,
    n_startup_trials: int = 60,
    metric: str = "log_loss",
    refit_every: int = 300,
    seed: int | None = 0,
    space: Callable[[Any], dict[str, Any]] | None = None,
    timeout: float | None = None,
    show_progress: bool = False,
) -> SearchResult:
    """Search hyperparameters with Optuna's TPE sampler.

    Parameters
    ----------
    home, away, outcome, times
        The full fixture list.  Sorted chronologically internally.
    validation_start, validation_end
        The window trials are scored on.  Everything before ``validation_start``
        is warm-up; everything after ``validation_end`` is held back, and the
        search never sees it.
    margins, finals
        Optional per-match margins and importance flags.  Supplying them adds
        the corresponding weighting branches to the search space.
    n_trials
        Configurations to evaluate.
    n_startup_trials
        Random trials before the TPE model takes over.  Too few and the sampler
        commits to whichever corner it stumbled into first; a rule of thumb is
        15-25% of ``n_trials``, and at least a few times the number of
        parameters.
    metric
        ``"log_loss"`` (default), ``"brier_score"``, ``"accuracy"`` or
        ``"classification_error"``.
    refit_every
        Matches forecast between calibration refits.
    seed
        Sampler seed.  Fixed by default so a search is reproducible; pass
        ``None`` for a fresh draw each run.
    space
        Override the sampling function.  Defaults to :func:`suggest_config`,
        bound to ``margins`` and ``finals``.
    timeout
        Optional wall-clock budget in seconds, honoured alongside ``n_trials``.
    show_progress
        Display Optuna's progress bar.

    Returns
    -------
    SearchResult
        Trials ranked best first, with the winning parameters and the study.

    Notes
    -----
    Every trial is scored on out-of-sample forecasts produced by
    :func:`~bscores.backtest.rolling_forecast`, so the causal guarantees hold
    inside the search exactly as they do outside it.  What the search cannot
    protect you from is selection on the validation window itself: the more
    trials you run, the more the best validation score flatters itself, which is
    precisely why the test window has to stay untouched until the end.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "optuna_search needs optuna; install it with: pip install 'bscores[tune]'"
        ) from exc

    from .backtest import rolling_forecast
    from .metrics import evaluate

    home_names = np.asarray(list(home), dtype=object)
    away_names = np.asarray(list(away), dtype=object)
    results: np.ndarray = np.asarray(outcome, dtype=np.float64).ravel()
    stamps: np.ndarray = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    if not (home_names.size == away_names.size == results.size == stamps.size):
        raise ValueError("home, away, outcome and times must be the same length")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_startup_trials < 1:
        raise ValueError("n_startup_trials must be at least 1")

    order = np.argsort(stamps, kind="stable")
    home_names, away_names = home_names[order], away_names[order]
    results, stamps = results[order], stamps[order]
    margin_values = None if margins is None else np.asarray(margins, dtype=np.float64)[order]
    final_values = None if finals is None else np.asarray(finals)[order]

    end = None if validation_end is None else float(as_days(validation_end))
    sample = space or (
        lambda trial: suggest_config(trial, margins=margin_values, finals=final_values)
    )

    def objective(trial: Any) -> float:
        config = sample(trial)
        forecast = rolling_forecast(
            home_names,
            away_names,
            results,
            stamps,
            model=config_to_model(config),
            initial_train=validation_start,
            refit_every=refit_every,
            weights=_arc_weights(config, margin_values, final_values),
            symmetric=bool(config.get("symmetric", False)),
            transform=config.get("transform", "identity"),
            keep_model=False,
        )
        if end is not None:
            forecast = forecast.between(None, end)
        scores = evaluate(forecast.outcome, forecast.probability)
        for name, value in scores.items():
            trial.set_user_attr(name, float(value))
        for name, value in config.items():
            trial.set_user_attr(f"config_{name}", value)
        return float(scores[metric])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        # multivariate/group are the settings that make TPE model interactions
        # and respect the conditional branches; both are flagged experimental
        # upstream, and the warning is not the caller's to act on.
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=True,   # model parameter interactions, not each in isolation
            group=True,          # ... respecting which branch each trial took
        )
    study = optuna.create_study(
        direction="maximize" if metric in _MAXIMISE else "minimize", sampler=sampler
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress
    )

    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {
            key[len("config_") :]: value
            for key, value in trial.user_attrs.items()
            if key.startswith("config_")
        }
        row.update(
            {k: v for k, v in trial.user_attrs.items() if not k.startswith("config_")}
        )
        row["trial"] = trial.number
        rows.append(row)
    rows.sort(key=lambda r: r[metric], reverse=metric in _MAXIMISE)

    best = study.best_trial
    assert best.value is not None
    best_config = {
        key[len("config_") :]: value
        for key, value in best.user_attrs.items()
        if key.startswith("config_")
    }
    return SearchResult(
        rows=rows,
        metric=metric,
        validation=(validation_start, validation_end),
        best_params=best_config,
        best_score=float(best.value),
        study=study,
    )


def build_tuned(
    config: dict[str, Any],
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    margins: Any = None,
    finals: Any = None,
) -> Any:
    """Fit a model on the whole fixture list using a searched configuration.

    Convenience for taking :attr:`SearchResult.best_params` straight to a
    usable model.
    """
    model = config_to_model(config)
    return model.fit(
        home,
        away,
        outcome,
        times,
        weights=_arc_weights(
            config,
            None if margins is None else np.asarray(margins, dtype=np.float64),
            finals,
        ),
        symmetric=bool(config.get("symmetric", False)),
        transform=config.get("transform", "identity"),
        model_draws=True,
    )


def suggest_elo(trial: Any) -> dict[str, Any]:
    """Sample one Elo configuration from a conditional space.

    The K schedule is the branch: a fixed ``k`` and Kovalchik's
    experience-decayed schedule take different parameters, and sampling both on
    every trial would waste most of the budget.
    """
    config: dict[str, Any] = {
        "home_advantage": trial.suggest_float("home_advantage", 0.0, 120.0),
        "spread": trial.suggest_float("spread", 200.0, 800.0, log=True),
    }
    if trial.suggest_categorical("schedule", ["fixed", "kovalchik"]) == "fixed":
        config["k"] = trial.suggest_float("k", 4.0, 80.0, log=True)
    else:
        config["k_scale"] = trial.suggest_float("k_scale", 50.0, 800.0, log=True)
        config["k_shape"] = trial.suggest_float("k_shape", 1.0, 25.0, log=True)
        config["k_power"] = trial.suggest_float("k_power", 0.1, 0.9)
    return config


def optuna_search_elo(
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    validation_start: Any,
    validation_end: Any = None,
    n_trials: int = 300,
    n_startup_trials: int = 60,
    metric: str = "log_loss",
    seed: int | None = 0,
    space: Callable[[Any], dict[str, Any]] | None = None,
    timeout: float | None = None,
    show_progress: bool = False,
) -> SearchResult:
    """Search Elo's hyperparameters with the same sampler and budget.

    :func:`~bscores.baselines.tune_elo` grid-searches a handful of values, which
    is fine against :func:`~bscores.tuning.grid_search`.  Against
    :func:`optuna_search` it is not: a comparison is only informative when both
    sides get the same search effort, so a B-score model tuned over hundreds of
    TPE trials should be measured against an Elo tuned the same way.

    Arguments mirror :func:`optuna_search`.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "optuna_search_elo needs optuna; install it with: pip install 'bscores[tune]'"
        ) from exc

    from .backtest import _resolve_start
    from .baselines import Elo
    from .metrics import evaluate

    home_names = np.asarray(list(home), dtype=object)
    away_names = np.asarray(list(away), dtype=object)
    results: np.ndarray = np.asarray(outcome, dtype=np.float64).ravel()
    stamps: np.ndarray = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    if not (home_names.size == away_names.size == results.size == stamps.size):
        raise ValueError("home, away, outcome and times must be the same length")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    order = np.argsort(stamps, kind="stable")
    home_names, away_names = home_names[order], away_names[order]
    results, stamps = results[order], stamps[order]

    start = _resolve_start(stamps, validation_start)
    stop = stamps.size if validation_end is None else _resolve_start(stamps, validation_end)
    if not start < stop:
        raise ValueError("validation_end must fall after validation_start")

    sample = space or suggest_elo

    def objective(trial: Any) -> float:
        config = sample(trial)
        probability = Elo(**config).run(home_names, away_names, results)
        scores = evaluate(results[start:stop], probability[start:stop])
        for name, value in scores.items():
            trial.set_user_attr(name, float(value))
        for name, value in config.items():
            trial.set_user_attr(f"config_{name}", value)
        return float(scores[metric])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials, multivariate=True, group=True
        )
    study = optuna.create_study(
        direction="maximize" if metric in _MAXIMISE else "minimize", sampler=sampler
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress
    )

    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {
            key[len("config_") :]: value
            for key, value in trial.user_attrs.items()
            if key.startswith("config_")
        }
        row.update({k: v for k, v in trial.user_attrs.items() if not k.startswith("config_")})
        row["trial"] = trial.number
        rows.append(row)
    rows.sort(key=lambda r: r[metric], reverse=metric in _MAXIMISE)

    best = study.best_trial
    assert best.value is not None
    return SearchResult(
        rows=rows,
        metric=metric,
        validation=(validation_start, validation_end),
        best_params={
            key[len("config_") :]: value
            for key, value in best.user_attrs.items()
            if key.startswith("config_")
        },
        best_score=float(best.value),
        study=study,
    )
