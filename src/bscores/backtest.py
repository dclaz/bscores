"""Out-of-sample evaluation.

Reproduces the paper's forecasting protocol (Sect. 3):

1. fit the probability model on every match up to a cut-off;
2. forecast the next ``refit_every`` matches;
3. fold those matches into the training set, refit, repeat.

The B-scores themselves need no such loop — they are a causal function of the
results that precede each match, so they are computed once for the whole fixture
list.  Only the logit coefficients are re-estimated, which is what makes the
back-test cheap enough to sweep over ``alpha``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._time import as_days
from .calibration import LogitCalibrator, Transform
from .models import BScoreModel

__all__ = ["BacktestResult", "rolling_forecast", "sweep_alpha"]


@dataclass
class BacktestResult:
    """Out-of-sample forecasts and the state that produced them."""

    times: np.ndarray
    """Match times, days since the epoch."""

    home: np.ndarray
    """Home competitor names."""

    away: np.ndarray
    """Away competitor names."""

    outcome: np.ndarray
    """Observed results on the ``{0, 0.5, 1}`` scale."""

    probability: np.ndarray
    """Forecast home-win probabilities."""

    home_score: np.ndarray
    """Home side's B-score as of just before the match."""

    away_score: np.ndarray
    """Away side's B-score as of just before the match."""

    train_size: np.ndarray
    """Number of matches the coefficients behind each forecast were fitted on."""

    coefficients: np.ndarray
    """One row of logit coefficients per refit."""

    refit_at: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    """Index into the test arrays where each refit took effect."""

    model: BScoreModel | None = None
    """The fitted model, for inspecting ratings after the fact."""

    def metrics(self, *, tol: float | None = None) -> dict[str, float]:
        """Log-loss, Brier score, accuracy and error rate over the test set."""
        from .metrics import DEFAULT_TOL, evaluate

        return evaluate(self.outcome, self.probability, tol=DEFAULT_TOL if tol is None else tol)

    def losses(self, kind: str = "log_loss") -> np.ndarray:
        """Per-match losses, for a Diebold-Mariano comparison.

        Parameters
        ----------
        kind
            ``"log_loss"`` or ``"brier_score"``.
        """
        from .metrics import DEFAULT_TOL

        if kind == "brier_score":
            return (self.outcome - self.probability) ** 2
        if kind == "log_loss":
            p = np.clip(self.probability, DEFAULT_TOL, 1.0 - DEFAULT_TOL)
            return -(self.outcome * np.log(p) + (1.0 - self.outcome) * np.log(1.0 - p))
        raise ValueError(f"unknown loss {kind!r}")

    def to_frame(self):  # pragma: no cover - thin pandas adapter
        """Return the forecasts as a ``pandas.DataFrame``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.times * 86_400_000_000, unit="us"),
                "home_team": self.home,
                "away_team": self.away,
                "outcome": self.outcome,
                "probability": self.probability,
                "home_score": self.home_score,
                "away_score": self.away_score,
                "train_size": self.train_size,
            }
        )

    def __len__(self) -> int:
        return int(self.probability.size)

    def __repr__(self) -> str:
        scores = self.metrics()
        return (
            f"BacktestResult(n={len(self)}, log_loss={scores['log_loss']:.4f}, "
            f"brier={scores['brier_score']:.4f}, accuracy={scores['accuracy']:.4f})"
        )


def _resolve_start(times: np.ndarray, initial_train: Any) -> int:
    """Index of the first forecast match."""
    if initial_train is None:
        return max(1, times.size // 2)
    if isinstance(initial_train, (int, np.integer)) and not isinstance(initial_train, bool):
        if initial_train < 0:
            raise ValueError("initial_train must be non-negative")
        return int(initial_train)
    if isinstance(initial_train, float) and 0.0 < initial_train < 1.0:
        return int(round(initial_train * times.size))
    cutoff = as_days(initial_train)
    return int(np.searchsorted(times, float(cutoff), side="left"))


def rolling_forecast(
    home: Sequence[str],
    away: Sequence[str],
    outcome: Any,
    times: Any,
    *,
    model: BScoreModel | None = None,
    alpha: float = 365.0,
    initial_train: Any = 0.5,
    refit_every: int = 300,
    weights: Sequence[float] | None = None,
    symmetric: bool = False,
    transform: Transform = "identity",
    ridge: float = 1e-6,
    fit_intercept: bool = True,
    keep_model: bool = True,
) -> BacktestResult:
    """Expanding-window out-of-sample forecast.

    Parameters
    ----------
    home, away, outcome, times
        The full fixture list.  It is sorted chronologically internally, so the
        caller's order does not matter.
    model
        A configured :class:`~bscores.BScoreModel`.  Built from ``alpha`` when
        omitted.  Any results already in it are kept, so a model warmed up on
        earlier seasons can be reused.
    alpha
        Memory parameter, in days, for the default model.
    initial_train
        Where the test period starts: an ``int`` count of matches, a ``float``
        fraction of the fixture list, or a date/timestamp.
    refit_every
        Matches forecast between refits.  The paper uses 300.
    symmetric, transform, ridge, fit_intercept
        Forwarded to :class:`~bscores.calibration.LogitCalibrator`.
    keep_model
        Attach the fitted model to the result.

    Returns
    -------
    BacktestResult
        Forecasts for matches from ``initial_train`` onwards.  Every probability
        is produced by coefficients fitted only on earlier matches, and by
        B-scores built only from earlier results.
    """
    home_names = np.asarray(list(home), dtype=object)
    away_names = np.asarray(list(away), dtype=object)
    results = np.asarray(outcome, dtype=np.float64).ravel()
    stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    n = home_names.size
    if not (away_names.size == results.size == stamps.size == n):
        raise ValueError(
            "home, away, outcome and times must be the same length, got "
            f"{n}, {away_names.size}, {results.size}, {stamps.size}"
        )
    if n == 0:
        raise ValueError("cannot back-test an empty fixture list")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    order = np.argsort(stamps, kind="stable")
    home_names, away_names = home_names[order], away_names[order]
    results, stamps = results[order], stamps[order]
    share = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()[order]

    engine = model if model is not None else BScoreModel(alpha=alpha)
    engine.add_matches(home_names, away_names, results, stamps, weights=share)
    home_scores, away_scores = engine.match_scores(
        home_names, away_names, stamps, inclusive=False
    )

    start = _resolve_start(stamps, initial_train)
    if start < 1:
        raise ValueError("initial_train must leave at least one match to train on")
    if start >= n:
        raise ValueError(
            f"initial_train leaves no matches to forecast ({start} of {n} used for training)"
        )

    probability = np.empty(n - start, dtype=np.float64)
    train_size = np.empty(n - start, dtype=np.int64)
    coefficients: list[np.ndarray] = []
    refit_at: list[int] = []

    calibrator = LogitCalibrator(
        fit_intercept=fit_intercept, symmetric=symmetric, transform=transform, ridge=ridge
    )
    cursor = start
    while cursor < n:
        stop = min(cursor + refit_every, n)
        calibrator.fit(home_scores[:cursor], away_scores[:cursor], results[:cursor])
        assert calibrator.beta_ is not None
        coefficients.append(calibrator.beta_.copy())
        refit_at.append(cursor - start)
        probability[cursor - start : stop - start] = calibrator.predict_proba(
            home_scores[cursor:stop], away_scores[cursor:stop]
        )
        train_size[cursor - start : stop - start] = cursor
        cursor = stop

    engine.calibrator = calibrator
    return BacktestResult(
        times=stamps[start:],
        home=home_names[start:],
        away=away_names[start:],
        outcome=results[start:],
        probability=probability,
        home_score=home_scores[start:],
        away_score=away_scores[start:],
        train_size=train_size,
        coefficients=np.array(coefficients),
        refit_at=np.array(refit_at, dtype=np.int64),
        model=engine if keep_model else None,
    )


def sweep_alpha(
    home: Sequence[str],
    away: Sequence[str],
    outcome: Any,
    times: Any,
    alphas: Sequence[float],
    *,
    metric: str = "log_loss",
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Back-test across a grid of memory parameters.

    The paper flags the choice of ``alpha`` as open ("worthy of investigation"),
    and this is the cheap way to answer it for a given competition: every
    ``alpha`` is a fresh causal sweep, evaluated strictly out of sample.

    Returns
    -------
    list[dict]
        One row per ``alpha``, sorted best-first on ``metric``, each holding the
        full metric set.
    """
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        result = rolling_forecast(
            home, away, outcome, times, alpha=float(alpha), keep_model=False, **kwargs
        )
        row = {"alpha": float(alpha)}
        row.update(result.metrics())
        rows.append(row)
    if metric not in rows[0]:
        raise ValueError(f"unknown metric {metric!r}")
    reverse = metric == "accuracy"
    return sorted(rows, key=lambda r: r[metric], reverse=reverse)
