"""Out-of-sample evaluation.

Reproduces the paper's forecasting protocol (Sect. 3):

1. fit the probability model on every match up to a cut-off;
2. forecast the next ``refit_every`` matches;
3. fold those matches into the training set, refit, repeat.

The B-scores themselves need no such loop — they are a causal function of the
results that precede each match, so they are computed once for the whole fixture
list.  Only the logit coefficients are re-estimated, which is what makes the
back-test cheap enough to sweep over ``alpha``.

For choosing hyperparameters, reach for :mod:`bscores.tuning` rather than
running this repeatedly and keeping the best: picking a setting on the same
matches you then report on inflates the result by however hard you searched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._time import as_days
from .calibration import LogitCalibrator, Transform
from .models import BScoreModel
from .typing import Names, Numbers

__all__ = ["BacktestResult", "rolling_forecast", "walk_forward"]


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

    def between(self, start: Any = None, end: Any = None) -> BacktestResult:
        """Restrict the forecasts to a time window, ``[start, end)``.

        Either bound may be ``None`` for open-ended.  Used to score a validation
        period separately from a test period without re-running the sweep.
        """
        keep = np.ones(self.times.size, dtype=bool)
        if start is not None:
            keep &= self.times >= float(as_days(start))
        if end is not None:
            keep &= self.times < float(as_days(end))
        if not keep.any():
            raise ValueError(f"no forecasts fall in [{start}, {end})")
        return BacktestResult(
            times=self.times[keep],
            home=self.home[keep],
            away=self.away[keep],
            outcome=self.outcome[keep],
            probability=self.probability[keep],
            home_score=self.home_score[keep],
            away_score=self.away_score[keep],
            train_size=self.train_size[keep],
            coefficients=self.coefficients,
            refit_at=self.refit_at,
            model=self.model,
        )

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
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    model: BScoreModel | None = None,
    alpha: float = 365.0,
    initial_train: Any = 0.5,
    refit_every: int = 300,
    weights: Numbers | None = None,
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

    calibrator = LogitCalibrator(
        fit_intercept=fit_intercept, symmetric=symmetric, transform=transform, ridge=ridge
    )
    probability, train_size, coefficients, refit_at = walk_forward(
        home_scores, away_scores, results, start, refit_every=refit_every, calibrator=calibrator
    )

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
        coefficients=coefficients,
        refit_at=refit_at,
        model=engine if keep_model else None,
    )


def walk_forward(
    home_score: np.ndarray,
    away_score: np.ndarray,
    outcome: np.ndarray,
    start: int,
    *,
    refit_every: int = 300,
    calibrator: LogitCalibrator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the expanding-window refit loop over pre-computed ratings.

    Split out from :func:`rolling_forecast` because the ratings are the
    expensive half and the refit loop is the cheap one: a sweep over calibration
    settings can reuse one causal rating pass for every variant it tries.

    Parameters
    ----------
    home_score, away_score
        Causal ratings for each fixture, from
        :meth:`~bscores.BScoreModel.match_scores`.
    outcome
        Results on the ``{0, 0.5, 1}`` scale, same order.
    start
        Index of the first match to forecast; everything before it is the
        initial training window.
    refit_every
        Matches forecast between refits.
    calibrator
        Fitted in place, so it holds the final coefficients on return.  A
        default :class:`~bscores.calibration.LogitCalibrator` is used when
        omitted.

    Returns
    -------
    tuple
        ``(probability, train_size, coefficients, refit_at)``.
    """
    n = outcome.size
    if not 0 < start < n:
        raise ValueError(f"start must lie in (0, {n}), got {start}")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    if calibrator is None:
        calibrator = LogitCalibrator()

    probability = np.empty(n - start, dtype=np.float64)
    train_size = np.empty(n - start, dtype=np.int64)
    coefficients: list[np.ndarray] = []
    refit_at: list[int] = []

    cursor = start
    while cursor < n:
        stop = min(cursor + refit_every, n)
        calibrator.fit(home_score[:cursor], away_score[:cursor], outcome[:cursor])
        assert calibrator.beta_ is not None
        coefficients.append(calibrator.beta_.copy())
        refit_at.append(cursor - start)
        probability[cursor - start : stop - start] = calibrator.predict_proba(
            home_score[cursor:stop], away_score[cursor:stop]
        )
        train_size[cursor - start : stop - start] = cursor
        cursor = stop

    return probability, train_size, np.array(coefficients), np.array(refit_at, dtype=np.int64)
