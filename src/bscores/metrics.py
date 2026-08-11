"""Forecast evaluation.

The paper scores predictions with the Brier score and the log-loss and compares
models with a Diebold-Mariano test; all three live here.

Every function takes ``outcome`` first and ``prediction`` second, and both may
carry fractional values — a draw scored ``0.5`` is a legitimate outcome.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_TOL",
    "log_loss",
    "brier_score",
    "accuracy",
    "classification_error",
    "evaluate",
    "diebold_mariano",
]

#: The default clip: the largest float64 eps below 1, i.e. ``2**-53``.
DEFAULT_TOL = float(np.finfo(np.float64).epsneg)


def _pair(outcome: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(outcome, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(
            f"outcome and prediction must be the same length, got {y.size} and {p.size}"
        )
    if y.size == 0:
        raise ValueError("cannot score an empty set of predictions")
    return y, p


def log_loss(outcome: Any, prediction: Any, tol: float = DEFAULT_TOL) -> float:
    """Mean binary cross-entropy.

    Parameters
    ----------
    outcome
        Observed results in ``[0, 1]``.
    prediction
        Forecast probabilities.
    tol
        Predictions are clipped to ``[tol, 1 - tol]``.  Raising it caps how much
        a single confident miss can dominate the average, which is the usual
        reason to touch it.
    """
    y, p = _pair(outcome, prediction)
    if not 0.0 <= tol < 0.5:
        raise ValueError(f"tol must lie in [0, 0.5), got {tol}")
    clipped = np.clip(p, tol, 1.0 - tol)
    return float(-np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)) / y.size)


def brier_score(outcome: Any, prediction: Any) -> float:
    """Mean squared error of the forecast probabilities."""
    y, p = _pair(outcome, prediction)
    return float(np.mean((y - p) ** 2))


def accuracy(outcome: Any, prediction: Any) -> float:
    """Fraction of matches whose rounded forecast matches the result.

    Rounding is half-to-even, so an exact ``0.5`` forecast is scored as a
    predicted loss rather than a coin toss.
    """
    y, p = _pair(outcome, prediction)
    return float(np.mean(np.round(p) == y))


def classification_error(outcome: Any, prediction: Any) -> float:
    """``1 - accuracy``."""
    return 1.0 - accuracy(outcome, prediction)


def evaluate(outcome: Any, prediction: Any, *, tol: float = DEFAULT_TOL) -> dict[str, float]:
    """All four losses at once, plus the sample size."""
    y, p = _pair(outcome, prediction)
    return {
        "n": float(y.size),
        "log_loss": log_loss(y, p, tol),
        "brier_score": brier_score(y, p),
        "accuracy": accuracy(y, p),
        "classification_error": classification_error(y, p),
    }


def diebold_mariano(
    loss_a: Any,
    loss_b: Any,
    *,
    horizon: int = 1,
    small_sample: bool = True,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive ability.

    Compares two loss series computed on the *same* forecast targets.  A
    negative statistic means model ``a`` lost less, i.e. forecast better — the
    sign convention used in the paper's Tables 2 and 3.

    Parameters
    ----------
    loss_a, loss_b
        Per-observation losses, same length.
    horizon
        Forecast horizon; sets how many autocovariance lags enter the long-run
        variance.  One-step-ahead forecasts (``1``) need no lags.
    small_sample
        Apply the Harvey-Leybourne-Newbold finite-sample correction.

    Returns
    -------
    tuple[float, float]
        ``(statistic, two_sided_p_value)``.  The p-value uses the normal
        approximation, which is what the ``t``-correction is calibrated against
        at these sample sizes.
    """
    a = np.asarray(loss_a, dtype=np.float64).ravel()
    b = np.asarray(loss_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"loss series must be the same length, got {a.size} and {b.size}")
    if a.size < 2:
        raise ValueError("need at least two observations")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    d = a - b
    n = d.size
    mean = float(np.mean(d))
    centred = d - mean

    variance = float(centred @ centred) / n
    for lag in range(1, horizon):
        cov = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * cov
    if variance <= 0.0:
        return 0.0, 1.0

    statistic = mean / math.sqrt(variance / n)
    if small_sample:
        correction = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        statistic *= math.sqrt(max(correction, 0.0))

    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, p_value
