"""Matplotlib figures for ratings, forecasts and searches.

Optional: ``pip install bscores[plot]``.  Nothing else in the package imports
this module, so the core stays numpy-only.

Every function takes an optional ``ax`` and returns the axes it drew on, so
figures compose:

>>> import matplotlib.pyplot as plt                    # doctest: +SKIP
>>> fig, (left, right) = plt.subplots(1, 2)            # doctest: +SKIP
>>> plot_ratings(history, ax=left)                     # doctest: +SKIP
>>> plot_calibration(outcome, probability, ax=right)   # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .diagnostics import calibration_curve
from .models import BScoreModel, RatingHistory

__all__ = [
    "plot_ratings",
    "plot_calibration",
    "plot_tuning",
    "plot_network",
    "plot_backtest",
    "plot_decay",
]


def _axes(ax: Any, **kwargs: Any) -> Any:
    """Return ``ax``, creating a figure if none was supplied."""
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "plotting needs matplotlib; install it with: pip install 'bscores[plot]'"
        ) from exc
    _, ax = plt.subplots(**kwargs)
    return ax


def _dates(times: np.ndarray) -> np.ndarray:
    """Float days since the epoch as datetime64, for a readable x axis."""
    return (times * 86_400_000_000).astype("datetime64[us]")


def plot_ratings(
    history: RatingHistory,
    *,
    competitors: Sequence[str] | None = None,
    top: int | None = 6,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Rating trajectories over time.

    Parameters
    ----------
    history
        From :meth:`~bscores.BScoreModel.score_history`.
    competitors
        Which series to draw.  Defaults to whoever is highest rated at the end.
    top
        How many to keep when ``competitors`` is not given.
    """
    ax = _axes(ax, figsize=(10, 5))
    if competitors is not None:
        names = list(competitors)
    else:
        leaders = np.argsort(-history.scores[-1])[: top or len(history.names)]
        names = [history.names[i] for i in leaders]
    dates = _dates(history.times)
    for name in names:
        ax.plot(dates, history.of(name), label=name, **kwargs)
    ax.set_xlabel("date")
    ax.set_ylabel("B-score")
    ax.set_title("Ratings over time")
    ax.legend(loc="upper left", fontsize="small", ncols=2)
    ax.margins(x=0.01)
    return ax


def plot_calibration(
    outcome: Any,
    prediction: Any,
    *,
    bins: int = 10,
    strategy: str = "quantile",
    label: str | None = None,
    ax: Any = None,
) -> Any:
    """Reliability diagram: forecast probability against observed frequency.

    Points on the diagonal mean the probabilities can be taken at face value.
    Marker area is proportional to how many matches fell in each bin, so a
    wayward point built on ten matches is visibly less damning than one built
    on two hundred.
    """
    ax = _axes(ax, figsize=(5.5, 5.5))
    curve = calibration_curve(outcome, prediction, bins=bins, strategy=strategy)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.6", label="perfect")
    sizes = 20.0 + 180.0 * curve["count"] / curve["count"].max()
    ax.plot(curve["predicted"], curve["observed"], linewidth=1)
    ax.scatter(curve["predicted"], curve["observed"], s=sizes, zorder=3, label=label)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if label:
        ax.legend(loc="upper left", fontsize="small")
    return ax


def plot_tuning(result: Any, key: str = "alpha", *, ax: Any = None, logx: bool = True) -> Any:
    """Best achievable score against one hyperparameter.

    Takes a :class:`~bscores.tuning.TuningResult`.  The shape of the curve is
    the useful part: a sharp minimum means the parameter is worth tuning, a flat
    one means it is not.
    """
    ax = _axes(ax, figsize=(7, 4.5))
    profile = sorted(result.sensitivity(key), key=lambda kv: (kv[0] is None, kv[0]))
    values = [v for v, _ in profile]
    scores = [s for _, s in profile]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    if numeric:
        ax.plot(values, scores, marker="o")
        if logx and min(values) > 0:
            ax.set_xscale("log")
    else:
        ax.bar([str(v) for v in values], scores)
    best_value, best_score = min(profile, key=lambda kv: kv[1])
    ax.axhline(best_score, linestyle="--", linewidth=1, color="0.6")
    ax.set_xlabel(key)
    ax.set_ylabel(f"best {result.metric}")
    ax.set_title(f"{result.metric} against {key} (best: {best_value})")
    return ax


def plot_decay(kernels: Sequence[Any], *, days: float = 1460.0, ax: Any = None) -> Any:
    """Weight against age for one or more decay kernels.

    Worth looking at before choosing one: the hyperbolic kernel's tail is much
    fatter than it looks in the formula, which is why an old result keeps
    influencing a rating long after it stops being informative.
    """
    ax = _axes(ax, figsize=(7, 4.5))
    ages = np.linspace(0.0, days, 400)
    for kernel in kernels:
        ax.plot(ages, kernel(ages), label=repr(kernel))
    ax.set_xlabel("age of result (days)")
    ax.set_ylabel("weight")
    ax.set_title("Decay kernels")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize="small")
    return ax


def plot_network(
    model: BScoreModel,
    *,
    at: Any = None,
    min_weight: float = 0.0,
    ax: Any = None,
) -> Any:
    """The loss network on a circular layout, competitors ordered by rating.

    Arcs run from loser to winner, with width and opacity following the decayed
    weight, and node size following the B-score.  Readable up to a few dozen
    competitors; beyond that use :func:`bscores.diagnostics.network_summary`.
    """
    ax = _axes(ax, figsize=(7, 7))
    scores = model.scores(at=at)
    n = scores.size
    if n == 0:
        raise ValueError("the model has no competitors to draw")

    order = np.argsort(-scores)
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    slot = np.empty(n, dtype=np.int64)
    slot[order] = np.arange(n)
    x, y = np.cos(angle[slot]), np.sin(angle[slot])

    matrix = model.network.matrix(at, inclusive=True)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    peak = dense.max() if dense.size else 0.0
    if peak > 0:
        for loser, winner in zip(*np.nonzero(dense > min_weight)):
            strength = dense[loser, winner] / peak
            ax.annotate(
                "",
                xy=(x[winner], y[winner]),
                xytext=(x[loser], y[loser]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": 0.3 + 2.2 * strength,
                    "alpha": 0.08 + 0.5 * strength,
                    "color": "C0",
                    "shrinkA": 8,
                    "shrinkB": 8,
                    "connectionstyle": "arc3,rad=0.12",
                },
            )
    ax.scatter(x, y, s=60 + 900 * scores / max(scores.max(), 1e-12), zorder=3, color="C1")
    for i, name in enumerate(model.players):
        ax.annotate(
            name,
            (x[i] * 1.14, y[i] * 1.14),
            ha="center",
            va="center",
            fontsize="small",
        )
    ax.set_title("Loss network (arcs point from loser to winner)")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_backtest(result: Any, *, baseline: Any = None, ax: Any = None) -> Any:
    """Cumulative log-loss advantage over a baseline, match by match.

    A rising line means the model is pulling ahead.  Flat stretches are where it
    is no better than the baseline, and dips are where it is worse — far more
    informative than a single average, because it shows *when* the edge came
    from.
    """
    ax = _axes(ax, figsize=(10, 4.5))
    model_loss = result.losses("log_loss")
    if baseline is None:
        rate = float(np.mean(result.outcome))
        reference = np.full(result.outcome.size, rate)
        label = f"base rate ({rate:.3f})"
    else:
        reference = np.asarray(baseline, dtype=np.float64).ravel()
        label = "baseline"
    clipped = np.clip(reference, 1e-15, 1 - 1e-15)
    baseline_loss = -(
        result.outcome * np.log(clipped) + (1 - result.outcome) * np.log(1 - clipped)
    )

    advantage = np.cumsum(baseline_loss - model_loss)
    ax.plot(_dates(result.times), advantage)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="0.6")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-loss saved")
    ax.set_title(f"Advantage over {label}")
    ax.margins(x=0.01)
    return ax
