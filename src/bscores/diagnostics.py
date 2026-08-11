"""Tools for interrogating ratings and forecasts.

Everything here answers a question you actually ask of a rating system:

* *Why is this competitor rated where it is?* — :func:`explain_rating`
  decomposes a B-score into the wins that produced it.  The decomposition is
  exact, not an approximation, because the eigenvector equation says a rating
  *is* the weighted sum of the ratings pointing at it.
* *Are the probabilities honest?* — :func:`calibration_curve` and
  :func:`reliability_table` bin forecasts against outcomes.
* *Is the network healthy enough to rate on?* — :func:`network_summary` reports
  density, connectivity and whether a cycle exists yet.
* *How much does the order actually move?* — :func:`rating_churn`.
* *Who has the wood on whom?* — :func:`head_to_head`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .centrality import _has_cycle, bonacich_centrality
from .metrics import _pair
from .models import BScoreModel, RatingHistory

__all__ = [
    "Contribution",
    "explain_rating",
    "calibration_curve",
    "reliability_table",
    "sharpness",
    "network_summary",
    "head_to_head",
    "rating_churn",
    "upset_rate",
]


@dataclass(frozen=True)
class Contribution:
    """One opponent's share of a competitor's rating."""

    opponent: str
    """Who was beaten."""

    weight: float
    """Decayed arc weight — how much, and how recently, they were beaten."""

    opponent_score: float
    """That opponent's own B-score."""

    contribution: float
    """Their share of this competitor's rating, in rating units."""

    share: float
    """The same, as a fraction of the total."""

    def __repr__(self) -> str:
        return (
            f"Contribution(opponent={self.opponent!r}, share={self.share:.1%}, "
            f"opponent_score={self.opponent_score:.4f})"
        )


def explain_rating(
    model: BScoreModel,
    name: str,
    *,
    at: Any = None,
    top: int | None = 10,
) -> list[Contribution]:
    """Break a competitor's B-score into the wins that built it.

    The eigenvector equation is :math:`x_i = \\rho^{-1} \\sum_j W_{ji} x_j`, so
    each competitor ``j`` that ``i`` has beaten contributes exactly
    ``W[j, i] * x_j / rho`` to ``i``'s rating.  Those terms sum to the rating,
    which makes this a decomposition rather than an attribution heuristic.

    Parameters
    ----------
    model
        A model with results in it.
    name
        Whose rating to explain.
    at
        Evaluate at this time; ``None`` means "now".
    top
        Keep only the largest ``top`` contributions, or ``None`` for all.

    Returns
    -------
    list[Contribution]
        Largest share first.

    Notes
    -----
    On a network with no directed cycle the spectral radius is zero and the
    equation above has no ``rho`` to divide by.  The raw weighted terms are
    reported instead, normalised to sum to one; :attr:`Contribution.contribution`
    is then on an arbitrary scale.

    Examples
    --------
    >>> from bscores import BScoreModel
    >>> model = BScoreModel(alpha=365.0)
    >>> for winner, loser, day in [("a", "b", 0), ("b", "c", 7), ("c", "a", 14),
    ...                            ("a", "c", 21), ("b", "a", 28)]:
    ...     _ = model.rate_result(winner, loser, at=float(day))
    >>> [c.opponent for c in explain_rating(model, "a")]
    ['b', 'c']
    """
    if name not in model.index:
        raise KeyError(f"unknown competitor {name!r}")
    node = model.index.get(name)
    scores = model.scores(at=at)
    matrix = model.network.matrix(at, inclusive=True)
    column = np.asarray(
        matrix[:, node].toarray().ravel() if hasattr(matrix, "toarray") else matrix[:, node],
        dtype=np.float64,
    )

    result = bonacich_centrality(matrix, return_info=True, tol=model.tol, shift=model.shift)
    rho = result.eigenvalue
    terms = column * scores
    scale = 1.0 / rho if rho > 0.0 else 1.0

    total = float(terms.sum())
    contributions = [
        Contribution(
            opponent=model.index[other],
            weight=float(column[other]),
            opponent_score=float(scores[other]),
            contribution=float(terms[other] * scale),
            share=float(terms[other] / total) if total > 0.0 else 0.0,
        )
        for other in np.flatnonzero(terms > 0.0)
    ]
    contributions.sort(key=lambda c: -c.share)
    return contributions if top is None else contributions[:top]


def calibration_curve(
    outcome: Any,
    prediction: Any,
    *,
    bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, np.ndarray]:
    """Bin forecasts and compare each bin's mean forecast with what happened.

    A well-calibrated model puts ``observed`` on top of ``predicted``: of the
    matches it called at 70%, about 70% should have been won.

    Parameters
    ----------
    outcome, prediction
        Results and forecast probabilities.
    bins
        Number of bins.
    strategy
        ``"uniform"`` splits ``[0, 1]`` evenly; ``"quantile"`` puts an equal
        number of forecasts in each bin, which is steadier when the forecasts
        cluster near 0.5.

    Returns
    -------
    dict
        ``predicted``, ``observed``, ``count`` and ``edges`` arrays.  Empty bins
        are dropped.
    """
    y, p = _pair(outcome, prediction)
    if bins < 1:
        raise ValueError("bins must be at least 1")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:
            edges = np.array([p.min(), np.nextafter(p.max(), np.inf)])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    counts = np.bincount(index, minlength=edges.size - 1)
    keep = counts > 0
    predicted = np.bincount(index, weights=p, minlength=edges.size - 1)[keep] / counts[keep]
    observed = np.bincount(index, weights=y, minlength=edges.size - 1)[keep] / counts[keep]
    return {
        "predicted": predicted,
        "observed": observed,
        "count": counts[keep],
        "edges": edges,
    }


def reliability_table(
    outcome: Any,
    prediction: Any,
    *,
    bins: int = 10,
    strategy: str = "uniform",
) -> str:
    """A printable version of :func:`calibration_curve`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.1, 0.9, 500)
    >>> y = (rng.random(500) < p).astype(float)
    >>> print(reliability_table(y, p, bins=4))
    predicted  observed  count     gap
        0.175     0.159     82  -0.016
        0.378     0.340    141  -0.037
        0.621     0.642    165  +0.021
        0.823     0.848    112  +0.025
    """
    curve = calibration_curve(outcome, prediction, bins=bins, strategy=strategy)
    lines = [f"{'predicted':>9}  {'observed':>8}  {'count':>5}  {'gap':>6}"]
    for predicted, observed, count in zip(
        curve["predicted"], curve["observed"], curve["count"]
    ):
        lines.append(
            f"{predicted:>9.3f}  {observed:>8.3f}  {int(count):>5}  {observed - predicted:>+6.3f}"
        )
    return "\n".join(lines)


def sharpness(prediction: Any) -> float:
    """How far forecasts stray from a permanent 50/50, on average.

    Calibration alone is easy to fake by always predicting the base rate.
    Sharpness is the other half of the picture: ``mean(|p - 0.5|) * 2``, so 0 is
    a model that never commits and 1 is one that always calls a certainty.
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("cannot measure the sharpness of no forecasts")
    return float(2.0 * np.mean(np.abs(p - 0.5)))


def network_summary(model: BScoreModel, *, at: Any = None) -> dict[str, Any]:
    """Structural health check on the result network.

    The B-score is only well defined once the network contains a directed cycle
    (see :func:`bscores.centrality.bonacich_centrality`), and it is only
    *informative* once the graph is reasonably dense.  This reports both.

    Returns
    -------
    dict
        Node and arc counts, density, whether a cycle exists, the spectral
        radius, how many competitors are unrated, and the solver route taken.
    """
    n = len(model.index)
    matrix = model.network.matrix(at, inclusive=True)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    arcs = int(np.count_nonzero(dense))
    possible = n * (n - 1)
    result = bonacich_centrality(matrix, return_info=True, tol=model.tol, shift=model.shift)
    return {
        "competitors": n,
        "results": model.network.n_events,
        "arcs": arcs,
        "density": float(arcs / possible) if possible else 0.0,
        "has_cycle": bool(_has_cycle(matrix)),
        "spectral_radius": float(result.eigenvalue),
        "unrated": int(np.sum(result.vector <= 0.0)),
        "solver": result.method,
        "total_weight": float(dense.sum()),
    }


def head_to_head(
    home: Sequence[str],
    away: Sequence[str],
    outcome: Any,
    *,
    competitors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Win counts for every pairing in a fixture list.

    Returns
    -------
    dict
        ``names`` plus a ``wins`` matrix where ``wins[i, j]`` counts times ``i``
        beat ``j``, and a ``played`` matrix of meetings.  Draws add 0.5 to each
        side's win count.
    """
    home_names = list(home)
    away_names = list(away)
    results = np.asarray(outcome, dtype=np.float64).ravel()
    if not (len(home_names) == len(away_names) == results.size):
        raise ValueError(
            "home, away and outcome must be the same length, got "
            f"{len(home_names)}, {len(away_names)}, {results.size}"
        )

    names = (
        sorted(set(home_names) | set(away_names)) if competitors is None else list(competitors)
    )
    position = {name: i for i, name in enumerate(names)}
    n = len(names)
    wins = np.zeros((n, n), dtype=np.float64)
    played = np.zeros((n, n), dtype=np.int64)

    for h, a, y in zip(home_names, away_names, results):
        if h not in position or a not in position:
            continue
        i, j = position[h], position[a]
        wins[i, j] += y
        wins[j, i] += 1.0 - y
        played[i, j] += 1
        played[j, i] += 1
    return {"names": names, "wins": wins, "played": played}


def rating_churn(history: RatingHistory, *, top: int = 8) -> dict[str, np.ndarray]:
    """How much the leaderboard moves between epochs.

    Parameters
    ----------
    history
        From :meth:`~bscores.BScoreModel.score_history`.
    top
        Size of the leaderboard whose membership is tracked.

    Returns
    -------
    dict
        ``times`` (one shorter than the history), ``rank_change`` — the mean
        absolute change in rank per competitor — and ``entered_top`` — how many
        competitors joined the top ``top`` at that epoch.  A rating system with
        a very short memory shows high churn; one with a long memory is nearly
        flat.
    """
    scores = history.scores
    if scores.shape[0] < 2:
        raise ValueError("need at least two epochs to measure churn")
    if not 0 < top <= scores.shape[1]:
        raise ValueError(f"top must lie in (0, {scores.shape[1]}]")

    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]

    rank_change = np.abs(np.diff(ranks, axis=0)).mean(axis=1)
    in_top = ranks < top
    entered = (in_top[1:] & ~in_top[:-1]).sum(axis=1)
    return {
        "times": history.times[1:],
        "rank_change": rank_change,
        "entered_top": entered.astype(np.int64),
    }


def upset_rate(outcome: Any, prediction: Any, *, threshold: float = 0.5) -> dict[str, float]:
    """How often the forecast favourite lost, split by confidence.

    Returns
    -------
    dict
        ``n_favoured`` matches where the forecast exceeded ``threshold``, the
        ``upset_rate`` among them, and ``expected`` — what the forecasts
        themselves implied.  A gap between the two is miscalibration, not bad
        luck.
    """
    y, p = _pair(outcome, prediction)
    favoured = p > threshold
    n = int(favoured.sum())
    if n == 0:
        return {"n_favoured": 0.0, "upset_rate": float("nan"), "expected": float("nan")}
    return {
        "n_favoured": float(n),
        "upset_rate": float(np.mean(y[favoured] < 0.5)),
        "expected": float(np.mean(1.0 - p[favoured])),
    }
