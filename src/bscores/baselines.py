"""Reference rating systems to benchmark B-scores against.

The paper's headline claim is comparative — B-scores beat Elo and friends on
log-loss and Brier score — so a like-for-like Elo lives here.  It is deliberately
small: enough to be a fair baseline in the AFL diagnostics, not a general
rating library.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._time import as_days
from .typing import Names

__all__ = ["Elo", "tune_elo", "DEFAULT_ELO_GRID"]

#: A reasonable search space for :func:`tune_elo`.  ``home_advantage`` is the
#: one that matters most on a home/away competition; the rest shape Kovalchik's
#: experience-decayed K schedule.
DEFAULT_ELO_GRID: dict[str, list[Any]] = {
    "home_advantage": [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
    "k_scale": [100.0, 175.0, 250.0, 400.0],
    "k_power": [0.2, 0.4, 0.6],
}


class Elo:
    """Elo ratings, Eqs. 4 and 5 of the paper.

    .. math::

        p_{i,j,t} = \\frac{1}{1 + 10^{(E_{j,t} - E_{i,t}) / 400}}, \\qquad
        E_{i,t+1} = E_{i,t} + K_{i,t}\\,(W_{i,t} - p_{i,j,t})

    Parameters
    ----------
    initial
        Starting rating for an unseen competitor.
    k
        Fixed update scale.  Leave as ``None`` for Kovalchik's experience-decayed
        schedule ``k_scale / (n_matches + k_shape) ** k_power``, which moves a
        newcomer's rating quickly and a veteran's slowly.
    k_scale, k_shape, k_power
        Parameters of that schedule.
    home_advantage
        Rating points added to the home side before computing the probability.
    spread
        Rating difference worth a factor of ``base`` in the odds.
    """

    def __init__(
        self,
        *,
        initial: float = 1500.0,
        k: float | None = None,
        k_scale: float = 250.0,
        k_shape: float = 5.0,
        k_power: float = 0.4,
        home_advantage: float = 0.0,
        base: float = 10.0,
        spread: float = 400.0,
    ) -> None:
        self.initial = float(initial)
        self.k = None if k is None else float(k)
        self.k_scale = float(k_scale)
        self.k_shape = float(k_shape)
        self.k_power = float(k_power)
        self.home_advantage = float(home_advantage)
        self.base = float(base)
        self.spread = float(spread)
        self._ratings: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def rating(self, name: str) -> float:
        """Current rating, or ``initial`` for an unseen competitor."""
        return self._ratings.get(name, self.initial)

    @property
    def ratings(self) -> dict[str, float]:
        """Copy of every rating held."""
        return dict(self._ratings)

    def _k(self, name: str) -> float:
        if self.k is not None:
            return self.k
        played = self._counts.get(name, 0)
        return self.k_scale / (played + self.k_shape) ** self.k_power

    def expect(self, home: str, away: str) -> float:
        """Probability the home side wins, before the match is played."""
        difference = self.rating(away) - (self.rating(home) + self.home_advantage)
        return 1.0 / (1.0 + self.base ** (difference / self.spread))

    def update(self, home: str, away: str, outcome: float) -> float:
        """Score one match and return the pre-match home-win probability."""
        expected = self.expect(home, away)
        home_k = self._k(home)
        away_k = self._k(away)
        self._ratings[home] = self.rating(home) + home_k * (outcome - expected)
        self._ratings[away] = self.rating(away) + away_k * ((1.0 - outcome) - (1.0 - expected))
        self._counts[home] = self._counts.get(home, 0) + 1
        self._counts[away] = self._counts.get(away, 0) + 1
        return expected

    def run(
        self,
        home: Names,
        away: Names,
        outcome: Any,
    ) -> np.ndarray:
        """Walk a fixture list in order, returning each pre-match probability.

        The fixtures must already be in chronological order; every probability
        is computed strictly before its match is used to update the ratings, so
        the output is directly comparable with
        :meth:`bscores.BScoreModel.predict_proba`.
        """
        results = np.asarray(outcome, dtype=np.float64).ravel()
        if len(home) != len(away) or results.size != len(home):
            raise ValueError(
                f"home/away/outcome length mismatch: {len(home)}, {len(away)}, {results.size}"
            )
        out = np.empty(results.size, dtype=np.float64)
        for i, (h, a, y) in enumerate(zip(home, away, results)):
            out[i] = self.update(h, a, float(y))
        return out

    def __repr__(self) -> str:
        schedule = f"k={self.k}" if self.k is not None else f"k_scale={self.k_scale}"
        return f"Elo({schedule}, competitors={len(self._ratings)})"


def tune_elo(
    home: Names,
    away: Names,
    outcome: Any,
    times: Any,
    *,
    grid: Mapping[str, Sequence[Any]] | None = None,
    validation_start: Any,
    validation_end: Any = None,
    metric: str = "log_loss",
) -> dict[str, Any]:
    """Choose Elo's hyperparameters on a validation window.

    A baseline nobody tuned is not a baseline, it is a straw man.  A B-score
    model picks up home advantage for free — the calibrating logit fits an
    intercept, and on a home/away competition that intercept *is* the home
    edge — while Elo has to be told about it through ``home_advantage``.
    Comparing a searched B-score model against a default Elo therefore flatters
    the B-scores for reasons that have nothing to do with the rating method.

    This gives Elo the same treatment: a search over the same validation window
    the B-score grid uses, leaving the test window untouched for both.

    Parameters
    ----------
    home, away, outcome, times
        The full fixture list.  Sorted chronologically internally, since Elo is
        sequential and the order it walks the fixtures in is the whole model.
    grid
        Values to try, defaulting to :data:`DEFAULT_ELO_GRID`.  Keys are
        :class:`Elo` constructor arguments.
    validation_start, validation_end
        The window to score on, as a date or a fraction of the fixture list.
        Everything before ``validation_start`` is warm-up.
    metric
        ``"log_loss"``, ``"brier_score"``, ``"accuracy"`` or
        ``"classification_error"``.

    Returns
    -------
    dict
        The winning constructor arguments, ready to splat into :class:`Elo`.

    Examples
    --------
    >>> from bscores.baselines import Elo, tune_elo
    >>> from bscores.datasets import load_afl
    >>> afl = load_afl(as_frame=False)                        # doctest: +SKIP
    >>> best = tune_elo(afl.home_team, afl.away_team, afl.outcome, afl.date,
    ...                 validation_start="2019-01-01",
    ...                 validation_end="2023-01-01")          # doctest: +SKIP
    >>> Elo(**best)                                           # doctest: +SKIP
    """
    from .backtest import _resolve_start
    from .metrics import evaluate

    maximise = metric in {"accuracy"}
    space = dict(DEFAULT_ELO_GRID if grid is None else grid)
    if not space:
        raise ValueError("grid must not be empty")

    home_names = np.asarray(list(home), dtype=object)
    away_names = np.asarray(list(away), dtype=object)
    results = np.asarray(outcome, dtype=np.float64).ravel()
    stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    if not (home_names.size == away_names.size == results.size == stamps.size):
        raise ValueError("home, away, outcome and times must be the same length")

    order = np.argsort(stamps, kind="stable")
    home_names, away_names = home_names[order], away_names[order]
    results, stamps = results[order], stamps[order]

    start = _resolve_start(stamps, validation_start)
    stop = stamps.size if validation_end is None else _resolve_start(stamps, validation_end)
    if not start < stop:
        raise ValueError(
            f"validation_end must fall after validation_start ({start} >= {stop})"
        )

    keys = list(space)
    best_params: dict[str, Any] | None = None
    best_score = -np.inf if maximise else np.inf
    for combination in itertools.product(*(list(space[k]) for k in keys)):
        params = dict(zip(keys, combination))
        probability = Elo(**params).run(home_names, away_names, results)
        score = evaluate(results[start:stop], probability[start:stop])[metric]
        better = score > best_score if maximise else score < best_score
        if best_params is None or better:
            best_params, best_score = params, score
    assert best_params is not None
    return best_params
