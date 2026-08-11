"""Reference rating systems to benchmark B-scores against.

The paper's headline claim is comparative — B-scores beat Elo and friends on
log-loss and Brier score — so a like-for-like Elo lives here.  It is deliberately
small: enough to be a fair baseline in the AFL diagnostics, not a general
rating library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Elo"]


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
        home: Sequence[str],
        away: Sequence[str],
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
