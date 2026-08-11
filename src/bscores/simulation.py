"""Monte Carlo season simulation.

Ratings answer "who is better"; a season simulation answers the question people
actually ask — "what are our chances of finishing top four?"  Feed it the
remaining fixtures and it plays the rest of the season a few thousand times.

Ratings are held fixed across a simulation rather than updated match by match.
That is the standard simplification, and it is a real one: it ignores the
feedback where an early upset would have changed later forecasts, so the spread
of simulated ladders is slightly narrower than reality.  It also makes the whole
thing one vectorised draw instead of thousands of centrality solves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import BScoreModel

__all__ = ["SeasonSimulation", "simulate_season"]


@dataclass(frozen=True)
class SeasonSimulation:
    """Outcome distribution over a simulated set of fixtures."""

    competitors: list[str]
    """Names, in the order the arrays are indexed."""

    points: np.ndarray
    """``(n_simulations, n_competitors)`` final points."""

    positions: np.ndarray
    """``(n_simulations, n_competitors)`` final ladder positions, 1 = best."""

    starting_points: np.ndarray
    """Points each competitor began the simulation with."""

    n_simulations: int
    """How many seasons were played."""

    def expected_points(self) -> dict[str, float]:
        """Mean final points per competitor, best first."""
        means = self.points.mean(axis=0)
        order = np.argsort(-means)
        return {self.competitors[i]: float(means[i]) for i in order}

    def finish_probability(self, place: int = 1) -> dict[str, float]:
        """Probability of finishing in exactly ``place``."""
        if not 1 <= place <= len(self.competitors):
            raise ValueError(f"place must lie in [1, {len(self.competitors)}]")
        probability = (self.positions == place).mean(axis=0)
        order = np.argsort(-probability)
        return {self.competitors[i]: float(probability[i]) for i in order}

    def top_n_probability(self, n: int = 4) -> dict[str, float]:
        """Probability of finishing in the top ``n`` — the finals question."""
        if not 1 <= n <= len(self.competitors):
            raise ValueError(f"n must lie in [1, {len(self.competitors)}]")
        probability = (self.positions <= n).mean(axis=0)
        order = np.argsort(-probability)
        return {self.competitors[i]: float(probability[i]) for i in order}

    def position_distribution(self, name: str) -> np.ndarray:
        """Probability of each finishing position for one competitor."""
        try:
            index = self.competitors.index(name)
        except ValueError:
            raise KeyError(f"unknown competitor {name!r}") from None
        counts = np.bincount(
            self.positions[:, index], minlength=len(self.competitors) + 1
        )[1:]
        return counts / self.n_simulations

    def to_frame(self):  # pragma: no cover - thin pandas adapter
        """Summary table: expected points and finishing probabilities."""
        import pandas as pd

        expected = self.expected_points()
        return pd.DataFrame(
            {
                "expected_points": pd.Series(expected),
                "p_first": pd.Series(self.finish_probability(1)),
                "p_top4": pd.Series(self.top_n_probability(4)),
            }
        ).sort_values("expected_points", ascending=False)

    def __repr__(self) -> str:
        leader, points = next(iter(self.expected_points().items()))
        return (
            f"SeasonSimulation({self.n_simulations} seasons, "
            f"{len(self.competitors)} competitors, favourite={leader!r} at {points:.1f} points)"
        )


def simulate_season(
    model: BScoreModel,
    home: Sequence[str],
    away: Sequence[str],
    *,
    n_simulations: int = 10_000,
    win_points: float = 4.0,
    draw_points: float = 2.0,
    draw_probability: float | None = None,
    standings: dict[str, float] | None = None,
    at: Any = None,
    seed: int | None = None,
) -> SeasonSimulation:
    """Play a fixture list ``n_simulations`` times using the model's forecasts.

    Parameters
    ----------
    model
        The rating model.  Calibrate it first — an uncalibrated model falls back
        to score shares, which are far less well behaved as probabilities.
    home, away
        The remaining fixtures.
    n_simulations
        Seasons to play.  10,000 puts the Monte Carlo error on a probability at
        well under a percentage point.
    win_points, draw_points
        The competition's points system.  The AFL default is 4 and 2.
    draw_probability
        Fixed probability of a draw, taken off the top of each match.  ``None``
        uses the model's own :meth:`~bscores.BScoreModel.predict_draw`, which is
        zero unless the model was fitted with ``model_draws=True``.
    standings
        Points already banked, by competitor.  Missing names start at zero.
    at
        Rate the competitors as at this time; ``None`` means "now".
    seed
        Seed for reproducibility.

    Returns
    -------
    SeasonSimulation

    Examples
    --------
    >>> from bscores import BScoreModel
    >>> model = BScoreModel(alpha=365.0)
    >>> for winner, loser, day in [("a", "b", 0), ("b", "c", 7), ("c", "a", 14)]:
    ...     _ = model.rate_result(winner, loser, at=float(day))
    >>> season = simulate_season(model, ["a", "b"], ["c", "a"], n_simulations=200, seed=0)
    >>> sorted(season.competitors)
    ['a', 'b', 'c']
    """
    home_names = list(home)
    away_names = list(away)
    if len(home_names) != len(away_names):
        raise ValueError(
            f"home/away length mismatch: {len(home_names)} vs {len(away_names)}"
        )
    if not home_names:
        raise ValueError("cannot simulate an empty fixture list")
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1")

    names = sorted(set(home_names) | set(away_names) | set(standings or {}))
    position = {name: i for i, name in enumerate(names)}
    n_teams = len(names)

    probability = np.empty(len(home_names), dtype=np.float64)
    drawn = np.empty(len(home_names), dtype=np.float64)
    for i, (h, a) in enumerate(zip(home_names, away_names)):
        probability[i] = model.predict_win([[h], [a]], at=at)[0]
        drawn[i] = (
            model.predict_draw([[h], [a]], at=at)
            if draw_probability is None
            else float(draw_probability)
        )
    if np.any(drawn < 0.0) or np.any(drawn >= 1.0):
        raise ValueError("draw probability must lie in [0, 1)")

    # Split each match into draw / home win / away win, then draw once per
    # simulation from that three-way categorical.
    home_win = probability * (1.0 - drawn)
    rng = np.random.default_rng(seed)
    roll = rng.random((n_simulations, len(home_names)))
    is_home_win = roll < home_win[None, :]
    is_draw = (roll >= home_win[None, :]) & (roll < (home_win + drawn)[None, :])

    points = np.zeros((n_simulations, n_teams), dtype=np.float64)
    if standings:
        for name, banked in standings.items():
            points[:, position[name]] += float(banked)
    starting = points[0].copy()

    home_index = np.array([position[name] for name in home_names])
    away_index = np.array([position[name] for name in away_names])
    home_gain = np.where(is_home_win, win_points, np.where(is_draw, draw_points, 0.0))
    away_gain = np.where(is_home_win, 0.0, np.where(is_draw, draw_points, win_points))
    np.add.at(points, (slice(None), home_index), home_gain)
    np.add.at(points, (slice(None), away_index), away_gain)

    # Ties on points are broken at random, so a tie splits the position evenly
    # rather than always favouring whoever sorts first.
    jitter = rng.random(points.shape) * 1e-9
    order = np.argsort(-(points + jitter), axis=1, kind="stable")
    positions = np.empty_like(order)
    rows = np.arange(n_simulations)[:, None]
    positions[rows, order] = np.arange(1, n_teams + 1)[None, :]

    return SeasonSimulation(
        competitors=names,
        points=points,
        positions=positions,
        starting_points=starting,
        n_simulations=n_simulations,
    )
