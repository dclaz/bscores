"""The B-score rating model.

:class:`BScoreModel` is the entry point.  It mirrors the shape of
`openskill.py <https://github.com/vivekjoshy/openskill.py>`_ — ``rating``,
``rate``, ``predict_win``, ``predict_rank``, ``ordinal`` — so swapping a
Plackett-Luce or Bradley-Terry rating system for B-scores is mostly a matter of
changing the import.

Where it necessarily differs: a B-score is a property of the *whole* network,
not of an individual, so the model is stateful.  Feeding it a result updates
everybody's rating, including competitors who did not play — that is the entire
point of the method (Sect. 2 of the paper).  Consequently :meth:`BScoreModel.
rate` mutates the model and returns the refreshed ratings, rather than being a
pure function of its arguments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._time import as_day, as_days
from .calibration import LogitCalibrator, Transform, fit_logistic, sigmoid
from .centrality import bonacich_centrality
from .decay import DecayKernel, as_kernel
from .network import LossNetwork, NodeIndex

__all__ = ["Rating", "RatingHistory", "BScoreModel"]

TeamLike = Any


@dataclass(frozen=True)
class Rating:
    """A competitor's standing at a point in time.

    Attributes
    ----------
    name
        Competitor identifier.
    score
        The B-score: this competitor's entry in the unit-norm Bonacich
        centrality vector.  Always in ``[0, 1]``; ``0`` means "no results yet,
        or only losses to competitors who never won".
    matches, wins, losses
        Raw counts (draws add ``0.5`` to each of wins and losses).
    rank
        1-based position on the leaderboard when the rating came from a ranked
        query, otherwise ``None``.
    """

    name: str
    score: float
    matches: int = 0
    wins: float = 0.0
    losses: float = 0.0
    rank: int | None = None

    def ordinal(self, *, scale: float = 1000.0, target: float = 0.0) -> float:
        """A display rating on a friendlier scale than ``[0, 1]``.

        Mirrors ``openskill``'s ``Rating.ordinal``: a monotone rescaling with no
        extra information in it.  The default puts a typical top competitor
        somewhere in the low hundreds.
        """
        return self.score * scale + target

    def __repr__(self) -> str:
        rank = "" if self.rank is None else f", rank={self.rank}"
        return f"Rating(name={self.name!r}, score={self.score:.6g}, matches={self.matches}{rank})"


@dataclass(frozen=True)
class RatingHistory:
    """B-scores for every competitor across a series of times."""

    times: np.ndarray
    """Evaluation times, days since the epoch, ascending."""

    scores: np.ndarray
    """``(len(times), len(names))`` matrix of B-scores."""

    names: list[str]
    """Column labels, in model id order."""

    def of(self, name: str) -> np.ndarray:
        """One competitor's score series."""
        try:
            column = self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown competitor {name!r}") from None
        return self.scores[:, column]

    def at(self, time: Any) -> np.ndarray:
        """Scores at the last evaluation time at or before ``time``."""
        stamp = as_day(time)
        position = int(np.searchsorted(self.times, stamp, side="right")) - 1
        if position < 0:
            return np.zeros(len(self.names))
        return self.scores[position]

    def to_frame(self):  # pragma: no cover - thin pandas adapter
        """Return a ``pandas.DataFrame`` indexed by date."""
        import pandas as pd

        index = pd.to_datetime(self.times * 86_400_000_000, unit="us").rename("date")
        return pd.DataFrame(self.scores, index=index, columns=self.names)

    def __repr__(self) -> str:
        return f"RatingHistory({len(self.times)} epochs x {len(self.names)} competitors)"


def _team_names(team: TeamLike) -> list[str]:
    """Normalise a team argument to a list of competitor names."""
    if isinstance(team, Rating):
        return [team.name]
    if isinstance(team, str):
        return [team]
    if isinstance(team, Iterable):
        names: list[str] = []
        for member in team:
            if isinstance(member, Rating):
                names.append(member.name)
            elif isinstance(member, str):
                names.append(member)
            else:
                raise TypeError(f"team members must be names or Ratings, got {member!r}")
        if not names:
            raise ValueError("teams must not be empty")
        return names
    raise TypeError(f"cannot interpret {team!r} as a team")


class BScoreModel:
    """B-score rating system.

    Parameters
    ----------
    alpha
        Memory parameter in days for the default hyperbolic kernel: a result
        halves in weight after ``alpha`` days.  The paper uses 365.
    kernel
        Override the kernel entirely — see :mod:`bscores.decay`.
    draw_weight
        Weight of each of the two arcs a drawn match contributes.  ``0.5`` (the
        default) splits one match's worth of skill transfer both ways; ``0``
        makes draws invisible to the network, matching a sport that has none.
    max_age
        Ignore results older than this many days.  Bounds the per-epoch cost of
        a long back-test at the price of truncating the kernel's tail.
    warm_start
        Seed each centrality solve with the previous epoch's answer.  Successive
        networks are nearly identical, so this typically cuts iterations by an
        order of magnitude.  Turn it off to make every solve independent.
    tol, max_iter, shift, regularization, solver
        Forwarded to :func:`bscores.centrality.bonacich_centrality`.
    sparse, dense_max_nodes
        Matrix representation controls, forwarded to
        :class:`bscores.network.LossNetwork`.

    Examples
    --------
    >>> model = BScoreModel(alpha=365.0)
    >>> _ = model.rate([["Geelong"], ["Carlton"]], at="2021-03-18")
    >>> _ = model.rate([["Carlton"], ["Essendon"]], at="2021-03-25")
    >>> round(model.rating("Geelong").score, 4) > 0
    True
    >>> probabilities = model.predict_win([["Geelong"], ["Essendon"]])
    >>> round(sum(probabilities), 10)
    1.0
    """

    def __init__(
        self,
        *,
        alpha: float = 365.0,
        kernel: DecayKernel | float | str | None = None,
        draw_weight: float = 0.5,
        max_age: float | None = None,
        warm_start: bool = True,
        tol: float = 1e-12,
        max_iter: int = 10_000,
        shift: float = 0.15,
        regularization: float = 0.0,
        solver: Literal["auto", "power", "dense"] = "auto",
        sparse: Literal["auto"] | bool = "auto",
        dense_max_nodes: int = 512,
    ) -> None:
        self.kernel = as_kernel(kernel, alpha=alpha)
        self.draw_weight = float(draw_weight)
        if self.draw_weight < 0.0:
            raise ValueError("draw_weight must be non-negative")
        self.warm_start = bool(warm_start)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.shift = float(shift)
        self.regularization = float(regularization)
        self.solver = solver

        self.index = NodeIndex()
        self.network = LossNetwork(
            self.kernel,
            sparse=sparse,
            dense_max_nodes=dense_max_nodes,
            max_age=max_age,
        )
        self.calibrator: LogitCalibrator | None = None
        self.draw_calibrator_: np.ndarray | None = None

        self._clock = 0.0
        self._version = 0
        self._cache: tuple[Any, np.ndarray] | None = None
        self._warm: np.ndarray | None = None
        self._matches = np.zeros(0, dtype=np.int64)
        self._wins = np.zeros(0, dtype=np.float64)
        self._losses = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Half-life of the decay kernel, in days."""
        return self.kernel.half_life

    @property
    def players(self) -> list[str]:
        """Known competitors, in the id order used by score vectors."""
        return self.index.names

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, name: object) -> bool:
        return name in self.index

    def _ensure_counters(self, size: int) -> None:
        current = self._matches.size
        if size <= current:
            return
        matches = np.zeros(size, dtype=np.int64)
        wins = np.zeros(size, dtype=np.float64)
        losses = np.zeros(size, dtype=np.float64)
        matches[:current] = self._matches
        wins[:current] = self._wins
        losses[:current] = self._losses
        self._matches, self._wins, self._losses = matches, wins, losses

    def _invalidate(self) -> None:
        self._version += 1
        self._cache = None

    def _warm_vector(self, size: int) -> np.ndarray | None:
        """Previous solution, padded to ``size`` so newcomers start positive.

        A newly added competitor has no entry in the cached eigenvector.  Seeding
        it with zero would leave it stuck at zero whenever its only opponents are
        also new, so pad with the mean of the existing scores instead.
        """
        if not self.warm_start or self._warm is None:
            return None
        warm = self._warm
        if warm.size == size:
            return warm
        if warm.size > size:  # pragma: no cover - the node count never shrinks
            return warm[:size]
        padded = np.empty(size, dtype=np.float64)
        padded[: warm.size] = warm
        fill = float(warm.mean()) if warm.size else 0.0
        padded[warm.size :] = max(fill, 1.0 / np.sqrt(size))
        return padded

    def _next_time(self, at: Any) -> float:
        if at is None:
            self._clock += 1.0
            return self._clock
        stamp = as_day(at)
        self._clock = max(self._clock, stamp)
        return stamp

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def rating(self, name: str, *, at: Any = None, default: bool = True) -> Rating:
        """Current (or historical) rating for one competitor.

        Unknown competitors come back with a zero score when ``default`` is set,
        which is the right answer: an unconnected node has no centrality.
        """
        if name not in self.index:
            if not default:
                raise KeyError(f"unknown competitor {name!r}")
            return Rating(name=name, score=0.0)
        node = self.index.get(name)
        scores = self.scores(at=at)
        return Rating(
            name=name,
            score=float(scores[node]),
            matches=int(self._matches[node]),
            wins=float(self._wins[node]),
            losses=float(self._losses[node]),
        )

    def ratings(self, names: Sequence[str] | None = None, *, at: Any = None) -> list[Rating]:
        """Ratings for ``names`` (default: everyone), in id order."""
        scores = self.scores(at=at)
        wanted = self.index.names if names is None else list(names)
        order = np.argsort(-scores, kind="stable")
        rank_of = {int(node): position + 1 for position, node in enumerate(order)}
        out: list[Rating] = []
        for name in wanted:
            if name not in self.index:
                out.append(Rating(name=name, score=0.0))
                continue
            node = self.index.get(name)
            out.append(
                Rating(
                    name=name,
                    score=float(scores[node]),
                    matches=int(self._matches[node]),
                    wins=float(self._wins[node]),
                    losses=float(self._losses[node]),
                    rank=rank_of[node],
                )
            )
        return out

    def leaderboard(self, top: int | None = None, *, at: Any = None) -> list[Rating]:
        """Ratings sorted best-first."""
        table = sorted(self.ratings(at=at), key=lambda r: (-r.score, r.name))
        return table if top is None else table[:top]

    def rate(
        self,
        teams: Sequence[TeamLike],
        *,
        ranks: Sequence[float] | None = None,
        at: Any = None,
        weight: float = 1.0,
    ) -> list[list[Rating]]:
        """Record a result and return the refreshed ratings.

        Parameters
        ----------
        teams
            Teams in finishing order (best first), unless ``ranks`` is given.
            A team is a list of competitor names, or a bare name for the common
            one-per-side case.
        ranks
            Explicit finishing positions, lower is better.  Equal ranks are
            draws.
        at
            Match time.  ``None`` advances an internal counter by one, which
            lets a caller with no calendar still get sensible decay (``alpha``
            is then measured in matches).
        weight
            Importance of this result.  Scales the skill transferred, so a
            play-off can count for more than a regular-season game.

        Returns
        -------
        list[list[Rating]]
            The updated ratings, in the shape the teams were passed in.
        """
        rosters = [_team_names(team) for team in teams]
        if len(rosters) < 2:
            raise ValueError("rate() needs at least two teams")
        if ranks is None:
            positions = np.arange(len(rosters), dtype=np.float64)
        else:
            positions = np.asarray(ranks, dtype=np.float64).ravel()
            if positions.shape != (len(rosters),):
                raise ValueError(f"ranks must have one entry per team ({len(rosters)})")

        time = self._next_time(at)
        node_ids = [[self.index.add(name) for name in roster] for roster in rosters]
        self.network.resize(len(self.index))
        self._ensure_counters(len(self.index))

        losers: list[int] = []
        winners: list[int] = []
        weights: list[float] = []
        for a in range(len(rosters)):
            for b in range(a + 1, len(rosters)):
                share = weight / (len(node_ids[a]) * len(node_ids[b]))
                if positions[a] == positions[b]:
                    if self.draw_weight <= 0.0:
                        continue
                    share *= self.draw_weight
                    pairs = [(x, y) for x in node_ids[a] for y in node_ids[b]]
                    for x, y in pairs:
                        losers += [x, y]
                        winners += [y, x]
                        weights += [share, share]
                    continue
                better, worse = (a, b) if positions[a] < positions[b] else (b, a)
                for loser in node_ids[worse]:
                    for winner in node_ids[better]:
                        losers.append(loser)
                        winners.append(winner)
                        weights.append(share)

        if losers:
            self.network.extend(losers, winners, np.full(len(losers), time), weights)

        for team, position in zip(node_ids, positions):
            beaten = float(np.sum(positions > position))
            lost_to = float(np.sum(positions < position))
            drew = float(np.sum(positions == position)) - 1.0
            for node in team:
                self._matches[node] += 1
                self._wins[node] += beaten + 0.5 * drew
                self._losses[node] += lost_to + 0.5 * drew

        self._invalidate()
        scores = self.scores()
        return [
            [
                Rating(
                    name=self.index[node],
                    score=float(scores[node]),
                    matches=int(self._matches[node]),
                    wins=float(self._wins[node]),
                    losses=float(self._losses[node]),
                )
                for node in team
            ]
            for team in node_ids
        ]

    def rate_result(
        self,
        winner: str,
        loser: str,
        *,
        at: Any = None,
        weight: float = 1.0,
        draw: bool = False,
    ) -> tuple[Rating, Rating]:
        """Record a single head-to-head result."""
        ranks = [0.0, 0.0] if draw else [0.0, 1.0]
        updated = self.rate([winner, loser], ranks=ranks, at=at, weight=weight)
        return updated[0][0], updated[1][0]

    def rate_many(
        self,
        winners: Sequence[str],
        losers: Sequence[str],
        times: Any = None,
        *,
        weights: Sequence[float] | None = None,
    ) -> BScoreModel:
        """Record many head-to-head results at once.

        Bulk ingestion skips the per-result rating refresh, so loading a whole
        season is one centrality solve rather than one per match.
        """
        winner_names = list(winners)
        loser_names = list(losers)
        if len(winner_names) != len(loser_names):
            raise ValueError(
                f"winners/losers length mismatch: {len(winner_names)} vs {len(loser_names)}"
            )
        if not winner_names:
            return self

        if times is None:
            stamps = self._clock + np.arange(1.0, len(winner_names) + 1.0)
        else:
            stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
            if stamps.size == 1 and len(winner_names) > 1:
                stamps = np.full(len(winner_names), float(stamps[0]))
            if stamps.shape != (len(winner_names),):
                raise ValueError(f"times length mismatch: {stamps.shape} vs {len(winner_names)}")
        self._clock = max(self._clock, float(stamps.max()))

        winner_ids = self.index.add_many(winner_names)
        loser_ids = self.index.add_many(loser_names)
        self.network.resize(len(self.index))
        self._ensure_counters(len(self.index))

        share = np.ones(len(winner_names)) if weights is None else np.asarray(weights, float)
        self.network.extend(loser_ids, winner_ids, stamps, share)

        np.add.at(self._matches, winner_ids, 1)
        np.add.at(self._matches, loser_ids, 1)
        np.add.at(self._wins, winner_ids, 1.0)
        np.add.at(self._losses, loser_ids, 1.0)
        self._invalidate()
        return self

    def add_matches(
        self,
        home: Sequence[str],
        away: Sequence[str],
        outcome: Any,
        times: Any,
        *,
        weights: Sequence[float] | None = None,
    ) -> BScoreModel:
        """Ingest a home/away fixture list scored on the ``{0, 0.5, 1}`` scale.

        ``outcome`` is from the home side's point of view: 1 home win, 0 away
        win, 0.5 draw.  Draws contribute an arc each way weighted by
        ``draw_weight``.
        """
        home_names = list(home)
        away_names = list(away)
        if len(home_names) != len(away_names):
            raise ValueError(
                f"home/away length mismatch: {len(home_names)} vs {len(away_names)}"
            )
        results = np.asarray(outcome, dtype=np.float64).ravel()
        if results.shape != (len(home_names),):
            raise ValueError(f"outcome length mismatch: {results.shape} vs {len(home_names)}")
        if np.any(results < 0.0) or np.any(results > 1.0):
            raise ValueError("outcome must lie in [0, 1]")
        stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
        if stamps.shape != (len(home_names),):
            raise ValueError(f"times length mismatch: {stamps.shape} vs {len(home_names)}")
        share = np.ones(len(home_names)) if weights is None else np.asarray(weights, float)
        if share.shape != (len(home_names),):
            raise ValueError(f"weights length mismatch: {share.shape} vs {len(home_names)}")
        if len(home_names) == 0:
            return self

        home_ids = self.index.add_many(home_names)
        away_ids = self.index.add_many(away_names)
        self.network.resize(len(self.index))
        self._ensure_counters(len(self.index))
        self._clock = max(self._clock, float(stamps.max()))

        home_win = results > 0.5
        away_win = results < 0.5
        drawn = ~(home_win | away_win)

        losers = np.concatenate(
            [away_ids[home_win], home_ids[away_win], home_ids[drawn], away_ids[drawn]]
        )
        winners = np.concatenate(
            [home_ids[home_win], away_ids[away_win], away_ids[drawn], home_ids[drawn]]
        )
        arc_times = np.concatenate(
            [stamps[home_win], stamps[away_win], stamps[drawn], stamps[drawn]]
        )
        arc_weights = np.concatenate(
            [
                share[home_win],
                share[away_win],
                share[drawn] * self.draw_weight,
                share[drawn] * self.draw_weight,
            ]
        )
        if self.draw_weight <= 0.0:
            keep = arc_weights > 0.0
            losers, winners = losers[keep], winners[keep]
            arc_times, arc_weights = arc_times[keep], arc_weights[keep]
        if losers.size:
            self.network.extend(losers, winners, arc_times, arc_weights)

        np.add.at(self._matches, home_ids, 1)
        np.add.at(self._matches, away_ids, 1)
        np.add.at(self._wins, home_ids, results)
        np.add.at(self._losses, home_ids, 1.0 - results)
        np.add.at(self._wins, away_ids, 1.0 - results)
        np.add.at(self._losses, away_ids, results)
        self._invalidate()
        return self

    # ------------------------------------------------------------------
    # scores
    # ------------------------------------------------------------------
    def scores(self, at: Any = None, *, inclusive: bool = True) -> np.ndarray:
        """B-scores for every known competitor, in id order.

        Parameters
        ----------
        at
            Evaluation time; ``None`` means "after everything recorded so far".
        inclusive
            Whether results timestamped exactly ``at`` count.  Pass ``False``
            for a rating a forecast of a match starting at ``at`` may use.
        """
        key = (self._version, None if at is None else as_day(at), inclusive)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]

        n = len(self.index)
        if n == 0:
            return np.zeros(0)
        self.network.resize(n)
        matrix = self.network.matrix(at, inclusive=inclusive, transposed=True)
        result = bonacich_centrality(
            matrix,
            transposed=True,
            x0=self._warm_vector(n),
            tol=self.tol,
            max_iter=self.max_iter,
            shift=self.shift,
            regularization=self.regularization,
            method=self.solver,
            return_info=True,
        )
        if self.warm_start:
            self._warm = result.vector
        self._cache = (key, result.vector)
        return result.vector

    def score_history(
        self,
        times: Any,
        *,
        inclusive: bool = False,
        columns: Sequence[int] | None = None,
    ) -> RatingHistory:
        """B-scores at a series of times, computed causally.

        Each epoch sees only results that happened before it (``inclusive=False``,
        the default), so the output is safe to use as a back-test feature.
        Duplicate times are solved once and shared.

        Parameters
        ----------
        times
            Evaluation times.  Need not be sorted or unique.
        columns
            Restrict the returned matrix to these node ids.  Worth using when
            sweeping a long history over many competitors, since the full
            matrix is ``len(unique times) x len(players)``.
        """
        stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
        n = len(self.index)
        self.network.resize(n)
        unique = np.unique(stamps)
        keep = None if columns is None else np.asarray(columns, dtype=np.int64)
        width = n if keep is None else keep.size
        out = np.zeros((unique.size, width), dtype=np.float64)
        if n == 0 or unique.size == 0:
            names = self.index.names if keep is None else [self.index[i] for i in keep]
            return RatingHistory(times=unique, scores=out, names=names)

        warm: np.ndarray | None = None
        matrices = self.network.iter_matrices(unique, inclusive=inclusive, transposed=True)
        for row, matrix in enumerate(matrices):
            result = bonacich_centrality(
                matrix,
                transposed=True,
                x0=warm if self.warm_start else None,
                tol=self.tol,
                max_iter=self.max_iter,
                shift=self.shift,
                regularization=self.regularization,
                method=self.solver,
                return_info=True,
            )
            if self.warm_start:
                warm = result.vector
            out[row] = result.vector if keep is None else result.vector[keep]

        names = self.index.names if keep is None else [self.index[i] for i in keep]
        return RatingHistory(times=unique, scores=out, names=names)

    def match_scores(
        self,
        home: Sequence[str],
        away: Sequence[str],
        times: Any,
        *,
        inclusive: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causal B-scores for both sides of each fixture.

        This is the feature builder behind :meth:`fit` and
        :func:`bscores.backtest.rolling_forecast`: for every fixture it returns
        the two competitors' scores as of just before kick-off.
        """
        home_names = list(home)
        away_names = list(away)
        if len(home_names) != len(away_names):
            raise ValueError(
                f"home/away length mismatch: {len(home_names)} vs {len(away_names)}"
            )
        stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
        if stamps.shape != (len(home_names),):
            raise ValueError(f"times length mismatch: {stamps.shape} vs {len(home_names)}")

        history = self.score_history(stamps, inclusive=inclusive)
        rows = np.searchsorted(history.times, stamps)
        home_ids = np.array(
            [self.index.get(n) if n in self.index else -1 for n in home_names], dtype=np.int64
        )
        away_ids = np.array(
            [self.index.get(n) if n in self.index else -1 for n in away_names], dtype=np.int64
        )
        home_scores = np.where(home_ids >= 0, history.scores[rows, np.maximum(home_ids, 0)], 0.0)
        away_scores = np.where(away_ids >= 0, history.scores[rows, np.maximum(away_ids, 0)], 0.0)
        return home_scores, away_scores

    # ------------------------------------------------------------------
    # calibration and prediction
    # ------------------------------------------------------------------
    def fit(
        self,
        home: Sequence[str],
        away: Sequence[str],
        outcome: Any,
        times: Any,
        *,
        ingest: bool = True,
        weights: Sequence[float] | None = None,
        sample_weight: Any = None,
        model_draws: bool = False,
        symmetric: bool = False,
        transform: Transform = "identity",
        ridge: float = 1e-6,
        fit_intercept: bool = True,
    ) -> BScoreModel:
        """Ingest a fixture list and calibrate Eq. 3 on it.

        Parameters
        ----------
        home, away, outcome, times
            The training fixtures.  ``outcome`` is on the ``{0, 0.5, 1}`` scale
            from the home side's point of view.
        ingest
            Add these results to the network first.  Leave it on unless the
            model has already seen them — the calibration features are computed
            causally either way, so ingesting cannot leak.
        model_draws
            Also fit a small logistic for the draw rate, enabling
            :meth:`predict_draw`.
        symmetric, transform, ridge, fit_intercept
            Forwarded to :class:`~bscores.calibration.LogitCalibrator`.
        """
        if ingest:
            self.add_matches(home, away, outcome, times, weights=weights)
        home_scores, away_scores = self.match_scores(home, away, times, inclusive=False)
        results = np.asarray(outcome, dtype=np.float64).ravel()

        self.calibrator = LogitCalibrator(
            fit_intercept=fit_intercept,
            symmetric=symmetric,
            transform=transform,
            ridge=ridge,
        ).fit(home_scores, away_scores, results, sample_weight=sample_weight)

        if model_draws:
            drawn = np.isclose(results, 0.5).astype(np.float64)
            design = np.column_stack(
                [np.ones_like(home_scores), np.abs(home_scores - away_scores)]
            )
            self.draw_calibrator_ = fit_logistic(
                design,
                drawn,
                sample_weight=None if sample_weight is None else np.asarray(sample_weight),
                penalize=np.array([False, True]),
                ridge=ridge,
            ).beta
        else:
            self.draw_calibrator_ = None
        return self

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`predict_win` uses the fitted logit."""
        return self.calibrator is not None and self.calibrator.is_fitted

    def _team_scores(self, teams: Sequence[TeamLike], at: Any, inclusive: bool) -> np.ndarray:
        scores = self.scores(at=at, inclusive=inclusive)
        totals = np.zeros(len(teams), dtype=np.float64)
        for position, team in enumerate(teams):
            for name in _team_names(team):
                if name in self.index:
                    totals[position] += scores[self.index.get(name)]
        return totals

    def predict_win(
        self,
        teams: Sequence[TeamLike],
        *,
        at: Any = None,
        inclusive: bool = True,
    ) -> list[float]:
        """Probability that each team wins.

        With two teams and a fitted calibrator this is Eq. 3, with ``teams[0]``
        in the home slot.  Otherwise it falls back to scores normalised to sum
        to one, which needs no training data — and when nobody has a score yet
        (an empty network) it returns a uniform prior.

        Returns
        -------
        list[float]
            One probability per team, summing to 1.
        """
        if len(teams) < 2:
            raise ValueError("predict_win() needs at least two teams")
        totals = self._team_scores(teams, at, inclusive)

        if len(teams) == 2 and self.is_calibrated:
            assert self.calibrator is not None
            probability = float(
                self.calibrator.predict_proba(totals[:1], totals[1:2])[0]
            )
            return [probability, 1.0 - probability]

        total = float(totals.sum())
        if total <= 0.0:
            return [1.0 / len(teams)] * len(teams)
        return [float(value / total) for value in totals]

    def predict_draw(
        self,
        teams: Sequence[TeamLike],
        *,
        at: Any = None,
        inclusive: bool = True,
    ) -> float:
        """Probability the match is drawn.

        Requires ``fit(..., model_draws=True)``; returns ``0.0`` otherwise,
        which is the right answer for a sport that cannot draw.
        """
        if self.draw_calibrator_ is None:
            return 0.0
        if len(teams) != 2:
            raise ValueError("predict_draw() supports exactly two teams")
        totals = self._team_scores(teams, at, inclusive)
        design = np.array([1.0, abs(float(totals[0] - totals[1]))])
        return float(sigmoid(design @ self.draw_calibrator_))

    def predict_rank(
        self,
        teams: Sequence[TeamLike],
        *,
        at: Any = None,
        inclusive: bool = True,
        tie_tol: float = 1e-9,
    ) -> list[tuple[int, float]]:
        """Predicted finishing rank and win probability for each team.

        Mirrors ``openskill``'s ``predict_rank``: ranks are 1-based, dense, and
        tied teams share a rank.

        Parameters
        ----------
        tie_tol
            Probabilities within this much of each other count as tied.  Scores
            that are equal by symmetry — two competitors with mirror-image
            records — come out of an iterative solve equal only to rounding, so
            an exact float comparison would split them arbitrarily.
        """
        probabilities = self.predict_win(teams, at=at, inclusive=inclusive)
        order = sorted(range(len(teams)), key=lambda i: -probabilities[i])
        ranks = [0] * len(teams)
        rank = 0
        previous: float | None = None
        for position, team in enumerate(order):
            if previous is None or probabilities[team] < previous - tie_tol:
                rank = position + 1
                previous = probabilities[team]
            ranks[team] = rank
        return [(ranks[i], probabilities[i]) for i in range(len(teams))]

    def predict_proba(
        self,
        home: Sequence[str],
        away: Sequence[str],
        times: Any,
        *,
        inclusive: bool = False,
    ) -> np.ndarray:
        """Vectorised home-win probabilities for a fixture list.

        Scores are taken causally (only results before each match), so this is
        the right call for evaluating a schedule already in the network.
        """
        home_scores, away_scores = self.match_scores(home, away, times, inclusive=inclusive)
        if self.is_calibrated:
            assert self.calibrator is not None
            return self.calibrator.predict_proba(home_scores, away_scores)
        total = home_scores + away_scores
        return np.where(total > 0.0, home_scores / np.where(total > 0.0, total, 1.0), 0.5)

    def __repr__(self) -> str:
        state = "calibrated" if self.is_calibrated else "uncalibrated"
        return (
            f"BScoreModel(kernel={self.kernel!r}, competitors={len(self.index)}, "
            f"results={self.network.n_events}, {state})"
        )
