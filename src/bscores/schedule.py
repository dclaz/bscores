"""Recovering a competition's structure from a bare fixture list.

Match archives often carry dates but no season or round number.  Both are easy
to infer, and both are wanted often enough to be worth doing once here:

* a **season** is a run of matches separated from the next by the off-season;
* a **round** is a maximal run of consecutive matches in which no competitor
  appears twice — which is what a round *is*, so the definition recovers byes,
  split rounds and finals weeks without needing a fixture template.

The main use is plotting.  A rating history drawn against the calendar spends a
third of its width on empty summers; drawn against
:func:`round_positions` the playing periods sit side by side.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._time import as_days
from .typing import Names

__all__ = ["infer_seasons", "infer_rounds", "round_positions", "round_labels"]

#: Days without a match that mark the boundary between two seasons.  Comfortably
#: longer than any mid-season break, shorter than any off-season.
DEFAULT_SEASON_GAP = 60.0


def infer_seasons(times: Any, *, gap: float = DEFAULT_SEASON_GAP) -> np.ndarray:
    """Label each match with a season, counting from 1.

    Parameters
    ----------
    times
        Match times, in any form :func:`bscores._time.as_days` accepts.  Need
        not be sorted.
    gap
        A break longer than this many days starts a new season.  The default
        separates competitions that run within a calendar year and those that
        straddle one, without either needing to be declared.

    Returns
    -------
    numpy.ndarray
        Integer season labels, aligned with ``times``.

    Examples
    --------
    >>> from bscores.schedule import infer_seasons
    >>> [int(s) for s in infer_seasons([0.0, 7.0, 400.0, 407.0])]
    [1, 1, 2, 2]
    """
    stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    if stamps.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not gap > 0.0:
        raise ValueError(f"gap must be positive, got {gap}")

    order = np.argsort(stamps, kind="stable")
    sorted_times = stamps[order]
    breaks = np.diff(sorted_times) > gap
    labels_sorted = np.concatenate([[1], 1 + np.cumsum(breaks)])
    labels = np.empty(stamps.size, dtype=np.int64)
    labels[order] = labels_sorted
    return labels


def infer_rounds(
    home: Names,
    away: Names,
    times: Any,
    *,
    season: Any = None,
    gap: float = DEFAULT_SEASON_GAP,
) -> np.ndarray:
    """Number the rounds within each season, counting from 1.

    A new round starts as soon as a competitor would play twice, so finals
    continue the numbering rather than restarting it, and a bye simply means
    fewer matches in that round.

    Parameters
    ----------
    home, away
        Competitor names for each match.
    times
        Match times.  Matches are processed in time order.  Matches sharing a
        timestamp — which is most of them, when the clock is day-resolution —
        are ordered by competitor name, so the answer depends only on the *set*
        of matches and not on the order they happen to be passed in.  Pass a
        finer clock (kick-off times rather than dates) when you have one and
        the real playing order matters.
    season
        Pre-computed season labels; inferred with :func:`infer_seasons` when
        omitted.
    gap
        Forwarded to :func:`infer_seasons`.

    Returns
    -------
    numpy.ndarray
        Integer round numbers, aligned with the inputs.

    Examples
    --------
    >>> from bscores.schedule import infer_rounds
    >>> [int(r) for r in infer_rounds(["a", "c", "a"], ["b", "d", "c"], [0.0, 1.0, 8.0])]
    [1, 1, 2]
    """
    home_names = list(home)
    away_names = list(away)
    stamps = np.atleast_1d(as_days(times)).astype(np.float64, copy=False)
    if not (len(home_names) == len(away_names) == stamps.size):
        raise ValueError(
            "home, away and times must be the same length, got "
            f"{len(home_names)}, {len(away_names)}, {stamps.size}"
        )
    if stamps.size == 0:
        return np.zeros(0, dtype=np.int64)

    labels = (
        infer_seasons(stamps, gap=gap)
        if season is None
        else np.asarray(season).ravel()
    )
    if labels.size != stamps.size:
        raise ValueError(f"season length mismatch: {labels.size} vs {stamps.size}")

    rounds = np.zeros(stamps.size, dtype=np.int64)
    # Same-timestamp matches must be walked in a defined order, or the greedy
    # boundary below lands differently depending on how the caller happened to
    # sort its rows.  Break those ties on the competitor names: arbitrary, but a
    # property of the match rather than of the row it arrived in.
    order = np.lexsort(
        (np.asarray(away_names, dtype=object), np.asarray(home_names, dtype=object), stamps, labels)
    )
    current_season: Any = None
    number = 0
    playing: set[str] = set()
    for index in order:
        if labels[index] != current_season:
            current_season, number, playing = labels[index], 1, set()
        elif home_names[index] in playing or away_names[index] in playing:
            number += 1
            playing = set()
        playing.add(home_names[index])
        playing.add(away_names[index])
        rounds[index] = number
    return rounds


def round_positions(season: Any, round_: Any) -> np.ndarray:
    """Map ``(season, round)`` pairs onto a single increasing axis.

    Consecutive rounds land on consecutive integers with no gap between
    seasons, which is the point: plotting against this collapses the
    off-season.

    Examples
    --------
    >>> from bscores.schedule import round_positions
    >>> [int(p) for p in round_positions([1, 1, 2], [1, 2, 1])]
    [0, 1, 2]
    """
    seasons = np.asarray(season).ravel()
    rounds = np.asarray(round_).ravel()
    if seasons.size != rounds.size:
        raise ValueError(f"season/round length mismatch: {seasons.size} vs {rounds.size}")
    if seasons.size == 0:
        return np.zeros(0, dtype=np.int64)

    pairs = np.stack([seasons, rounds], axis=1)
    unique = np.unique(pairs, axis=0)
    lookup = {(int(s), int(r)): i for i, (s, r) in enumerate(unique)}
    return np.array([lookup[(int(s), int(r))] for s, r in pairs], dtype=np.int64)


def round_labels(season: Any, round_: Any, *, final: Any = None) -> np.ndarray:
    """Human-readable ``"2023 R5"`` / ``"2023 F2"`` labels.

    Parameters
    ----------
    season, round_
        Season and round numbers.
    final
        Optional boolean mask marking finals, which are then numbered ``F1``,
        ``F2``, ... from the first finals round of that season.
    """
    seasons = np.asarray(season).ravel()
    rounds = np.asarray(round_).ravel()
    if seasons.size != rounds.size:
        raise ValueError(f"season/round length mismatch: {seasons.size} vs {rounds.size}")

    if final is None:
        return np.array([f"{s} R{r}" for s, r in zip(seasons, rounds)], dtype=object)

    flags = np.asarray(final).astype(bool).ravel()
    if flags.size != seasons.size:
        raise ValueError(f"final length mismatch: {flags.size} vs {seasons.size}")

    first_final = {
        int(s): int(rounds[flags & (seasons == s)].min())
        for s in np.unique(seasons)
        if np.any(flags & (seasons == s))
    }
    out = []
    for s, r, is_final in zip(seasons, rounds, flags):
        if is_final and int(s) in first_final:
            out.append(f"{s} F{int(r) - first_final[int(s)] + 1}")
        else:
            out.append(f"{s} R{r}")
    return np.array(out, dtype=object)
