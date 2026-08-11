"""Bundled data.

The package ships an AFL match archive so the examples, tests and diagnostics
all run offline.  Regenerate it from the raw workbook with
``python scripts/build_afl_dataset.py``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

__all__ = ["MatchData", "load_afl", "afl_data_path"]

_AFL_RESOURCE = "afl_matches.csv"


@dataclass(frozen=True)
class MatchData:
    """A fixture list as plain numpy arrays.

    Attributes
    ----------
    date
        Match date, ``datetime64[D]``.
    date_time
        Local kick-off, ``datetime64[s]``.
    home_team, away_team, venue
        Object arrays of strings.
    home_score, away_score, margin
        Integer scores; ``margin`` is home minus away.
    outcome
        Home-side result on the ``{0, 0.5, 1}`` scale.
    final
        Whether the match was a play-off.
    home_odds, away_odds
        Bookmaker decimal odds.
    """

    date: np.ndarray
    date_time: np.ndarray
    home_team: np.ndarray
    away_team: np.ndarray
    venue: np.ndarray
    home_score: np.ndarray
    away_score: np.ndarray
    margin: np.ndarray
    outcome: np.ndarray
    final: np.ndarray
    home_odds: np.ndarray
    away_odds: np.ndarray

    def __len__(self) -> int:
        return int(self.outcome.size)

    @property
    def teams(self) -> list[str]:
        """Sorted list of every competitor appearing in the fixture list."""
        return sorted(set(self.home_team.tolist()) | set(self.away_team.tolist()))

    def __repr__(self) -> str:
        return (
            f"MatchData({len(self)} matches, {len(self.teams)} teams, "
            f"{self.date.min()} to {self.date.max()})"
        )


def afl_data_path():
    """Path to the packaged AFL CSV, as an ``importlib.resources`` traversable."""
    return resources.files("bscores.data").joinpath(_AFL_RESOURCE)


def _read_rows() -> list[dict[str, str]]:
    with (
        resources.as_file(afl_data_path()) as path,
        path.open("rt", newline="", encoding="utf-8") as handle,
    ):
        return list(csv.DictReader(handle))


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(row[key]) if row[key] not in ("", "NA") else np.nan for row in rows])


def load_afl(*, as_frame: bool = True) -> Any:
    """Load AFL match results and betting odds, 2009-06-19 to 2026-08-02.

    3533 matches between 18 teams, sorted chronologically.  Sourced from
    https://www.aussportsbetting.com/historical_data/afl.xlsx.

    Venue names carry the sponsor in use at the time the archive was published,
    not at the time of the match, so a ground may appear under one name across
    its whole history.

    Parameters
    ----------
    as_frame
        Return a ``pandas.DataFrame`` (requires pandas).  ``False`` returns a
        :class:`MatchData` of numpy arrays and needs nothing beyond numpy.

    Returns
    -------
    pandas.DataFrame or MatchData

    Examples
    --------
    >>> from bscores.datasets import load_afl
    >>> matches = load_afl(as_frame=False)
    >>> len(matches)
    3533
    """
    if as_frame:
        import pandas as pd

        with resources.as_file(afl_data_path()) as path:
            frame = pd.read_csv(path, parse_dates=["date", "date_time"])
        return frame

    rows = _read_rows()
    return MatchData(
        date=np.array([row["date"] for row in rows], dtype="datetime64[D]"),
        date_time=np.array([row["date_time"] for row in rows], dtype="datetime64[s]"),
        home_team=np.array([row["home_team"] for row in rows], dtype=object),
        away_team=np.array([row["away_team"] for row in rows], dtype=object),
        venue=np.array([row["venue"] for row in rows], dtype=object),
        home_score=np.array([int(row["home_score"]) for row in rows], dtype=np.int64),
        away_score=np.array([int(row["away_score"]) for row in rows], dtype=np.int64),
        margin=np.array([int(row["margin"]) for row in rows], dtype=np.int64),
        outcome=_float_column(rows, "outcome"),
        final=np.array([row["final"] == "True" for row in rows], dtype=bool),
        home_odds=_float_column(rows, "home_odds"),
        away_odds=_float_column(rows, "away_odds"),
    )
