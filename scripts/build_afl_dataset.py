"""Rebuild the packaged AFL dataset from ``data/afl.xlsx``.

Produces ``src/bscores/data/afl_matches.csv``, the file shipped with the package
and returned by :func:`bscores.datasets.load_afl`.

The raw workbook is the historical AFL archive published at
https://www.aussportsbetting.com/historical_data/afl.xlsx

Usage::

    python scripts/build_afl_dataset.py [--source data/afl.xlsx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "afl.xlsx"
DEFAULT_DEST = REPO_ROOT / "src" / "bscores" / "data" / "afl_matches.csv"

#: Columns of the published dataset, in order.
COLUMNS = [
    "date",
    "date_time",
    "home_team",
    "away_team",
    "venue",
    "home_score",
    "away_score",
    "margin",
    "outcome",
    "final",
    "home_odds",
    "away_odds",
]


def match_outcome(margin: pd.Series) -> pd.Series:
    """Map a home-team margin onto ``{0, 0.5, 1}``.

    A home win scores 1, an away win scores 0 and a draw scores 0.5, which is
    what a probability-scale loss function needs.
    """
    return (margin + 0.5).clip(lower=0.0, upper=1.0)


def build(source: Path = DEFAULT_SOURCE) -> pd.DataFrame:
    """Read the raw workbook and return the tidied match table."""
    raw = pd.read_excel(source, skiprows=1)
    def clean(name: str) -> str:
        for old, new in (("?", ""), ("(", ""), (")", "")):
            name = name.replace(old, new)
        return name.strip().lower().replace(" ", "_")

    raw.columns = [clean(c) for c in raw.columns]

    date = pd.to_datetime(raw["date"]).dt.normalize()
    kick_off = raw["kick_off_local"].astype(str)
    date_time = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + kick_off)

    margin = raw["home_score"].astype("int64") - raw["away_score"].astype("int64")

    out = pd.DataFrame(
        {
            "date": date.dt.date.astype("string"),
            "date_time": date_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "home_team": raw["home_team"].astype("string").str.strip(),
            "away_team": raw["away_team"].astype("string").str.strip(),
            "venue": raw["venue"].astype("string").str.strip(),
            "home_score": raw["home_score"].astype("int64"),
            "away_score": raw["away_score"].astype("int64"),
            "margin": margin,
            "outcome": match_outcome(margin).astype("float64"),
            "final": raw["play_off_game"].notna(),
            "home_odds": raw["home_odds"].astype("float64"),
            "away_odds": raw["away_odds"].astype("float64"),
        }
    )[COLUMNS]

    # The workbook is published newest-first; a rating system reads history
    # forwards, so store it chronologically and stably.
    out = out.sort_values(["date_time", "home_team"], kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    frame = build(args.source)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    # Plain CSV, not gzip: the file lives in git, where a diffable text blob
    # is worth more than the 200 KB compression would save.
    frame.to_csv(args.dest, index=False, lineterminator="\n")
    print(f"wrote {len(frame)} matches to {args.dest} ({args.dest.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
