"""Command line interface.

Installed as the ``bscores`` console script, and runnable as
``python -m bscores``.  Everything it does is available from the Python API;
the CLI exists so a fixture list in a CSV can be rated, forecast or tuned
without writing a script.

    bscores info
    bscores rate --data afl --top 10
    bscores forecast --data results.csv --alpha 120 --kernel exponential
    bscores tune --data afl --validation-start 2019-01-01 --validation-end 2023-01-01
    bscores simulate --data afl --simulations 20000

``--data afl`` uses the bundled archive; anything else is read as a CSV.  Add
``--format json`` to any subcommand for machine-readable output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .backtest import rolling_forecast
from .datasets import load_afl
from .diagnostics import network_summary
from .models import BScoreModel
from .simulation import simulate_season
from .tuning import grid_search

__all__ = ["main", "build_parser"]

BUNDLED = "afl"


class Fixtures:
    """A fixture list read from a CSV or the bundled dataset."""

    def __init__(
        self,
        home: np.ndarray,
        away: np.ndarray,
        outcome: np.ndarray,
        times: np.ndarray,
        source: str,
    ) -> None:
        self.home = home
        self.away = away
        self.outcome = outcome
        self.times = times
        self.source = source

    def __len__(self) -> int:
        return int(self.outcome.size)


def read_fixtures(args: argparse.Namespace) -> Fixtures:
    """Load the fixture list named by ``--data``."""
    if args.data == BUNDLED:
        matches = load_afl(as_frame=False)
        return Fixtures(
            matches.home_team, matches.away_team, matches.outcome, matches.date, BUNDLED
        )

    path = Path(args.data)
    if not path.is_file():
        raise SystemExit(f"no such file: {path} (use --data afl for the bundled archive)")
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, str]] = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise SystemExit(f"{path} contains no rows")

    missing = [
        column
        for column in (args.home_col, args.away_col, args.outcome_col, args.date_col)
        if column not in rows[0]
    ]
    if missing:
        raise SystemExit(
            f"{path} is missing column(s) {', '.join(missing)}; "
            f"available: {', '.join(rows[0])}"
        )
    return Fixtures(
        np.array([row[args.home_col] for row in rows], dtype=object),
        np.array([row[args.away_col] for row in rows], dtype=object),
        np.array([float(row[args.outcome_col]) for row in rows], dtype=np.float64),
        np.array([row[args.date_col] for row in rows], dtype="datetime64[s]"),
        str(path),
    )


def build_model(args: argparse.Namespace) -> BScoreModel:
    return BScoreModel(
        alpha=args.alpha,
        kernel=args.kernel,
        regularization=args.regularization,
        max_age=args.max_age,
    )


def emit(payload: dict[str, Any], lines: Sequence[str], args: argparse.Namespace) -> None:
    """Print either JSON or the prepared text block."""
    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("\n".join(lines))


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------
def command_info(args: argparse.Namespace) -> int:
    matches = load_afl(as_frame=False)
    data: dict[str, Any] = {
        "name": BUNDLED,
        "matches": len(matches),
        "competitors": len(matches.teams),
        "first": str(matches.date.min()),
        "last": str(matches.date.max()),
        "seasons": [int(matches.season.min()), int(matches.season.max())],
    }
    emit(
        {"version": __version__, "bundled_dataset": data},
        [
            f"bscores {__version__}",
            f"bundled dataset: {data['matches']} matches, "
            f"{data['competitors']} competitors",
            f"                 {data['first']} to {data['last']} "
            f"(seasons {data['seasons'][0]}-{data['seasons'][1]})",
        ],
        args,
    )
    return 0


def command_rate(args: argparse.Namespace) -> int:
    fixtures = read_fixtures(args)
    model = build_model(args)
    model.add_matches(fixtures.home, fixtures.away, fixtures.outcome, fixtures.times)

    board = model.leaderboard(top=args.top)
    payload = {
        "source": fixtures.source,
        "matches": len(fixtures),
        "alpha": args.alpha,
        "kernel": args.kernel,
        "ratings": [
            {
                "rank": rating.rank,
                "name": rating.name,
                "score": rating.score,
                "matches": rating.matches,
                "wins": rating.wins,
            }
            for rating in board
        ],
        "network": network_summary(model),
    }
    lines = [f"{'#':>3}  {'competitor':<24}{'B-score':>10}{'played':>8}{'won':>8}"]
    lines += [
        f"{rating.rank or 0:>3}  {rating.name:<24}{rating.score:>10.4f}"
        f"{rating.matches:>8}{rating.wins:>8.1f}"
        for rating in board
    ]
    emit(payload, lines, args)
    return 0


def command_forecast(args: argparse.Namespace) -> int:
    fixtures = read_fixtures(args)
    result = rolling_forecast(
        fixtures.home,
        fixtures.away,
        fixtures.outcome,
        fixtures.times,
        model=build_model(args),
        initial_train=args.initial_train,
        refit_every=args.refit_every,
        transform=args.transform,
    )
    metrics = result.metrics()
    payload = {
        "source": fixtures.source,
        "forecast": len(result),
        "alpha": args.alpha,
        "kernel": args.kernel,
        "metrics": metrics,
        "coefficients": result.coefficients.tolist(),
    }
    emit(
        payload,
        [
            f"forecast {len(result)} of {len(fixtures)} matches out of sample",
            f"  log-loss  {metrics['log_loss']:.4f}",
            f"  Brier     {metrics['brier_score']:.4f}",
            f"  accuracy  {metrics['accuracy']:.4f}",
        ],
        args,
    )
    return 0


def command_tune(args: argparse.Namespace) -> int:
    fixtures = read_fixtures(args)
    grid: dict[str, list[Any]] = {"alpha": list(args.alpha_grid)}
    if args.kernel_grid:
        grid["kernel"] = list(args.kernel_grid)
    if args.transform_grid:
        grid["transform"] = list(args.transform_grid)

    search = grid_search(
        fixtures.home,
        fixtures.away,
        fixtures.outcome,
        fixtures.times,
        grid=grid,
        metric=args.metric,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        refit_every=args.refit_every,
    )
    payload = {
        "source": fixtures.source,
        "metric": args.metric,
        "configurations": len(search),
        "rating_passes": search.n_rating_passes,
        "best": search.best,
        "top": search.top(args.top),
    }
    lines = [
        f"{len(search)} configurations, {search.n_rating_passes} rating passes",
        f"best {args.metric} = {search.best_score:.4f} at {search.best_params}",
        "",
        f"top {min(args.top, len(search))}:",
    ]
    keys = list(grid)
    lines += [
        "  " + "  ".join(f"{key}={row[key]}" for key in keys)
        + f"   {args.metric}={row[args.metric]:.4f}"
        for row in search.top(args.top)
    ]
    emit(payload, lines, args)
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    fixtures = read_fixtures(args)
    model = build_model(args)
    model.fit(fixtures.home, fixtures.away, fixtures.outcome, fixtures.times)

    teams = model.players
    home, away = [], []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home.append(teams[i])
            away.append(teams[j])

    season = simulate_season(
        model, home, away, n_simulations=args.simulations, seed=args.seed
    )
    expected = season.expected_points()
    first = season.finish_probability(1)
    top_n = season.top_n_probability(args.finals)
    payload = {
        "source": fixtures.source,
        "fixtures": len(home),
        "simulations": args.simulations,
        "standings": [
            {
                "name": name,
                "expected_points": expected[name],
                "p_first": first[name],
                f"p_top{args.finals}": top_n[name],
            }
            for name in expected
        ],
    }
    lines = [
        f"{len(home)} fixtures, {args.simulations} simulated seasons",
        "",
        f"  {'competitor':<24}{'exp. points':>12}{'P(first)':>10}{f'P(top {args.finals})':>11}",
    ]
    lines += [
        f"  {name:<24}{expected[name]:>12.1f}{first[name]:>10.1%}{top_n[name]:>11.1%}"
        for name in expected
    ]
    emit(payload, lines, args)
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
def _add_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        default=BUNDLED,
        help=f"CSV path, or {BUNDLED!r} for the bundled archive (default: %(default)s)",
    )
    for flag, default in (
        ("--home-col", "home_team"),
        ("--away-col", "away_team"),
        ("--outcome-col", "outcome"),
        ("--date-col", "date"),
    ):
        parser.add_argument(flag, default=default, help="CSV column (default: %(default)s)")


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alpha", type=float, default=365.0, help="half-life in days (default: %(default)s)"
    )
    parser.add_argument(
        "--kernel",
        default="hyperbolic",
        choices=["hyperbolic", "exponential", "uniform"],
        help="decay kernel (default: %(default)s)",
    )
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument(
        "--max-age", type=float, default=None, help="ignore results older than N days"
    )


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", default="text", choices=["text", "json"])


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser, exposed for testing and documentation."""
    parser = argparse.ArgumentParser(
        prog="bscores",
        description="Rate and forecast pairwise contests by eigenvector centrality.",
    )
    parser.add_argument("--version", action="version", version=f"bscores {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="package and bundled-dataset summary")
    _add_format(info)
    info.set_defaults(func=command_info)

    rate = subparsers.add_parser("rate", help="rate competitors from a fixture list")
    _add_data_options(rate)
    _add_model_options(rate)
    _add_format(rate)
    rate.add_argument("--top", type=int, default=None, help="show only the best N")
    rate.set_defaults(func=command_rate)

    forecast = subparsers.add_parser("forecast", help="expanding-window out-of-sample forecast")
    _add_data_options(forecast)
    _add_model_options(forecast)
    _add_format(forecast)
    forecast.add_argument(
        "--initial-train",
        default="0.5",
        help="match count, fraction, or date at which forecasting starts (default: %(default)s)",
    )
    forecast.add_argument("--refit-every", type=int, default=300)
    forecast.add_argument(
        "--transform", default="identity", choices=["identity", "log", "sqrt"]
    )
    forecast.set_defaults(func=command_forecast)

    tune = subparsers.add_parser("tune", help="search hyperparameters on a validation window")
    _add_data_options(tune)
    _add_format(tune)
    tune.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[21.0, 60.0, 120.0, 365.0, 730.0],
        help="half-lives to try (default: %(default)s)",
    )
    tune.add_argument(
        "--kernel-grid", nargs="*", default=["hyperbolic", "exponential"],
        choices=["hyperbolic", "exponential", "uniform"],
    )
    tune.add_argument(
        "--transform-grid", nargs="*", default=[], choices=["identity", "log", "sqrt"]
    )
    tune.add_argument("--validation-start", default="0.5")
    tune.add_argument("--validation-end", default=None)
    tune.add_argument("--refit-every", type=int, default=300)
    tune.add_argument(
        "--metric",
        default="log_loss",
        choices=["log_loss", "brier_score", "accuracy", "classification_error"],
    )
    tune.add_argument("--top", type=int, default=10)
    tune.set_defaults(func=command_tune)

    simulate = subparsers.add_parser("simulate", help="Monte Carlo a round-robin season")
    _add_data_options(simulate)
    _add_model_options(simulate)
    _add_format(simulate)
    simulate.add_argument("--simulations", type=int, default=10_000)
    simulate.add_argument("--finals", type=int, default=4, help="size of the finals cut")
    simulate.add_argument("--seed", type=int, default=None)
    simulate.set_defaults(func=command_simulate)

    return parser


def _coerce_split(value: str | None) -> Any:
    """Read a split point as a fraction, a match count, or a date."""
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and number > 1 else number


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for attribute in ("initial_train", "validation_start", "validation_end"):
        if hasattr(args, attribute):
            setattr(args, attribute, _coerce_split(getattr(args, attribute)))
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as error:
        print(f"bscores: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    raise SystemExit(main())
