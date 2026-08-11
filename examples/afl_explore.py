"""Exploring AFL ratings with the diagnostics toolkit.

Walks through the questions you actually want to ask of a rating system:

* who is rated where, and *why*;
* is the network healthy enough to rate on;
* are the forecast probabilities honest;
* how much does the order churn;
* what are the chances of finishing top four.

Usage::

    python examples/afl_explore.py [--alpha 120] [--plot out/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bscores import BScoreModel
from bscores.datasets import load_afl
from bscores.diagnostics import (
    explain_rating,
    head_to_head,
    network_summary,
    rating_churn,
    reliability_table,
    sharpness,
    upset_rate,
)
from bscores.simulation import simulate_season
from bscores.tuning import AFL_TUNED
from bscores.weights import margin_weight

TEST_START = "2023-01-01"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def build(matches, alpha: float) -> BScoreModel:
    """A model using the tuned settings, fitted on the whole archive."""
    model = BScoreModel(
        alpha=alpha,
        kernel=AFL_TUNED["kernel"],
        regularization=AFL_TUNED["regularization"],
    )
    weights = margin_weight(matches.margin, scheme="linear", scale=24.0, cap=3.0)
    return model.fit(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        weights=weights,
        transform=AFL_TUNED["transform"],
        model_draws=True,
    )


def show_leaderboard(model: BScoreModel) -> None:
    rule("Leaderboard")
    print(f"{'#':>2}  {'team':<18}{'B-score':>9}{'played':>8}{'won':>8}")
    for rating in model.leaderboard():
        print(
            f"{rating.rank or 0:>2}  {rating.name:<18}{rating.score:>9.4f}"
            f"{rating.matches:>8}{rating.wins:>8.1f}"
        )


def show_explanations(model: BScoreModel) -> None:
    rule("Why is the top side rated where it is?")
    leader = model.leaderboard(top=1)[0]
    print(f"{leader.name} scores {leader.score:.4f}.  That decomposes exactly into:\n")
    print(f"  {'beat':<18}{'share':>8}{'their rating':>14}{'arc weight':>12}")
    for part in explain_rating(model, leader.name, top=8):
        print(
            f"  {part.opponent:<18}{part.share:>7.1%}{part.opponent_score:>14.4f}"
            f"{part.weight:>12.2f}"
        )
    print(
        "\n  Every row is a competitor this side has beaten, weighted by how\n"
        "  recently and by how highly that competitor is itself rated.  The\n"
        "  shares sum to 1 because the eigenvector equation says a rating *is*\n"
        "  the weighted sum of the ratings pointing at it."
    )


def show_network(model: BScoreModel) -> None:
    rule("Network health")
    for key, value in network_summary(model).items():
        if isinstance(value, float):
            print(f"  {key:<18} {value:.4f}")
        else:
            print(f"  {key:<18} {value}")


def show_calibration(matches, model: BScoreModel) -> None:
    rule(f"Are the probabilities honest?  ({TEST_START} onwards, out of sample)")
    from bscores.backtest import rolling_forecast

    weights = margin_weight(matches.margin, scheme="linear", scale=24.0, cap=3.0)
    result = rolling_forecast(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        model=BScoreModel(
            alpha=AFL_TUNED["alpha"],
            kernel=AFL_TUNED["kernel"],
            regularization=AFL_TUNED["regularization"],
        ),
        initial_train=TEST_START,
        weights=weights,
        transform=AFL_TUNED["transform"],
    )
    print(reliability_table(result.outcome, result.probability, bins=6, strategy="quantile"))
    print(
        f"\n  sharpness  {sharpness(result.probability):.3f}"
        "  (0 = never commits, 1 = always certain)"
    )
    upsets = upset_rate(result.outcome, result.probability, threshold=0.6)
    print(
        f"  of {int(upsets['n_favoured'])} matches called at better than 60%, "
        f"{upsets['upset_rate']:.1%} were upsets against {upsets['expected']:.1%} expected"
    )
    return result


def show_head_to_head(matches, model: BScoreModel) -> None:
    rule("Head to head, top four sides, whole archive")
    top = [r.name for r in model.leaderboard(top=4)]
    table = head_to_head(
        matches.home_team, matches.away_team, matches.outcome, competitors=top
    )
    print(f"  {'':<18}" + "".join(f"{name[:10]:>12}" for name in top))
    for i, name in enumerate(top):
        cells = "".join(
            "           -" if i == j else f"{table['wins'][i, j]:>7.1f}/{table['played'][i, j]:<4}"
            for j in range(len(top))
        )
        print(f"  {name:<18}{cells}")
    print("\n  cells are wins / meetings, row team against column team")


def show_churn(matches, model: BScoreModel) -> None:
    rule("How much does the order actually move?")
    dates = np.unique(matches.date)
    history = model.score_history(dates)
    for label, alpha in (("30 days", 30.0), ("120 days (tuned)", 120.0), ("1095 days", 1095.0)):
        other = BScoreModel(alpha=alpha, kernel=AFL_TUNED["kernel"])
        other.add_matches(matches.home_team, matches.away_team, matches.outcome, matches.date)
        churn = rating_churn(other.score_history(dates), top=8)
        print(
            f"  half-life {label:<18} mean rank change {churn['rank_change'].mean():.3f}"
            f"   top-8 entries {int(churn['entered_top'].sum())}"
        )
    print("\n  A shorter memory tracks form and churns; a longer one is steadier.")
    return history


def show_simulation(matches, model: BScoreModel) -> None:
    rule("Simulating the rest of a season")
    teams = model.players
    home, away = [], []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home.append(teams[i])
            away.append(teams[j])
    season = simulate_season(model, home, away, n_simulations=20_000, seed=0)
    print(f"  {len(home)} fixtures, {season.n_simulations} simulated seasons\n")
    print(f"  {'team':<18}{'exp. points':>12}{'P(first)':>10}{'P(top 4)':>10}")
    expected = season.expected_points()
    first = season.finish_probability(1)
    top4 = season.top_n_probability(4)
    for name in expected:
        print(f"  {name:<18}{expected[name]:>12.1f}{first[name]:>10.1%}{top4[name]:>10.1%}")
    print(
        "\n  Ratings are held fixed across each simulated season, so the spread\n"
        "  is a little narrower than reality would give."
    )


def save_plots(directory: Path, matches, model, history, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bscores.decay import Exponential, Hyperbolic
    from bscores.plotting import (
        plot_backtest,
        plot_calibration,
        plot_decay,
        plot_network,
        plot_ratings,
    )

    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "ratings": lambda: plot_ratings(history, top=6, x="round", schedule=matches),
        "calibration": lambda: plot_calibration(result.outcome, result.probability, bins=8),
        "network": lambda: plot_network(model),
        "backtest": lambda: plot_backtest(result),
        "decay": lambda: plot_decay([Hyperbolic(120.0), Exponential(120.0), Hyperbolic(365.0)]),
    }
    for name, build_figure in figures.items():
        ax = build_figure()
        path = directory / f"afl_{name}.png"
        ax.figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=AFL_TUNED["alpha"])
    parser.add_argument("--plot", type=Path, default=None, help="directory to save figures in")
    args = parser.parse_args()

    matches = load_afl(as_frame=False)
    print(repr(matches))
    model = build(matches, args.alpha)

    show_leaderboard(model)
    show_explanations(model)
    show_network(model)
    result = show_calibration(matches, model)
    show_head_to_head(matches, model)
    history = show_churn(matches, model)
    show_simulation(matches, model)

    if args.plot is not None:
        rule("Figures")
        save_plots(args.plot, matches, model, history, result)


if __name__ == "__main__":
    main()
