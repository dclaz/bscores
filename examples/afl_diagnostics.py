"""B-score diagnostics on the AFL archive.

Runs the paper's evaluation protocol over 2534 AFL matches (2009-2022):
ratings, an out-of-sample forecast comparison against Elo and the home-ground
base rate, a Diebold-Mariano test on the loss differences, a sweep over the
memory parameter, and the betting ROI surface from Definition 1.

Usage::

    python examples/afl_diagnostics.py [--alpha 365] [--quick]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bscores import (
    BScoreModel,
    Elo,
    diebold_mariano,
    evaluate,
    roi,
    rolling_forecast,
    sweep_alpha,
)
from bscores.datasets import load_afl

ALPHA_GRID = [7.0, 14.0, 21.0, 30.0, 60.0, 90.0, 180.0, 365.0, 730.0, 1825.0, 3650.0]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def show_ratings(matches, alpha: float) -> None:
    rule(f"Ratings as at {matches.date.max()} (alpha = {alpha:g} days)")
    model = BScoreModel(alpha=alpha)
    model.add_matches(matches.home_team, matches.away_team, matches.outcome, matches.date)
    print(f"{'#':>2}  {'team':<18}{'B-score':>9}{'ordinal':>9}{'played':>8}{'won':>7}")
    for rating in model.leaderboard():
        print(
            f"{rating.rank or 0:>2}  {rating.name:<18}{rating.score:>9.4f}"
            f"{rating.ordinal():>9.1f}{rating.matches:>8}{rating.wins:>7.1f}"
        )
    print(f"\n||scores||_2 = {np.linalg.norm(model.scores()):.6f}  (unit by construction)")


def show_forecasts(matches, alpha: float) -> tuple[object, np.ndarray]:
    rule(f"Out-of-sample forecasts (alpha = {alpha:g}, refit every 300 matches)")
    started = time.perf_counter()
    result = rolling_forecast(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        alpha=alpha,
        initial_train=0.5,
        refit_every=300,
    )
    elapsed = time.perf_counter() - started

    start = len(matches) - len(result)
    elo_probability = Elo().run(matches.home_team, matches.away_team, matches.outcome)[start:]
    base_rate = np.full(len(result), matches.outcome[:start].mean())

    rows = {
        "B-score": result.probability,
        "Elo": elo_probability,
        "home base rate": base_rate,
    }
    print(f"test set: {len(result)} matches from {matches.date[start]}")
    print(f"{'model':<16}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}")
    for name, probability in rows.items():
        scores = evaluate(result.outcome, probability)
        print(
            f"{name:<16}{scores['log_loss']:>10.4f}{scores['brier_score']:>9.4f}"
            f"{scores['accuracy']:>10.4f}"
        )

    rule("Diebold-Mariano tests (negative favours B-scores)")
    bscore_loss = result.losses("log_loss")
    for name, probability in rows.items():
        if name == "B-score":
            continue
        other = -(
            result.outcome * np.log(np.clip(probability, 1e-15, 1 - 1e-15))
            + (1 - result.outcome) * np.log(1 - np.clip(probability, 1e-15, 1 - 1e-15))
        )
        statistic, p_value = diebold_mariano(bscore_loss, other)
        print(f"  vs {name:<16} DM = {statistic:>7.3f}   p = {p_value:.4f}")

    rule("Fitted logit coefficients (Eq. 3)")
    print(f"{'refit at':>9}{'beta_0':>10}{'beta_1':>10}{'beta_2':>10}")
    for index, beta in zip(result.refit_at, result.coefficients):
        print(f"{int(result.train_size[index]):>9}{beta[0]:>10.4f}{beta[1]:>10.4f}{beta[2]:>10.4f}")

    print(f"\nwall clock: {elapsed:.2f}s for {len(matches)} matches")
    return result, base_rate


def show_alpha_sweep(matches) -> None:
    rule("Memory parameter sweep (out-of-sample log-loss)")
    rows = sweep_alpha(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        ALPHA_GRID,
        initial_train=0.5,
        refit_every=300,
    )
    print(f"{'alpha (days)':>13}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}")
    for row in sorted(rows, key=lambda r: r["alpha"]):
        marker = "  <- best" if row is rows[0] else ""
        print(
            f"{row['alpha']:>13.0f}{row['log_loss']:>10.4f}{row['brier_score']:>9.4f}"
            f"{row['accuracy']:>10.4f}{marker}"
        )
    print(
        "\nThe paper uses alpha = 365 to mirror the 52-week ATP/WTA ranking window.\n"
        "An AFL season is 22 rounds, and form turns over faster than that."
    )


def show_betting(matches, result) -> None:
    rule("Betting ROI, Definition 1 (flat 1-unit stakes, best available odds)")
    start = len(matches) - len(result)
    home_odds = matches.home_odds[start:]
    away_odds = matches.away_odds[start:]

    print(f"{'r':>6}{'q':>6}{'bets':>7}{'ROI':>9}")
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
        for min_implied in (0.0, 0.3):
            home = roi(
                result.outcome,
                result.probability,
                home_odds,
                threshold=threshold,
                min_implied=min_implied,
            )
            away = roi(
                1.0 - result.outcome,
                1.0 - result.probability,
                away_odds,
                threshold=threshold,
                min_implied=min_implied,
            )
            bets = home["n_bets"] + away["n_bets"]
            if bets == 0:
                continue
            profit = home["profit"] + away["profit"]
            print(f"{threshold:>6.2f}{min_implied:>6.2f}{int(bets):>7}{profit / bets:>9.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=365.0, help="memory parameter, in days")
    parser.add_argument("--quick", action="store_true", help="skip the alpha sweep")
    args = parser.parse_args()

    matches = load_afl(as_frame=False)
    print(f"{matches!r}")

    show_ratings(matches, args.alpha)
    result, _ = show_forecasts(matches, args.alpha)
    if not args.quick:
        show_alpha_sweep(matches)
    show_betting(matches, result)


if __name__ == "__main__":
    main()
