"""Bayesian hyperparameter search on the AFL archive, both sides.

``examples/afl_tuning.py`` grid-searches a box of parameters.  This searches a
*conditional* space with Optuna's TPE sampler — window widths only sampled for
window kernels, margin-scheme parameters only when margin weighting is on — and
gives Elo an identical budget with the identical sampler, so the comparison
measures the rating methods rather than how hard each was searched.

Usage::

    python examples/afl_optuna.py [--trials 600] [--warmup 120]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bscores.backtest import rolling_forecast
from bscores.baselines import Elo
from bscores.datasets import load_afl
from bscores.metrics import diebold_mariano, evaluate
from bscores.search import _arc_weights, config_to_model, optuna_search, optuna_search_elo

VALIDATION_START = "2019-01-01"
TEST_START = "2023-01-01"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def describe(params: dict) -> str:
    """One-line rendering of a configuration."""
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=600, help="trials per model")
    parser.add_argument(
        "--warmup",
        type=int,
        default=120,
        help="random trials before TPE takes over (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matches = load_afl(as_frame=False)
    print(matches)

    rule(f"Searching {args.trials} trials each, {args.warmup} random before TPE")
    started = time.perf_counter()
    bscore = optuna_search(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        margins=matches.margin,
        finals=matches.final,
        validation_start=VALIDATION_START,
        validation_end=TEST_START,
        n_trials=args.trials,
        n_startup_trials=args.warmup,
        seed=args.seed,
    )
    elo = optuna_search_elo(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        validation_start=VALIDATION_START,
        validation_end=TEST_START,
        n_trials=args.trials,
        n_startup_trials=args.warmup,
        seed=args.seed,
    )
    print(f"  {2 * args.trials} trials in {time.perf_counter() - started:.0f}s")
    print(f"\n  B-score  validation {bscore.best_score:.4f}\n    {describe(bscore.best_params)}")
    print(f"\n  Elo      validation {elo.best_score:.4f}\n    {describe(elo.best_params)}")

    rule("Best five B-score trials (validation)")
    keys = ("alpha", "kernel", "transform", "regularization", "margin_scheme")
    for row in bscore.top(5):
        shown = describe({k: row[k] for k in keys if k in row})
        print(f"  {row['log_loss']:.4f}  {shown}")

    rule(f"Held-out test window ({TEST_START} onwards) — scored once")
    config = bscore.best_params
    tuned = rolling_forecast(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        model=config_to_model(config),
        initial_train=TEST_START,
        weights=_arc_weights(config, matches.margin.astype(float), matches.final),
        symmetric=bool(config.get("symmetric", False)),
        transform=config.get("transform", "identity"),
    )
    test = matches.date >= np.datetime64(TEST_START)
    observed = matches.outcome[test]
    elo_probability = Elo(**elo.best_params).run(
        matches.home_team, matches.away_team, matches.outcome
    )[test]

    print(f"  test set: {int(test.sum())} matches\n")
    print(f"  {'model':<26}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}")
    rows = (("B-score, TPE-tuned", tuned.probability), ("Elo, TPE-tuned", elo_probability))
    for name, probability in rows:
        scores = evaluate(observed, probability)
        print(
            f"  {name:<26}{scores['log_loss']:>10.4f}{scores['brier_score']:>9.4f}"
            f"{scores['accuracy']:>10.4f}"
        )

    def pointwise(probability):
        clipped = np.clip(probability, 1e-15, 1 - 1e-15)
        return -(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))

    statistic, p_value = diebold_mariano(
        pointwise(tuned.probability), pointwise(elo_probability)
    )
    print(
        f"\n  Diebold-Mariano, B-score vs Elo (negative favours B-score):"
        f"  {statistic:+.3f}  p = {p_value:.4f}"
    )
    print(
        "\n  Both sides had the same sampler, the same budget and the same\n"
        "  validation window, so what is left is the rating method — and on\n"
        "  828 matches it is not enough to separate them."
    )


if __name__ == "__main__":
    main()
