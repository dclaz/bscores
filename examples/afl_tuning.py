"""Hyperparameter search on the bundled AFL data.

Reproduces how :data:`bscores.tuning.AFL_TUNED` was chosen.  The protocol is the
point: everything is selected on a validation window and scored once on a test
window that the search never saw.

.. code-block:: text

    2009-06 .. 2018-09   warm-up      1922 matches, model builds a history
    2019-03 .. 2022-09   validation    783 matches, the search runs here
    2023-03 .. 2026-08   test          828 matches, reported once at the end

Usage::

    python examples/afl_tuning.py [--stage all|1|2|3|final]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bscores import Elo, evaluate, rolling_forecast, tune_elo
from bscores.datasets import load_afl
from bscores.metrics import diebold_mariano
from bscores.tuning import grid_search, refit_best
from bscores.weights import importance_weight, margin_weight

VALIDATION_START = "2019-01-01"
TEST_START = "2023-01-01"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def weight_options(matches) -> dict[str, np.ndarray]:
    """Candidate per-match arc weightings for the search to choose between."""
    options = {
        "uniform": np.ones(len(matches)),
        "finals": importance_weight(matches.final, weight=2.0),
        "mov": margin_weight(matches.margin, scheme="linear", scale=24.0, cap=3.0),
    }
    for scheme in ("sqrt", "log"):
        options[f"mov_{scheme}"] = margin_weight(
            matches.margin, scheme=scheme, scale=24.0, cap=3.0
        )
    return options


def search(matches, grid, options=None, **kwargs):
    started = time.perf_counter()
    result = grid_search(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        grid=grid,
        weight_options=options,
        validation_start=VALIDATION_START,
        validation_end=TEST_START,
        **kwargs,
    )
    print(
        f"{len(result)} configurations, {result.n_rating_passes} rating passes, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return result


def show(result, keys, n=8) -> None:
    for row in result.top(n):
        shown = "  ".join(f"{k}={row[k]}" for k in keys)
        print(f"  {shown:<62} log-loss={row['log_loss']:.4f}  acc={row['accuracy']:.4f}")


def stage_one(matches) -> None:
    rule("Stage 1 — kernel shape and memory length")
    result = search(
        matches,
        {
            "alpha": [7.0, 14.0, 21.0, 30.0, 60.0, 90.0, 120.0, 180.0, 365.0, 730.0, 3650.0],
            "kernel": ["hyperbolic", "exponential"],
            "transform": ["identity", "log", "sqrt"],
        },
    )
    show(result, ("alpha", "kernel", "transform"))
    print("\n  best achievable per value:")
    for key in ("kernel", "transform"):
        profile = "   ".join(f"{v}={s:.4f}" for v, s in result.sensitivity(key))
        print(f"    {key:<11} {profile}")


def stage_two(matches) -> None:
    rule("Stage 2 — what each result is worth")
    options = weight_options(matches)
    result = search(
        matches,
        {
            "alpha": [60.0, 90.0, 120.0, 150.0, 180.0],
            "kernel": ["exponential"],
            "transform": ["sqrt"],
            "weights": list(options),
            "draw_weight": [0.0, 0.5, 1.0],
            "symmetric": [False, True],
        },
        options,
    )
    show(result, ("alpha", "weights", "draw_weight", "symmetric"))
    print("\n  best achievable per value:")
    for key in ("weights", "draw_weight", "symmetric"):
        profile = "   ".join(f"{v}={s:.4f}" for v, s in result.sensitivity(key))
        print(f"    {key:<12} {profile}")


def stage_three(matches) -> None:
    rule("Stage 3 — is the hyperbolic kernel's problem really its tail?")
    result = search(
        matches,
        {
            "alpha": [21.0, 60.0, 120.0, 240.0],
            "kernel": ["hyperbolic", "exponential"],
            "transform": ["sqrt"],
            "max_age": [None, 365.0, 730.0, 1460.0],
        },
    )

    def best(**match):
        rows = [
            row
            for row in result.rows
            if all(row[key] == value for key, value in match.items())
        ]
        return min(rows, key=lambda row: row["log_loss"])

    uncapped = best(kernel="hyperbolic", max_age=None)
    capped = min(
        (row for row in result.rows if row["kernel"] == "hyperbolic" and row["max_age"]),
        key=lambda row: row["log_loss"],
    )
    exponential = best(kernel="exponential", max_age=None)

    print(f"  {'configuration':<44}{'log-loss':>10}")
    for label, row in (
        (f"hyperbolic, no cut-off (alpha={row_alpha(uncapped)})", uncapped),
        (f"hyperbolic, max_age={capped['max_age']:.0f} (alpha={row_alpha(capped)})", capped),
        (f"exponential, no cut-off (alpha={row_alpha(exponential)})", exponential),
    ):
        print(f"  {label:<44}{row['log_loss']:>10.4f}")

    print(
        f"\n  Truncating the tail is worth "
        f"{uncapped['log_loss'] - capped['log_loss']:.4f} of log-loss to the\n"
        f"  hyperbolic kernel, taking it to within "
        f"{abs(capped['log_loss'] - exponential['log_loss']):.4f} of the exponential\n"
        "  one.  The heavy tail is the whole difference: with no cut-off a decade\n"
        "  of stale results still carries weight, and an archive holds thousands."
    )


def row_alpha(row) -> str:
    return f"{row['alpha']:.0f}"


def final(matches) -> dict:
    rule("Final search on the validation window")
    options = weight_options(matches)
    result = search(
        matches,
        {
            "alpha": [80.0, 100.0, 120.0, 150.0, 180.0],
            "kernel": ["exponential"],
            "transform": ["identity", "sqrt"],
            "weights": ["uniform", "mov"],
            "regularization": [0.0, 1e-3, 1e-2, 3e-2],
        },
        options,
    )
    show(result, ("alpha", "transform", "weights", "regularization"))
    chosen = result.best_params
    print(f"\n  chosen: {chosen}")

    rule(f"Held-out test window ({TEST_START} onwards) — scored once")
    tuned = refit_best(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        chosen,
        weight_options=options,
        test_start=TEST_START,
    )
    reference = {
        "B-score, tuned": tuned,
        "B-score, paper alpha=365": rolling_forecast(
            matches.home_team,
            matches.away_team,
            matches.outcome,
            matches.date,
            alpha=365.0,
            initial_train=TEST_START,
        ),
    }
    n_test = len(tuned)
    start = len(matches) - n_test
    # Give Elo the same tuning budget on the same validation window.  A B-score
    # model gets home advantage for free — the calibrating logit fits an
    # intercept — while Elo has to be told about it, so a default Elo is not a
    # like-for-like baseline no matter how faithful it is to the paper.
    elo_params = tune_elo(
        matches.home_team,
        matches.away_team,
        matches.outcome,
        matches.date,
        validation_start=VALIDATION_START,
        validation_end=TEST_START,
    )
    elo_tuned = Elo(**elo_params).run(
        matches.home_team, matches.away_team, matches.outcome
    )[start:]
    elo_default = Elo().run(matches.home_team, matches.away_team, matches.outcome)[start:]
    base_rate = np.full(n_test, matches.outcome[:start].mean())

    print(f"  test set: {n_test} matches\n")
    print(f"  {'model':<28}{'log-loss':>10}{'Brier':>9}{'accuracy':>10}")
    for name, res in reference.items():
        scores = res.metrics()
        print(
            f"  {name:<28}{scores['log_loss']:>10.4f}{scores['brier_score']:>9.4f}"
            f"{scores['accuracy']:>10.4f}"
        )
    for name, probability in (
        ("Elo, tuned", elo_tuned),
        ("Elo, paper defaults", elo_default),
        ("home base rate", base_rate),
    ):
        scores = evaluate(matches.outcome[start:], probability)
        print(
            f"  {name:<28}{scores['log_loss']:>10.4f}{scores['brier_score']:>9.4f}"
            f"{scores['accuracy']:>10.4f}"
        )

    settings = ", ".join(f"{k}={v:g}" for k, v in elo_params.items())
    print(f"\n  Elo was tuned on the same validation window: {settings}")

    print("\n  Diebold-Mariano vs the tuned model (negative favours tuned):")
    observed = matches.outcome[start:]

    def pointwise(probability):
        clipped = np.clip(probability, 1e-15, 1 - 1e-15)
        return -(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))

    tuned_loss = tuned.losses("log_loss")
    comparisons = {
        "Elo, tuned": pointwise(elo_tuned),
        "Elo, paper defaults": pointwise(elo_default),
        "paper alpha=365": reference["B-score, paper alpha=365"].losses("log_loss"),
        "home base rate": pointwise(base_rate),
    }
    for name, other in comparisons.items():
        statistic, p_value = diebold_mariano(tuned_loss, other)
        print(f"    vs {name:<18} DM = {statistic:>7.3f}   p = {p_value:.4f}")

    validation_score = result.best_score
    test_score = tuned.metrics()["log_loss"]
    direction = "better" if test_score < validation_score else "worse"
    print(
        f"\n  The tuned model scores {validation_score:.4f} on validation and "
        f"{test_score:.4f} on test —\n"
        f"  {direction} out of sample, because the two windows differ in how\n"
        "  predictable they happened to be, not because the model changed.\n"
        "  Only the gaps between models within one window mean anything."
    )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "1", "2", "3", "final"])
    args = parser.parse_args()

    matches = load_afl(as_frame=False)
    print(repr(matches))
    if args.stage in ("all", "1"):
        stage_one(matches)
    if args.stage in ("all", "2"):
        stage_two(matches)
    if args.stage in ("all", "3"):
        stage_three(matches)
    if args.stage in ("all", "final"):
        final(matches)


if __name__ == "__main__":
    main()
