"""Throughput of the rating engine as the competition grows.

Measures the two costs that matter for a back-test: assembling ``W_t`` at every
epoch, and solving for its principal eigenvector.  Run it after touching
:mod:`bscores.network` or :mod:`bscores.centrality`.

Usage::

    python examples/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from bscores import BScoreModel
from bscores.decay import Exponential, Hyperbolic

#: Fixtures are spread over ten years of whole days, so several matches share a
#: date exactly as a real round does.  The engine solves once per distinct date,
#: which is what makes the epoch count — not the match count — the driver.
SPAN_DAYS = 3650


def synthetic(n_teams: int, n_matches: int, seed: int = 0):
    """A fixture list with a latent strength order and a 30% upset rate."""
    rng = np.random.default_rng(seed)
    teams = np.array([f"t{i}" for i in range(n_teams)])
    home_id = rng.integers(0, n_teams, n_matches)
    away_id = (home_id + 1 + rng.integers(0, n_teams - 1, n_matches)) % n_teams
    stronger_at_home = home_id < away_id
    upset = rng.random(n_matches) < 0.30
    outcome = np.where(stronger_at_home ^ upset, 1.0, 0.0)
    times = np.sort(rng.integers(0, SPAN_DAYS, n_matches).astype(float))
    return teams[home_id], teams[away_id], outcome, times


def time_it(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    print("Causal rating sweep: one centrality solve per distinct match date\n")
    print(f"{'teams':>7}{'matches':>9}{'epochs':>8}{'seconds':>10}{'us/epoch':>11}")
    for n_teams, n_matches in [(18, 2_500), (64, 10_000), (128, 25_000), (256, 50_000)]:
        home, away, outcome, times = synthetic(n_teams, n_matches)
        model = BScoreModel(alpha=365.0)
        model.add_matches(home, away, outcome, times)
        epochs = np.unique(times).size
        seconds = time_it(lambda m=model, t=times: m.score_history(t), repeats=1)
        print(
            f"{n_teams:>7}{n_matches:>9}{epochs:>8}{seconds:>10.2f}"
            f"{1e6 * seconds / epochs:>11.0f}"
        )

    print("\nWarm starting the eigenvector solve\n")
    home, away, outcome, times = synthetic(64, 10_000)
    for warm in (False, True):
        model = BScoreModel(alpha=365.0, warm_start=warm)
        model.add_matches(home, away, outcome, times)
        seconds = time_it(lambda m=model, t=times: m.score_history(t), repeats=1)
        print(f"  warm_start={str(warm):<6} {seconds:.2f}s")

    print("\nDecay kernel: memoryless kernels stream instead of rebuilding\n")
    home, away, outcome, times = synthetic(64, 25_000)
    for kernel in (Hyperbolic(365.0), Exponential(365.0)):
        model = BScoreModel(kernel=kernel)
        model.add_matches(home, away, outcome, times)
        streams = model.network.can_stream(np.unique(times))
        seconds = time_it(lambda m=model, t=times: m.score_history(t), repeats=1)
        print(f"  {kernel!r:<24} streaming={str(streams):<6} {seconds:.2f}s")

    print("\nTruncating the kernel tail with max_age\n")
    home, away, outcome, times = synthetic(64, 25_000)
    for max_age in (None, 730.0, 365.0):
        model = BScoreModel(alpha=365.0, max_age=max_age)
        model.add_matches(home, away, outcome, times)
        seconds = time_it(lambda m=model, t=times: m.score_history(t), repeats=1)
        print(f"  max_age={str(max_age):<6} {seconds:.2f}s")


if __name__ == "__main__":
    main()
