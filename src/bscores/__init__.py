"""B-scores: rating competitors by eigenvector centrality.

A Python implementation of the method in

    Arcagni, A., Candila, V. & Grassi, R. (2023).
    *A new model for predicting the winner in tennis based on the eigenvector
    centrality.* Annals of Operations Research 325, 615-632.
    https://doi.org/10.1007/s10479-022-04594-7

Results form a directed network in which an arc points from the loser to the
winner, weighted by how recently the match was played.  A competitor's rating —
its **B-score** — is its entry in the principal eigenvector of that network, so
it is high when the competitors it has beaten are themselves highly rated.  The
consequence that sets the method apart from Elo and Bradley-Terry: *every* match
moves *every* rating, including those of competitors who did not play.

Quick start
-----------

>>> from bscores import BScoreModel
>>> model = BScoreModel(alpha=365.0)
>>> _ = model.rate([["Geelong"], ["Carlton"]], at="2021-03-18")
>>> _ = model.rate([["Carlton"], ["Essendon"]], at="2021-03-25")
>>> [round(p, 3) for p in model.predict_win([["Geelong"], ["Essendon"]])]
[1.0, 0.0]

On a real fixture list, calibrate the logit (Eq. 3) and forecast out of sample:

>>> from bscores import rolling_forecast
>>> from bscores.datasets import load_afl
>>> afl = load_afl(as_frame=False)
>>> result = rolling_forecast(
...     afl.home_team, afl.away_team, afl.outcome, afl.date,
...     alpha=365.0, initial_train=0.5, refit_every=300,
... )
>>> result.metrics()["log_loss"] < 0.69
True

Tune before you trust a default: ``alpha`` moves accuracy further than the
choice of rating system does.  See :mod:`bscores.tuning`.

Where things live
-----------------

======================== ====================================================
:mod:`bscores.models`    ``BScoreModel``, ``Rating``, ``RatingHistory``
:mod:`bscores.network`   ``LossNetwork`` — dated results in, ``W_t`` out
:mod:`bscores.centrality` the eigenvector solve, and its degenerate fallback
:mod:`bscores.decay`     ``Hyperbolic``, ``Exponential``, ``Uniform``, ``Window``
:mod:`bscores.weights`   margin-of-victory and match-importance arc weights
:mod:`bscores.calibration` the logit that turns ratings into probabilities
:mod:`bscores.metrics`   log-loss, Brier, Diebold-Mariano, betting ROI
:mod:`bscores.backtest`  ``rolling_forecast`` — expanding-window evaluation
:mod:`bscores.tuning`    ``grid_search`` over a validation window
:mod:`bscores.diagnostics` explain a rating, check calibration, inspect the network
:mod:`bscores.simulation` Monte Carlo season outcomes
:mod:`bscores.plotting`  matplotlib figures (optional extra)
:mod:`bscores.baselines` ``Elo``, for comparison
:mod:`bscores.datasets`  ``load_afl`` — 3533 AFL matches, bundled
======================== ====================================================
"""

from __future__ import annotations

from .backtest import BacktestResult, rolling_forecast, walk_forward
from .baselines import Elo
from .calibration import LogitCalibrator, fit_logistic, sigmoid
from .centrality import (
    EigenResult,
    bonacich_centrality,
    in_strength,
    neumann_centrality,
    out_strength,
)
from .datasets import load_afl
from .decay import DecayKernel, Exponential, Hyperbolic, Uniform, Window, as_kernel
from .diagnostics import (
    calibration_curve,
    explain_rating,
    head_to_head,
    network_summary,
    rating_churn,
    reliability_table,
    sharpness,
    upset_rate,
)
from .metrics import (
    accuracy,
    brier_score,
    classification_error,
    diebold_mariano,
    evaluate,
    log_loss,
    roi,
)
from .models import BScoreModel, Rating, RatingHistory
from .network import LossNetwork, NodeIndex
from .simulation import SeasonSimulation, simulate_season
from .tuning import TuningResult, grid_search, refit_best
from .weights import importance_weight, margin_weight

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # model
    "BScoreModel",
    "Rating",
    "RatingHistory",
    # network and centrality
    "LossNetwork",
    "NodeIndex",
    "bonacich_centrality",
    "neumann_centrality",
    "in_strength",
    "out_strength",
    "EigenResult",
    # decay kernels and arc weights
    "DecayKernel",
    "Hyperbolic",
    "Exponential",
    "Uniform",
    "Window",
    "as_kernel",
    "margin_weight",
    "importance_weight",
    # calibration
    "LogitCalibrator",
    "fit_logistic",
    "sigmoid",
    # evaluation
    "log_loss",
    "brier_score",
    "accuracy",
    "classification_error",
    "evaluate",
    "diebold_mariano",
    "roi",
    "rolling_forecast",
    "walk_forward",
    "BacktestResult",
    "Elo",
    # tuning
    "grid_search",
    "refit_best",
    "TuningResult",
    # diagnostics
    "explain_rating",
    "calibration_curve",
    "reliability_table",
    "sharpness",
    "network_summary",
    "head_to_head",
    "rating_churn",
    "upset_rate",
    # simulation
    "simulate_season",
    "SeasonSimulation",
    # data
    "load_afl",
]
